"""Live engine: turns the OpenF1 sample stream into smooth car positions a few seconds behind real time.

Same method as build_replay.py (speed telemetry integrated along the line, corrected towards the position feed;
lateral offset smoothed; cars follow the computed racing line), but every step only uses samples that have
already arrived. The picture is shown DELAY seconds behind the newest data, so the smoothing still has a few
seconds of "future" around each frame.

The track model (track.npz) comes from build_replay.py run with TRACK_ONLY=1 on an earlier session of the same
weekend (practice or qualifying), so it is ready before the race starts.
"""
import numpy as np

def smooth_at(t, y, tg, sigma, K):
    """local quadratic fit of y(t) evaluated at tg (Gaussian weights), one-sided near the newest sample"""
    c = np.searchsorted(t, tg)
    idx = np.clip(c[:, None] + np.arange(-K // 2, K // 2)[None], 0, len(t) - 1)
    u = t[idx] - tg[:, None]
    w = np.exp(-0.5 * (u / sigma) ** 2) + 1e-9
    A = np.stack([np.ones_like(u), u, u * u], -1)
    M = np.einsum("nk,nki,nkj->nij", w, A, A) + np.eye(3) * 1e-6
    b = np.einsum("nk,nki,nk->ni", w, A, y[idx])
    out = np.linalg.solve(M, b[..., None])[:, 0, 0]
    out[tg <= t[0]] = y[0]; out[tg >= t[-1]] = y[-1]
    return out

class LiveEngine:
    WINDOW = 40.0   # seconds of history used for each step

    def __init__(self, track_npz, nums, hz=4, delay=4.0):
        T = np.load(track_npz)
        self.LC, self.LT, self.LN, self.rl_off = T["LC"], T["LT"], T["LN"], T["rl_off"]
        self.S, self.R, self.TT = float(T["xf_s"]), T["xf_R"], T["xf_t"]
        self.NL = len(self.LC); self.hz = hz; self.delay = delay; self.nums = list(nums)
        # raster lookup: nearest line point for any position within 30 m
        RAD = 30; LC = self.LC
        self.g0 = LC.min(0) - RAD - 2
        self.gw, self.gh = (np.ceil((LC.max(0) - LC.min(0)) + 2 * RAD + 4)).astype(int)
        bd = np.full((self.gh, self.gw), np.inf); bi = np.zeros((self.gh, self.gw), int)
        oz, ox = np.mgrid[-RAD:RAD + 1, -RAD:RAD + 1]
        for i, p in enumerate(LC):
            cx, cz = int(p[0] - self.g0[0]), int(p[1] - self.g0[1])
            d2 = (self.g0[0] + cx + ox + 0.5 - p[0]) ** 2 + (self.g0[1] + cz + oz + 0.5 - p[1]) ** 2
            sl = (slice(cz - RAD, cz + RAD + 1), slice(cx - RAD, cx + RAD + 1))
            better = d2 < bd[sl]; bd[sl][better] = d2[better]; bi[sl][better] = i
        self.bd, self.bi = bd, bi
        self.loc = {n: [] for n in nums}; self.spd = {n: [] for n in nums}
        self.last_s = {}; self.last_xy = {}
        self.t_start = None; self.s_start = {}
        self.t_out = None

    # ---------- input ----------
    def add_location(self, n, t, x, y):
        if x or y: self.loc[n].append((t, *((self.S * (self.R @ np.array([x, y], float))) + self.TT)))
    def add_car_data(self, n, t, speed, throttle, brake):
        if throttle is not None and throttle > 100: return   # frozen packet marker
        self.spd[n].append((t, (speed or 0) / 3.6))
    def add_race_control(self, t, message):
        if self.t_start is None and (message.startswith("RACE START") or message.startswith("STANDING START")): self.t_start = t

    # ---------- geometry ----------
    def on_line(self, p):
        ix = np.clip((p[:, 0] - self.g0[0]).astype(int), 0, self.gw - 1); iz = np.clip((p[:, 1] - self.g0[1]).astype(int), 0, self.gh - 1)
        i = self.bi[iz, ix]; ok = self.bd[iz, ix] < 28 ** 2
        d = p - self.LC[i]
        return i + (d * self.LT[i]).sum(1), (d * self.LN[i]).sum(1), ok
    def line_point(self, s, lat):
        s = np.mod(s, self.NL); i = np.floor(s).astype(int) % self.NL; f = (s - np.floor(s))[:, None]
        base = self.LC[i] * (1 - f) + self.LC[(i + 1) % self.NL] * f
        nrm = self.LN[i] * (1 - f) + self.LN[(i + 1) % self.NL] * f
        return base + nrm * lat[:, None]

    # ---------- output ----------
    def step(self, now):
        """frames (times, {n: xy array}) between the previous call and now - delay"""
        t_end = now - self.delay
        if self.t_out is None: self.t_out = t_end - 1.0
        k0 = int(np.floor(self.t_out * self.hz)) + 1; k1 = int(np.floor(t_end * self.hz))
        if k1 < k0: return np.zeros(0), {}
        tg = np.arange(k0, k1 + 1) / self.hz
        self.t_out = tg[-1]
        out = {}
        for n in self.nums:
            xy = self._car(n, tg, now)
            if xy is not None: out[n] = xy
        return tg, out

    def _car(self, n, tg, now):
        lo = tg[0] - self.WINDOW
        L = [r for r in self.loc[n] if r[0] >= lo]
        V = [r for r in self.spd[n] if r[0] >= lo - 2]
        self.loc[n] = L; self.spd[n] = V
        if len(L) < 6:
            return np.repeat(self.last_xy[n][None], len(tg), 0) if n in self.last_xy else None
        La = np.array(L); t, p = La[:, 0], La[:, 1:3]
        o = np.argsort(t); t, p = t[o], p[o]
        s, lat, ok = self.on_line(p)
        if ok.sum() < 6:   # far from the line (garage): just smooth x/z
            xy = np.column_stack([smooth_at(t, p[:, 0], tg, 0.8, 28), smooth_at(t, p[:, 1], tg, 0.8, 28)])
            self.last_xy[n] = xy[-1]; return xy
        t, s, lat = t[ok], s[ok], lat[ok]
        prog = np.unwrap(s * 2 * np.pi / self.NL) * self.NL / (2 * np.pi)
        # integrated speed on a fine grid covering the window
        gq = np.arange(t[0], max(t[-1], tg[-1]) + 0.25, 0.25)
        if len(V) >= 4:
            Va = np.array(V); Va = Va[np.argsort(Va[:, 0])]
            v = np.interp(gq, Va[:, 0], Va[:, 1])
        else:
            v = np.gradient(np.interp(gq, t, prog), gq)
        pv = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(gq))])
        corr = prog - np.interp(t, gq, pv)
        corr = np.array([np.median(corr[max(0, q - 8):q + 9]) for q in range(len(corr))])
        pg = np.interp(tg, gq, pv) + smooth_at(t, corr, tg, 2.5, 72)
        pg = np.maximum.accumulate(pg)
        sg = np.mod(pg, self.NL)
        # never step backwards between calls
        if n in self.last_s:
            d = (sg - self.last_s[n] + self.NL / 2) % self.NL - self.NL / 2
            sg = np.where((d < 0) & (d > -40), self.last_s[n], sg)
        latg = smooth_at(t, lat, tg, 0.8, 28)
        w = np.clip(1 - (np.abs(latg) - 3) / 5, 0, 1)
        # grid and first metres: stay on the feed's own line (the stagger is drawn by the page)
        if self.t_start is None or tg[0] < self.t_start: w[:] = 0
        else:
            if n not in self.s_start: self.s_start[n] = sg[0]
            ds = (sg - self.s_start[n]) % self.NL
            w *= np.clip((ds - 150) / 450, 0, 1) * (ds < self.NL - 50) + (ds >= self.NL - 50)
            if tg[0] - self.t_start > 120: w = np.clip(1 - (np.abs(latg) - 3) / 5, 0, 1)
        lat_f = latg + w * np.interp(sg, np.arange(self.NL), self.rl_off, period=self.NL)
        xy = self.line_point(sg, lat_f)
        self.last_s[n] = sg[-1]; self.last_xy[n] = xy[-1]
        return xy
