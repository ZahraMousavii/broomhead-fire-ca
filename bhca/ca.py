"""Cellular-automaton fire spread: Rothermel local behaviour + Huygens elliptical wavelets.

Every burnable cell carries a local fire-behaviour ellipse (head-fire ROS, length-to-breadth ratio, direction) from the
Rothermel model. A burning cell offers fire to its 16 neighbours (8 Moore + 8 knight moves); the fire needs
distance / R(theta) minutes to cross an edge, R(theta) being the Anderson ellipse radius in that direction. A cell
ignites when the first offer arrives, so the wavelets' envelope is the Huygens fire front (Finney 2002 minimum
travel time, Richards 1990 / FARSITE elliptical wavelets).

Two solvers share the same rule:
  HuygensCA.advance / run   time-stepped synchronous CA (states, snapshots, animation, time-varying weather)
  solve_arrival_mtt         scipy shortest-path form of the same rule, used inside calibration loops
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import rothermel as rt
from .moisture import simard_emc

UNBURNED, BURNING, BURNT, UNBURNABLE = 0, 1, 2, 3

_MOORE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_KNIGHT = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
OFFSETS = np.array(_MOORE + _KNIGHT)                                   # (dr, dc), row grows southwards
AZIMUTH = np.degrees(np.arctan2(OFFSETS[:, 1], -OFFSETS[:, 0])) % 360.0  # compass bearing of each neighbour
DIST = np.hypot(OFFSETS[:, 0], OFFSETS[:, 1])                          # in cells
PAD = 2


@dataclass
class CAParams:
    lb_scale: float = 1.0          # multiplier on the effective wind fed to Anderson's LB formula
    lb_max: float = 8.0
    emc_offset: float = 0.0        # % added to Simard EMC
    tau_h: tuple = (1.0, 10.0, 100.0)   # dead-fuel time-lags (h)
    edge_mode: str = "mean"        # "mean": average source/target ROS along an edge; "source": source cell only
    ros_scale: float = 1.0         # global multiplier on ROS (calibration handle)
    ros_mult: dict = field(default_factory=dict)   # per-fuel-model ROS multipliers (calibration handle)
    burn_min: float = 2.0          # minimum minutes a cell stays "burning" in the state map (display only)


@dataclass
class Weather:
    """Hourly weather stream starting at ignition time (wind in km/h at 20 ft, direction the wind blows FROM)."""
    wind_kmh: np.ndarray
    wind_from: np.ndarray
    temp_c: np.ndarray
    rh: np.ndarray

    @classmethod
    def constant(cls, wind_kmh, wind_from, temp_c=18.0, rh=63.0):
        return cls(np.array([wind_kmh], float), np.array([wind_from], float), np.array([temp_c], float), np.array([rh], float))

    @classmethod
    def from_run(cls, run):
        w = run.wx
        return cls(w["wind_kmh"], w["wind_dir"], w["temp"], w["rh"])

    def at(self, t_min):
        i = np.clip((np.asarray(t_min) // 60).astype(int), 0, len(self.wind_kmh) - 1)
        return self.wind_kmh[i], self.wind_from[i]

    @property
    def emc_pct(self):
        return float(np.mean(simard_emc(self.temp_c, self.rh)))


class FireEnvironment:
    """Landscape + fuel models + weather + calibration parameters. Independent of where the fire is lit."""

    def __init__(self, land, fuels: dict, weather: Weather, moisture0: dict, params: CAParams | None = None):
        self.land, self.weather, self.params = land, weather, params or CAParams()
        self.geo = land.geo
        self.H, self.W = land.geo.shape
        self.fuels = fuels
        self.moisture0 = {int(k): np.asarray(v, float)[:5] for k, v in moisture0.items()}
        self.static = {k: rt.rothermel_static(fm) for k, fm in fuels.items()}
        self.burnable = np.zeros((self.H, self.W), bool)
        for k in fuels:
            if k in self.moisture0:
                self.burnable |= (land.fuel == k)
        self.burnable &= land.domain
        self._cls_idx = {k: np.flatnonzero(((land.fuel == k) & self.burnable).ravel()) for k in fuels}

    def moisture_for(self, k, t_min):
        t = np.asarray(t_min, float)
        p = self.params
        e = self.weather.emc_pct + p.emc_offset
        m0 = self.moisture0[k]
        M = np.empty(t.shape + (5,))
        tau = np.asarray(p.tau_h, float)
        M[..., :3] = e + (m0[:3] - e) * np.exp(-(t[..., None] / 60.0) / tau)
        M[..., 3:] = m0[3:]
        return M

    def fields(self, t_min):
        """Local fire behaviour of every cell if it ignites at time t_min (scalar or (H,W) array).

        Returns flat arrays over the H*W grid: R_head (m/min), lb, az_max (deg), IR, tau (Rothermel, English)."""
        H, W, p = self.H, self.W, self.params
        n = H * W
        out = dict(R_head=np.zeros(n), lb=np.ones(n), az_max=np.zeros(n), IR=np.zeros(n), tau=np.zeros(n))
        tt = np.broadcast_to(np.asarray(t_min, float), (H, W)).ravel()
        ws, wd = self.weather.at(tt)
        ws, wd = np.broadcast_to(ws, (n,)), np.broadcast_to(wd, (n,))
        slope, aspect = self.land.slope.ravel(), self.land.aspect.ravel()
        for k, idx in self._cls_idx.items():
            if idx.size == 0 or self.static[k] is None:
                continue
            base = rt.rothermel_moist(self.static[k], self.moisture_for(k, tt[idx]))
            hf = rt.head_fire(base, ws[idx], wd[idx], slope[idx], aspect[idx], lb_scale=p.lb_scale, lb_max=p.lb_max)
            out["R_head"][idx] = hf["R_head"] * p.ros_scale * p.ros_mult.get(k, 1.0)
            out["lb"][idx], out["az_max"][idx] = hf["lb"], hf["az_max"]
            out["IR"][idx], out["tau"][idx] = base["IR"], base["tau"]
        return out


def _pad(a, fill):
    out = np.full((a.shape[0] + 2 * PAD, a.shape[1] + 2 * PAD), fill, dtype=a.dtype)
    out[PAD:-PAD, PAD:-PAD] = a
    return out


def _edge_time(env, F, src, k, mode):
    """Minutes for fire to cross the edge from flat-padded cells `src` towards neighbour k."""
    d = DIST[k] * env.geo.cellsize
    Rs = rt.radial_ros(F["R_head"][src], F["lb"][src], F["az_max"][src], AZIMUTH[k])
    ts = d / np.maximum(Rs, 1e-9)
    if mode == "mean":
        dst = src + env._off[k]
        Rd = rt.radial_ros(F["R_head"][dst], F["lb"][dst], F["az_max"][dst], AZIMUTH[k])
        return 0.5 * (ts + d / np.maximum(Rd, 1e-9))
    return ts


def _prepare(env):
    if getattr(env, "_off", None) is None:
        Wp = env.W + 2 * PAD
        env._off = OFFSETS[:, 0] * Wp + OFFSETS[:, 1]
        env._burn_p = _pad(env.burnable, False).ravel()
        r, c = np.mgrid[0:env.H, 0:env.W]
        env._pidx = ((r + PAD) * Wp + (c + PAD)).ravel()          # grid flat index -> padded flat index


def _padded_fields(env, F):
    out = {}
    for key, a in F.items():
        p = np.zeros(env._burn_p.shape, float) if key != "lb" else np.ones(env._burn_p.shape, float)
        p[env._pidx] = a
        out[key] = p
    return out


class HuygensCA:
    """Time-stepped cellular automaton. States: 0 unburned, 1 burning, 2 burnt, 3 unburnable."""

    def __init__(self, env: FireEnvironment, ignition: np.ndarray, refresh_min=10.0, dt=None):
        _prepare(env)
        self.env, self.refresh, self.dt_fixed = env, refresh_min, dt
        Np = env._burn_p.size
        self.t = 0.0
        self.t_arr = np.full(Np, np.inf)
        self.done = np.zeros(Np, bool)
        self.IR_cell = np.zeros(Np)
        self.tau_cell = np.zeros(Np)
        ign = env._pidx[np.flatnonzero((ignition & env.burnable).ravel())]
        if ign.size == 0:
            raise ValueError("Ignition does not intersect any burnable cell.")
        self.t_arr[ign] = 0.0
        self.sched = ign.copy()
        self._next_refresh = 0.0
        self.F = None

    def _refresh(self, t):
        self.F = _padded_fields(self.env, self.env.fields(t))
        self._next_refresh = t + self.refresh
        rmax = float(self.F["R_head"].max())
        self.dt = self.dt_fixed or float(np.clip(0.5 * self.env.geo.cellsize / max(rmax, 1e-6), 0.004, 1.0))

    def _ignite(self, cells):
        env, F = self.env, self.F
        self.done[cells] = True
        self.IR_cell[cells] = F["IR"][cells]
        self.tau_cell[cells] = F["tau"][cells]
        t0 = self.t_arr[cells]
        new = []
        for k in range(16):
            dst = cells + env._off[k]
            ok = env._burn_p[dst]
            if not ok.any():
                continue
            s, d, t = cells[ok], dst[ok], t0[ok]
            cand = t + _edge_time(env, F, s, k, env.params.edge_mode)
            imp = cand < self.t_arr[d] - 1e-12
            if imp.any():
                np.minimum.at(self.t_arr, d[imp], cand[imp])
                self.done[d[imp]] = False
                new.append(d[imp])
        if new:
            self.sched = np.unique(np.concatenate([self.sched] + new))

    def advance(self, t_new):
        while self.t < t_new - 1e-12:
            if self.F is None or self.t >= self._next_refresh - 1e-12:
                self._refresh(self.t)
            t1 = min(self.t + self.dt, t_new, self._next_refresh)
            while self.sched.size:
                is_due = self.t_arr[self.sched] < t1
                due, self.sched = self.sched[is_due], self.sched[~is_due]
                if due.size == 0:
                    break
                self._ignite(due)
            self.t = t1
        return self

    def run(self, t_end=None, callback=None, every=5.0):
        """Advance until the fire has burnt out (t_end None) or to t_end minutes, optionally calling callback(t, ca)."""
        t_stop = t_end if t_end is not None else np.inf
        next_cb = 0.0
        while self.t < t_stop and (self.sched.size or (t_end is not None)):
            target = min(self.t + max(every, 1e-3), t_stop)
            self.advance(target)
            if callback is not None and self.t >= next_cb:
                callback(self.t, self)
                next_cb = self.t + every
            if not self.sched.size and t_end is None:
                break
        return self

    # ---- outputs --------------------------------------------------------------------------------------------
    def _unpad(self, a):
        H, W = self.env.H, self.env.W
        return a.reshape(H + 2 * PAD, W + 2 * PAD)[PAD:-PAD, PAD:-PAD]

    @property
    def arrival(self):
        a = self._unpad(self.t_arr).copy()
        a[~np.isfinite(a)] = np.nan
        return a

    def state(self, t=None):
        t = self.t if t is None else t
        a = self._unpad(self.t_arr)
        tau = np.maximum(self._unpad(self.tau_cell), self.env.params.burn_min)
        st = np.full(a.shape, UNBURNED, np.uint8)
        st[a <= t] = BURNT
        st[(a <= t) & (t - a < tau)] = BURNING
        st[~self.env.burnable] = UNBURNABLE
        return st

    def flame_length(self, ros=None):
        """Byram flame length (m) per cell from the CA's own local front speed (1/|grad T|)."""
        if ros is None:
            ros = self.local_ros()
        IR, tau = self._unpad(self.IR_cell), self._unpad(self.tau_cell)
        fl = rt.flame_length(IR, tau, ros)
        fl[~np.isfinite(self.arrival) | (IR <= 0)] = np.nan
        return fl

    def local_ros(self):
        gy, gx = np.gradient(self.arrival, self.env.geo.cellsize)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.hypot(gx, gy)


# ------------------------------------------------------------------------------------------------------------------
def solve_arrival_mtt(env: FireEnvironment, ignition: np.ndarray, passes=3, T_init=None):
    """Arrival times (min, NaN = unreached) from the same wavelet rule via scipy's Dijkstra.

    Dead-fuel moisture (hence ROS) depends on time, so the edge costs are re-evaluated at the previous pass' arrival
    times; 2-3 passes converge because the moisture drift is slow compared with the spread."""
    _prepare(env)
    H, W = env.H, env.W
    burn = env.burnable
    cid = -np.ones(env._burn_p.size, int)
    cells_p = env._pidx[burn.ravel()]
    cid[cells_p] = np.arange(cells_p.size)
    src_nodes = cid[env._pidx[(ignition & burn).ravel()]]
    if src_nodes.size == 0:
        raise ValueError("Ignition does not intersect any burnable cell.")
    T = np.zeros((H, W)) if T_init is None else np.where(np.isfinite(T_init), T_init, 0.0)
    dist = None
    for _ in range(passes):
        F = _padded_fields(env, env.fields(T))
        rows, cols, wts = [], [], []
        for k in range(16):
            dst = cells_p + env._off[k]
            ok = env._burn_p[dst]
            if not ok.any():
                continue
            s = cells_p[ok]
            rows.append(cid[s]); cols.append(cid[dst[ok]])
            wts.append(_edge_time(env, F, s, k, env.params.edge_mode))
        g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(cells_p.size,) * 2)
        dist = dijkstra(g, directed=True, indices=src_nodes, min_only=True)
        Tn = np.full((H, W), np.nan)
        flat = Tn.ravel()
        flat[np.flatnonzero(burn.ravel())] = np.where(np.isfinite(dist), dist, np.nan)
        T = np.where(np.isfinite(Tn), Tn, 0.0)
    return Tn
