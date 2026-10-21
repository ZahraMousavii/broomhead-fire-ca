"""Two-stage calibration of the Rothermel/Huygens CA against FARSITE output.

Stage 1  cell level   fit heather fuel-model parameters, wind adjustment, ellipse shape and dead-fuel moisture
                      conditioning so Rothermel + Huygens reproduce FARSITE's local ROS and flame-length grids.
Stage 2  scenario     fine-tune a few parameters end-to-end so the CA's arrival-time maps match FARSITE's, while
                      keeping the stage-1 (ROS, flame length) residuals in the objective.
"""
from __future__ import annotations
import copy, json, time
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares

from . import rothermel as rt
from .moisture import simard_emc
from .ca import CAParams, FireEnvironment, Weather, solve_arrival_mtt

CLASSES = (33, 34, 35, 36)
PER_CLASS = ["w1", "w10", "w100", "wlw", "sav1", "savlw", "depth", "mx"]
GLOBALS = ["waf", "lb_scale", "emc_off", "tau1", "tau10"]
START = dict(w1=0.6, w10=0.5, w100=0.2, wlw=1.5, sav1=2500, savlw=1500, depth=1.5, mx=30,
             waf=0.25, lb_scale=1.0, emc_off=0.0, tau1=1.0, tau10=10.0)
LOWER = dict(w1=0.02, w10=1e-3, w100=1e-3, wlw=1e-3, sav1=1200, savlw=600, depth=0.3, mx=10,
             waf=0.05, lb_scale=0.3, emc_off=-6, tau1=0.05, tau10=1.0)
UPPER = dict(w1=6, w10=8, w100=6, wlw=12, sav1=4500, savlw=3500, depth=5, mx=80,
             waf=1.0, lb_scale=2.0, emc_off=6, tau1=30, tau10=200)


@dataclass
class ModelSpec:
    fuels: dict            # code -> FuelModel
    params: CAParams

    def to_json(self, path, extra=None):
        d = dict(fuels={str(k): f.to_dict() for k, f in self.fuels.items()},
                 params={k: v for k, v in asdict(self.params).items()})
        d["params"]["ros_mult"] = {str(k): v for k, v in self.params.ros_mult.items()}
        d["params"]["tau_h"] = list(self.params.tau_h)
        if extra:
            d["provenance"] = extra
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        fuels = {int(k): rt.FuelModel(**v) for k, v in d["fuels"].items()}
        p = dict(d["params"])
        p["ros_mult"] = {int(k): v for k, v in p.get("ros_mult", {}).items()}
        p["tau_h"] = tuple(p["tau_h"])
        return cls(fuels, CAParams(**p))

    def write_fmd(self, path):
        with open(path, "w") as f:
            f.write("ENGLISH\n")
            for fm in self.fuels.values():
                f.write(fm.to_fmd_line() + "\n")


def vector_to_spec(x, free_names=None, base: ModelSpec | None = None, share_waf=True):
    """Parameter vector (stage-1 layout: 8 params x 4 classes, then 5 globals) -> ModelSpec."""
    g = {n: x[len(CLASSES) * len(PER_CLASS) + i] for i, n in enumerate(GLOBALS)}
    fuels = {}
    for ci, k in enumerate(CLASSES):
        d = {n: float(x[ci * len(PER_CLASS) + i]) for i, n in enumerate(PER_CLASS)}
        fuels[k] = rt.FuelModel(k, f"HEATHER{k}", waf=float(g["waf"]), **d)
    ros_mult = dict(base.params.ros_mult) if base else {}
    ros_scale = base.params.ros_scale if base else 1.0
    params = CAParams(lb_scale=float(g["lb_scale"]), emc_offset=float(g["emc_off"]),
                      tau_h=(float(g["tau1"]), float(g["tau10"]), 100.0), ros_scale=ros_scale, ros_mult=ros_mult)
    return ModelSpec(fuels, params)


def spec_to_vector(spec: ModelSpec):
    x = []
    for k in CLASSES:
        fm = spec.fuels[k]
        x += [getattr(fm, n) for n in PER_CLASS]
    p = spec.params
    x += [spec.fuels[CLASSES[0]].waf, p.lb_scale, p.emc_offset, p.tau_h[0], p.tau_h[1]]
    return np.array(x, float)


# ---------------------------------------------------------------------------------------------------- stage 1
class CellLevelProblem:
    """Residuals between Rothermel+Huygens local predictions and FARSITE per-cell ROS / flame length."""

    def __init__(self, table):
        self.t = table
        self.m0 = np.stack([table[k] for k in ("m1", "m10", "m100", "mlh", "mlw")], 1)
        self.emc0 = simard_emc(table["temp"], table["rh"])
        self.th = table["T"] / 60.0
        self.idx = {k: np.flatnonzero(table["fuel"] == k) for k in CLASSES}

    def predict(self, spec: ModelSpec, k):
        t, i, p = self.t, self.idx[k], spec.params
        st = rt.rothermel_static(spec.fuels[k])
        e = self.emc0[i] + p.emc_offset
        tau = np.asarray(p.tau_h, float)
        M = self.m0[i].copy()
        M[:, :3] = e[:, None] + (self.m0[i][:, :3] - e[:, None]) * np.exp(-self.th[i][:, None] / tau[None, :])
        b = rt.rothermel_moist(st, M)
        hf = rt.head_fire(b, t["ws"][i], t["wd"][i], t["slope"][i], t["aspect"][i], lb_scale=p.lb_scale, lb_max=p.lb_max)
        ros = rt.front_normal_speed(hf["R_head"], hf["lb"], hf["az_max"], t["naz"][i]) * p.ros_scale * p.ros_mult.get(k, 1.0)
        fl = rt.flame_length(b["IR"], b["tau"], t["ros"][i])
        return ros, fl

    def residuals(self, spec):
        out = []
        for k in CLASSES:
            i = self.idx[k]
            if i.size == 0:
                continue
            ros, fl = self.predict(spec, k)
            w = 1.0 / np.sqrt(i.size)
            out.append(np.log(np.maximum(ros, 1e-6) / self.t["ros"][i]) * w)
            out.append(np.log(np.maximum(fl, 1e-6) / self.t["fl"][i]) * w)
        return np.concatenate(out)

    def report(self, spec):
        rows = {}
        for k in CLASSES:
            i = self.idx[k]
            ros, fl = self.predict(spec, k)
            r1, r2 = np.log(ros / self.t["ros"][i]), np.log(fl / self.t["fl"][i])
            rows[k] = dict(n=int(i.size), ros_bias=float(np.mean(r1)), ros_mae=float(np.mean(np.abs(r1))),
                           fl_bias=float(np.mean(r2)), fl_mae=float(np.mean(np.abs(r2))))
        return rows


def fit_cell_level(table, n_starts=3, max_nfev=80, seed=0, verbose=True):
    prob = CellLevelProblem(table)
    names = PER_CLASS * len(CLASSES) + GLOBALS
    x0 = np.array([START[n] for n in names]); lo = np.array([LOWER[n] for n in names]); hi = np.array([UPPER[n] for n in names])
    f = lambda x: prob.residuals(vector_to_spec(x))
    best = None
    for s in range(n_starts):
        rng = np.random.default_rng(seed + s)
        xs = np.clip(x0 * np.exp(rng.normal(0, 0.3, x0.size)) if s else x0.copy(), lo, hi)
        t = time.time()
        sol = least_squares(f, xs, bounds=(lo, hi), x_scale=np.maximum(np.abs(xs), 0.1), loss="soft_l1",
                            f_scale=0.1, max_nfev=max_nfev)
        if verbose:
            print(f"  stage-1 start {s}: cost {sol.cost:.4f}  nfev {sol.nfev}  ({time.time() - t:.0f}s)", flush=True)
        if best is None or sol.cost < best.cost:
            best = sol
    return vector_to_spec(best.x), prob, best


# ---------------------------------------------------------------------------------------------------- stage 2
class ArrivalProblem:
    """End-to-end residuals: CA arrival time vs FARSITE arrival time on sampled burned cells of each run."""

    def __init__(self, runs, land, env_builder, cells_per_run=400, seed=0, passes=2):
        self.runs, self.land, self.build, self.passes = runs, land, env_builder, passes
        rng = np.random.default_rng(seed)
        self.samples, self.ign = [], []
        for r in runs:
            ok = np.flatnonzero((np.isfinite(r.arrival) & (r.arrival > 1.0)).ravel())
            self.samples.append(rng.choice(ok, min(cells_per_run, ok.size), replace=False))
            self.ign.append(r.ignition)

    def residuals(self, spec):
        out = []
        for r, s, ign in zip(self.runs, self.samples, self.ign):
            env = self.build(r, self.land, spec)
            T = solve_arrival_mtt(env, ign, passes=self.passes).ravel()[s]
            Tf = r.arrival.ravel()[s]
            T = np.where(np.isfinite(T), T, 2.0 * np.nanmax(r.arrival))
            out.append((T - Tf) / (Tf + 2.0) / np.sqrt(s.size))
        return np.concatenate(out)


def fit_arrival_level(spec0: ModelSpec, cell_problem: CellLevelProblem, runs, land, env_builder,
                      max_nfev=12, cell_weight=1.0, verbose=True, seed=0):
    """Free parameters: per-class ROS multipliers and Mx, plus waf, lb_scale, emc_off, tau1 (12 in total)."""
    arr = ArrivalProblem(runs, land, env_builder, seed=seed)
    base_x = spec_to_vector(spec0)
    ix = {}
    names = PER_CLASS * len(CLASSES) + GLOBALS
    for j, n in enumerate(names):
        ix.setdefault(n, []).append(j)
    free_vec_idx = [j for n in ("mx", "waf", "lb_scale", "emc_off", "tau1") for j in ix[n]]

    def unpack(z):
        x = base_x.copy()
        x[free_vec_idx] = z[:len(free_vec_idx)]
        spec = vector_to_spec(x, base=spec0)
        spec.params.ros_mult = {k: float(m) for k, m in zip(CLASSES, z[len(free_vec_idx):])}
        return spec

    z0 = np.concatenate([base_x[free_vec_idx], np.ones(len(CLASSES))])
    lo = np.concatenate([[LOWER[names[j]] for j in free_vec_idx], np.full(len(CLASSES), 0.6)])
    hi = np.concatenate([[UPPER[names[j]] for j in free_vec_idx], np.full(len(CLASSES), 1.6)])
    z0 = np.clip(z0, lo + 1e-9, hi - 1e-9)
    f = lambda z: np.concatenate([arr.residuals(unpack(z)), cell_weight * cell_problem.residuals(unpack(z))])
    t = time.time()
    sol = least_squares(f, z0, bounds=(lo, hi), x_scale=np.maximum(np.abs(z0), 0.1), loss="soft_l1", f_scale=0.1,
                        max_nfev=max_nfev, diff_step=0.02)
    if verbose:
        print(f"  stage-2: cost {sol.cost:.4f}  nfev {sol.nfev}  ({time.time() - t:.0f}s)", flush=True)
    return unpack(sol.x), sol
