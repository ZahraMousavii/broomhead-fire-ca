"""Pair every FARSITE run with its scenario definition, run the CA from those inputs and score it."""
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
import numpy as np

from .fzp import load_fzp, FzpRun
from .landscape import ignition_mask, ignition_mask_from_fzp
from .scenarios import FAMILIES, Scenario, check_inputs, resolve_run_file
from .ca import FireEnvironment, CAParams, solve_arrival_mtt
from .tables import front_normal_azimuth
from . import rothermel as rt

BASELINE_LANDSCAPE = "newBroomhead.tif"
BASELINE_FUELFILE = "fuel model-joe modelling.fmd"
SCENARIO_IDS = [f"W{i:02d}" for i in range(1, 30)]  # W01-W29 (W29 added 2026-10)


@dataclass
class Case:
    """One FARSITE run (the target) tied to the scenario definition (the input) that the CA is driven with."""
    run: FzpRun
    scn: Scenario
    family: dict
    issues: list = field(default_factory=list)      # differences between the FARSITE run's inputs and the scenario
    notes: list = field(default_factory=list)       # harmless remarks
    ignition_ok: bool = True                        # ignition of the family == FARSITE's t=0 cells (up to a small shift)
    ignition_shift: tuple = (0, 0)                  # (rows, cols) by which FARSITE's ignition cells are offset from the shapefile raster
    ignition: np.ndarray = None                     # ignition cells given to the CA
    matrix_T: float = np.nan                        # arrival time listed in the run matrix
    t_end_matches_matrix: bool = True

    @property
    def usable(self):
        return (not self.issues) and self.ignition_ok

    # inputs come from the scenario (folder files + matrix), never from the FARSITE log
    @property
    def wind_kmh(self):
        return self.scn.wind_kmh

    @property
    def wind_from_deg(self):
        return self.scn.wind_from

    @property
    def moisture(self):
        return self.scn.moisture

    @property
    def weather(self):
        return self.scn.weather

    @property
    def wx(self):
        w = self.scn.weather
        return dict(temp=w.temp_c, rh=w.rh, wind_kmh=w.wind_kmh, wind_dir=w.wind_from)

    @property
    def run_id(self):
        return self.scn.id

    @property
    def name(self):
        return f"{self.family['label']}/{self.scn.id}"

    def __getattr__(self, item):                    # arrival, ros, flame_length, geo, ... come from the FARSITE run
        if item in ("run", "scn", "family"):
            raise AttributeError(item)
        return getattr(self.run, item)


def family_resolved(family):
    """Whether this ignition line can be simulated at all: it has a shapefile, or its cells can be read from its own
    completed FARSITE runs (ignition_source == 'fzp')."""
    return bool(family.get("folder")) and bool(family.get("ignition") or family.get("ignition_source") == "fzp")


def family_ignition(root, family, geo):
    if family.get("ignition"):
        return ignition_mask(root, family["ignition"], geo)
    if family.get("ignition_source") == "fzp":
        return ignition_mask_from_fzp(os.path.join(root, "output", family["folder"]))
    raise ValueError(f"{family['label']}: no ignition shapefile and no ignition_source to derive one from")


def best_shift(ign, zero, max_shift=3):
    """Shift (rows, cols) of `ign` that best overlaps FARSITE's t=0 cells, and the IoU reached."""
    best = ((0, 0), -1.0)
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            sh = np.roll(np.roll(ign, dr, 0), dc, 1)
            iou = float((sh & zero).sum() / max((sh | zero).sum(), 1))
            if iou > best[1] + 1e-12 or (abs(iou - best[1]) < 1e-12 and abs(dr) + abs(dc) < abs(best[0][0]) + abs(best[0][1])):
                best = ((dr, dc), iou)
    return best


def discover_cases(root, scenarios):
    """All W01-W28 FARSITE runs of the six ignition families (patch-design runs are not considered).

    Returns (cases, excluded) where excluded lists (path, reason) for files that could not be used at all."""
    cases, excluded = [], []
    for fam in FAMILIES:
        if not family_resolved(fam):
            continue                                   # ignition geometry not resolved for this line yet
        ign_cache = None
        for sid in SCENARIO_IDS:
            scn = scenarios.get(sid)
            folder = os.path.join(root, "output", fam["folder"])
            p = resolve_run_file(folder, sid)
            if scn is not None and fam["key"] in scn.missing:
                if p is not None:
                    excluded.append((os.path.relpath(p, root),
                                     "blank in the run matrix (FARSITE could not complete this run); this leftover "
                                     "file is not used even though it exists - use the CA instead"))
                continue
            if p is None:
                continue
            try:
                run = load_fzp(p)
            except Exception as e:
                excluded.append((os.path.relpath(p, root), f"incomplete FARSITE package ({str(e)[:60]})"))
                continue
            if run.landscape != BASELINE_LANDSCAPE or run.custom_fuel_file != BASELINE_FUELFILE:
                excluded.append((os.path.relpath(p, root), f"other landscape / fuel file ({run.landscape})"))
                continue
            scn = scenarios[sid]
            case = Case(run, scn, fam)
            case.issues = check_inputs(scn, run.wx, run.moisture)
            zero = np.isfinite(run.arrival) & (run.arrival == 0)
            if ign_cache is None:
                ign_cache = family_ignition(root, fam, run.geo)
            ign = ign_cache
            shift, iou = best_shift(ign, zero)
            case.ignition_shift = shift
            case.ignition_ok = iou == 1.0
            case.ignition = ign if shift == (0, 0) else zero      # same geometry, but FARSITE placed it a few metres away
            if shift != (0, 0) and case.ignition_ok:
                case.notes.append(f"FARSITE ignition sits {shift[0]} rows / {shift[1]} cols from the shapefile raster "
                                  f"(the run used the WGS84 copy of the shapefile); CA is lit on FARSITE's t=0 cells")
            if not case.ignition_ok:
                case.issues.append("FARSITE run was ignited with a different line than this family's ignition line")
            case.matrix_T = scn.reference_times.get(fam["key"], np.nan)
            case.t_end_matches_matrix = bool(np.isfinite(case.matrix_T) and abs(float(np.nanmax(run.arrival)) - case.matrix_T) < 0.05)
            if not case.t_end_matches_matrix:
                case.issues.append(f"FARSITE file ends at {float(np.nanmax(run.arrival)):.2f} min but the run matrix lists "
                                   f"{case.matrix_T:g} min for this run")
            cases.append(case)
    return cases, excluded


def make_env(case, land, fuels, params: CAParams):
    scn = case.scn if isinstance(case, Case) else case
    return FireEnvironment(land, fuels, scn.weather, scn.moisture, params)


def simulate(env, ignition, passes=3):
    return solve_arrival_mtt(env, ignition, passes=passes)


def derived_outputs(env, T):
    """CA analogues of FARSITE's ROS and flame-length grids from an arrival-time grid."""
    _, speed = front_normal_azimuth(T, env.geo.cellsize)
    F = env.fields(np.nan_to_num(T, nan=0.0))
    IR, tau = F["IR"].reshape(T.shape), F["tau"].reshape(T.shape)
    fl = rt.flame_length(IR, tau, speed)
    fl[~np.isfinite(T) | (IR <= 0)] = np.nan
    return speed, fl


def score(T_ca, run, env=None):
    """Metrics of a CA arrival grid against a FARSITE run (FARSITE runs stop at a finite time T_end)."""
    Tf = run.arrival
    T_end = float(np.nanmax(Tf))
    f_b, c_b = np.isfinite(Tf), np.isfinite(T_ca) & (T_ca <= T_end)
    both = f_b & np.isfinite(T_ca)
    err = (T_ca - Tf)[both]
    # reference cell = the last cell FARSITE burned among the cells the CA can also burn (FARSITE's very last cell is
    # sometimes a barrier cell of the CA, e.g. an edge cell of class 1 or 91)
    ref = np.unravel_index(np.argmax(np.where(both, Tf, -np.inf)), Tf.shape)
    out = dict(T_end=T_end, ref_farsite_min=float(Tf[ref]), ref_ca_min=float(T_ca[ref]),
               ca_T_full_burn=float(np.nanmax(T_ca)), n_farsite=int(f_b.sum()), n_ca=int(c_b.sum()),
               mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)),
               rel_mae=float(np.mean(np.abs(err) / np.maximum(Tf[both], 1.0))),
               corr=float(np.corrcoef(T_ca[both], Tf[both])[0, 1]) if both.sum() > 2 else np.nan)
    for frac in (0.25, 0.5, 1.0):
        t = frac * T_end
        a, b = f_b & (Tf <= t), np.isfinite(T_ca) & (T_ca <= t)
        out[f"iou_{int(frac * 100)}"] = float((a & b).sum() / max((a | b).sum(), 1))
    if env is not None:
        speed, fl = derived_outputs(env, T_ca)
        ok = both & np.isfinite(run.ros) & np.isfinite(speed) & (run.arrival > 3)
        if ok.sum() > 10:
            out["ros_logerr_mae"] = float(np.mean(np.abs(np.log(np.maximum(speed[ok], 1e-6) / run.ros[ok]))))
            okf = ok & np.isfinite(fl) & np.isfinite(run.flame_length)
            out["fl_logerr_mae"] = float(np.mean(np.abs(np.log(np.maximum(fl[okf], 1e-6) / run.flame_length[okf]))))
    return out
