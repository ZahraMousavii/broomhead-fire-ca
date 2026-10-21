"""Turn FARSITE output grids into per-cell supervised training tables."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def front_normal_azimuth(arrival, cellsize=1.0):
    """Outward normal azimuth (deg from N) and 1/|grad T| speed (m/min) from an arrival-time grid."""
    a = np.array(arrival, float)
    with np.errstate(invalid="ignore"):
        gy, gx = np.gradient(a, cellsize)
    ge, gn = gx, -gy
    mag = np.hypot(ge, gn)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = 1.0 / mag
    return np.degrees(np.arctan2(ge, gn)) % 360.0, speed


def clean_mask(run, land, classes, speed_tol=0.2, min_arrival=0.0):
    """Cells where FARSITE's local front speed is a trustworthy target: burned, interior to one fuel class,
    all eight neighbours burned, and ROS consistent with the arrival-time gradient."""
    az, speed = front_normal_azimuth(run.arrival, run.geo.cellsize)
    fu = land.fuel
    fmax = ndi.maximum_filter(fu, size=3)
    fmin = ndi.minimum_filter(fu, size=3)
    finite3 = ndi.minimum_filter(np.isfinite(run.arrival).astype(np.uint8), size=3) > 0
    ok = (np.isin(fu, list(classes)) & (fmax == fmin) & finite3 & np.isfinite(run.ros) & np.isfinite(run.flame_length)
          & (run.arrival > min_arrival) & (run.ros > 0) & (run.flame_length > 0) & land.domain)
    with np.errstate(invalid="ignore"):
        ok &= np.abs(run.ros / speed - 1.0) < speed_tol
    return ok, az


def build_table(runs, land, classes, per_class=1500, seed=0, speed_tol=0.2):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("run", "fuel", "ws", "wd", "slope", "aspect", "naz", "ros", "fl", "T", "temp", "rh", "m1", "m10", "m100", "mlh", "mlw")}
    for ri, run in enumerate(runs):
        ok, az = clean_mask(run, land, classes, speed_tol)
        for k in classes:
            idx = np.flatnonzero((ok & (land.fuel == k)).ravel())
            if idx.size == 0:
                continue
            if idx.size > per_class:
                idx = rng.choice(idx, per_class, replace=False)
            mo = run.moisture[k]
            cols["run"].append(np.full(idx.size, ri)); cols["fuel"].append(np.full(idx.size, k))
            cols["ws"].append(np.full(idx.size, run.wind_kmh)); cols["wd"].append(np.full(idx.size, run.wind_from_deg))
            cols["slope"].append(land.slope.ravel()[idx]); cols["aspect"].append(land.aspect.ravel()[idx])
            cols["naz"].append(az.ravel()[idx]); cols["ros"].append(run.ros.ravel()[idx].astype(float))
            cols["fl"].append(run.flame_length.ravel()[idx].astype(float))
            cols["T"].append(run.arrival.ravel()[idx].astype(float))
            cols["temp"].append(np.full(idx.size, float(np.mean(run.wx["temp"])))); cols["rh"].append(np.full(idx.size, float(np.mean(run.wx["rh"]))))
            for nm, v in zip(("m1", "m10", "m100", "mlh", "mlw"), mo):
                cols[nm].append(np.full(idx.size, v))
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in cols.items()}
