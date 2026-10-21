"""Dead-fuel moisture conditioning: first-order time-lag relaxation toward the weather equilibrium moisture."""
from __future__ import annotations
import numpy as np


def simard_emc(temp_c, rh):
    """Equilibrium moisture content (%) of fine dead fuel (Simard 1968), temp in deg C, RH in %."""
    t = np.asarray(temp_c, float) * 9.0 / 5.0 + 32.0
    h = np.asarray(rh, float)
    lo = 0.03229 + 0.281073 * h - 0.000578 * h * t
    mid = 2.22749 + 0.160107 * h - 0.014784 * t
    hi = 21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t - 0.483199 * h
    return np.where(h < 10, lo, np.where(h < 50, mid, hi))


def condition_dead_moisture(m0_pct, t_min, emc_pct, tau_h=(1.0, 10.0, 100.0)):
    """Dead moisture [1h,10h,100h] (%) after t_min minutes of constant weather.

    m0_pct: (...,3) initial moisture; t_min broadcastable to m0_pct[...,0]; emc_pct equilibrium moisture (%)."""
    m0 = np.asarray(m0_pct, float)
    t_h = np.asarray(t_min, float)[..., None] / 60.0
    tau = np.asarray(tau_h, float)
    return emc_pct + (m0 - emc_pct) * np.exp(-t_h / tau)


def moisture_at(m0_pct, t_min, temp_c, rh, emc_offset=0.0, tau_h=(1.0, 10.0, 100.0)):
    """Full 5-value moisture set (%) at time t: dead classes relax to EMC(+offset); live values stay constant."""
    m0 = np.asarray(m0_pct, float)
    t = np.asarray(t_min, float)
    emc = float(np.mean(simard_emc(temp_c, rh))) + emc_offset
    out = np.empty(np.broadcast(m0[..., 0], t).shape + (5,))
    out[..., :3] = condition_dead_moisture(m0[..., :3], t, emc, tau_h)
    out[..., 3:] = m0[..., 3:]
    return out
