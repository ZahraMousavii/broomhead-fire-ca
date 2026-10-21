"""Rothermel (1972) surface-fire spread with Albini (1976) weighting, Rothermel (1983) wind-slope vector
combination, Anderson (1983) length-to-breadth ratio and Byram flame length.

Fuel-model units follow FlamMap .fmd files (ENGLISH): loads in tons/ac, SAV in 1/ft, depth in ft,
moisture of extinction in %, heat content in Btu/lb. Outputs are metric (m/min, m, kW/m).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import numpy as np

TONS_AC_TO_LB_FT2 = 2000.0 / 43560.0
FT_TO_M = 0.3048
KMH_TO_FTMIN = 1000.0 / 60.0 / FT_TO_M          # 54.68 ft/min per km/h
BTU_FT2_TO_KJ_M2 = 11.3565
RHO_P, S_T, S_E = 32.0, 0.0555, 0.010
SAV_10H, SAV_100H = 109.0, 30.0


@dataclass
class FuelModel:
    code: int
    name: str
    w1: float = 0.0            # 1-h dead load, tons/ac
    w10: float = 0.0
    w100: float = 0.0
    wlh: float = 0.0           # live herbaceous
    wlw: float = 0.0           # live woody
    dynamic: bool = False      # kept for .fmd export; curing transfer is not applied (heather has no herb load)
    sav1: float = 2000.0       # 1/ft
    savlh: float = 1500.0
    savlw: float = 1500.0
    depth: float = 1.0         # ft
    mx: float = 25.0           # dead moisture of extinction, %
    hd: float = 8000.0         # Btu/lb
    hl: float = 8000.0
    waf: float = 0.4           # midflame wind adjustment factor (calibrated, not in .fmd)

    def to_dict(self):
        return asdict(self)

    def to_fmd_line(self):
        return (f"{self.code} {self.name} {self.w1:.4f} {self.w10:.4f} {self.w100:.4f} {self.wlh:.4f} {self.wlw:.4f} "
                f"{'dynamic' if self.dynamic else 'static'} {self.sav1:.0f} {self.savlh:.0f} {self.savlw:.0f} "
                f"{self.depth:.3f} {self.mx:.1f} {self.hd:.0f} {self.hl:.0f} {self.name}_fitted")


ANDERSON_FM1 = FuelModel(1, "FM1", w1=0.74, sav1=3500, savlh=1500, savlw=1500, depth=1.0, mx=12.0, waf=0.36)


def _size_class(sav):
    for k, lim in enumerate((1200.0, 192.0, 96.0, 48.0, 16.0)):
        if sav >= lim:
            return k
    return 5


def rothermel_static(fm: FuelModel):
    """Moisture-independent part of Rothermel's model (weighting factors, packing ratio, wind/slope coefficients)."""
    w = np.array([fm.w1, fm.w10, fm.w100, fm.wlh, fm.wlw]) * TONS_AC_TO_LB_FT2
    sav = np.array([fm.sav1, SAV_10H, SAV_100H, fm.savlh, fm.savlw], float)
    cat = np.array([0, 0, 0, 1, 1])
    hcat = np.array([fm.hd, fm.hd, fm.hd, fm.hl, fm.hl], float)
    A = np.where(w > 0, sav * w / RHO_P, 0.0)
    A_cat = np.array([A[cat == 0].sum(), A[cat == 1].sum()])
    if w.sum() <= 0 or A_cat.sum() <= 0:
        return None
    f_cat = A_cat / A_cat.sum()
    f_ij, g_ij = np.zeros(5), np.zeros(5)
    for c in (0, 1):
        idx = np.where(cat == c)[0]
        if A_cat[c] <= 0:
            continue
        f_ij[idx] = A[idx] / A_cat[c]
        cls = np.array([_size_class(sav[j]) for j in idx])
        for a, j in enumerate(idx):
            g_ij[j] = f_ij[idx][cls == cls[a]].sum() if w[j] > 0 else 0.0
    sig_cat = np.array([(f_ij[cat == c] * sav[cat == c]).sum() for c in (0, 1)])
    sigma = float((f_cat * sig_cat).sum())
    rho_b = float(w.sum() / fm.depth)
    beta = rho_b / RHO_P
    ratio = beta / (3.348 * sigma ** -0.8189)
    s15 = sigma ** 1.5
    Aexp = 1.0 / (4.774 * sigma ** 0.1 - 7.27)
    gam = (s15 / (495.0 + 0.0594 * s15)) * ratio ** Aexp * np.exp(Aexp * (1.0 - ratio))
    xi = np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    wd_s = w[:3] * np.exp(-138.0 / sav[:3])
    wl_s = w[3:] * np.exp(-500.0 / sav[3:])
    return dict(w=w, sav=sav, cat=cat, hcat=hcat, A_cat=A_cat, f_cat=f_cat, f_ij=f_ij, g_ij=g_ij, sigma=sigma,
                rho_b=rho_b, beta=beta, ratio=ratio, gam=gam, xi=xi, eps=np.exp(-138.0 / sav), wd_s=wd_s, wl_s=wl_s,
                mx_dead=fm.mx / 100.0, tau=384.0 / sigma, waf=fm.waf,
                C=7.47 * np.exp(-0.133 * sigma ** 0.55), B=0.02526 * sigma ** 0.54,
                E=0.715 * np.exp(-3.59e-4 * sigma))


def rothermel_moist(st, moist_pct):
    """Moisture-dependent part, vectorised over rows of moist_pct (n,5) [1h,10h,100h,herb,woody] in %.

    Returns a dict of arrays (n,) that, together with the static terms, defines the fire behaviour."""
    M = np.atleast_2d(np.asarray(moist_pct, float))[:, :5] / 100.0
    n = M.shape[0]
    w, cat, f_ij, g_ij = st["w"], st["cat"], st["f_ij"], st["g_ij"]
    mx_dead = st["mx_dead"]
    if st["wl_s"].sum() > 0 and st["wd_s"].sum() > 0:
        Wr = st["wd_s"].sum() / st["wl_s"].sum()
        mf_fine = (M[:, :3] * st["wd_s"]).sum(1) / st["wd_s"].sum()
        mx_live = np.maximum(mx_dead, 2.9 * Wr * (1.0 - mf_fine / mx_dead) - 0.226)
    else:
        mx_live = np.full(n, mx_dead)
    mx_cat = [np.full(n, mx_dead), mx_live]
    eta_s = min(0.174 * S_E ** -0.19, 1.0)
    IR_sum = np.zeros(n)
    sink = np.zeros(n)
    qig = 250.0 + 1116.0 * M
    for c in (0, 1):
        if st["A_cat"][c] <= 0:
            continue
        sel = cat == c
        wn = (g_ij[sel] * w[sel]).sum() * (1.0 - S_T)
        h = (f_ij[sel] * st["hcat"][sel]).sum()
        mf = (M[:, sel] * f_ij[sel]).sum(1)
        r = np.minimum(mf / mx_cat[c], 1.0)
        eta_m = np.maximum(1.0 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3, 0.0)
        IR_sum += wn * h * eta_m * eta_s
        sink += st["f_cat"][c] * (qig[:, sel] * (f_ij[sel] * st["eps"][sel])).sum(1)
    IR = st["gam"] * IR_sum
    R0 = IR * st["xi"] / (st["rho_b"] * sink)
    out = dict(R0=R0, IR=IR, mx_live=mx_live)
    for k in ("tau", "beta", "ratio", "C", "B", "E", "sigma", "rho_b", "waf"):
        out[k] = np.full(n, st[k])
    return out


def rothermel_base(fm: FuelModel, moist_pct):
    """Scalar convenience wrapper: dict of floats for one fuel model and one moisture set."""
    st = rothermel_static(fm)
    if st is None:
        return dict(R0=0.0, IR=0.0, tau=0.0, beta=1e-6, ratio=1.0, C=0.0, B=1.0, E=1.0, sigma=1.0, rho_b=0.0,
                    waf=fm.waf, mx_live=0.0)
    return {k: float(v[0]) for k, v in rothermel_moist(st, moist_pct).items()}


def anderson_lb(u_mph, scale=1.0, lb_max=8.0):
    u = np.asarray(u_mph, float) * scale
    lb = 0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397
    return np.clip(lb, 1.0, lb_max)


def head_fire(base, wind_kmh, wind_from_deg, slope_deg, aspect_deg, waf=None, lb_scale=1.0, lb_max=8.0,
              slope_scale=1.0):
    """Vectorised Rothermel head-fire ROS, direction and ellipse shape.

    `base` is a dict whose values are floats or arrays broadcastable with the terrain arrays.
    Returns dict with R_head (m/min), lb, az_max (deg clockwise from N, direction of travel), u_eff (mph).
    """
    R0, IR, ratio = base["R0"], base["IR"], base["ratio"]
    C, B, E, beta = base["C"], base["B"], base["E"], base["beta"]
    waf = base["waf"] if waf is None else waf
    U = np.asarray(wind_kmh, float) * KMH_TO_FTMIN * waf                     # midflame, ft/min
    phi_w = C * np.power(np.maximum(U, 1e-9), B) * np.power(ratio, -E)
    phi_w = np.where(U > 0, phi_w, 0.0)
    tan_s = np.tan(np.radians(np.asarray(slope_deg, float)))
    phi_s = slope_scale * 5.275 * np.power(beta, -0.3) * tan_s ** 2
    az_w = np.radians(np.asarray(wind_from_deg, float) + 180.0)
    az_up = np.radians(np.asarray(aspect_deg, float) + 180.0)
    om = az_up - az_w
    X = phi_w + phi_s * np.cos(om)
    Y = phi_s * np.sin(om)
    phi_e = np.hypot(X, Y)
    az_max = az_w + np.arctan2(Y, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_eff = np.power(np.maximum(phi_e, 0.0) * np.power(ratio, E) / np.where(C > 0, C, 1.0), 1.0 / B)
    u_lim = 0.9 * IR
    over = u_eff > u_lim
    phi_lim = C * np.power(np.maximum(u_lim, 1e-9), B) * np.power(ratio, -E)
    phi_e = np.where(over, phi_lim, phi_e)
    u_eff = np.where(over, u_lim, u_eff)
    r_head = R0 * (1.0 + phi_e) * FT_TO_M
    lb = anderson_lb(u_eff / 88.0, lb_scale, lb_max)
    return dict(R_head=r_head, lb=lb, az_max=np.degrees(az_max) % 360.0, u_eff=u_eff / 88.0, R0=R0 * FT_TO_M)


def ellipse_axes(R_head, lb):
    """Semi-axes (A, B) and focal offset c of the wavelet whose rear focus is the ignition point."""
    e = np.sqrt(np.maximum(1.0 - 1.0 / lb ** 2, 0.0))
    A = R_head / (1.0 + e)
    return A, A / lb, A * e, e


def radial_ros(R_head, lb, az_max, az_dir):
    """Elliptical spread rate (m/min) from the ignition point along azimuth az_dir (Anderson 1983)."""
    e = np.sqrt(np.maximum(1.0 - 1.0 / lb ** 2, 0.0))
    d = np.radians(np.asarray(az_dir, float) - az_max)
    return R_head * (1.0 - e) / (1.0 - e * np.cos(d))


def front_normal_speed(R_head, lb, az_max, az_normal):
    """Huygens envelope normal speed (support function of the ellipse) for a front whose outward normal is az_normal."""
    A, B, c, _ = ellipse_axes(R_head, lb)
    psi = np.radians(np.asarray(az_normal, float) - az_max)
    return c * np.cos(psi) + np.sqrt((A * np.cos(psi)) ** 2 + (B * np.sin(psi)) ** 2)


def flame_length(IR, tau, ros_m_min):
    """Byram flame length (m) from reaction intensity (Btu/ft2/min), residence time (min) and ROS (m/min)."""
    ha = IR * tau * BTU_FT2_TO_KJ_M2                                      # kJ/m2
    ib = ha * np.maximum(ros_m_min, 0.0) / 60.0                           # kW/m
    return 0.0775 * np.power(ib, 0.46)
