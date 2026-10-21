"""Run the calibrated CA for any wind, moisture and temperature you choose, and write the results to a folder."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np

from .landscape import load_landscape, find_project_root
from .calibrate import ModelSpec
from .ca import FireEnvironment, Weather, solve_arrival_mtt
from .evaluate import derived_outputs, family_ignition, family_resolved
from .gridio import write_asc
from .scenarios import FAMILY_BY_KEY, FAMILIES

MOISTURE_PRESETS = {
    "wet": [30, 28, 26, 120, 80],
    "spring": [12, 15, 18, 90, 70],
    "summer": [8, 11, 14, 60, 140],
    "drought": [5, 7, 9, 30, 60],
}
# calibrated range of the training data (wind 7-30 km/h, see the report); outside it the model is extrapolating
CALIBRATED_WIND_KMH = (7.0, 30.0)


@dataclass
class CustomRun:
    wind_speed_kmh: float = 22.0               # 20-ft wind speed, km/h
    wind_from_deg: float = 225.0               # direction the wind blows FROM, degrees (225 = south-west)
    moisture: str | list = "summer"            # preset name, or [1h, 10h, 100h, live herb, live woody] in %
    temperature_c: float = 18.0
    relative_humidity_pct: float = 63.0
    ignition: str = "east"                     # east | north | mid_horizontal | mid_vertical | rect10 | rect5 | west | south | southwest
    fuel_raster: str = "mergedPatchesAndWhole.asc"
    engine: str = "mtt"                        # mtt (fast) or ca (time-stepped automaton)
    lb_scale_override: float | None = None     # optional change to the fitted ellipse scale
    ros_multiplier: float = 1.0                # optional global change to rate of spread
    out_dir: str = "custom_run_output"
    model_path: str = None
    root: str = None
    make_animation: bool = False               # also write fire_spread.gif (built from the arrival-time grid)
    animation_frames: int = 80
    notes: list = field(default_factory=list)
    frames: list = field(default=None, repr=False)   # filled by run_custom when make_animation: [(t_min, PIL image)]
    fuel_frames: list = field(default=None, repr=False)   # same times, fire spreading over the fuel-model map

    def moisture_values(self):
        if isinstance(self.moisture, str):
            return list(MOISTURE_PRESETS[self.moisture.lower()])
        vals = [float(v) for v in self.moisture]
        if len(vals) != 5:
            raise ValueError("moisture must be a preset name or five values [1h, 10h, 100h, live herb, live woody]")
        return vals


def run_custom(p: CustomRun):
    root = p.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = p.model_path or os.path.join(code_dir, "..", "results", "model_final.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"trained model not found: {model_path}")
    if p.ignition not in FAMILY_BY_KEY:
        raise ValueError(f"ignition must be one of {[f['key'] for f in FAMILIES]}")
    fam = FAMILY_BY_KEY[p.ignition]
    if not family_resolved(fam):
        raise ValueError(f"ignition line {p.ignition} has no geometry in this project")

    land = load_landscape(root, p.fuel_raster)
    spec = ModelSpec.from_json(model_path)
    params = spec.params
    if p.lb_scale_override is not None:
        params.lb_scale = float(p.lb_scale_override)
    params.ros_scale = float(params.ros_scale) * float(p.ros_multiplier)

    wind_kmh = float(p.wind_speed_kmh)
    if not (CALIBRATED_WIND_KMH[0] <= wind_kmh <= CALIBRATED_WIND_KMH[1]):
        p.notes.append(f"wind {wind_kmh:g} km/h is outside the calibrated range "
                       f"{CALIBRATED_WIND_KMH[0]:g}-{CALIBRATED_WIND_KMH[1]:g} km/h; results are extrapolated")
    m = p.moisture_values()
    moist = {k: m for k in spec.fuels}
    weather = Weather.constant(wind_kmh, p.wind_from_deg, p.temperature_c, p.relative_humidity_pct)
    env = FireEnvironment(land, spec.fuels, weather, moist, params)
    ign = family_ignition(root, fam, land.geo)

    if p.engine == "ca":
        from .ca import HuygensCA
        T = HuygensCA(env, ign).run().arrival
    else:
        T = solve_arrival_mtt(env, ign)
    ros, fl = derived_outputs(env, T)

    os.makedirs(p.out_dir, exist_ok=True)
    write_asc(os.path.join(p.out_dir, "arrival_time_min.asc"), T, land.geo)
    write_asc(os.path.join(p.out_dir, "rate_of_spread_m_min.asc"), ros, land.geo)
    write_asc(os.path.join(p.out_dir, "flame_length_m.asc"), fl, land.geo)
    intensity = viz_fireline_intensity(fl)
    write_asc(os.path.join(p.out_dir, "fireline_intensity_kw_m.asc"), intensity, land.geo)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import viz
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    tmax = float(np.nanmax(T))
    viz._panel(ax, land, T, "turbo_r", 0, tmax, f"CA arrival time - {fam['label']}", "min",
               np.linspace(0, tmax, 9)[1:-1])
    ax.set_title(f"{fam['label']} | wind {wind_kmh:g} km/h from {p.wind_from_deg:.0f} deg | "
                 f"moisture {m[0]:g}/{m[1]:g}/{m[2]:g}/{m[3]:g}/{m[4]:g} %", fontsize=9)
    map_path = os.path.join(p.out_dir, "arrival_map.png")
    fig.tight_layout(); fig.savefig(map_path, dpi=130); plt.close(fig)
    int_path = viz.plot_intensity_map(land, intensity, os.path.join(p.out_dir, "intensity_map.png"), T=T,
                                      ignition=ign, burnable=env.burnable, title=f"Fire intensity - {fam['label']} | wind {wind_kmh:g} km/h from {p.wind_from_deg:.0f} deg")
    anim_path = fuel_anim_path = None
    if p.make_animation:
        p.frames = spread_animation(env, ign, T, fl, p, title=fam["label"])
        anim_path = viz.save_gif(p.frames, os.path.join(p.out_dir, "fire_spread.gif"))
        p.fuel_frames = viz.fuel_spread_frames(land, T, ign, env.burnable, spec.fuels, n_frames=p.animation_frames,
                                               wind_from_deg=p.wind_from_deg, wind_kmh=wind_kmh, title=fam["label"])
        fuel_anim_path = viz.save_gif(p.fuel_frames, os.path.join(p.out_dir, "fuel_spread.gif"))

    result = dict(
        ignition_line=fam["label"], wind_kmh=wind_kmh, wind_from_deg=p.wind_from_deg,
        moisture_1h_10h_100h_herb_woody=m, temperature_c=p.temperature_c,
        relative_humidity_pct=p.relative_humidity_pct, ignition_cells=int(ign.sum()),
        burnable_cells=int(env.burnable.sum()), cells_reached=int(np.isfinite(T).sum()),
        time_to_burn_all_min=round(tmax, 1), mean_rate_of_spread_m_min=round(float(np.nanmean(ros)), 3),
        max_flame_length_m=round(float(np.nanmax(fl)), 2),
        mean_fireline_intensity_kw_m=round(float(np.nanmean(intensity)), 1),
        max_fireline_intensity_kw_m=round(float(np.nanmax(intensity)), 1), out_dir=os.path.abspath(p.out_dir),
        map=os.path.abspath(map_path), intensity_map=os.path.abspath(int_path),
        animation=os.path.abspath(anim_path) if anim_path else None,
        fuel_animation=os.path.abspath(fuel_anim_path) if fuel_anim_path else None, notes=p.notes,
    )
    return result, T, ros, fl


def viz_fireline_intensity(fl):
    from .viz import fireline_intensity
    return fireline_intensity(fl)


def spread_animation(env, ign, T, fl, p: CustomRun, title=""):
    """Frames showing the fire spreading from the ignition cells, coloured by fireline-intensity class."""
    from . import viz
    return viz.spread_frames(env.land, T, ign, env.burnable, flame_length=fl, n_frames=p.animation_frames,
                             wind_from_deg=p.wind_from_deg, wind_kmh=float(p.wind_speed_kmh), title=title)
