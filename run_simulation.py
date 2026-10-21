#!/usr/bin/env python3
"""Run the calibrated Broomhead cellular automaton for one scenario / ignition line.

Scenario mode (inputs read from the run matrix and the 'weather and fuel moisture' folder):
    python run_simulation.py --model model_final.json --scenario W06 --line east --out sim_W06_east --compare --animate
Free mode (your own wind and moisture):
    python run_simulation.py --model model_final.json --ignition line-east.shp --wind 22 --wind-from 225 \
        --moisture summer --temp 18 --rh 63 --out sim_custom

--line: east | north | mid_horizontal | mid_vertical | rect10 | rect5 | west | south | southwest (or give --ignition <shapefile name/path>).
--compare looks up the matching FARSITE run (scenario mode only) and writes a side-by-side map plus scores.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.landscape import load_landscape, find_project_root, ignition_mask
from bhca.gridio import write_asc
from bhca.calibrate import ModelSpec
from bhca.ca import FireEnvironment, Weather, solve_arrival_mtt, HuygensCA
from bhca.scenarios import load_scenarios, FAMILY_BY_KEY
from bhca.evaluate import derived_outputs, score, discover_cases, family_ignition
from bhca import viz

MOISTURE = {"wet": [30, 28, 26, 120, 80], "spring": [12, 15, 18, 90, 70], "summer": [8, 11, 14, 60, 140], "drought": [5, 7, 9, 30, 60]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--scenario", default=None, help="W01 ... W28")
    ap.add_argument("--line", default=None, choices=sorted(FAMILY_BY_KEY))
    ap.add_argument("--ignition", default=None, help="ignition shapefile name or path (free mode)")
    ap.add_argument("--wind", type=float); ap.add_argument("--wind-from", type=float)
    ap.add_argument("--moisture", default="summer", help="wet | spring | summer | drought | 1h,10h,100h,herb,woody in %%")
    ap.add_argument("--temp", type=float, default=18.0); ap.add_argument("--rh", type=float, default=63.0)
    ap.add_argument("--fuel", default="mergedPatchesAndWhole.asc")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--engine", choices=["mtt", "ca"], default="mtt", help="mtt = fast shortest-path form; ca = explicit time-stepped automaton")
    a = ap.parse_args()

    root = a.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.out, exist_ok=True)
    land = load_landscape(root, a.fuel)
    spec = ModelSpec.from_json(a.model)
    case = None
    if a.scenario:
        scen = load_scenarios(root)[a.scenario]
        weather, moist = scen.weather, scen.moisture
        print(f"scenario {scen.id}: wind from {scen.wind_from:.0f} deg at {scen.wind_kmh:.0f} km/h, {scen.moisture_class} "
              f"(T {scen.temp_c:.0f} C, RH {scen.rh:.0f} %), fuel moisture {scen.moisture[35]} from {scen.fms_file}; weather {scen.wxs_file}")
        if a.line:
            fam = FAMILY_BY_KEY[a.line]
            if not fam.get("ignition"):
                ap.error(f"no ignition shapefile is resolved yet for --line {a.line}; pass --ignition <shapefile> instead")
            ign = family_ignition(root, fam, land.geo)
            if a.compare:
                cases, _ = discover_cases(root, {scen.id: scen, **load_scenarios(root)})
                case = next((c for c in cases if c.family["key"] == a.line and c.scn.id == a.scenario), None)
                if case is not None:
                    ign = case.ignition
        else:
            ign = ignition_mask(root, a.ignition, land.geo)
    else:
        if a.wind is None or a.wind_from is None or not a.ignition:
            ap.error("free mode needs --ignition, --wind and --wind-from (or use --scenario)")
        m = MOISTURE[a.moisture] if a.moisture in MOISTURE else [float(v) for v in a.moisture.split(",")]
        moist = {k: m for k in spec.fuels}
        weather = Weather.constant(a.wind, a.wind_from, a.temp, a.rh)
        ign = ignition_mask(root, a.ignition, land.geo)
    env = FireEnvironment(land, spec.fuels, weather, moist, spec.params)
    print(f"ignition cells: {int(ign.sum())}   burnable cells: {int(env.burnable.sum())}")

    T = HuygensCA(env, ign).run().arrival if a.engine == "ca" else solve_arrival_mtt(env, ign)
    ros, fl = derived_outputs(env, T)
    write_asc(os.path.join(a.out, "ca_arrival_time_min.asc"), T, land.geo)
    write_asc(os.path.join(a.out, "ca_rate_of_spread_m_min.asc"), ros, land.geo)
    write_asc(os.path.join(a.out, "ca_flame_length_m.asc"), fl, land.geo)
    print(f"fire reaches {int(np.isfinite(T).sum())} cells; last arrival {np.nanmax(T):.1f} min")

    if case is not None:
        s = score(T, case.run, env)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
        viz.plot_run_comparison(case, T, land, os.path.join(a.out, "comparison_with_farsite.png"))
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        viz._panel(ax, land, T, "turbo", 0, np.nanmax(T), "CA arrival time", "min", np.linspace(0, np.nanmax(T), 10)[1:-1])
        fig.savefig(os.path.join(a.out, "ca_arrival_map.png"), dpi=130); plt.close(fig)
    if a.animate:
        viz.make_gif(env, ign, os.path.join(a.out, "ca_animation.gif"))
        print("animation written")


if __name__ == "__main__":
    main()
