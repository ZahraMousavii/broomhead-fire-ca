#!/usr/bin/env python3
"""Regenerate the evaluation figures from a trained model: python make_figures.py --results ../results"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.landscape import load_landscape, find_project_root
from bhca.scenarios import load_scenarios
from bhca.evaluate import discover_cases, make_env, simulate
from bhca.tables import build_table
from bhca import calibrate as cal, viz
from run_training import split_of

SHOW = [("east", "W06"), ("east", "W16"), ("north", "W12"), ("mid_horizontal", "W20"), ("east", "W17")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = a.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    land = load_landscape(root)
    spec = cal.ModelSpec.from_json(os.path.join(a.results, "model_final.json"))
    cases, _ = discover_cases(root, load_scenarios(root))
    usable = [c for c in cases if c.usable]
    df = pd.read_csv(os.path.join(a.results, "evaluation_per_run.csv"))
    viz.plot_evaluation(df[df.usable], os.path.join(a.results, "fig_evaluation_summary.png"))
    train = [c for c in usable if split_of(c) == "train"]
    prob = cal.CellLevelProblem(build_table(train, land, cal.CLASSES, per_class=1000))
    viz.plot_cell_level(prob, spec, os.path.join(a.results, "fig_cell_level_fit.png"))
    pairs = []
    for key, sid in SHOW:
        c = next(x for x in usable if x.family["key"] == key and x.scn.id == sid)
        T = simulate(make_env(c, land, spec.fuels, spec.params), c.ignition)
        tag = split_of(c).replace(" ", "_")
        m = c.scn.moisture[35]
        viz.plot_run_comparison(c, T, land, os.path.join(a.results, f"fig_compare_{key}_{sid}_{tag}.png"),
                                title=f"{c.name} [{split_of(c)}] | wind {c.wind_kmh:.0f} km/h from {c.wind_from_deg:.0f} deg | "
                                      f"1-h moisture {m[0]:.0f}%, live woody {m[4]:.0f}%, T {c.scn.temp_c:.0f} C, RH {c.scn.rh:.0f}%")
        pairs.append((f"{c.name}\n[{split_of(c)}]", c.arrival, T))
    viz.plot_arrival_scatter(pairs, os.path.join(a.results, "fig_arrival_scatter.png"))


if __name__ == "__main__":
    main()
