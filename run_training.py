#!/usr/bin/env python3
"""Train the Rothermel/Huygens cellular automaton on the FARSITE (.fzp) runs of the Broomhead project.

    python run_training.py --out <output folder> [--root <Broomhead_final>] [--quick]

Inputs of every scenario W01-W28 (wind, temperature, RH, fuel moisture) come from the run matrix and the
'weather and fuel moisture' folder; the FARSITE .fzp files only supply the target grids (arrival time, ROS, flame length).
Only W01-W28 runs are used (patch-design runs are not part of the CA), and the North line uses output/North ignition line/new.

Split (baseline landscape, runs whose FARSITE inputs equal the scenario definition):
  train              East ignition line, all weather codes except the hold-out codes below
  held-out weather   East ignition line, W04 W10 W16 W22 W28   (unseen wind / moisture combinations)
  held-out ignition  every run of the other five ignition geometries
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.landscape import load_landscape, find_project_root
from bhca.scenarios import load_scenarios
from bhca.evaluate import discover_cases, make_env, simulate, score
from bhca.tables import build_table
from bhca import calibrate as cal

HOLDOUT_WEATHER = ("W04", "W10", "W16", "W22", "W28")
STAGE2_RUNS = ("W01", "W03", "W06", "W09", "W12", "W13", "W17", "W18", "W20", "W21", "W24", "W26")


def split_of(case):
    if case.family["key"] == "east":
        return "held-out weather" if case.scn.id in HOLDOUT_WEATHER else "train"
    return "held-out ignition"


def evaluate_cases(cases, land, spec, tag):
    """Run the CA from scenario inputs + ignition shapefile only, then compare with the FARSITE grids."""
    rows = []
    for c in cases:
        env = make_env(c, land, spec.fuels, spec.params)
        T = simulate(env, c.ignition)
        s = score(T, c.run, env)
        s.update(scenario=c.scn.id, family=c.family["label"], family_key=c.family["key"], model=tag, split=split_of(c),
                 usable=c.usable, moisture_class=c.scn.moisture_class, wind_kmh=c.scn.wind_kmh, wind_from=c.scn.wind_from,
                 temp_c=c.scn.temp_c, rh=c.scn.rh, matrix_T=c.matrix_T, farsite_T_end=s.pop("T_end"),
                 issues="; ".join(c.issues), notes="; ".join(c.notes))
        rows.append(s)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny budget, only to smoke-test the pipeline")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--stage2-nfev", type=int, default=12)
    a = ap.parse_args()
    root = a.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    land = load_landscape(root)
    scen = load_scenarios(root)
    cases, excluded = discover_cases(root, scen)
    usable = [c for c in cases if c.usable]
    print(f"{len(cases)} W01-W28 FARSITE runs found, {len(usable)} consistent with their scenario definition, "
          f"{len(cases) - len(usable)} flagged, {len(excluded)} unreadable")
    pd.DataFrame([dict(run=c.name, issues="; ".join(c.issues)) for c in cases if not c.usable] +
                 [dict(run=p, issues=r) for p, r in excluded]).to_csv(os.path.join(a.out, "excluded_runs.csv"), index=False)
    train = [c for c in usable if split_of(c) == "train"]
    print(f"train runs: {len(train)}   held-out weather: {sum(split_of(c) == 'held-out weather' for c in usable)}   "
          f"held-out ignition: {sum(split_of(c) == 'held-out ignition' for c in usable)}")

    tab = build_table(train, land, cal.CLASSES, per_class=1000 if not a.quick else 200)
    print(f"stage-1 table: {len(tab['ros'])} cells", flush=True)
    spec1, prob, sol1 = cal.fit_cell_level(tab, n_starts=1 if a.quick else a.starts, max_nfev=5 if a.quick else 80)
    spec1.to_json(os.path.join(a.out, "model_stage1_cell_level.json"))
    print("stage-1 cell-level report:", json.dumps(prob.report(spec1)), flush=True)

    s2 = [c for c in train if c.scn.id in STAGE2_RUNS]
    builder = lambda case, land_, spec: make_env(case, land_, spec.fuels, spec.params)
    spec2, sol2 = cal.fit_arrival_level(spec1, prob, s2, land, builder, max_nfev=2 if a.quick else a.stage2_nfev)
    prov = dict(trained=time.strftime("%Y-%m-%d %H:%M"), train_runs=[c.name for c in train], stage2_runs=[c.name for c in s2],
                holdout_weather=list(HOLDOUT_WEATHER), stage1_cost=float(sol1.cost), stage2_cost=float(sol2.cost),
                note="Effective (calibrated) heather fuel parameters; individual values are not uniquely identifiable.")
    spec2.to_json(os.path.join(a.out, "model_final.json"), prov)
    spec2.write_fmd(os.path.join(a.out, "fitted_heather_fuels.fmd"))

    df = pd.concat([evaluate_cases(cases, land, spec1, "stage1"), evaluate_cases(cases, land, spec2, "final")])
    df.to_csv(os.path.join(a.out, "evaluation_per_run.csv"), index=False)
    du = df[df.usable]
    summ = (du.groupby(["model", "split"])[["mae", "rel_mae", "bias", "corr", "iou_50", "iou_100", "ros_logerr_mae", "fl_logerr_mae"]]
              .mean().round(3))
    summ["n_runs"] = du.groupby(["model", "split"]).size()
    summ.to_csv(os.path.join(a.out, "evaluation_summary.csv"))
    print(summ.to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
