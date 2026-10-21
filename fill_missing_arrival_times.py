#!/usr/bin/env python3
"""Compute a CA arrival time for every run-matrix cell that is blank because FARSITE could not complete that run.

    python fill_missing_arrival_times.py --model ../results/model_final.json --out ../results

For each blank cell, the leftover output/<family>/<scenario>.fzp file (if any) is never opened - the CA is run purely
from the scenario's own inputs (run matrix wind, weather-folder temperature/RH/fuel moisture) and the family's ignition
shapefile, exactly like every other simulation in run_all_scenarios.py. The reported time is the CA's own last arrival
(the same quantity the run matrix records for a completed FARSITE run: the time the fire stops spreading).

Families whose ignition shapefile is not yet resolved (west, south) are skipped and listed separately.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.landscape import load_landscape, find_project_root
from bhca.scenarios import load_scenarios, FAMILIES, FAMILY_BY_KEY, find_run_matrix, resolve_run_file
from bhca.calibrate import ModelSpec
from bhca.evaluate import make_env, simulate, family_ignition, family_resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--matrix", default=None, help="run-matrix .xlsx to use instead of the project default")
    a = ap.parse_args()
    root = a.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.out, exist_ok=True)
    land = load_landscape(root)
    spec = ModelSpec.from_json(a.model)
    matrix_path = a.matrix or find_run_matrix(root)
    scen = load_scenarios(root, matrix_path=matrix_path)
    print(f"run matrix: {os.path.basename(matrix_path)}")

    rows = []
    for sid, s in scen.items():
        for key in sorted(s.missing):
            fam = FAMILY_BY_KEY[key]
            leftover = resolve_run_file(os.path.join(root, "output", fam.get("folder") or ""), sid)
            row = dict(scenario=sid, ignition_line=fam["label"], wind_from_deg=s.wind_from, wind_kmh=s.wind_kmh,
                       moisture_class=s.moisture_class, leftover_file_ignored=leftover is not None)
            if not family_resolved(fam):
                row["ca_arrival_time_min"] = np.nan
                row["status"] = "not computed: no ignition shapefile resolved for this line yet"
            else:
                ign = family_ignition(root, fam, land.geo)
                env = make_env(s, land, spec.fuels, spec.params)
                T = simulate(env, ign)
                row["ca_arrival_time_min"] = round(float(np.nanmax(T)), 1)
                row["status"] = "computed by CA"
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(a.out, "missing_arrival_times_filled_by_CA.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten {out_csv}")


if __name__ == "__main__":
    main()
