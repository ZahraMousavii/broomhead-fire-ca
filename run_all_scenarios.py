#!/usr/bin/env python3
"""Run the calibrated CA for every scenario W01-W28 on every ignition line, from the input files only.

    python run_all_scenarios.py --model ../results/model_final.json --out ../results

For each (ignition line, scenario) the CA is given
  * the scenario's wind (run matrix), temperature / RH (.wxs) and fuel moisture (.fms) from 'weather and fuel moisture',
  * the ignition line shapefile of the family,
  * the landscape layers (elevation, slope, aspect, fuel model),
and nothing from the FARSITE runs. The FARSITE runs (where they exist and match the scenario) are used only to score it.
Outputs: all_scenarios_results.csv, fig_scenarios_<line>.png (28 arrival-time maps per ignition line), fig_matrix_vs_ca.png.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.landscape import load_landscape, find_project_root
from bhca.scenarios import load_scenarios, FAMILIES
from bhca.calibrate import ModelSpec
from bhca.evaluate import discover_cases, make_env, simulate, score, family_ignition, family_resolved, SCENARIO_IDS
from bhca import viz
from run_training import split_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--skip-figures", action="store_true", help="only write the CSV (the maps take most of the run time)")
    ap.add_argument("--figure-lines", default=None, help="comma-separated ignition-line keys to draw (default: all)")
    a = ap.parse_args()
    root = a.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.out, exist_ok=True)
    land = load_landscape(root)
    spec = ModelSpec.from_json(a.model)
    scen = load_scenarios(root)
    cases, _ = discover_cases(root, scen)
    by_key = {(c.family["key"], c.scn.id): c for c in cases}
    rows, grids = [], {}
    t0 = time.time()
    for fam in FAMILIES:
        if not family_resolved(fam):
            print(f"skipping {fam['label']}: no ignition geometry resolved for this line yet")
            continue
        canon = family_ignition(root, fam, land.geo)
        for sid in SCENARIO_IDS:
            s = scen[sid]
            case = by_key.get((fam["key"], sid))
            # the ignition of the family, except where FARSITE lit the same line a few metres away (then its own t=0 cells)
            ign = case.ignition if (case is not None and case.usable) else canon
            env = make_env(s, land, spec.fuels, spec.params)
            T = simulate(env, ign)
            grids[(fam["key"], sid)] = T
            m = s.moisture[35]
            row = dict(ignition_line=fam["label"], scenario=sid, sub_test=s.phase, wind_from_deg=s.wind_from, wind_kmh=s.wind_kmh,
                       moisture_class=s.moisture_class, temp_c=s.temp_c, rh_pct=s.rh, fm_1h=m[0], fm_10h=m[1], fm_100h=m[2],
                       fm_live_herb=m[3], fm_live_woody=m[4], fms_file=s.fms_file, wxs_file=s.wxs_file,
                       matrix_arrival_min=s.reference_times.get(fam["key"], np.nan), ca_full_burn_min=float(np.nanmax(T)),
                       farsite_run=("none" if case is None else ("consistent" if case.usable else "inconsistent with scenario")),
                       ignition_cells=int(ign.sum()))
            if case is not None:
                sc = score(T, case.run, env)
                row.update(split=split_of(case), farsite_last_arrival_min=sc["T_end"], ref_cell_farsite_min=sc["ref_farsite_min"],
                           ref_cell_ca_min=sc["ref_ca_min"],
                           **{k: sc[k] for k in ("mae", "rel_mae", "bias", "corr", "iou_50", "iou_100")})
                if case.issues:
                    row["note"] = "; ".join(case.issues)
                elif case.notes:
                    row["note"] = "; ".join(case.notes)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["ca_vs_farsite_pct_at_ref_cell"] = 100 * (df.ref_cell_ca_min - df.ref_cell_farsite_min) / df.ref_cell_farsite_min
    df.to_csv(os.path.join(a.out, "all_scenarios_results.csv"), index=False)
    print(f"{len(df)} CA simulations in {time.time() - t0:.0f}s")

    ext = viz._extent(land.geo)
    hs = np.where(land.domain, viz.hillshade(land.elevation, land.geo.cellsize), np.nan)
    figs = [] if a.skip_figures else [f for f in FAMILIES if not a.figure_lines or f['key'] in a.figure_lines.split(',')]
    for fam in figs:
        fig, axs = plt.subplots(4, 7, figsize=(24, 17))
        for ax, sid in zip(axs.ravel(), SCENARIO_IDS):
            T = grids[(fam["key"], sid)]
            s = scen[sid]
            ax.imshow(hs, cmap="gray", extent=ext, vmin=0, vmax=1.2)
            tmax = float(np.nanmax(T))
            ax.imshow(np.ma.masked_invalid(T), cmap="turbo", extent=ext, vmin=0, vmax=tmax, alpha=0.85)
            X, Y = land.geo.cell_centres()
            ax.contour(X, Y, T, levels=np.linspace(0, tmax, 9)[1:-1], colors="k", linewidths=0.3)
            to = np.radians((s.wind_from + 180) % 360)                      # direction the wind blows towards
            cx, cy = ext[0] + 0.86 * (ext[1] - ext[0]), ext[2] + 0.88 * (ext[3] - ext[2])
            L = 0.09 * (ext[1] - ext[0])
            ax.annotate("", xy=(cx + L * np.sin(to), cy + L * np.cos(to)), xytext=(cx - L * np.sin(to), cy - L * np.cos(to)),
                        arrowprops=dict(arrowstyle="-|>", color="w", lw=2.2, ec="k"))
            ax.set_title(f"{sid}  wind from {s.wind_from:.0f}° {s.wind_kmh:.0f} km/h\n{s.moisture_class} ({s.temp_c:.0f}°C, RH {s.rh:.0f}%)  "
                         f"burn-out {tmax:.0f} min", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"CA arrival time (min) - {fam['label']} - every scenario driven by its own wind, temperature/RH and fuel moisture "
                     f"(white arrow = direction the wind blows towards)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(a.out, f"fig_scenarios_{fam['key']}.png"), dpi=70)
        plt.close(fig)

    d = df.dropna(subset=["ref_cell_farsite_min", "ref_cell_ca_min"])
    if not a.skip_figures:
        fig, ax = plt.subplots(figsize=(7, 6.4))
        for fam in FAMILIES:
            q = d[(d.ignition_line == fam["label"]) & (d.farsite_run == "consistent")]
            ax.scatter(q.ref_cell_farsite_min, q.ref_cell_ca_min, s=22, label=fam["label"], alpha=0.85)
        lim = [5, 1100]
        ax.plot(lim, lim, "k-", lw=1); ax.plot(lim, [0.8 * v for v in lim], "k:", lw=0.8); ax.plot(lim, [1.2 * v for v in lim], "k:", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("FARSITE arrival time at the last-burned cell, min"); ax.set_ylabel("CA arrival time at the same cell, min")
        ax.set_title("Time to burn to the last cell: FARSITE vs cellular automaton (dotted: +/-20%)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_matrix_vs_ca.png"), dpi=130); plt.close(fig)
    ok = d[d.farsite_run == "consistent"]
    print(f"CA vs FARSITE arrival time at the last-burned cell ({len(ok)} consistent runs): median |error| "
          f"{np.nanmedian(np.abs(ok.ca_vs_farsite_pct_at_ref_cell)):.1f} %, within 20 %: {(np.abs(ok.ca_vs_farsite_pct_at_ref_cell) <= 20).mean() * 100:.0f} %")


if __name__ == "__main__":
    main()
