#!/usr/bin/env python3
"""Consistency checks (run after training: python run_selfcheck.py --model model_final.json)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.landscape import load_landscape, find_project_root
from bhca.calibrate import ModelSpec
from bhca.ca import HuygensCA, solve_arrival_mtt
from bhca.scenarios import load_scenarios
from bhca.evaluate import discover_cases, make_env
from bhca.tables import front_normal_azimuth
from bhca import rothermel as rt


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = a.root or find_project_root(os.path.dirname(os.path.abspath(__file__)))
    ok = True
    b = rt.rothermel_base(rt.ANDERSON_FM1, [6, 7, 8, 60, 90])
    ok &= check("Rothermel FM1 no-wind ROS ~4.4 ft/min (hand calculation)", abs(b["R0"] - 4.39) < 0.05, f"({b['R0']:.2f})")
    lb, Rh = 2.0, 10.0
    e = np.sqrt(1 - 1 / lb ** 2)
    ok &= check("ellipse head/back rates", np.isclose(rt.radial_ros(Rh, lb, 0, 0), Rh) and np.isclose(rt.radial_ros(Rh, lb, 0, 180), Rh * (1 - e) / (1 + e)))
    ok &= check("support function >= radial rate", rt.front_normal_speed(Rh, lb, 0, 60) >= rt.radial_ros(Rh, lb, 0, 60) - 1e-9)

    land = load_landscape(root)
    scen = load_scenarios(root)
    ok &= check("28 scenarios read from run matrix + weather and fuel moisture folder", len(scen) == 28)
    cases, excluded = discover_cases(root, scen)
    ok &= check("no patch-design runs among the cases", all(c.scn.id.startswith("W") for c in cases))
    ok &= check("North line cases come from output/North ignition line/new", all("new" in c.family["folder"] for c in cases if c.family["key"] == "north"))
    east = [c for c in cases if c.family["key"] == "east"]
    ok &= check("East-line FARSITE runs use exactly the scenario inputs", all(not c.issues for c in east), f"({len(east)} runs)")
    ok &= check("FARSITE last-arrival time == run-matrix value (East line)", all(c.t_end_matches_matrix for c in east))
    ok &= check("ignition shapefile raster == FARSITE t=0 cells (East, W06)", east[5].ignition_shift == (0, 0) and east[5].ignition_ok)

    spec = ModelSpec.from_json(a.model)
    c = east[5]
    env = make_env(c, land, spec.fuels, spec.params)
    T1 = solve_arrival_mtt(env, c.ignition, passes=1)
    T2 = HuygensCA(env, c.ignition, refresh_min=1e9).run().arrival
    both = np.isfinite(T1) & np.isfinite(T2)
    d = float(np.nanmax(np.abs(T1 - T2)[both]))
    ok &= check("CA engine == Dijkstra (static conditions)", d < 1e-6 and (np.isfinite(T1) == np.isfinite(T2)).all(), f"(max diff {d:.2e} min)")
    _, sp = front_normal_azimuth(c.arrival)
    sel = np.isfinite(c.ros) & np.isfinite(sp) & (c.arrival > 5)
    med = float(np.median(c.ros[sel] / sp[sel]))
    ok &= check("FARSITE ROS ~ 1/|grad T|", abs(med - 1) < 0.02, f"(median ratio {med:.3f})")

    west = [c for c in cases if c.family["key"] == "west"]
    south = [c for c in cases if c.family["key"] == "south"]
    ok &= check("West ignition line: FARSITE runs found and usable", len(west) >= 20 and all(c.usable for c in west), f"({len(west)} runs)")
    ok &= check("South ignition line: FARSITE runs found and usable", len(south) >= 20 and all(c.usable for c in south), f"({len(south)} runs)")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
