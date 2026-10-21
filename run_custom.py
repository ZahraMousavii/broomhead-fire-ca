#!/usr/bin/env python3
"""Run the CA with your own parameters, from a JSON file or from the command line.

    python run_custom.py                           # uses custom_run_config.json next to this file
    python run_custom.py my_scenario.json
    python run_custom.py --wind-kmh 29 --wind-from 270 --moisture drought --ignition west --out results_custom

Config keys (all optional): wind_speed_kmh (km/h), wind_from_deg, moisture (preset name or [1h,10h,100h,herb,woody] in %),
temperature_c, relative_humidity_pct, ignition (east|north|mid_horizontal|mid_vertical|rect10|rect5|west|south|southwest),
engine (mtt|ca), lb_scale_override, ros_multiplier, out_dir, make_animation.
"""
import argparse, dataclasses, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhca.custom_run import CustomRun, run_custom


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(here, "custom_run_config.json"))
    ap.add_argument("--wind-kmh", type=float); ap.add_argument("--wind-from", type=float)
    ap.add_argument("--moisture"); ap.add_argument("--temp", type=float); ap.add_argument("--rh", type=float)
    ap.add_argument("--ignition"); ap.add_argument("--out")
    a = ap.parse_args()
    cfg = {}
    if os.path.exists(a.config):
        cfg = json.load(open(a.config))
    elif a.config != os.path.join(here, "custom_run_config.json"):
        ap.error(f"config file not found: {a.config}")
    mapping = dict(wind_kmh="wind_speed_kmh", wind_from="wind_from_deg", moisture="moisture", temp="temperature_c",
                   rh="relative_humidity_pct", ignition="ignition", out="out_dir")
    for k, v in vars(a).items():
        if k in mapping and v is not None:
            cfg[mapping[k]] = v
    fields = {f.name for f in dataclasses.fields(CustomRun)}
    unknown = set(cfg) - fields
    if unknown:
        ap.error(f"unknown config keys: {sorted(unknown)}")
    cfg.setdefault("out_dir", os.path.join(here, "..", "custom_run_output"))
    run = CustomRun(**cfg)
    result, *_ = run_custom(run)
    for k, v in result.items():
        print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
