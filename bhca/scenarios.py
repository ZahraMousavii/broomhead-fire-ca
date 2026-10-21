"""Scenario definitions W01-W28: wind from the run matrix, weather and fuel moisture from 'weather and fuel moisture'.

Sources (all read from the project folder, nothing is hard-coded):
  Broomhead_Fire_Simulation_Run_Matrix_all.xlsx       'Run Matrix' sheet     wind direction / speed / moisture class of W01-W28,
                                                       and the arrival time FARSITE reached for each ignition line (blank if
                                                       FARSITE could not complete that run)
                                                       'FlamMap Input Reference' sheet   temperature and RH of each moisture class
  weather and fuel moisture/fuel moisture/*.fms       initial fuel moisture (1 h, 10 h, 100 h, live herb, live woody) per fuel model
  weather and fuel moisture/weather/*.wxs             hourly temperature / RH / precipitation / cloud cover (168 h)
The .wxs files in the folder are the last-saved version of files that were edited between FARSITE runs, so they cannot
identify a scenario by name. A scenario's weather file is therefore chosen by its temperature and RH (the moisture class),
and its wind speed and direction are the run-matrix values (constant, as in every FARSITE run).

A blank arrival-time cell in the Run Matrix means FARSITE did not complete that run (it does not mean the ignition line
does not apply). Any leftover file of the same name under output/<family>/ is a partial or otherwise invalid attempt and
must never be used as ground truth for that (family, scenario) pair, even if it happens to be readable - see
Scenario.missing and evaluate.discover_cases, which drops those combinations before a FARSITE file is ever opened.
"""
from __future__ import annotations
import glob, os, re
from dataclasses import dataclass, field
import numpy as np

from .ca import Weather

CLASS_KEYS = (("wet", "wet"), ("spring", "spring"), ("summer", "summer"), ("extreme", "drought"))

MATRIX_CANDIDATES = ("Broomhead_Fire_Simulation_Run_Matrix_all_new.xlsx", "Broomhead_Fire_Simulation_Run_Matrix_all.xlsx",
                     "Broomhead_Fire_Simulation_Run_Matrix (4)run2.xlsx")


def cell_minutes(v):
    """Arrival time from a run-matrix cell. Where a cell holds two values 'a/b', the value after the slash is used.
    Trailing annotations some cells carry (e.g. '(old)', '?', '???') are stripped before parsing."""
    s = str(v).strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        raise ValueError(f"no number found in run-matrix cell {v!r}")
    return float(m.group(0))


def resolve_run_file(folder_path, sid):
    """Path of the FARSITE output for one run, or None. When two files share the run ID (e.g. W04.fzp and W04-new.fzp,
    or W11-NEW.fzp), the one whose name carries 'new' (any case) is used."""
    if not os.path.isdir(folder_path):
        return None
    plain, newer = None, None
    for name in sorted(os.listdir(folder_path)):
        m = re.fullmatch(rf"{sid}(-new)?\.fzp", name, re.IGNORECASE)
        if not m:
            continue
        if m.group(1):
            newer = os.path.join(folder_path, name)
        else:
            plain = os.path.join(folder_path, name)
    return newer or plain

# ignition geometry of every family of FARSITE runs (folder inside output/ and the ignition shapefile it corresponds to).
# 'folder'/'ignition' is None for a line whose shapefile is not yet resolved in this project folder: the CA cannot be run
# for it and discover_cases skips it, but its column is still read from the run matrix (reference_times / missing).
FAMILIES = [
    dict(key="east", label="East ignition line", folder="East ignition line", ignition="line-east.shp", match=("east",)),
    dict(key="north", label="North ignition line", folder="North ignition line/new", ignition="line-north.shp", match=("north",)),
    dict(key="mid_horizontal", label="Middle horizontal line", folder="middle_horzintal", ignition="ig-midle-horzintall.shp", match=("horizintall", "horizintal", "horizontal")),
    dict(key="mid_vertical", label="Middle vertical line", folder="middle_vertical", ignition="ig-midle-verticall.shp", match=("vertical",)),
    dict(key="rect10", label="Middle rectangle 10 m", folder="middle-10Rec", ignition="ig-rectangle-middlee.shp", match=("rectangle-10m", "rectangle-10")),
    dict(key="rect5", label="Middle rectangle 5 m", folder="middle-5Rec", ignition="ig-rectangle-middlle-5m.shp", match=("rectangle-5m", "rectangle-5")),
    # West/South have no shapefile in the project (their FARSITE run logs name a file that only ever existed on the
    # machine that ran FlamMap); their ignition cells are instead read straight from their own completed FARSITE runs
    # (bhca.landscape.ignition_mask_from_fzp) - see family_ignition() in bhca.evaluate.
    dict(key="west", label="West ignition line", folder="West ignition line", ignition=None, ignition_source="fzp", match=("west",)),
    dict(key="south", label="South ignition line", folder="South ignition line", ignition=None, ignition_source="fzp", match=("south",)),
    # Southwest has a shapefile ('ignition line/sowthwest/south-west.shp') but no FARSITE runs in this project yet, so
    # its 'folder' (output/Southwest ignition line) does not exist: discover_cases simply finds 0 runs for it, same
    # as any other family would with an empty output folder, and the CA can still be run from the shapefile alone.
    dict(key="southwest", label="Southwest ignition line", folder="Southwest ignition line", ignition="south-west.shp",
         match=("southwest", "south-west", "south west")),
]
FAMILY_BY_KEY = {f["key"]: f for f in FAMILIES}


def _match_arrival_columns(header_row):
    """Map each 'Arrival Time(min)__...' column index to a family key, in the order FAMILIES lists them.

    Two families (rect10, rect5) share the same 'rectangl' keyword, so the first matching column goes to the family
    that has not been assigned yet, in FAMILIES order - this is why FAMILIES lists rect10 before rect5."""
    assigned = {}
    used_cols = set()
    for fam in FAMILIES:
        for i, h in enumerate(header_row):
            if i in used_cols or not h or "arrival time" not in str(h).lower():
                continue
            if any(m in str(h).lower() for m in fam["match"]):
                assigned[fam["key"]] = i
                used_cols.add(i)
                break
    return assigned


def _class_key(text):
    t = str(text).lower()
    for needle, key in CLASS_KEYS:
        if needle in t:
            return key
    raise ValueError(f"unknown fuel-moisture scenario: {text}")


def read_fms(path):
    """{fuel model code: [1h, 10h, 100h, live herb, live woody] in %}."""
    out = {}
    for line in open(path):
        v = line.split()
        if len(v) >= 6:
            out[int(v[0])] = [float(x) for x in v[1:6]]
    return out


def read_wxs(path):
    """Hourly RAWS weather as arrays: temp (C), rh (%), precip, wind_kmh, wind_dir, cloud."""
    rows = []
    for line in open(path):
        v = line.split()
        if len(v) >= 10 and v[0].isdigit() and len(v[0]) == 4:
            rows.append([float(x) for x in v[:10]])
    a = np.array(rows)
    return dict(temp=a[:, 4], rh=a[:, 5], precip=a[:, 6], wind_kmh=a[:, 7], wind_dir=a[:, 8], cloud=a[:, 9])


@dataclass
class Scenario:
    id: str
    phase: str
    wind_from: float
    wind_kmh: float
    moisture_class: str
    class_label: str
    moisture: dict                    # fuel code -> [5]
    fms_file: str
    wxs_file: str
    wxs_wind_matches: bool            # True if the chosen .wxs already carries this scenario's wind
    wxs: dict                         # hourly arrays from the .wxs
    matrix_dead_text: str
    matrix_live_text: str
    reference_times: dict = field(default_factory=dict)   # family key -> matrix arrival time (min), only where present
    missing: frozenset = field(default_factory=frozenset)  # family keys whose matrix cell is blank for this scenario
                                                            # (FARSITE could not complete that run; use the CA instead)

    @property
    def temp_c(self):
        return float(np.mean(self.wxs["temp"]))

    @property
    def rh(self):
        return float(np.mean(self.wxs["rh"]))

    @property
    def weather(self) -> Weather:
        n = len(self.wxs["temp"])
        return Weather(np.full(n, self.wind_kmh), np.full(n, self.wind_from), self.wxs["temp"].copy(), self.wxs["rh"].copy())

    def moisture_row(self, fuel=35):
        return self.moisture[fuel]


def find_run_matrix(root):
    for name in MATRIX_CANDIDATES:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {MATRIX_CANDIDATES} found under {root}")


def load_scenarios(root, matrix_path=None):
    import openpyxl
    matrix = matrix_path or find_run_matrix(root)
    wb = openpyxl.load_workbook(matrix, data_only=True)
    header = next(wb["Run Matrix"].iter_rows(values_only=True))
    arrival_col = _match_arrival_columns(header)   # family key -> column index, for whichever columns this file has
    ref = {}
    for row in wb["FlamMap Input Reference"].iter_rows(values_only=True):
        if row[0] and isinstance(row[1], (int, float)) and isinstance(row[2], (int, float)):
            ref[_class_key(row[0])] = (float(row[1]), float(row[2]))
    fdir = os.path.join(root, "weather and fuel moisture")
    fms = {p: read_fms(p) for p in sorted(glob.glob(os.path.join(fdir, "fuel moisture", "*.fms")))}
    wxs = {p: read_wxs(p) for p in sorted(glob.glob(os.path.join(fdir, "weather", "*.wxs")))}
    scen = {}
    for row in wb["Run Matrix"].iter_rows(min_row=3, values_only=True):
        if not (row[0] and re.fullmatch(r"W\d\d", str(row[0]))):
            continue
        cls = _class_key(row[10])
        m1 = float(str(row[5]).split("/")[0])
        fms_hit = [p for p, d in fms.items() if abs(d[35][0] - m1) < 1e-9]
        if len(fms_hit) != 1:
            raise ValueError(f"{row[0]}: could not match a unique .fms file for 1-h moisture {m1}: {fms_hit}")
        T, RH = ref[cls]
        wd, ws = float(row[3]), float(row[4])
        cands = [p for p, d in wxs.items() if abs(d["temp"].mean() - T) < 1e-6 and abs(d["rh"].mean() - RH) < 1e-6]
        if not cands:
            raise ValueError(f"{row[0]}: no .wxs file with T={T} RH={RH}")
        exact = [p for p in cands if np.all(wxs[p]["wind_kmh"] == ws) and np.all(wxs[p]["wind_dir"] == wd)]
        pick = (exact or cands)[0]
        ref_times, missing = {}, set()
        for key, ci in arrival_col.items():
            v = row[ci]
            if v is None or str(v).strip() == "":
                missing.add(key)          # column exists for this family, but this scenario's cell is blank
            else:
                ref_times[key] = cell_minutes(v)
        scen[row[0]] = Scenario(
            id=row[0], phase=str(row[1]), wind_from=wd, wind_kmh=ws, moisture_class=cls, class_label=str(row[10]).split("--")[0],
            moisture=fms[fms_hit[0]], fms_file=os.path.basename(fms_hit[0]), wxs_file=os.path.basename(pick),
            wxs_wind_matches=bool(exact), wxs=wxs[pick],
            matrix_dead_text=" / ".join(str(row[i]) for i in (5, 6, 7)), matrix_live_text=" / ".join(str(row[i]) for i in (8, 9)),
            reference_times=ref_times, missing=frozenset(missing))
    return scen


def check_inputs(scn: Scenario, run_wx, run_moisture, fuel=35):
    """Differences between a FARSITE run's embedded inputs and the scenario definition (empty list = identical)."""
    issues = []
    ws, wd = float(np.mean(run_wx["wind_kmh"])), float(np.mean(run_wx["wind_dir"]))
    if abs(ws - scn.wind_kmh) > 1e-6:
        issues.append(f"wind speed {ws:g} km/h in FARSITE run, {scn.wind_kmh:g} in scenario")
    if abs(((wd - scn.wind_from + 180) % 360) - 180) > 1e-6:
        issues.append(f"wind from {wd:g} deg in FARSITE run, {scn.wind_from:g} in scenario")
    t, rh = float(np.mean(run_wx["temp"])), float(np.mean(run_wx["rh"]))
    if abs(t - scn.temp_c) > 1e-6 or abs(rh - scn.rh) > 1e-6:
        issues.append(f"temperature/RH {t:g} C/{rh:g}% in FARSITE run, {scn.temp_c:g} C/{scn.rh:g}% in scenario")
    if [float(x) for x in run_moisture[fuel]] != [float(x) for x in scn.moisture[fuel]]:
        issues.append(f"fuel moisture {run_moisture[fuel]} in FARSITE run, {scn.moisture[fuel]} in scenario")
    return issues
