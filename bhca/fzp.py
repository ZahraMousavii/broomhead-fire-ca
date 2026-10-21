"""Reader for FlamMap/FARSITE .fzp output packages (zip archives of binary grids + a text run log)."""
from __future__ import annotations
import os, re, struct, zipfile
from dataclasses import dataclass, field
import numpy as np
from .gridio import GridGeo

NODATA = -9999.0
_GRID_KEYS = {"Farsite Arrival Time": "arrival", "Farsite Flame Length": "flame_length",
              "Farsite Rate Of Spread": "ros"}


def _base(s):
    return os.path.basename(s.strip().replace("\\", "/"))


@dataclass
class FzpRun:
    path: str
    family: str                      # folder name, e.g. "East ignition line"
    run_id: str                      # file stem, e.g. "W06"
    geo: GridGeo
    arrival: np.ndarray              # minutes since ignition, NaN = never burned
    flame_length: np.ndarray         # m, NaN where not burned / ignition cells
    ros: np.ndarray                  # m/min (local front-normal speed), NaN where undefined
    landscape: str = ""
    ignition_file: str = ""
    custom_fuel_file: str = ""
    moisture: dict = field(default_factory=dict)   # fuel model -> [1h,10h,100h,herb,woody] in %
    wx: dict = field(default_factory=dict)         # arrays: wind_kmh, wind_dir, temp, rh
    timestep: float = 5.0

    @property
    def wind_kmh(self):
        return float(np.mean(self.wx["wind_kmh"])) if self.wx else np.nan

    @property
    def wind_from_deg(self):
        return float(np.mean(self.wx["wind_dir"])) if self.wx else np.nan

    @property
    def wind_is_constant(self):
        return bool(self.wx) and np.ptp(self.wx["wind_kmh"]) == 0 and np.ptp(self.wx["wind_dir"]) == 0

    def moisture_for(self, fuel):
        return self.moisture.get(int(fuel))

    @property
    def name(self):
        return f"{self.family}/{self.run_id}"


def _decode_grid(buf):
    nr, nc = struct.unpack("<ii", buf[:8])
    dx, dy = struct.unpack("<dd", buf[8:24])
    xhi, xlo, yhi, ylo = struct.unpack("<dddd", buf[24:56])
    off = len(buf) - nr * nc * 4
    name = buf[64:off].split(b"\0")[0].decode("latin1")
    a = np.frombuffer(buf[off:], "<f4").reshape(nr, nc).astype(np.float32)
    a = np.where(a <= NODATA + 1, np.nan, a)
    return name, a, GridGeo(nc, nr, xlo, ylo, dx)


def load_fzp(path) -> FzpRun:
    z = zipfile.ZipFile(path)
    grids, geo, log = {}, None, None
    for n in z.namelist():
        info = z.getinfo(n)
        if info.file_size < 200_000:
            head = z.read(n)
            if head.startswith(b"#Run Log"):
                log = head.decode("latin1").replace("\r", "")
                continue
        if 500_000 <= info.file_size <= 2_000_000:
            name, a, g = _decode_grid(z.read(n))
            if name in _GRID_KEYS:
                grids[_GRID_KEYS[name]] = a
                geo = g
    if len(grids) < 3 or log is None:
        raise ValueError(f"{path}: not a complete FARSITE output package")
    g = lambda pat: (re.search(pat, log).group(1).strip() if re.search(pat, log) else "")
    m = re.search(r"FUEL_MOISTURES_DATA: (\d+)\n((?:.*\n)+?)\n", log)
    moist = {}
    if m:
        for line in m.group(2).strip().split("\n"):
            v = line.split()
            moist[int(v[0])] = [float(x) for x in v[1:6]]
    raws = re.findall(r"^(\d{4}) (\d+) (\d+) (\d+) (-?[\d.]+) (\d+) ([\d.]+) ([\d.]+) ([\d.]+) (\d+)", log, re.M)
    wx = {}
    if raws:
        wx = dict(temp=np.array([float(r[4]) for r in raws]), rh=np.array([float(r[5]) for r in raws]),
                  wind_kmh=np.array([float(r[7]) for r in raws]), wind_dir=np.array([float(r[8]) for r in raws]))
    rel = os.path.relpath(path)
    return FzpRun(path=path, family=os.path.basename(os.path.dirname(path)),
                  run_id=os.path.splitext(os.path.basename(path))[0], geo=geo,
                  arrival=grids["arrival"], flame_length=grids["flame_length"], ros=grids["ros"],
                  landscape=_base(g(r"#Landscape File: (.*)")), ignition_file=_base(g(r"FARSITE_IGNITION_FILE: (.*)")),
                  custom_fuel_file=_base(g(r"CUSTOM_FUELS_FILE: (.*)")), moisture=moist, wx=wx,
                  timestep=float(g(r"FARSITE_TIMESTEP: (.*)") or 5))
