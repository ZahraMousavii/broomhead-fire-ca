"""Landscape layers (elevation, slope, aspect, fuel model) and project path helpers."""
from __future__ import annotations
import glob, os
from dataclasses import dataclass
import numpy as np
from .gridio import GridGeo, read_asc, read_shapefile_parts, rasterize_polylines

BURNABLE_DEFAULT = (33, 34, 35, 36)


def find_project_root(start=None):
    p = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(p, "newarea")) and os.path.isdir(os.path.join(p, "output")):
            return p
        q = os.path.dirname(p)
        if q == p:
            raise FileNotFoundError("Could not find the Broomhead project root (needs 'newarea' and 'output').")
        p = q


@dataclass
class Landscape:
    geo: GridGeo
    elevation: np.ndarray
    slope: np.ndarray            # degrees
    aspect: np.ndarray           # degrees, direction the slope faces (downslope azimuth)
    fuel: np.ndarray             # int fuel-model codes, 0 = nodata
    domain: np.ndarray           # True inside the study polygon (valid elevation)

    def burnable(self, codes):
        return np.isin(self.fuel, list(codes)) & self.domain & np.isfinite(self.slope)


def load_landscape(root, fuel_file="mergedPatchesAndWhole.asc"):
    n = os.path.join(root, "newarea")
    elev, geo = read_asc(os.path.join(n, "elevation.asc"))
    slope, _ = read_asc(os.path.join(n, "slope.asc"))
    asp, _ = read_asc(os.path.join(n, "aspect.asc"))
    fp = fuel_file if os.path.isabs(fuel_file) else (
        os.path.join(n, fuel_file) if os.path.exists(os.path.join(n, fuel_file)) else
        glob.glob(os.path.join(n, "**", fuel_file), recursive=True)[0])
    fuel, _ = read_asc(fp)
    fuel = np.nan_to_num(fuel, nan=0).astype(int)
    domain = np.isfinite(elev)
    return Landscape(geo, elev, np.where(np.isfinite(slope), slope, 0.0), np.where(np.isfinite(asp), asp, 0.0),
                     fuel, domain & np.isfinite(slope))


def find_ignition_shapefile(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    hits = [h for h in hits if os.sep + "output" + os.sep not in h and "C-output" not in h]
    if not hits:
        raise FileNotFoundError(name)
    hits.sort(key=lambda h: (("ignition line" not in h), len(h)))
    return hits[0]


def ignition_mask(root, name, geo):
    return rasterize_polylines(read_shapefile_parts(find_ignition_shapefile(root, name)), geo)


def ignition_mask_from_fzp(folder_path):
    """Ignition cells of a FARSITE run family read straight from its own output, for a line whose shapefile was never
    delivered into the project (its FARSITE run log names a file that only ever existed on the machine that ran
    FlamMap). Every run of a family ignites the same cells, so any one completed, readable run in the folder gives
    the mask; this checks that against a second run when more than one is available."""
    import glob as _glob
    from .fzp import load_fzp
    mask = None
    for p in sorted(_glob.glob(os.path.join(folder_path, "*.fzp"))):
        try:
            r = load_fzp(p)
        except Exception:
            continue
        z = np.isfinite(r.arrival) & (r.arrival == 0)
        if mask is None:
            mask = z
        elif not np.array_equal(mask, z):
            raise ValueError(f"{p}: ignition cells differ from other runs in {folder_path} - not a single fixed ignition line")
    if mask is None:
        raise FileNotFoundError(f"no readable .fzp found in {folder_path} to derive an ignition mask from")
    return mask
