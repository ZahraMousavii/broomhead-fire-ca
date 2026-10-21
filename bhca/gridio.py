"""Minimal raster / shapefile I/O (numpy only) for the Broomhead project layers."""
from __future__ import annotations
import struct
from dataclasses import dataclass
import numpy as np


@dataclass
class GridGeo:
    """North-up grid geometry: row 0 is the northern edge; (xll, yll) is the lower-left corner."""
    ncols: int
    nrows: int
    xll: float
    yll: float
    cellsize: float

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    def xy_to_rc(self, x, y):
        col = np.floor((np.asarray(x) - self.xll) / self.cellsize).astype(int)
        row = self.nrows - 1 - np.floor((np.asarray(y) - self.yll) / self.cellsize).astype(int)
        return row, col

    def cell_centres(self):
        cols = self.xll + (np.arange(self.ncols) + 0.5) * self.cellsize
        rows = self.yll + (self.nrows - 1 - np.arange(self.nrows) + 0.5) * self.cellsize
        return np.meshgrid(cols, rows)


def read_asc(path):
    """Read an ESRI ASCII grid. Returns (array float64 with NaN for NODATA, GridGeo)."""
    hdr = {}
    with open(path, "r") as f:
        for _ in range(6):
            key, val = f.readline().split()
            hdr[key.lower()] = float(val)
        data = np.loadtxt(f, dtype=np.float64)
    nod = hdr.get("nodata_value", -9999.0)
    data = np.where(np.isclose(data, nod, rtol=1e-6, atol=1e-6) | (data < -1e30), np.nan, data)
    geo = GridGeo(int(hdr["ncols"]), int(hdr["nrows"]), hdr["xllcorner"], hdr["yllcorner"], hdr["cellsize"])
    return data, geo


def write_asc(path, arr, geo: GridGeo, nodata=-9999.0, fmt="%.4f"):
    a = np.where(np.isfinite(arr), arr, nodata)
    with open(path, "w") as f:
        f.write(f"ncols {geo.ncols}\nnrows {geo.nrows}\nxllcorner {geo.xll:.6f}\n"
                f"yllcorner {geo.yll:.6f}\ncellsize {geo.cellsize:.6f}\nNODATA_value {nodata}\n")
        np.savetxt(f, a, fmt=fmt)


def read_shapefile_parts(path):
    """Return a list of (n,2) coordinate arrays, one per polyline/polygon part (types 3, 5, 13, 15)."""
    with open(path, "rb") as f:
        d = f.read()
    parts_out, i = [], 100
    while i + 8 <= len(d):
        _, clen = struct.unpack(">ii", d[i:i + 8])
        c = d[i + 8:i + 8 + clen * 2]
        i += 8 + clen * 2
        t = struct.unpack("<i", c[:4])[0]
        if t in (3, 5, 13, 15):
            npart, npts = struct.unpack("<ii", c[36:44])
            idx = list(struct.unpack(f"<{npart}i", c[44:44 + 4 * npart])) + [npts]
            pts = np.frombuffer(c[44 + 4 * npart:44 + 4 * npart + 16 * npts], "<f8").reshape(-1, 2)
            for a, b in zip(idx[:-1], idx[1:]):
                parts_out.append(pts[a:b].copy())
    return parts_out


def rasterize_polylines(parts, geo: GridGeo, step_frac=0.25):
    """Burn polylines onto the grid: every cell touched by the densified line is True."""
    mask = np.zeros(geo.shape, dtype=bool)
    for p in parts:
        for a, b in zip(p[:-1], p[1:]):
            L = float(np.hypot(*(b - a)))
            n = max(int(np.ceil(L / (geo.cellsize * step_frac))), 1) + 1
            t = np.linspace(0.0, 1.0, n)
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            r, c = geo.xy_to_rc(x, y)
            ok = (r >= 0) & (r < geo.nrows) & (c >= 0) & (c < geo.ncols)
            mask[r[ok], c[ok]] = True
    return mask
