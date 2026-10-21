"""Plots and animation for the Broomhead CA."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from .ca import UNBURNED, BURNING, BURNT, UNBURNABLE


def hillshade(elev, cellsize=1.0, az=315.0, alt=45.0):
    e = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    gy, gx = np.gradient(e, cellsize)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(360 - az + 90), np.radians(90 - alt)
    hs = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def _extent(geo):
    return (geo.xll, geo.xll + geo.ncols * geo.cellsize, geo.yll, geo.yll + geo.nrows * geo.cellsize)


def _panel(ax, land, data, cmap, vmin, vmax, title, label, levels=None):
    ext = _extent(land.geo)
    ax.imshow(np.where(land.domain, hillshade(land.elevation, land.geo.cellsize), np.nan), cmap="gray", extent=ext, vmin=0, vmax=1.2)
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, extent=ext, vmin=vmin, vmax=vmax, alpha=0.85)
    if levels is not None and np.isfinite(data).any():
        X, Y = land.geo.cell_centres()
        ax.contour(X, Y, np.where(np.isfinite(data), data, np.nan), levels=levels, colors="k", linewidths=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    return im


def plot_run_comparison(run, T_ca, land, path, title=None):
    Tf = run.arrival
    tmax = np.nanmax(Tf)
    Tc = np.where(np.isfinite(T_ca) & (T_ca <= tmax), T_ca, np.nan)
    diff = np.where(np.isfinite(Tf) & np.isfinite(T_ca), T_ca - Tf, np.nan)
    lim = max(1.0, float(np.nanpercentile(np.abs(diff), 98)))
    levels = np.linspace(0, tmax, 9)[1:-1]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    _panel(axs[0], land, Tf, "turbo", 0, tmax, "FARSITE arrival time", "min", levels)
    _panel(axs[1], land, Tc, "turbo", 0, tmax, "Cellular automaton arrival time", "min", levels)
    _panel(axs[2], land, diff, "RdBu_r", -lim, lim, "CA minus FARSITE", "min (red = CA later)")
    fig.suptitle(title or f"{run.name}  |  wind {run.wind_kmh:.0f} km/h from {run.wind_from_deg:.0f} deg", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_arrival_scatter(pairs, path, max_pts=40000, seed=0):
    """pairs: list of (label, FARSITE arrival grid, CA arrival grid)."""
    n = len(pairs)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, (lab, Tf, Tc) in zip(axs[0], pairs):
        m = np.isfinite(Tf) & np.isfinite(Tc)
        i = np.flatnonzero(m.ravel())
        i = rng.choice(i, min(max_pts, i.size), replace=False)
        x, y = Tf.ravel()[i], Tc.ravel()[i]
        ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        lim = max(x.max(), y.max()) * 1.02
        ax.plot([0, lim], [0, lim], "r-", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("FARSITE arrival (min)"); ax.set_ylabel("CA arrival (min)")
        ax.set_title(lab, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_evaluation(df, path):
    splits = ["train", "held-out weather", "held-out ignition"]
    models = [m for m in ("stage1", "final") if m in set(df["model"])]
    colors = {"stage1": "#9ecae1", "final": "#08519c"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, ttl) in zip(axs, [("rel_mae", "Relative arrival-time error (MAE / T)"), ("iou_100", "Burned-area IoU at end of run"),
                                    ("corr", "Correlation of arrival times")]):
        w = 0.36
        for j, m in enumerate(models):
            data = [df[(df.model == m) & (df.split == s)][col].dropna().values for s in splits]
            pos = np.arange(len(splits)) + (j - (len(models) - 1) / 2) * w
            bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(colors[m]); b.set_alpha(0.9)
            for med in bp["medians"]:
                med.set_color("k")
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels([f"{s}\n(n={int(((df.model == models[-1]) & (df.split == s)).sum())})" for s in splits], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend([plt.Rectangle((0, 0), 1, 1, fc=colors[m]) for m in models], ["stage 1 (cell-level)" if m == "stage1" else "final (stage 2)" for m in models], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cell_level(prob, spec, path):
    from .calibrate import CLASSES
    fig, axs = plt.subplots(2, len(CLASSES), figsize=(3.6 * len(CLASSES), 7))
    for j, k in enumerate(CLASSES):
        i = prob.idx[k]
        ros, fl = prob.predict(spec, k)
        for r, (obs, pred, nm) in enumerate([(prob.t["ros"][i], ros, "ROS (m/min)"), (prob.t["fl"][i], fl, "flame length (m)")]):
            ax = axs[r, j]
            ax.hexbin(np.log10(obs), np.log10(pred), gridsize=45, bins="log", cmap="viridis", mincnt=1)
            lo, hi = np.log10(min(obs.min(), pred.min())), np.log10(max(obs.max(), pred.max()))
            ax.plot([lo, hi], [lo, hi], "r-", lw=1)
            ax.set_title(f"fuel {k}: {nm}", fontsize=9)
            ax.set_xlabel("FARSITE (log10)"); ax.set_ylabel("Rothermel-Huygens (log10)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


_PAL = np.array([[236, 236, 236], [230, 60, 20], [70, 70, 70], [255, 255, 255]], np.uint8)   # unburned, burning, burnt, barrier


def frame_rgb(land, state, t_min=None, scale=2):
    hs = hillshade(land.elevation, land.geo.cellsize)
    base = (np.where(land.domain, 150 + 90 * hs, 255)).astype(np.uint8)
    img = np.stack([base] * 3, -1).astype(float)
    burnt = state == BURNT
    burning = state == BURNING
    img[burnt] = 0.35 * img[burnt] + 0.65 * np.array([60, 50, 50])
    img[burning] = np.array([255, 110, 20])
    img[(state == UNBURNABLE) & land.domain] = 0.55 * img[(state == UNBURNABLE) & land.domain] + 0.45 * np.array([90, 150, 90])
    im = Image.fromarray(img.astype(np.uint8)).resize((land.geo.ncols * scale, land.geo.nrows * scale), Image.NEAREST)
    if t_min is not None:
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        d.rectangle([4, 4, 118, 22], fill=(255, 255, 255))
        d.text((8, 8), f"t = {t_min:6.1f} min", fill=(0, 0, 0))
    return im


def make_gif(env, ignition, path, n_frames=90, T_total=None, scale=2, fps=12):
    """Animate the time-stepped CA (states: burning front in orange, burnt-out cells dark)."""
    from .ca import HuygensCA, solve_arrival_mtt
    if T_total is None:
        T_total = float(np.nanmax(solve_arrival_mtt(env, ignition, passes=1)))
    every = T_total / n_frames
    frames = []
    ca = HuygensCA(env, ignition)
    def cb(t, c):
        frames.append(frame_rgb(env.land, c.state(t), t, scale))
    ca.run(t_end=T_total * 1.001, callback=cb, every=every)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return ca


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


# Fireline-intensity classes and colours, the same as the patch-crossing / perimeter-intensity figures
# (C-output/2026-10-04_2214_patch_crossing_perimeter_intensity_new_matrix/broomhead_ca/patch_crossing_figures.py)
I_BINS = [0, 500, 2000, 4000, 10000, 1e9]
I_LABELS = ["< 500", "500-2000", "2000-4000", "4000-10000", "> 10000"]
I_NOTE = ["hand tools can hold", "machinery needed", "aircraft / indirect", "indirect attack only", "extreme"]
I_CMAP = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0.12, 0.95, 5)))
I_NORM = matplotlib.colors.BoundaryNorm(I_BINS, I_CMAP.N)
C_PATCH = "#8f8a80"        # managed patch / non-fuel
C_UNBURNED = "#dfe6cf"     # burnable heather the fire has not reached
C_IGNITION = "#0072b2"
C_PERIMETER = "#1f1f1f"
C_FRONT = "#ff0000"        # advancing fire front in the animation


def _rgb255(hex_or_rgba):
    return np.array(matplotlib.colors.to_rgb(hex_or_rgba)) * 255.0


def fireline_intensity(flame_length_m):
    """Byram fireline intensity (kW/m) from flame length (m): FL = 0.0775 * I^0.46."""
    return np.power(np.asarray(flame_length_m, float) / 0.0775, 1.0 / 0.46)


def intensity_class(I):
    """0..4 = the five intensity classes (I_LABELS); -1 where there is no intensity."""
    c = np.digitize(np.nan_to_num(I, nan=-1.0), I_BINS[1:-1])
    return np.where(np.isfinite(I), c, -1)


def perimeter_step(T):
    t95 = float(np.nanpercentile(T, 95)) if np.isfinite(T).any() else 0.0
    for s in (2, 5, 10, 15, 20, 30, 60, 120):
        if t95 / s <= 8:
            return s
    return 240


def plot_intensity_map(land, I, path, T=None, ignition=None, burnable=None, title=""):
    """Fireline intensity in the five classes, with fire perimeters every N minutes (same style as the
    perimeter-intensity figures)."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(8.5, 7.8))
    dom = land.domain
    nofuel = dom & ~burnable if burnable is not None else np.zeros_like(dom)
    ax.imshow(np.where(dom, 1, np.nan), cmap=ListedColormap([C_UNBURNED]), interpolation="nearest")
    ax.imshow(np.where(nofuel, 1, np.nan), cmap=ListedColormap([C_PATCH]), interpolation="nearest")
    ax.imshow(np.where(np.isfinite(I), I, np.nan), cmap=I_CMAP, norm=I_NORM, interpolation="nearest")
    step = None
    if T is not None and np.isfinite(T).any():
        step = perimeter_step(T)
        Tm = np.where(np.isfinite(T), T, np.nanmax(T) + 10 * step)
        lev = np.arange(step, np.nanmax(T) + step, step)
        if lev.size:
            cs = ax.contour(Tm, levels=lev, colors=C_PERIMETER, linewidths=0.6)
            ax.clabel(cs, fmt="%d", fontsize=6.5, inline_spacing=1)
    if ignition is not None:
        ax.contour(np.asarray(ignition, float), levels=[0.5], colors=[C_IGNITION], linewidths=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + (f"\nfire perimeters every {step} min" if step else ""), fontsize=9.5)
    h = [Patch(fc=I_CMAP(i), ec="none", label=f"{I_LABELS[i]} kW/m ({I_NOTE[i]})") for i in range(5)]
    h += [Patch(fc=C_PATCH, label="managed patch / non-fuel"), Patch(fc=C_UNBURNED, label="heather not reached by the fire"),
          plt.Line2D([], [], color=C_PERIMETER, lw=1, label="fire perimeter"),
          plt.Line2D([], [], color=C_IGNITION, lw=2, label="ignition")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96)); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def _ignition_mask(ignition):
    ig = np.asarray(ignition, bool).copy()           # thicken the ignition line so it stays visible
    for ax_ in (0, 1):
        for sh in (-1, 1):
            ig |= np.roll(np.asarray(ignition, bool), sh, axis=ax_)
    return ig


def _frame_canvas(img, scale, header, rows, title="", wind_from_deg=None, wind_kmh=None, subheader=""):
    """Turn an RGB grid into one animation frame: header text, wind arrow and legend rows below the map."""
    from PIL import ImageDraw
    Hc, Wc = img.shape[:2]
    W, H = Wc * scale, Hc * scale
    head, foot = (52 if subheader else 34), 16 + 24 * len(rows)
    f_big, f_small = _font(16), _font(12)
    pic = Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    canvas = Image.new("RGB", (W, H + head + foot), (255, 255, 255))
    canvas.paste(pic, (0, head))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), header, fill=(0, 0, 0), font=f_big)
    if subheader:
        d.text((8, 30), subheader, fill=(60, 60, 60), font=f_small)
    if title:
        d.text((W - 8, 10), title, fill=(60, 60, 60), font=f_small, anchor="ra")
    if wind_from_deg is not None:                # arrow points where the wind blows TO
        cx, cy, r = W - 40, head + 40, 24
        a = np.radians(wind_from_deg + 180.0)
        dx, dy = np.sin(a), -np.cos(a)
        tip, tail = (cx + r * dx, cy + r * dy), (cx - r * dx, cy - r * dy)
        d.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 18], fill=(255, 255, 255))
        d.line([tail, tip], fill=(20, 20, 20), width=3)
        px, py = -dy, dx
        d.polygon([tip, (tip[0] - 9 * dx + 5 * px, tip[1] - 9 * dy + 5 * py),
                   (tip[0] - 9 * dx - 5 * px, tip[1] - 9 * dy - 5 * py)], fill=(20, 20, 20))
        lab = "wind" if wind_kmh is None else f"{wind_kmh:g} km/h"
        d.text((cx, cy + r + 4), lab, fill=(20, 20, 20), font=f_small, anchor="ma")
    y = H + head + 8
    for items in rows:
        x = 8
        for col, lab in items:
            d.rectangle([x, y, x + 14, y + 14], fill=tuple(int(v) for v in col), outline=(90, 90, 90))
            d.text((x + 19, y + 1), lab, fill=(0, 0, 0), font=f_small)
            x += 34 + int(d.textlength(lab, font=f_small))
        y += 24
    return canvas


def spread_frames(land, T, ignition, burnable, flame_length=None, n_frames=80, scale=2, wind_from_deg=None,
                  wind_kmh=None, title=""):
    """Frames of the fire spreading over the land, drawn from an arrival-time grid T (minutes).

    Burned ground is coloured by fireline intensity class (from the flame-length grid), with the colours of the
    perimeter-intensity figures: pale yellow < 500 kW/m ... dark red > 10000 kW/m. The red line is the advancing
    fire front, blue the ignition, grey the managed patches / non-fuel, pale green heather not reached yet.
    Returns a list of (t_min, PIL.Image)."""
    T = np.asarray(T, float)
    reached = np.isfinite(T)
    t_end = float(np.nanmax(T)) if reached.any() else 0.0
    edge_min = max(t_end / max(n_frames, 1), 0.5)
    n_burn = max(int((burnable & land.domain).sum()), 1)
    hs = hillshade(land.elevation, land.geo.cellsize)
    shade = (0.82 + 0.18 * hs)[..., None]                     # a hint of the terrain under every colour
    base = np.full(land.domain.shape + (3,), 255.0)
    base[land.domain] = _rgb255(C_UNBURNED)
    base[land.domain & ~burnable] = _rgb255(C_PATCH)
    base = np.where(land.domain[..., None], base * shade, base)
    class_rgb = np.array([_rgb255(I_CMAP(i)) for i in range(5)])
    if flame_length is not None:
        cls = intensity_class(fireline_intensity(flame_length))
        fire = np.where((cls >= 0)[..., None], class_rgb[np.clip(cls, 0, 4)], class_rgb[0]) * shade
    else:
        fire = np.broadcast_to(class_rgb[2], base.shape) * shade
    ig = _ignition_mask(ignition)
    rows = [[(class_rgb[i], f"{I_LABELS[i]} kW/m") for i in range(5)],
            [(_rgb255(C_FRONT), "fire front"), (_rgb255(C_IGNITION), "ignition"),
             (_rgb255(C_PATCH), "managed patch / non-fuel"), (_rgb255(C_UNBURNED), "heather not reached")]]
    frames = []
    for t in np.linspace(0.0, t_end, n_frames + 1):
        img = base.copy()
        burnt = reached & (T <= t)
        img[burnt] = fire[burnt]
        if t < t_end:
            img[burnt & (t - T < edge_min)] = _rgb255(C_FRONT)
        img[ig] = _rgb255(C_IGNITION)
        pct = 100.0 * burnt.sum() / n_burn
        frames.append((float(t), _frame_canvas(img, scale, f"t = {t:6.1f} min    burned {pct:5.1f} %", rows,
                                               title, wind_from_deg, wind_kmh)))
    return frames


# Fuel models, light green = shortest fuel bed -> dark green = tallest (heather classes 33-36 of the landscape)
FUEL_GREENS = ["#c7e9c0", "#74c476", "#31a354", "#00441b"]
FUEL_GREYS = ["#d2d2d2", "#a4a4a4", "#767676", "#474747"]     # the same classes once burned (darker = taller)


def fuel_spread_frames(land, T, ignition, burnable, fuels, n_frames=80, scale=2, wind_from_deg=None,
                       wind_kmh=None, title=""):
    """Frames of the fire spreading over the fuel-model map.

    Unburned fuel is green, light for the shortest fuel bed to dark for the tallest (ordered by the fuel model's
    depth); once a cell has burned it turns grey (darker grey = taller fuel), so you can see which fuels the fire
    runs through. Red = advancing fire front, blue = ignition, warm grey = managed patch / non-fuel. The header
    gives the share of each fuel class burned so far. Uses the same times as spread_frames, so the two stay in step.
    fuels: {code: FuelModel} of the burnable fuel models. Returns a list of (t_min, PIL.Image)."""
    T = np.asarray(T, float)
    reached = np.isfinite(T)
    t_end = float(np.nanmax(T)) if reached.any() else 0.0
    edge_min = max(t_end / max(n_frames, 1), 0.5)
    codes = sorted((k for k in fuels if (land.fuel == k).any()), key=lambda k: fuels[k].depth)
    pal_g = [FUEL_GREENS[round(i * (len(FUEL_GREENS) - 1) / max(len(codes) - 1, 1))] for i in range(len(codes))]
    pal_b = [FUEL_GREYS[round(i * (len(FUEL_GREYS) - 1) / max(len(codes) - 1, 1))] for i in range(len(codes))]
    hs = hillshade(land.elevation, land.geo.cellsize)
    shade = (0.82 + 0.18 * hs)[..., None]
    base = np.full(land.domain.shape + (3,), 255.0)
    base[land.domain] = _rgb255(C_PATCH)
    burnt_rgb = base.copy()
    masks = []
    for k, g, b in zip(codes, pal_g, pal_b):
        m = (land.fuel == k) & burnable
        base[m], burnt_rgb[m] = _rgb255(g), _rgb255(b)
        masks.append(m)
    base = np.where(land.domain[..., None], base * shade, base)
    burnt_rgb = np.where(land.domain[..., None], burnt_rgb * shade, burnt_rgb)
    n_k = [max(int(m.sum()), 1) for m in masks]
    ig = _ignition_mask(ignition)
    rows = [[(_rgb255(g), f"fuel {k} ({fuels[k].depth * 0.3048:.2f} m)") for k, g in zip(codes, pal_g)],
            [(_rgb255(b), f"{k} burned") for k, b in zip(codes, pal_b)],
            [(_rgb255(C_FRONT), "fire front"), (_rgb255(C_IGNITION), "ignition"),
             (_rgb255(C_PATCH), "managed patch / non-fuel")]]
    frames = []
    for t in np.linspace(0.0, t_end, n_frames + 1):
        img = base.copy()
        burnt = reached & (T <= t)
        img[burnt] = burnt_rgb[burnt]
        if t < t_end:
            img[burnt & (t - T < edge_min)] = _rgb255(C_FRONT)
        img[ig] = _rgb255(C_IGNITION)
        share = "   ".join(f"{k}: {100.0 * (burnt & m).sum() / n:4.0f}%" for k, m, n in zip(codes, masks, n_k))
        frames.append((float(t), _frame_canvas(img, scale, f"t = {t:6.1f} min    fuel models", rows, title,
                                               wind_from_deg, wind_kmh, subheader="burned so far by fuel  " + share)))
    return frames


def save_gif(frames, path, fps=12, hold_s=2.0):
    ims = [im for _, im in frames]
    ims += [ims[-1]] * int(hold_s * fps)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path
