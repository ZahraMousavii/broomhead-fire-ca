# broomhead-fire-ca

A cellular-automaton fire-spread model for heather moorland, built around the Rothermel (1972) surface-fire
equations and a Huygens-wavelet propagation rule (the same principle FARSITE itself uses). The model is
calibrated against real FARSITE runs from the Broomhead moorland management project (Peak District, UK) and
used throughout that project to test patch-cutting designs, wind directions and weather scenarios that FARSITE
itself was never run for.

If you just want to point the model at your own landscape/fuel/weather without any of this project's specific
data or scenario machinery, see **[fire-ca-panel](https://github.com/ZahraMousavii/fire-ca-panel)** instead -
an interactive app built on the same physics. This repo is the research/calibration side: the scenario
definitions, the two-stage fit against FARSITE, and the command-line pipeline used to produce every result in
the Broomhead project.

## What's here vs. what isn't

The `bhca` package and the scripts below are everything needed to run and calibrate the model. The project's
own input data - the site's elevation/slope/aspect/fuel rasters, the FARSITE `.fzp` output used as calibration
ground truth, the run matrix, weather/moisture files and ignition-line shapefiles - is not included, since it's
the client's own survey and modelling data rather than something generic. Without it, `run_training.py` and
`run_all_scenarios.py` have nothing to calibrate or evaluate against, but every module is otherwise ordinary,
readable Python with no missing pieces - `bhca.rothermel`, `bhca.ca` and `bhca.moisture` implement the fire
physics on their own and can be used (or read) independently of any of this project's own files.

## How it works

Every burnable cell carries a local fire-behaviour ellipse (head-fire rate of spread, length-to-breadth ratio,
direction of travel) from the Rothermel model. A burning cell offers fire to its 16 neighbours (8 compass + 8
knight-move directions, which approximate a smooth ellipse far better than 8 alone); the time to cross an edge
is distance divided by the Anderson (1983) elliptical radius of the source cell in that direction, and a cell
ignites at the time of the first offer that reaches it. `bhca.ca.solve_arrival_mtt` solves this as a Dijkstra
shortest-path problem (fast, used for almost every result); `bhca.ca.HuygensCA` solves the identical rule by
synchronous time-stepping instead, which is slower but can drive an animation and time-varying weather.

Dead-fuel moisture relaxes exponentially toward the weather's equilibrium moisture content (Simard 1968) with
calibrated time lags; live-fuel moisture is held constant (heather has no herbaceous component that cures).
Flame length and fireline intensity are read off with Byram's formulas from the reaction intensity, residence
time and local rate of spread.

Calibration (`bhca.calibrate`) is two stages of nonlinear least squares against FARSITE: first the fuel models
themselves, fit against FARSITE's own per-cell rate-of-spread and flame-length grids; then a handful of global
parameters (wind-adjustment factor, ellipse shape, moisture time lags, per-fuel-class multipliers) fit so that
the whole-landscape arrival-time field matches FARSITE's, while the cell-level fit stays in the same objective
so the second stage can't drift away from validated local fire behaviour just to chase arrival times.

## Install

Python 3.10+, no GDAL or other native dependencies:

```
pip install -r requirements.txt
```

## Run

```
cd bhca/..            # this folder, next to your copy of the project's input data
python run_training.py      --out ../results                                   # calibrate (~8 min)
python run_selfcheck.py     --model ../results/model_final.json                # consistency checks
python run_all_scenarios.py --model ../results/model_final.json --out ../results   # every scenario x every line
python make_figures.py      --results ../results
python build_workbook.py    --results ../results --out ../results              # Excel workbook
python run_simulation.py    --model ../results/model_final.json --scenario W06 --line east --out sim_W06 --compare --animate
```

`--line` takes any key in `bhca.scenarios.FAMILIES` (currently `east`, `north`, `mid_horizontal`,
`mid_vertical`, `rect10`, `rect5`, `west`, `south`, `southwest`). Free mode runs the CA on its own, without a
FARSITE run to compare against:

```
python run_simulation.py --model ../results/model_final.json --out sim_free \
    --ignition line-east.shp --wind 22 --wind-from 225 --moisture summer --temp 18 --rh 63
```

`--wind-from` is the direction the wind blows *from* (225 = south-west). `run_custom.py` and
`custom_run_config.json` give the same free-mode run from a JSON file instead of flags; `Broomhead_CA_panel.ipynb`
is a small Jupyter/`ipywidgets` version of the same thing for interactive parameter sweeps (`pip install
ipywidgets` and open it in Jupyter or VS Code's notebook editor).

## Files

| file | role |
|---|---|
| `bhca/gridio.py` | `.asc` and shapefile I/O, numpy only |
| `bhca/fzp.py` | reads FARSITE `.fzp` packages (arrival time, flame length, ROS grids, run log) |
| `bhca/landscape.py` | landscape layers, project-root discovery, ignition rasterising |
| `bhca/scenarios.py` | scenario definitions from the run matrix and weather/moisture files; ignition families; input consistency checks |
| `bhca/rothermel.py` | Rothermel spread, wind-slope vector combination, Anderson ellipse, Byram flame length |
| `bhca/moisture.py` | dead-fuel moisture conditioning toward the weather equilibrium |
| `bhca/ca.py` | the cellular automaton (`HuygensCA`) and its fast shortest-path form (`solve_arrival_mtt`) |
| `bhca/tables.py` | supervised training tables built from FARSITE grids |
| `bhca/calibrate.py` | two-stage calibration, model save/load, `.fmd` export |
| `bhca/evaluate.py` | ties FARSITE runs to scenarios, runs the CA, scores it against FARSITE |
| `bhca/custom_run.py` | one-off runs from a config file or plain keyword arguments, outside the scenario/calibration machinery |
| `bhca/viz.py` | maps, comparison figures, animation |
| `run_training.py`, `run_all_scenarios.py`, `run_simulation.py`, `run_selfcheck.py`, `make_figures.py`, `build_workbook.py`, `run_custom.py` | command-line entry points |

## Limitations

- The calibration target is FARSITE, not observed fire behaviour on the ground.
- Not modelled: spotting, crown fire, fire acceleration (FARSITE's own output doesn't show them here either).
  Wind is held constant within a scenario.
- Fuel-moisture sensitivity is slightly weaker than FARSITE's own (the CA runs a little fast in summer/drought
  conditions, a little slow when the fuel is wet) - noted here because it's a real, measured bias, not something
  masked by the fit.

## License

MIT - see [LICENSE](LICENSE).
