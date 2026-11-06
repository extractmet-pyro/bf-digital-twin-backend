# BF-TWIN backend — Blast Furnace Digital Twin API

**Extractmet Private Limited** · FastAPI service behind the BF-TWIN operator desk.

Hybrid physics–AI digital twin of an ironmaking blast furnace producing liquid hot metal:
first-principles transient model (shaft, cohesive/dripping zones, raceway, hearth, wall heat loss),
process simulator with thermal closure, physics-informed neural surrogate, NSGA-II
multi-objective optimisation, GRU-based nonlinear MPC and soft sensors.

## Repository layout
```
app/
  main.py            FastAPI app (all endpoints, CORS, gzip, optional /ui static mount)
  core/              physics: thermo, materials & presets, geometry, heat loss, slag,
                     furnace (transient solver), simulate (steady/dynamic/report), economics
  ai/                physics (balance operator), nn (PINN MLP), gbm, gru, nsga2, surrogate, control
  data/              shipped artefacts: base_report.json, presets/, dynamic/, study_results.json,
                     surrogate_*.npz, gru_thermal.npz, softsensor_gbm.json, validation.json, lhs_dataset.npz
tools/
  gen_lhs.py         Latin-hypercube first-principles data set (≈25 min on 2 cores)
  train_all.py       trains surrogates, runs the optimisation / GRU / soft-sensor / control studies
  build_cache.py     pre-computes preset reports, scenario trajectories and validation.json
tests/               pytest: balance closure, closure target, API smoke tests
render.yaml          Render Blueprint (web service)
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```
One-command desk + API (frontend folder next to this one):
```bash
FRONTEND_DIR=../bf-digital-twin-frontend uvicorn app.main:app --port 8000
# open http://localhost:8000/ui/
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Rebuild model artefacts (optional — they are shipped in app/data)
```bash
python tools/gen_lhs.py          # first-principles LHS data set
python tools/train_all.py        # surrogates + studies  (or: python tools/train_all.py soft,control)
python tools/build_cache.py      # presets, scenario caches, validation report
```

## Deploy on Render
1. Push this folder to a GitHub repository `bf-digital-twin-backend`.
2. Render → **New → Blueprint** → pick the repository → **Apply** (uses `render.yaml`).
3. Wait for `/api/health` to turn green; note the URL, e.g. `https://bf-twin-api.onrender.com`.
4. Set `ALLOWED_ORIGINS` to your frontend URL (comma-separated list allowed).

Environment variables: `ALLOWED_ORIGINS` (CORS, default `*`), `FRONTEND_DIR` (optional static UI),
`PYTHON_VERSION` (Render build), `OMP_NUM_THREADS=1`.

## Main endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status + artefact availability |
| GET | `/api/meta` | presets, materials, refractories, scenarios |
| GET | `/api/base?preset=BF-2800` | pre-computed base case |
| POST | `/api/steady` | steady state with overrides (`geometry`, `operation`, `materials`, `lining`, `cooling`, `utilities`, `closure`) |
| POST | `/api/dynamic` | transient trajectory (`scenario`, `hours`, `events`) |
| POST | `/api/geometry` | furnace profile and lining drawing |
| POST | `/api/heatloss` | wall heat-loss evaluation of a lining design |
| POST | `/api/raceway`, `/api/slag` | sub-model what-ifs |
| POST | `/api/optimise` | NSGA-II on the physics-informed surrogate |
| POST | `/api/surrogate/predict` | surrogate KPIs (with thermal closure for 6 inputs) |
| POST | `/api/mpc/advice` | one NMPC advisory move |
| GET | `/api/study`, `/api/validation`, `/api/presets/summary`, `/api/thermo` | study, validation and reference data |

© 2026 Extractmet Private Limited. Proprietary — see LICENSE.
