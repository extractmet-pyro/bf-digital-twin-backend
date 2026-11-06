"""API smoke tests (FastAPI TestClient)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient   # noqa: E402
from app.main import app                    # noqa: E402

c = TestClient(app)


def test_health_and_meta():
    assert c.get("/api/health").json()["status"] == "ok"
    m = c.get("/api/meta").json()
    assert "BF-2800" in m["catalogue"]["presets"]


def test_base_and_geometry():
    r = c.get("/api/base").json()
    assert r["kpi"]["hmt_C"] > 1400
    g = c.post("/api/geometry", json={"preset": "BF-1050"}).json()
    assert g["summary"]["working_volume_m3"] > 1000


def test_steady_override():
    r = c.post("/api/steady", json={"operation": {"pci": 180}, "closure": True})
    assert r.status_code == 200
    assert abs(r.json()["kpi"]["hmt_C"] - 1500) < 6


def test_heatloss_and_slag_and_raceway():
    h = c.post("/api/heatloss", json={}).json()
    assert h["total_MW"] > 1
    s = c.post("/api/slag", json={"composition": {"CaO": 36, "SiO2": 34, "Al2O3": 18, "MgO": 8}, "T_C": 1500}).json()
    assert s["viscosity_Pa_s"] > 0
    w = c.post("/api/raceway", json={"operation": {"pci": 150}}).json()
    assert 1800 < w["T_raft_C"] < 2500


def test_ai_endpoints():
    if c.get("/api/health").json()["artifacts"]["surrogate_pinn.npz"]:
        p = c.post("/api/surrogate/predict", json={"x": [[4850, 1170, 4.5, 170, 12, 25]]}).json()
        assert 1400 < p["y"][0]["hmt"] < 1600
        o = c.post("/api/optimise", json={"pop": 24, "gens": 10}).json()
        assert len(o["front"]) > 0
