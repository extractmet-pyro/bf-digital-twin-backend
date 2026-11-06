"""Core physics tests: balances close every step, steady closure hits target, heat loss sane."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.simulate import solve_steady, kpi_block, full_report, run_dynamic   # noqa: E402
from app.core.geometry import Geometry                                            # noqa: E402
from app.core.materials import default_inputs, PRESETS                             # noqa: E402
from app.core import thermo as TH                                                 # noqa: E402
from app.core import slag as SL                                                   # noqa: E402


@pytest.fixture(scope="module")
def base():
    return solve_steady(None)


def test_steady_closure_hits_target(base):
    k = kpi_block(base)
    assert abs(k["hmt_C"] - 1500.0) < 5.0
    assert 280 < k["coke_rate"] < 420
    assert 0.40 < k["eta_CO"] < 0.56


def test_element_and_enthalpy_balance_close(base):
    rep = full_report(base)
    assert rep["balances"]["max_rel_element_residual"] < 1e-9
    assert abs(rep["balances"]["enthalpy_step_residual_MJ"]) < 1e-3


def test_dynamic_balances_every_step(base):
    _, res = run_dynamic(bf=copy.deepcopy(base), scenario="pci_trip", hours=6.0, record_every=1)
    assert max(abs(r["bal_max_rel"]) for r in res["trajectory"]) < 1e-9
    hmt = [r["hmt_C"] for r in res["trajectory"]]
    assert min(hmt) < hmt[0]          # losing PCI cools the hearth


def test_geometry_all_presets():
    for key in PRESETS:
        g = Geometry(default_inputs(key)["geometry"], 36)
        assert abs(g.summary()["working_volume_m3"] - PRESETS[key]["V_work"]) / PRESETS[key]["V_work"] < 0.02


def test_thermo_equilibria():
    # Fe/FeO with CO: equilibrium CO fraction ~ 0.6-0.75 between 800 and 1200 C
    bg = TH.baur_glaessner()
    i = int(np.argmin(np.abs(np.array(bg["T_C"]) - 1000)))
    assert 55 < bg["FeO_Fe_CO"][i] < 80


def test_slag_properties():
    r = SL.slag_report({"CaO": 36, "SiO2": 34, "Al2O3": 18, "MgO": 8.5, "MnO": 0.5, "FeO": 0.4, "TiO2": 0.9}, 1773.15)
    assert 0.1 < r["viscosity_Pa_s"] < 2.0
    assert 1.0 < r["B2"] < 1.1
