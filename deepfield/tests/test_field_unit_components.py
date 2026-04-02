"""unit tests for field package objects (see tutorials/01.Basics, 02.Grid, 04.Wells, 05.Faults)."""
import pandas as pd
from pathlib import Path

import numpy as np
import pytest

from deepfield.field.aquifer import Aquifers
from deepfield.field.base_component import BaseComponent
from deepfield.field.faults import Faults
from deepfield.field.grids import OrthogonalGrid
from deepfield.field.states import States
from deepfield import Field

EGG_MINI_DATA = Path(__file__).resolve().parent / "data" / "egg_mini" / "Egg_Mini.DATA"
OPEN_EGG_DATA = Path(__file__).resolve().parents[2] / "open_data" / "egg" / "Egg_Model_ECL.DATA"


def test_base_component_setitem_and_copy():
    c = BaseComponent()
    c["FOO"] = np.array([1.0, 2.0])
    assert "FOO" in c
    assert np.array_equal(c.foo, [1.0, 2.0])
    d = c.copy()
    assert d.foo.shape == c.foo.shape
    d.foo[0] = 99.0
    assert c.foo[0] == 1.0


def test_aquifers_empty_structure():
    a = Aquifers()
    assert a.names == ()
    assert list(a.items()) == []


def test_faults_root_only():
    f = Faults()
    assert f.root.name == "FIELD"
    assert f.root.ntype == "group"


def test_states_construct_with_dates():
    s = States(dates=pd.to_datetime([]))
    assert s.n_timesteps == 0


@pytest.fixture(scope="module")
def egg_mini_field():
    assert EGG_MINI_DATA.is_file()
    return Field(str(EGG_MINI_DATA)).load(include_binary=False)


def test_orthogonal_grid_type_and_geometry(egg_mini_field):
    g = egg_mini_field.grid
    assert isinstance(g, OrthogonalGrid)
    vol = g.cell_volumes
    assert vol.shape == (g.actnum_ids.size,)
    assert np.all(vol > 0)


def test_rock_ravel_active_subset(egg_mini_field):
    r = egg_mini_field.rock
    aid = egg_mini_field.grid.actnum_ids
    poro_r = r.poro.ravel(order="F")
    active_poro = poro_r[aid]
    assert active_poro.size == aid.size
    assert np.allclose(active_poro, 0.2)


def test_tables_swof_dataframe(egg_mini_field):
    swof = egg_mini_field.tables.SWOF
    assert swof.shape[0] >= 2


def test_wells_indexing(egg_mini_field):
    w = egg_mini_field.wells
    assert "PROD1" in w
    prod = w["PROD1"]
    assert prod.name == "PROD1"


@pytest.mark.skipif(not OPEN_EGG_DATA.is_file(), reason="open_data/egg deck not available")
def test_open_data_egg_matches_tutorial_grid_dims():
    # tutorials/02.Grid.ipynb: model_egg = Field('../open_data/egg/Egg_Model_ECL.DATA').load(include_binary=False)
    m = Field(str(OPEN_EGG_DATA)).load(include_binary=False)
    assert tuple(np.asarray(m.grid.dimens).ravel()) == (60, 60, 7)
    assert "PORO" in m.rock
