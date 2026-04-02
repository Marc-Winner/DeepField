"""integration: load egg-style mini deck (tutorials/02.Grid.ipynb uses egg with include_binary=False)."""
from pathlib import Path

import numpy as np
import pytest

from deepfield import Field
from deepfield.field.wells import Network

EGG_MINI_DATA = Path(__file__).resolve().parent / "data" / "egg_mini" / "Egg_Mini.DATA"


@pytest.fixture(scope="module")
def egg_mini_model():
    assert EGG_MINI_DATA.is_file(), EGG_MINI_DATA
    return Field(str(EGG_MINI_DATA)).load(include_binary=False)


def test_field_loads_egg_mini(egg_mini_model):
    m = egg_mini_model
    assert tuple(np.asarray(m.grid.dimens).ravel()) == (4, 4, 2)
    assert isinstance(m.grid.actnum, np.ndarray)
    assert tuple(m.grid.actnum.shape) == (4, 4, 2)


def test_rock_spatial_matches_grid(egg_mini_model):
    m = egg_mini_model
    assert "PORO" in m.rock
    assert tuple(m.rock.poro.shape) == (4, 4, 2)
    assert np.allclose(m.rock.poro, 0.2)


def test_tables_swof_from_props(egg_mini_model):
    m = egg_mini_model
    assert "SWOF" in m.tables
    swof = m.tables.SWOF
    assert swof.shape[0] >= 2
    assert swof.shape[1] >= 3


def test_wells_network_and_compdat(egg_mini_model):
    m = egg_mini_model
    assert isinstance(m.wells, Network)
    names = set(m.wells.main_branches)
    assert names == {"INJ1", "INJ2", "PROD1", "PROD2"}
    assert m.meta.get("MODEL_TYPE") == "ECL"


def test_network_schedule_keywords(egg_mini_model):
    m = egg_mini_model
    assert "NETWORK" in m.wells.root
    assert int(m.wells.root.network.iloc[0]["NODMAX"]) == 24
    assert "SEP" in m.wells
    assert "TVRK" in m.wells
    assert m.wells["PROD1"].VFP is not None
    assert m.wells["PROD1"].VFP.number == 1
    assert np.isclose(m.wells["SEP"].PRESS, 0.005)


def test_faults_and_aquifers_present(egg_mini_model):
    m = egg_mini_model
    assert m.faults.root.name == "FIELD"
    assert m.aquifers.names == ()


def test_states_empty_or_start_dated(egg_mini_model):
    m = egg_mini_model
    # no restart in mini deck; states may be empty
    if m.states.attributes:
        assert m.states.n_timesteps >= 1


def test_field_components_registry(egg_mini_model):
    m = egg_mini_model
    assert set(m.components) >= {"grid", "rock", "wells", "tables", "faults", "aquifers", "states"}
