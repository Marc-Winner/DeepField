from pathlib import Path

import pytest

from ..field import Field


def _network_demo_data_path():
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parents[1]  # .../DeepField/deepfield/tests -> .../DeepField
    egg_crm_root = project_root.parent / "egg-crm"
    return egg_crm_root / "tnav_models" / "network_demo" / "NETWORK_DEMO.DATA"


def test_dump_raises_on_deferred_keywords(tmp_path):
    data_path = _network_demo_data_path()
    model = Field(str(data_path), loglevel="ERROR", lazy_keywords=("VFPPROD",)).load()

    out_base = tmp_path / "dump_model"
    with pytest.raises(ValueError, match=r"load_lazy"):
        model.dump(path=str(out_base), mode="w", data=True, results=False)


def test_vfpprod_dumped_before_dates(tmp_path):
    data_path = _network_demo_data_path()
    model = Field(str(data_path), loglevel="ERROR", lazy_keywords=("VFPPROD",)).load()
    model.load_lazy(keywords=("VFPPROD",), raise_errors=True)

    out_base = tmp_path / "dump_model"
    model.dump(path=str(out_base), mode="w", data=True, results=False)

    title = model.meta.get("TITLE", "Untitled")
    schedule_inc = out_base / title / "INCLUDE" / "schedule.inc"
    content = schedule_inc.read_text(encoding="utf-8", errors="ignore")

    vfp_pos = content.find("VFPPROD")
    dates_pos = content.find("DATES")
    assert vfp_pos != -1, "VFPPROD must be present in dumped schedule.inc"
    assert dates_pos != -1, "DATES must be present in dumped schedule.inc"
    assert vfp_pos < dates_pos, "VFPPROD must be emitted before the first DATES block"

