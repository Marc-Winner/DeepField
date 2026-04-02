"""VFP dump behaviour using egg_mini deck (VFPTABL, VFPPROD, VFPINJ)."""
from pathlib import Path

import pytest

from deepfield import Field

EGG_MINI_DATA = Path(__file__).resolve().parent / "data" / "egg_mini" / "Egg_Mini.DATA"


def test_dump_raises_on_deferred_keywords(tmp_path):
    assert EGG_MINI_DATA.is_file(), EGG_MINI_DATA
    model = Field(str(EGG_MINI_DATA), loglevel="ERROR", lazy_keywords=("VFPPROD",)).load(
        include_binary=False
    )

    # dump() joins path + title then mkdir(title_dir); parent must exist (pytest tmp_path does)
    with pytest.raises(ValueError, match=r"load_lazy"):
        model.dump(path=str(tmp_path), mode="w", data=True, results=False)


def test_vfp_tables_dumped_before_dates_when_present(tmp_path):
    assert EGG_MINI_DATA.is_file(), EGG_MINI_DATA
    model = Field(str(EGG_MINI_DATA), loglevel="ERROR", lazy_keywords=("VFPPROD",)).load(
        include_binary=False
    )
    model.load_lazy(keywords=("VFPPROD",), raise_errors=True)

    model.dump(path=str(tmp_path), mode="w", data=True, results=False)

    title = model.meta.get("TITLE", "Untitled")
    schedule_inc = tmp_path / title / "INCLUDE" / "schedule.inc"
    content = schedule_inc.read_text(encoding="utf-8", errors="ignore")

    vfp_prod_pos = content.find("VFPPROD")
    vfp_inj_pos = content.find("VFPINJ")
    dates_pos = content.find("DATES")

    assert vfp_prod_pos != -1, "VFPPROD must be present in dumped schedule.inc"
    assert vfp_inj_pos != -1, "VFPINJ must be present in dumped schedule.inc"
    if dates_pos != -1:
        assert vfp_prod_pos < dates_pos, "VFPPROD must be emitted before the first DATES block"
        assert vfp_inj_pos < dates_pos, "VFPINJ must be emitted before the first DATES block"
