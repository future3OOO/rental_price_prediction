import os
import pandas as pd
from prediction import _canonicalize_furnishings, _normalize_aliases2


def test_canonicalize_furnishings_lowercase():
    assert _canonicalize_furnishings("Furnished") == "furnished"
    assert _canonicalize_furnishings("UNFURNISHED") == "unfurnished"
    assert _canonicalize_furnishings("Partially Furnished") == "partially furnished"
    assert _canonicalize_furnishings("fully furnished") == "furnished"
    assert _canonicalize_furnishings(None) == "unfurnished"


def test_normalize_aliases2_maps_and_lowercases():
    df = pd.DataFrame({
        "Furnishings": ["Furnished", "PARTIALLY FURNISHED", None],
        "floor_size": [100, 120, 80],
    })
    expected = {"Furnishings", "Floor Size (sqm)", "floor_size"}
    out = _normalize_aliases2(df, expected)
    assert out.loc[0, "Furnishings"] == "furnished"
    assert out.loc[1, "Furnishings"] == "partially furnished"
    assert out.loc[2, "Furnishings"] == "unfurnished"
    # ensure size alias copied
    assert "Floor Size (sqm)" in out.columns
    assert out["Floor Size (sqm)"].iloc[0] == 100

