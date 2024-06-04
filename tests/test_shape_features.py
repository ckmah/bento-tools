import pytest
import bento as bt

from . import conftest


@pytest.fixture(scope="module")
def shape_features_data(small_data):
    bt.tl.analyze_shapes(
        sdata=small_data,
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
        feature_names=conftest.SHAPE_FEATURES,
        feature_kws=conftest.OPENING_PARAMS,
        progress=False,
    )

    return small_data


# Test case to check if multiple shape features are calculated for multiple shapes
def test_multiple_shapes_multiple_features(shape_features_data):
    # Check if cell_boundaries shape features are calculated
    assert all(
        c in shape_features_data.shapes["cell_boundaries"].columns
        for c in conftest.CELL_FEATURES
    )

    # Check if nucleus_boundaries shape features are calculated
    for feature in conftest.NUCLEUS_FEATURES:
        assert feature in shape_features_data.shapes["nucleus_boundaries"].columns


# Test case to check if shape_stats function calculates area, aspect_ratio and density for both cell_boundaries and nucleus_boundaries
def test_shape_stats(shape_features_data):
    bt.tl.shape_stats(sdata=shape_features_data)

    # Check if cell_boundaries and nucleus_boundaries shape features are calculated
    cell_columns = shape_features_data.shapes["cell_boundaries"].columns
    assert "cell_boundaries_area" in cell_columns
    assert "cell_boundaries_aspect_ratio" in cell_columns
    assert "cell_boundaries_density" in cell_columns
    nucleus_columns = shape_features_data.shapes["nucleus_boundaries"].columns
    assert "nucleus_boundaries_area" in nucleus_columns
    assert "nucleus_boundaries_aspect_ratio" in nucleus_columns
    assert "nucleus_boundaries_density" in nucleus_columns
