import pytest
import bento as bt

from . import conftest


@pytest.fixture(scope="module")
def point_features_data(small_data):
    bt.tl.analyze_points(
        sdata=small_data,
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
        feature_names=conftest.POINT_FEATURES,
        groupby=["feature_name"],
        recompute=False,
        progress=True,
    )

    return small_data


# Test case to check if multiple shape features are calculated for multiple shapes
def test_multiple_shapes_multiple_groups(point_features_data):
    point_features = (
        conftest.POINT_CELL_FEATURE_NAMES
        + conftest.POINT_NUCLEUS_FEATURE_NAMES
        + conftest.POINT_ONLY_FEATURE_NAMES
    )

    # Check if cell_boundaries and nucleus_boundaries point features are calculated
    assert all(
        feature
        in point_features_data.table.uns[
            "cell_boundaries_feature_name_features"
        ].columns
        for feature in point_features
    )
