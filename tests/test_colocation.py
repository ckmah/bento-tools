import pytest

import bento as bt


@pytest.fixture(scope="module")
def colocation_data(small_data):
    bt.tl.coloc_quotient(small_data, shapes=["cell_boundaries"])
    bt.tl.colocation(small_data, ranks=range(1, 3 + 1), plot_error=False)

    return small_data


def test_coloc_quotient(colocation_data):
    # Check that clq is in small_data.table.uns
    assert "clq" in colocation_data.table.uns

    # Check that cell_boundaries is in colocation_data.table.uns["clq"]
    assert "cell_boundaries" in colocation_data.table.uns["clq"]

    coloc_quotient_features = [
        "feature_name",
        "neighbor",
        "clq",
        "cell_boundaries",
        "log_clq",
        "compartment",
    ]
    # Check columns are in clq["cell_boundaries"]
    for feature in coloc_quotient_features:
        assert feature in colocation_data.table.uns["clq"]["cell_boundaries"]

    # Check that tensor is in colocation_data.table.uns
    assert "tensor" in colocation_data.table.uns

    # Check that tensor_labels is in colocation_data.table.uns
    assert "tensor_labels" in colocation_data.table.uns

    # Check that tensor_names is in colocation_data.table.uns
    assert "tensor_names" in colocation_data.table.uns

    # Check keys are in tensor_labels
    for feature in colocation_data.table.uns["tensor_names"]:
        assert feature in colocation_data.table.uns["tensor_labels"]

    # Check that factors is in colocation_data.table.uns
    assert "factors" in colocation_data.table.uns

    # Check that keys are in factors
    for i in range(1, 3):
        assert i in colocation_data.table.uns["factors"]

    # Check that factors_error is in colocation_data.table.uns
    assert "factors_error" in colocation_data.table.uns
    assert "rmse" in colocation_data.table.uns["factors_error"]
    assert "rank" in colocation_data.table.uns["factors_error"]
