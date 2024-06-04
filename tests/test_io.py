
from . import conftest


def test_points_indexing(small_data):
    # Check points indexing
    assert "cell_boundaries" in small_data.points["transcripts"].columns
    assert "nucleus_boundaries" in small_data.points["transcripts"].columns


def test_shapes_indexing(small_data):
    # Check shapes are added to sdata
    assert "cell_boundaries" in small_data.shapes["cell_boundaries"].columns
    assert "cell_boundaries" in small_data.shapes["nucleus_boundaries"].columns
    assert "nucleus_boundaries" in small_data.shapes["cell_boundaries"].columns

    # check shape indexing is accurate in both directions
    assert small_data["cell_boundaries"][
        "nucleus_boundaries"
    ].to_dict() == conftest.CELL_TO_NUCLEUS_MAP
    assert small_data["nucleus_boundaries"][
        "cell_boundaries"
    ].to_dict() == conftest.NUCLEUS_TO_CELL_MAP


def test_points_attrs(small_data):
    # Check points attrsf
    assert "transform" in small_data.points["transcripts"].attrs.keys()
    assert (
        small_data.points["transcripts"].attrs["spatialdata_attrs"]["feature_key"]
        == "feature_name"
    )
    assert (
        small_data.points["transcripts"].attrs["spatialdata_attrs"]["instance_key"]
        == "cell_boundaries"
    )


def test_shapes_attrs(small_data):
    # Check shapes attrs
    assert "transform" in small_data.shapes["cell_boundaries"].attrs.keys()
    assert "transform" in small_data.shapes["nucleus_boundaries"].attrs.keys()


def test_index_dtypes(small_data):
    # Check index dtypes
    assert small_data.shapes["cell_boundaries"].index.dtype == "object"
    assert small_data.shapes["nucleus_boundaries"].index.dtype == "object"
    assert small_data.points["transcripts"].index.dtype == "int64"
