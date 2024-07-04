import bento as bt
import pandas as pd
import geopandas as gpd
import spatialdata as sd
import dask.dataframe as dd


def test_sample_data():
    sdata = bt.ds.sample_data()

    # Check if the returned object is an instance of `bento.SpatialData`
    assert isinstance(sdata, sd.SpatialData)

    # Check if the required keys are present in the `sdata` object
    assert "transcripts" in sdata.points
    assert "cell_boundaries" in sdata.shapes
    assert "nucleus_boundaries" in sdata.shapes

    # Check if the types of the keys are correct
    assert isinstance(sdata.points["transcripts"], dd.DataFrame)
    assert isinstance(sdata.shapes["cell_boundaries"], gpd.GeoDataFrame)
    assert isinstance(sdata.shapes["nucleus_boundaries"], gpd.GeoDataFrame)

    # Check if the `feature_name` column is present in the `transcripts` DataFrame
    assert "feature_name" in sdata.points["transcripts"]

    # Check if the `cell_boundaries` and `nucleus_boundaries` shapes are present
    assert "cell_boundaries" in sdata.shapes
    assert "nucleus_boundaries" in sdata.shapes
    assert isinstance(sdata.shapes["cell_boundaries"], gpd.GeoDataFrame)
    assert isinstance(sdata.shapes["nucleus_boundaries"], gpd.GeoDataFrame)
