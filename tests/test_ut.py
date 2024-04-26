import unittest
import bento as bt
import spatialdata as sd
import pandas as pd
import geopandas as gpd
import dask as dd


class testUt(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.data = sd.read_zarr(f"{datadir}/merfish_sample.zarr")
        self.data = bt.io.prep(
            self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

    def test_get_points(self):
        pd_sync = bt.ut.get_points(
            sdata=self.data, points_key="transcripts", astype="pandas", sync=True
        )

        self.assertTrue(type(pd_sync) == pd.DataFrame)

        gdf_sync = bt.ut.get_points(
            sdata=self.data, points_key="transcripts", astype="geopandas", sync=True
        )

        self.assertTrue(type(gdf_sync) == gpd.GeoDataFrame)

        dd_sync = bt.ut.get_points(
            sdata=self.data, points_key="transcripts", astype="dask", sync=True
        )

        self.assertTrue(type(dd_sync) == dd.dataframe.core.DataFrame)

    def test_get_shape(self):
        sync = bt.ut.get_shape(
            sdata=self.data, shape_key="nucleus_boundaries", sync=True
        )

        self.assertTrue(type(sync) == gpd.GeoSeries)

    def test_set_points_metadata(self):
        list_metadata = [0] * len(self.data.points["transcripts"])
        series_metadata = pd.Series(list_metadata)
        dataframe_metadata = pd.DataFrame(
            {"0": list_metadata, "1": list_metadata, "2": list_metadata}
        )
        column_names = [
            "list_metadata",
            "series_metadata",
            "dataframe_metadata0",
            "dataframe_metadata1",
            "dataframe_metadata2",
        ]

        bt.ut.set_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata=list_metadata,
            columns=column_names[0],
        )
        bt.ut.set_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata=series_metadata,
            columns=column_names[1],
        )
        bt.ut.set_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata=dataframe_metadata,
            columns=[column_names[2], column_names[3], column_names[4]],
        )
        for column in column_names:
            self.assertTrue(column in self.data.points["transcripts"])

    def test_set_shape_metadata(self):
        list_metadata = [0] * len(self.data.shapes["cell_boundaries"])
        series_metadata = pd.Series(list_metadata)
        dataframe_metadata = pd.DataFrame(
            {"0": list_metadata, "1": list_metadata, "2": list_metadata}
        )
        column_names = [
            "list_metadata",
            "series_metadata",
            "dataframe_metadata0",
            "dataframe_metadata1",
            "dataframe_metadata2",
        ]

        bt.ut.set_shape_metadata(
            sdata=self.data,
            shape_key="cell_boundaries",
            metadata=list_metadata,
            column_names=column_names[0],
        )
        bt.ut.set_shape_metadata(
            sdata=self.data,
            shape_key="cell_boundaries",
            metadata=series_metadata,
            column_names=column_names[1],
        )
        bt.ut.set_shape_metadata(
            sdata=self.data,
            shape_key="cell_boundaries",
            metadata=dataframe_metadata,
            column_names=[column_names[2], column_names[3], column_names[4]],
        )
        for column in column_names:
            self.assertTrue(column in self.data.shapes["cell_boundaries"])

    def test_get_points_metadata(self):
        list_metadata = [0] * len(self.data.points["transcripts"])
        series_metadata = pd.Series(list_metadata)
        column_names = ["list_metadata", "series_metadata"]

        bt.ut.set_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata=list_metadata,
            columns=column_names[0],
        )
        bt.ut.set_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata=series_metadata,
            columns=column_names[1],
        )

        pd_metadata_single = bt.ut.get_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata_keys=column_names[0],
            astype="pandas",
        )
        dd_metadata_single = bt.ut.get_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata_keys=column_names[0],
            astype="dask",
        )
        pd_metadata = bt.ut.get_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata_keys=[column_names[0], column_names[1]],
            astype="pandas",
        )
        dd_metadata = bt.ut.get_points_metadata(
            sdata=self.data,
            points_key="transcripts",
            metadata_keys=[column_names[0], column_names[1]],
            astype="dask",
        )

        self.assertTrue(type(pd_metadata_single) == pd.DataFrame)
        self.assertTrue(column_names[0] in pd_metadata_single)

        self.assertTrue(type(dd_metadata_single) == dd.dataframe.core.DataFrame)
        self.assertTrue(column_names[0] in dd_metadata_single)

        self.assertTrue(type(pd_metadata) == pd.DataFrame)
        self.assertTrue(type(dd_metadata) == dd.dataframe.core.DataFrame)
        self.assertTrue("list_metadata" in pd_metadata_single)
        self.assertTrue("list_metadata" in dd_metadata_single)
        for column in column_names:
            self.assertTrue(column in pd_metadata)
            self.assertTrue(column in dd_metadata)

    def test_get_shape_metadata(self):
        list_metadata = [0] * len(self.data.shapes["cell_boundaries"])
        series_metadata = pd.Series(list_metadata)
        column_names = ["list_metadata", "series_metadata"]

        bt.ut.set_shape_metadata(
            sdata=self.data,
            shape_key="cell_boundaries",
            metadata=list_metadata,
            column_names=column_names[0],
        )
        bt.ut.set_shape_metadata(
            sdata=self.data,
            shape_key="cell_boundaries",
            metadata=series_metadata,
            column_names=column_names[1],
        )
        metadata_single = bt.ut.get_shape_metadata(
            sdata=self.data, shape_key="cell_boundaries", metadata_keys=column_names[0]
        )
        metadata = bt.ut.get_shape_metadata(
            sdata=self.data,
            shape_key="cell_boundaries",
            metadata_keys=[column_names[0], column_names[1]],
        )

        self.assertTrue(type(metadata_single) == pd.DataFrame)
        self.assertTrue(column_names[0] in metadata_single)

        self.assertTrue(type(metadata) == pd.DataFrame)
        for column in column_names:
            self.assertTrue(column in metadata)
