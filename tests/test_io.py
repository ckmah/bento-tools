import unittest
import bento as bt
import spatialdata as sd


class TestIO(unittest.TestCase):
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

    def test_points_indexing(self):
        # Check points indexing
        self.assertTrue("cell_boundaries" in self.data.points["transcripts"].columns)
        self.assertTrue("nucleus_boundaries" in self.data.points["transcripts"].columns)

    def test_shapes_indexing(self):
        # Check shapes indexing
        self.assertTrue(
            "cell_boundaries" in self.data.shapes["cell_boundaries"].columns
        )
        self.assertTrue(
            "cell_boundaries" in self.data.shapes["nucleus_boundaries"].columns
        )
        self.assertTrue(
            "nucleus_boundaries" in self.data.shapes["cell_boundaries"].columns
        )

    def test_points_attrs(self):
        # Check points attrs
        self.assertTrue("transform" in self.data.points["transcripts"].attrs.keys())
        self.assertTrue(
            self.data.points["transcripts"].attrs["spatialdata_attrs"]["feature_key"]
            == "feature_name"
        )
        self.assertTrue(
            self.data.points["transcripts"].attrs["spatialdata_attrs"]["instance_key"]
            == "cell_boundaries"
        )

    def test_shapes_attrs(self):
        # Check shapes attrs
        self.assertTrue("transform" in self.data.shapes["cell_boundaries"].attrs.keys())
        self.assertTrue(
            "transform" in self.data.shapes["nucleus_boundaries"].attrs.keys()
        )

    def test_index_dtypes(self):
        # Check index dtypes
        self.assertTrue(self.data.shapes["cell_boundaries"].index.dtype == "object")
        self.assertTrue(self.data.shapes["nucleus_boundaries"].index.dtype == "object")
        self.assertTrue(self.data.points["transcripts"].index.dtype == "int64")
