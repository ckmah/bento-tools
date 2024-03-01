import unittest
import bento as bt
import spatialdata as sd


class TestIO(unittest.TestCase): 
    def setUp(self):
        self.data = sd.read_zarr("/mnt/d/sdata/xenium_rep1_io/small_data.zarr")

    def test_format_sdata(self):
        bt.io.format_sdata(
            self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            shape_names=["cell_boundaries", "nucleus_boundaries"],
        )

        # Check points indexing
        self.assertTrue("cell_boundaries" in self.data.points["transcripts"].columns)
        self.assertTrue("nucleus_boundaries" in self.data.points["transcripts"].columns)

        # Check shapes indexing
        self.assertTrue("cell_boundaries" in self.data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries" in self.data.shapes["cell_boundaries"].columns)
