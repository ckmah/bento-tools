import unittest
import bento as bt
import spatialdata as sd
import matplotlib.pyplot as plt


# Test if plotting functions run without error
class TestSignaturesPlotting(unittest.TestCase):
    def setUp(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.imgdir = "/".join(bt.__file__.split("/")[:-2]) + "/tests/img/"
        self.data = sd.read_zarr(f"{datadir}/small_data.zarr")
        self.data = bt.io.format_sdata(
            sdata=self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        bt.tl.coloc_quotient(
            self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            feature_key="feature_name",
            shapes=["cell_boundaries"]
        )

        bt.tl.colocation(
            self.data, 
            ranks=range(1, 6), 
            instance_key="cell_boundaries", 
            feature_key="feature_name"
        )

    def test_colocation_plotting(self):
        bt.pl.colocation(self.data, rank=5, fname=f"{self.imgdir}signatures/colocation")
        plt.figure()
