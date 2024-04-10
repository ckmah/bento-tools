import unittest
import bento as bt
import spatialdata as sd
import matplotlib.pyplot as plt


# Test if plotting functions run without error
class TestLpPlotting(unittest.TestCase):
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
        bt.tl.lp(
            sdata=self.data,
            instance_key="cell_boundaries",
            nucleus_key="nucleus_boundaries",
            groupby="feature_name",
        )

    def test_lp_dist_plotting(self):
        bt.pl.lp_dist(self.data, fname=f"{self.imgdir}lp/lp_dist")
        plt.figure()

    def test_lp_genes_plotting(self):
        bt.pl.lp_genes(
            self.data, 
            groupby="feature_name",
            points_key="transcripts",
            instance_key="cell_boundaries",
            fname=f"{self.imgdir}lp/lp_genes"
        )
        plt.figure()

    def test_lp_diff_discrete_plotting(self):
        area_binary = []
        median = self.data.shapes['cell_boundaries']['cell_boundaries_area'].median()
        for i in range(len(self.data.shapes['cell_boundaries'])):
            cell_boundaries_area = self.data.shapes['cell_boundaries']['cell_boundaries_area'][i]
            if cell_boundaries_area > median:
                area_binary.append("above")
            else:
                area_binary.append("below")
        self.data.shapes['cell_boundaries']['area_binary'] = area_binary

        bt.tl.lp_diff_discrete(self.data, phenotype="area_binary")
        bt.pl.lp_diff_discrete(
            self.data, 
            phenotype="area_binary", 
            fname=f"{self.imgdir}lp/lp_diff_discrete.png"
        )
        plt.figure()
