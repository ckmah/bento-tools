import unittest
import bento as bt
import matplotlib as mpl
import matplotlib.pyplot as plt

adata = bt.ds.sample_data()


# Test if plotting functions run without error
class TestPlotting(unittest.TestCase):
    def test_points(self):
        bt.pl.points(adata)

        bt.pl.points(adata, hue="gene", legend=False)

        genes = ["MALAT1", "TLN1", "SPTBN1"]
        bt.pl.points(adata[:, genes], hue="gene", legend=False)

        self.assertTrue(True)

    def test_density(self):
        bt.pl.density(adata)

        self.assertTrue(True)

    def test_shapes(self):
        bt.pl.shapes(adata)

        bt.pl.shapes(adata, color_style="fill")

        bt.pl.shapes(adata, hue="cell", color_style="fill")

        self.assertTrue(True)
