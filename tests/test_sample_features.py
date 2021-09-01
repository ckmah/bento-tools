import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_proximity_cell_inner(self):
        bento.tl.proximity(data, "cell_shape", position="inner")
        self.assertTrue("cell_inner_proximity" in data.layers)

    def test_proximity_nucleus_outer(self):
        bento.tl.proximity(data, "nucleus_shape", position="outer")
        self.assertTrue("nucleus_outer_proximity" in data.layers)

    def test_asymmetry_cell_inner(self):
        bento.tl.asymmetry(data, "cell_shape", position="inner")
        self.assertTrue("cell_inner_asymmetry" in data.layers)

    def test_asymmetry_nucleus_outer(self):
        bento.tl.asymmetry(data, "nucleus_shape", position="outer")
        self.assertTrue("nucleus_outer_asymmetry" in data.layers)
