import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_proximity_cell(self):
        bento.tl.ShapeProximity("cell_shape").transform(data)
        self.assertTrue("cell_inner_proximity" in data.layers)
        self.assertTrue("cell_outer_proximity" in data.layers)

    def test_proximity_nucleus(self):
        bento.tl.ShapeProximity("nucleus_shape").transform(data)
        self.assertTrue("nucleus_inner_proximity" in data.layers)
        self.assertTrue("nucleus_outer_proximity" in data.layers)

    def test_asymmetry_cell(self):
        bento.tl.ShapeAsymmetry("cell_shape").transform(data)
        self.assertTrue("cell_inner_asymmetry" in data.layers)
        self.assertTrue("cell_outer_asymmetry" in data.layers)

    def test_asymmetry_nucleus(self):
        bento.tl.ShapeAsymmetry("nucleus_shape").transform(data)
        self.assertTrue("nucleus_inner_asymmetry" in data.layers)
        self.assertTrue("nucleus_outer_asymmetry" in data.layers)

    def test_ripley(self):
        model = bento.tl.Ripley()
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_opening_enrichment(self):
        model = bento.tl.CellOpenEnrichment()
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))
