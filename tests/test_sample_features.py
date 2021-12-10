import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_proximity_cell(self):
        model = bento.tl.ShapeProximity("cell_shape")
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_proximity_nucleus(self):
        model = bento.tl.ShapeProximity("nucleus_shape")
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_asymmetry_cell(self):
        model = bento.tl.ShapeAsymmetry("cell_shape")
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_asymmetry_nucleus(self):
        model = bento.tl.ShapeAsymmetry("nucleus_shape")
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_ripley(self):
        model = bento.tl.Ripley()
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_point_dispersion(self):
        model = bento.tl.PointDispersion()
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

    def test_shape_dispersion(self):
        model = bento.tl.ShapeDispersion("nucleus_shape")
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))

def test_shape_enrichment(self):
        model = bento.tl.ShapeEnrichment("nucleus_shape")
        model.transform(data)
        self.assertTrue(
            all(name in data.layers for name in model.metadata.keys()))