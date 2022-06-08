import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_analyze(self):
        features = list(bento.tl.sample_features.keys())
        bento.tl.analyze_samples(data, features, chunksize=100)

    # def test_proximity_cell(self):
    #     bento.tl.analyze(data, bento.tl.ShapeProximity("cell_shape"), chunks=1)
    #     self.assertTrue("cell_inner_proximity" in data.layers)
    #     self.assertTrue("cell_outer_proximity" in data.layers)

    # def test_proximity_nucleus(self):
    #     bento.tl.analyze(data, bento.tl.ShapeProximity("nucleus_shape"), chunks=1)
    #     self.assertTrue("nucleus_inner_proximity" in data.layers)
    #     self.assertTrue("nucleus_outer_proximity" in data.layers)

    # def test_asymmetry_cell(self):
    #     bento.tl.analyze(data, bento.tl.ShapeAsymmetry("cell_shape"), chunks=1)
    #     self.assertTrue("cell_inner_asymmetry" in data.layers)
    #     self.assertTrue("cell_outer_asymmetry" in data.layers)

    # def test_asymmetry_nucleus(self):
    #     bento.tl.analyze(data, bento.tl.ShapeAsymmetry("nucleus_shape"), chunks=1)
    #     self.assertTrue("nucleus_inner_asymmetry" in data.layers)
    #     self.assertTrue("nucleus_outer_asymmetry" in data.layers)

    # def test_ripley_stats(self):
    #     bento.tl.analyze(data, bento.tl.RipleyStats(), chunks=1)
    #     self.assertTrue("l_max" in data.layers)
    #     self.assertTrue("l_max_gradient" in data.layers)
    #     self.assertTrue("l_min_gradient" in data.layers)
    #     self.assertTrue("l_monotony" in data.layers)
    #     self.assertTrue("l_half_radius" in data.layers)

    # def test_point_dispersion(self):
    #     bento.tl.analyze(data, bento.tl.PointDispersion(), chunks=1)
    #     self.assertTrue("point_dispersion" in data.layers)

    # def test_shape_dispersion(self):
    #     bento.tl.analyze(data, bento.tl.ShapeDispersion("nucleus_shape"), chunks=1)
    #     self.assertTrue("nucleus_dispersion" in data.layers)

    # def test_shape_enrichment(self):
    #     bento.tl.analyze(data, bento.tl.ShapeEnrichment("nucleus_shape"), chunks=1)
    #     self.assertTrue("nucleus_enrichment" in data.layers)
