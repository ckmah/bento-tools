import unittest
import bento

data = bento.datasets.sample_data()


class TestPointFeatures(unittest.TestCase):
    def test_analyze(self):
        features = list(bento.tl.point_features.keys())

        # Simplest case, single parameters
        bento.tl.analyze_points(data, 'cell_shape', features[0], groupby=None, progress=False)
        print(data.obsm['point_features'].shape)
        self.assertTrue("point_features" in data.obsm)


        # Multiple shapes, features, and gene groupby
        n_layers = len(data.layers)
        bento.tl.analyze_points(data, ['cell_shape', 'nucleus_shape'], features, groupby="gene", progress=False)
        self.assertTrue(len(data.layers) > n_layers)
