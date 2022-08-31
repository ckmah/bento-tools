import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_analyze(self):
        features = list(bento.tl.sample_feature_fns.keys())
        bento.tl.analyze_samples(data, ['cell_shape'], features, progress=False)
        bento.tl.analyze_samples(data, ['cell_shape'], features, groupby=None, progress=False)
        self.assertTrue("sp" in data.obsm)
