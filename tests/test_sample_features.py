import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_analyze(self):
        features = list(bento.tl.sample_features.keys())
        bento.tl.analyze_samples(data, features)