import unittest
import bento as bt

data = bt.ds.sample_data()[:5, :5]
bt.sync(data)

# Ad a missing shape for testing
nucleus_shapes = data.obs["nucleus_shape"]
nucleus_shapes[1] = None

features = list(bt.tl.list_shape_features().keys())


class TestShapeFeatures(unittest.TestCase):
    # Simplest case, single shape and feature
    def test_single_shape_single_feature(self):
        # Test shape name with/without suffix
        bt.tl.analyze_shapes(data, "cell", "area")
        bt.tl.analyze_shapes(data, "cell_shape", "area")
        self.assertTrue("cell_area" in data.obs)

    def test_single_shape_multi_feature(self):
        # Test all features
        bt.tl.analyze_shapes(data, "cell", features)
        feature_keys = [f"cell_{f}" for f in features]
        self.assertTrue(f in data.obs for f in feature_keys)

    def test_missing_shape(self):
        # Test missing nucleus shapes
        bt.tl.analyze_shapes(data, "nucleus", features)
        feature_keys = [f"nucleus_{f}" for f in features]
        self.assertTrue(f in data.obs for f in feature_keys)
        self.assertTrue(data.obs[f].isna()[1] for f in feature_keys)
