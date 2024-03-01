import unittest
import bento as bt
import spatialdata as sd


class TestShapeFeatures(unittest.TestCase):
    def setUp(self):
        self.data = sd.read_zarr("/mnt/d/sdata/xenium_rep1_io/small_data.zarr")
        self.shape_features = bt.tl.list_shape_features().keys()

    # Simplest case, single shape and feature
    def test_single_shape_single_feature(self):
        bt.tl.analyze_shapes(self.data, "cell_boundaries", "area")
        self.assertTrue(
            "cell_boundaries_area" in self.data.shapes["cell_boundaries"].columns
        )

    # def test_single_shape_multi_feature(self):
    #     # Test all features
    #     bt.tl.analyze_shapes(self.data, "cell", features)
    #     feature_keys = [f"cell_{f}" for f in features]
    #     self.assertTrue(f in self.data.obs for f in feature_keys)

    # def test_missing_shape(self):
    #     # Test missing nucleus shapes
    #     bt.tl.analyze_shapes(self.data, "nucleus", features)
    #     feature_keys = [f"nucleus_{f}" for f in features]
    #     self.assertTrue(f in self.data.obs for f in feature_keys)
    #     self.assertTrue(self.data.obs[f].isna()[1] for f in feature_keys)
