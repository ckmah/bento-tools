# import unittest
# import bento as bt

# data = bt.ds.sample_data()[:5, :5]
# bt.sync(data)

# # Ad a missing shape for testing
# nucleus_shapes = data.obs["nucleus_shape"]
# nucleus_shapes[1] = None

# features = list(bt.tl.list_point_features().keys())


# class TestPointFeatures(unittest.TestCase):
#     def test_single_feature(self):
#         # Simplest case, single parameters
#         bt.tl.analyze_points(data, "cell_shape", features[0], groupby=None)

#         self.assertTrue("cell_features" in data.uns)
#         self.assertTrue(data.uns["cell_features"].shape[0] == data.n_obs)

#     def test_multiple_shapes(self):
#         # Multiple shapes, and features
#         bt.tl.analyze_points(
#             data, ["cell_shape", "nucleus_shape"], features, groupby=None
#         )

#         self.assertTrue("cell_features" in data.uns)
#         self.assertTrue(data.uns["cell_features"].shape[0] == data.n_obs)

#     def test_multiple_shapes_features_groupby(self):
#         # Multiple shapes, features, and gene groupby
#         bt.tl.analyze_points(
#             data, ["cell_shape", "nucleus_shape"], features, groupby="gene"
#         )

#         output_key = "cell_gene_features"
#         n_groups = data.uns["points"].groupby(["cell", "gene"], observed=True).ngroups
#         self.assertTrue(data.uns[output_key].shape[0] == n_groups)
