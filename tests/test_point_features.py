import unittest
import bento as bt

data = bt.ds.sample_data()[:5, :5]
bt.sync(data)


class TestPointFeatures(unittest.TestCase):
    def test_analyze(self):
        features = list(bt.tl.list_point_features().keys())

        # Simplest case, single parameters
        bt.tl.analyze_points(data, "cell_shape", features[0], groupby=None)

        self.assertTrue("cell_features" in data.uns)
        self.assertTrue(data.uns["cell_features"].shape[0] == data.n_obs)

        # Multiple shapes, features, and gene groupby
        bt.tl.analyze_points(
            data, ["cell_shape", "nucleus_shape"], features, groupby="gene"
        )

        n_groups = data.uns["points"].groupby(["cell", "gene"], observed=True).ngroups
        self.assertTrue(data.uns["cell_gene_features"].shape[0] == n_groups)
