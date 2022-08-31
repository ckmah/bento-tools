import unittest
import bento

data = bento.datasets.sample_data()


class TestSignatures(unittest.TestCase):
    def test_to_tensor(self):
        bento.tl.to_tensor(data, [None])
        tensor = data.uns['tensor']
        self.assertTrue(tensor.shape == (1, data.n_obs, data.n_vars))

    def test_signatures(self):
        bento.tl.lp(data)
        rank = 3
        bento.tl.lp_signatures(data, ranks=rank, nruns=2)

        signame = f"r{rank}_signatures"
        # Check that signatures are correct size
        self.assertTrue(data.uns[signame].shape == (len(bento.utils.PATTERN_NAMES), rank))
        self.assertTrue(data.obsm[signame].shape == (data.n_obs, rank))
        self.assertTrue(data.varm[signame].shape == (data.n_vars, rank))

        # Check that there are no nans
        self.assertFalse(data.uns[signame].isna().any(axis=None))
        self.assertFalse(data.obsm[signame].isna().any(axis=None))
        self.assertFalse(data.varm[signame].isna().any(axis=None))