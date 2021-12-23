import unittest
import bento

data = bento.datasets.sample_data()


class TestTensorTools(unittest.TestCase):
    def test_decompose_tensor(self):
        bento.tl.intracellular_patterns(data)
        N_FACTORS = 1
        bento.tl.decompose_tensor(data, N_FACTORS)

        dim_names = list(data.uns['tensor_labels'].keys())
        self.assertTrue(set(dim_names) == set(bento.tl.TENSOR_DIM_NAMES))

        n_features = len(bento.PATTERN_NAMES)
        n_cells = data.n_obs
        n_genes = data.n_vars
        tensor_shape = (n_features, n_cells, n_genes)
        self.assertTrue(data.uns["tensor"].shape == tensor_shape)

        for i, name in enumerate(bento.tl.TENSOR_DIM_NAMES):
            dim_len = data.uns['tensor_loadings'][name].values.shape[0]
            self.assertTrue(dim_len == tensor_shape[i])

            n_factors = len(data.uns['tensor_loadings'][name].keys())
            self.assertTrue(n_factors == N_FACTORS)
