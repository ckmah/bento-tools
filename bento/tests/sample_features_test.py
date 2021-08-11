from ..tools import extract
from ..datasets import sample_data

data = sample_data()
import unittest


class TestSampleFeatures(unittest.TestCase):
    def test_extract(self):
        adata = extract(data, "cyto_distance_to_cell", copy=True)
        self.assertTrue("cyto_distance_to_cell" in adata.layers)

    def test_extract_multicore(self):
        adata = extract(data, "cyto_distance_to_cell", n_cores=2, copy=True)
        self.assertTrue("cyto_distance_to_cell" in adata.layers)
