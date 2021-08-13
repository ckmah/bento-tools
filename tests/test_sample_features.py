import unittest
import bento

data = bento.datasets.sample_data()


class TestSampleFeatures(unittest.TestCase):
    def test_extract(self):
        adata = bento.tl.extract(data, "cyto_distance_to_cell", copy=True)
        self.assertTrue("cyto_distance_to_cell" in adata.layers)

    def test_extract_multicore(self):
        adata = bento.tl.extract(data, "cyto_distance_to_cell", n_jobs=2, copy=True)
        self.assertTrue("cyto_distance_to_cell" in adata.layers)

    def test_CytoDistanceToNucleus(self):
        adata = bento.tl.extract(data, "cyto_distance_to_nucleus", copy=True)
        self.assertTrue("cyto_distance_to_nucleus" in adata.layers)