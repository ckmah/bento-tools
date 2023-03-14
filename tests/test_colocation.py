import unittest
import bento

data = bento.datasets.sample_data()


class TestColocation(unittest.TestCase):
    def test_coloc_quotient(self):
        bento.tl.coloc_quotient(data)
        self.assertTrue("clq" in data.uns)

    def test_colocation(self):
        bento.tl.coloc_quotient(data, radius=20, min_points=10, min_cells=0)
        bento.tl.colocation(data, ranks=[3], iterations=3)
        self.assertTrue("clq" in data.uns)
