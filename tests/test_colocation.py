import unittest
import bento

data = bento.datasets.sample_data()


class TestColocation(unittest.TestCase):
    def test_coloc_quotient(self):
        bento.tl.coloc_quotient(data)
        self.assertTrue("coloc_quotient" in data.uns)
