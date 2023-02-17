import unittest
import bento

data = bento.datasets.sample_data()


class TestPatterns(unittest.TestCase):
    def test_lp(self):
        bento.tl.lp(data)

        # Check if "lp" and "lpp" are in data.obsm
        self.assertTrue("lp" in data.uns.keys() and "lpp" in data.uns.keys())
