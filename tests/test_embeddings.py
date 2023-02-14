import unittest
import bento

data = bento.datasets.sample_data()


class TestEmbeddings(unittest.TestCase):
    def test_pt_embed(self):
        bento.tl.pt_embed(data, radius=30)
        self.assertTrue("pt_embed" in data.uns)
