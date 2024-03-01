# import unittest
# import bento as bt

# data = bt.ds.sample_data()

# rank = 3


# class TestColocation(unittest.TestCase):
#     def test_coloc_quotient(self):
#         bt.tl.coloc_quotient(data)
#         self.assertTrue("clq" in data.uns)

#     def test_colocation(self):
#         bt.tl.coloc_quotient(data, radius=20, min_points=10, min_cells=0)
#         bt.tl.colocation(data, ranks=[rank], iterations=3)
#         self.assertTrue("clq" in data.uns)

#     def test_plot(self):
#         bt.pl.colocation(data, rank=rank)
#         self.assertTrue(True)
