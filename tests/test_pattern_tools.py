import unittest
import bento

data = bento.datasets.sample_data()
bento.tl.rasterize_cells(data, '/tmp/patpred', overwrite=False)
bento.tl.predict_patterns(data, '/tmp/patpred', model='multilabel')


class TestPatterns(unittest.TestCase):
    def test_diff_spots(self):
        data.obs['batch'] = ['0', '1']
        bento.tl.spots_diff(data, "batch")
        print(data.uns['diff_batch'].head())
        self.assertTrue("diff_batch" in data.uns.keys())