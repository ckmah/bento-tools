import unittest
import bento

data = bento.datasets.sample_data()
bento.tl.rasterize_cells(data, '/tmp/patpred')
bento.tl.predict_patterns(data, '/tmp/patpred', model='multilabel')


class TestPlotting(unittest.TestCase):
    def test_gene_patterns_default(self):
        fig = bento.pl.gene_patterns(data, data.var_names[0])
        self.assertTrue(fig)