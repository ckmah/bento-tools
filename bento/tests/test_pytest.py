from ..io import read_geodata
from ..tools import prepare_features
from anndata import AnnData
from unittest import TestCase

class IOTest(TestCase):
    def test_read_geodata(self):
        data = read_geodata(
            points_path='examples/spots.shp',
            cell_path='examples/cell_membrane.shp',
            other={'nucleus': 'examples/nucleus.shp'})
        self.assertTrue((type(data) == AnnData) and ('x' in data.var_names) and ('y' in dat.var_names))

class Tools(TestCase):
    def test_prepare_features(self):
        data =
