from ..io import read_geodata
from anndata import AnnData
import pytest

@pytest.fixture
def adata():
    return read_geodata(
        points='examples/spots.shp',
        cell='examples/cell_membrane.shp',
        other={'nucleus': 'examples/nucleus.shp'})

def test_read_geodata(adata):
    assert isinstance(adata, AnnData)

def test_points_format(adata):
    assert 'points' in adata.uns_keys()

def test_masks_format(adata):
    assert 'masks' in adata.uns_keys()