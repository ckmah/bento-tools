import bento
from anndata import AnnData


def test_raw_to_anndata():
    path_prefix = 'examples/data/cell1'
    assert type(bento.prepare(spots_path=f'{path_prefix}/spots.csv',
                                    cell_img_path=f'{path_prefix}/cell_membrane.png',
                                    nucleus_img_path=f'{path_prefix}/nucleus.png')) is AnnData
