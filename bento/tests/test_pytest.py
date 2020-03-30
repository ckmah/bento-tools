import bento

def test_zeros():
    mtx = bento.zeros()
    assert mtx.shape == (2,2) and mtx.sum() == 0
