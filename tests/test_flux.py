import unittest
import bento as bt

data = bt.ds.sample_data()
radius = 50
n_neighbors = 20
res = 0.02


class TestFlux(unittest.TestCase):
    def test_flux_radius(self):
        bt.tl.flux(data, method="radius", radius=radius, res=res)

        self.assertTrue(
            key in data.uns.keys() for key in ["flux", "flux_embed", "color"]
        )
        self.assertTrue(data.uns["flux"].shape[0] == data.uns["cell_raster"].shape[0])
        self.assertTrue(
            data.uns["flux_embed"].shape[0] == data.uns["cell_raster"].shape[0]
        )
        self.assertTrue(data.uns["flux_color"].flatten()[0][0] == "#")

    def test_flux_knn(self):
        bt.tl.flux(data, method="knn", n_neighbors=n_neighbors, res=res)

        self.assertTrue(
            key in data.uns.keys() for key in ["flux", "flux_embed", "flux_color"]
        )
        self.assertTrue(data.uns["flux"].shape[0] == data.uns["cell_raster"].shape[0])
        self.assertTrue(
            data.uns["flux_embed"].shape[0] == data.uns["cell_raster"].shape[0]
        )
        self.assertTrue(data.uns["flux_color"].flatten()[0][0] == "#")

    def test_fluxmap(self):
        bt.tl.flux(data, method="radius", radius=radius, res=res)
        bt.tl.fluxmap(data, n_clusters=range(2, 4), train_size=0.2, res=res)
        bt.tl.fluxmap(data, n_clusters=3, train_size=1, res=res)
        self.assertTrue("fluxmap" in data.uns["cell_raster"])
        self.assertTrue(
            [
                f in data.uns["points"].columns
                for f in ["fluxmap0", "fluxmap1", "fluxmap2"]
            ]
        )
        self.assertTrue(
            [
                f in data.obs.columns
                for f in ["fluxmap0_shape", "fluxmap1_shape", "fluxmap2_shape"]
            ]
        )
