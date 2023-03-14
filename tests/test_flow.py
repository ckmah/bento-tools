import unittest
import bento

data = bento.datasets.sample_data()


class TestFlux(unittest.TestCase):
    def test_flux_radius(self):
        bento.tl.flux(data, method="radius", radius=50, render_resolution=0.5)

        self.assertTrue(
            key in data.uns.keys() for key in ["flux", "flux_embed", "flux_vis"]
        )
        self.assertTrue(data.uns["flux"].shape[0] == data.uns["cell_raster"].shape[0])
        self.assertTrue(
            data.uns["flux_embed"].shape[0] == data.uns["cell_raster"].shape[0]
        )
        self.assertTrue(
            data.uns["flux_vis"].shape[0] == data.uns["cell_raster"].shape[0]
        )

    def test_flux_knn(self):
        bento.tl.flux(data, method="knn", n_neighbors=20, render_resolution=0.5)

        self.assertTrue(
            key in data.uns.keys() for key in ["flux", "flux_embed", "flux_vis"]
        )
        self.assertTrue(data.uns["flux"].shape[0] == data.uns["cell_raster"].shape[0])
        self.assertTrue(
            data.uns["flux_embed"].shape[0] == data.uns["cell_raster"].shape[0]
        )
        self.assertTrue(
            data.uns["flux_vis"].shape[0] == data.uns["cell_raster"].shape[0]
        )

    def test_fluxmap(self):
        bento.tl.flux(data, method="radius", radius=50, render_resolution=0.5)
        bento.tl.fluxmap(
            data, n_clusters=range(2, 5), train_size=1, render_resolution=0.5
        )
        bento.tl.fluxmap(data, n_clusters=3, train_size=1, render_resolution=0.5)
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
