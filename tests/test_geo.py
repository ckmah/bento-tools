import unittest
import bento

data = bento.datasets.sample_data()


class TestGeo(unittest.TestCase):
    def test_rename_cells(self):
        bento.tl.flow(data, method="radius", radius=200, render_resolution=1)
        bento.tl.flowmap(data, 2, train_size=1, render_resolution=1)
        bento.geo.rename_shapes(
            data,
            {"flowmap1_shape": "flowmap3_shape", "flowmap2_shape": "flowmap4_shape"},
            points_key=["points", "cell_raster"],
            points_encoding=["onhot", "label"],
        )

        new_names = ["flowmap3_shape", "flowmap4_shape"]
        self.assertTrue([f in data.obs.columns for f in new_names])
        self.assertTrue([f in data.uns["points"].columns for f in new_names])
        self.assertTrue([f in data.uns["cell_raster"]["flowmap"] for f in ["3", "4"]])
