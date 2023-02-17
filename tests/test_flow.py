import unittest
import bento

data = bento.datasets.sample_data()


class TestFlow(unittest.TestCase):
    def test_flow_radius(self):
        bento.tl.flow(data, method="radius", radius=50, render_resolution=0.5)

        self.assertTrue(
            key in data.uns.keys() for key in ["flow", "flow_embed", "flow_vis"]
        )
        self.assertTrue(data.uns["flow"].shape[0] == data.uns["cell_raster"].shape[0])
        self.assertTrue(
            data.uns["flow_embed"].shape[0] == data.uns["cell_raster"].shape[0]
        )
        self.assertTrue(
            data.uns["flow_vis"].shape[0] == data.uns["cell_raster"].shape[0]
        )

    def test_flow_knn(self):
        bento.tl.flow(data, method="knn", n_neighbors=20, render_resolution=0.5)

        self.assertTrue(
            key in data.uns.keys() for key in ["flow", "flow_embed", "flow_vis"]
        )
        self.assertTrue(data.uns["flow"].shape[0] == data.uns["cell_raster"].shape[0])
        self.assertTrue(
            data.uns["flow_embed"].shape[0] == data.uns["cell_raster"].shape[0]
        )
        self.assertTrue(
            data.uns["flow_vis"].shape[0] == data.uns["cell_raster"].shape[0]
        )

    def test_flowmap(self):
        bento.tl.flow(data, method="radius", radius=50, render_resolution=0.5)
        bento.tl.flowmap(
            data, n_clusters=range(2, 5), train_size=1, render_resolution=0.5
        )
        bento.tl.flowmap(data, n_clusters=3, train_size=1, render_resolution=0.5)
        self.assertTrue("flowmap" in data.uns["cell_raster"])
        self.assertTrue(
            [
                f in data.uns["points"].columns
                for f in ["flowmap0", "flowmap1", "flowmap2"]
            ]
        )
        self.assertTrue(
            [
                f in data.obs.columns
                for f in ["flowmap0_shape", "flowmap1_shape", "flowmap2_shape"]
            ]
        )
