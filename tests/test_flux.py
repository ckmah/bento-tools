import unittest
import bento as bt
import spatialdata as sd


class TestFlux(unittest.TestCase):
    def setUp(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.data = sd.read_zarr(f"{datadir}/small_data.zarr")
        self.data = bt.io.format_sdata(
            sdata=self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        bt.tl.flux(
            sdata=self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            feature_key="feature_name",
            res=1,
            radius=10
        )

    def test_flux(self):
        # Check that cell_boundaries_raster is in self.data.points
        self.assertTrue("cell_boundaries_raster" in self.data.points)

        # Check that flux_genes is in self.data.table.uns
        self.assertTrue("flux_genes" in self.data.table.uns)
        genes = self.data.table.uns["flux_genes"]

        # Check that flux_variance_ratio is in self.data.table.uns
        self.assertTrue("flux_variance_ratio" in self.data.table.uns)

        # Check columns are added in cell_boundaries_raster
        for gene in genes:
            self.assertTrue(gene in self.data.points["cell_boundaries_raster"].columns)

        for i in range(10):
            self.assertTrue(
                f"flux_embed_{i}" in self.data.points["cell_boundaries_raster"].columns
            )

    def test_fluxmap(self):
        bt.tl.fluxmap(
            sdata=self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            res=1,
            train_size=1,
            n_clusters=3,
            plot_error=False
        )

        self.assertTrue("fluxmap" in self.data.points["cell_boundaries_raster"].columns)
        for i in range(1, 4):
            self.assertTrue(
                f"fluxmap{i}_boundaries" in self.data.points["transcripts"].columns
            )
            self.assertTrue(f"fluxmap{i}_boundaries" in self.data.shapes)
