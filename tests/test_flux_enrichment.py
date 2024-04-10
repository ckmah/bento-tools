import unittest
import bento as bt
import spatialdata as sd


class TestFluxEnrichement(unittest.TestCase):
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
        )
        bt.tl.fluxmap(
            sdata=self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            n_clusters=3,
        )

        self.fe_fazal2019_features = [
            "Cytosol",
            "ER Lumen",
            "ERM",
            "Lamina",
            "Nuclear Pore",
            "Nucleolus",
            "Nucleus",
            "OMM",
        ]
        self.fe_xia2019_features = ["ER", "Nucleus"]

    def test_fe_fazal2019(self):
        bt.tl.fe_fazal2019(self.data)

        # Check that cell_boundaries_raster is in self.data.points
        self.assertTrue("cell_boundaries_raster" in self.data.points)

        # Check that fe_stats is in self.data.table.uns
        self.assertTrue("fe_stats" in self.data.table.uns)

        # Check that fe_ngenes is in self.data.table.uns
        self.assertTrue("fe_ngenes" in self.data.table.uns)

        # Check columns are in cell_boundaries_raster, fe_stats, abd fe_ngenes
        for feature in self.fe_fazal2019_features:
            self.assertTrue(
                f"flux_{feature}" in self.data.points["cell_boundaries_raster"]
            )
            self.assertTrue(feature in self.data.table.uns["fe_stats"])
            self.assertTrue(feature in self.data.table.uns["fe_ngenes"])

    def test_fe_xia2019(self):
        bt.tl.fe_xia2019(self.data)

        # Check that cell_boundaries_raster is in self.data.points
        self.assertTrue("cell_boundaries_raster" in self.data.points)

        # Check that fe_stats is in self.data.table.uns
        self.assertTrue("fe_stats" in self.data.table.uns)

        # Check that fe_ngenes is in self.data.table.uns
        self.assertTrue("fe_ngenes" in self.data.table.uns)

        # Check columns are in cell_boundaries_raster, fe_stats, abd fe_ngenes
        for feature in self.fe_xia2019_features:
            self.assertTrue(
                f"flux_{feature}" in self.data.points["cell_boundaries_raster"]
            )
            self.assertTrue(feature in self.data.table.uns["fe_stats"])
            self.assertTrue(feature in self.data.table.uns["fe_ngenes"])
