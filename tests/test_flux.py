import os
import unittest

import matplotlib.pyplot as plt
import spatialdata as sd

import bento as bt
from unittest.mock import patch


class TestFlux(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.imgdir = "/".join(bt.__file__.split("/")[:-2]) + "/tests/img/flux"
        os.makedirs(self.imgdir, exist_ok=True)
        self.data = sd.read_zarr(f"{datadir}/merfish_sample.zarr")
        self.data = bt.io.prep(
            sdata=self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )
        self.res=0.1
        bt.tl.flux(
            sdata=self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            feature_key="feature_name",
            res=self.res,
            radius=0.5,
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
            res=self.res,
            train_size=1,
            n_clusters=3,
            plot_error=False,
        )
        self.assertTrue("fluxmap" in self.data.points["cell_boundaries_raster"].columns)
        for i in range(1, 4):
            self.assertTrue(
                f"fluxmap{i}" in self.data.points["transcripts"].columns
            )
            self.assertTrue(f"fluxmap{i}" in self.data.shapes)

    # @patch("matplotlib.pyplot.savefig")
    # def test_flux_plot(self, mock_savefig):
    #     bt.pl.flux(self.data, res=1, fname=f"{self.imgdir}/flux.png")

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

    # def test_fluxmap_plot(self):
    #     bt.tl.fluxmap(
    #         sdata=self.data,
    #         points_key="transcripts",
    #         instance_key="cell_boundaries",
    #         res=self.res,
    #         train_size=1,
    #         n_clusters=3,
    #         plot_error=False,
    #     )
    #     plt.figure()
    #     bt.pl.fluxmap(self.data, fname=f"{self.imgdir}/fluxmap.png")

    # def test_fe_plot(self):
    #     # TODO this test is so slow
    #     bt.tl.fluxmap(
    #         sdata=self.data,
    #         points_key="transcripts",
    #         instance_key="cell_boundaries",
    #         res=self.res,
    #         train_size=1,
    #         n_clusters=3,
    #         plot_error=False,
    #     )
    #     bt.tl.fe_fazal2019(self.data)

    #     plt.figure()
    #     bt.pl.fe(
    #         self.data,
    #         "flux_OMM",
    #         res=self.res,
    #         shapes=["cell_boundaries", "fluxmap1_boundaries"],
    #         fname=f"{self.imgdir}/fe_flux_OMM_fluxmap1.png",
    #     )
