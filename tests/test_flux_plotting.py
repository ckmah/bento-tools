import unittest
import bento as bt
import spatialdata as sd
import matplotlib.pyplot as plt


# Test if plotting functions run without error
class TestFluxPlotting(unittest.TestCase):
    def setUp(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.imgdir = "/".join(bt.__file__.split("/")[:-2]) + "/tests/img/"
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

    def test_flux_plotting(self):
        bt.pl.flux(self.data, res=1, fname=f"{self.imgdir}flux/flux")
        plt.figure()

    def test_fluxmap_plotting(self):
        bt.tl.fluxmap(
            sdata=self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            res=1,
            train_size=1,
            n_clusters=3,
            plot_error=False
        )
        bt.pl.fluxmap(self.data, fname=f"{self.imgdir}flux/fluxmap")
        plt.figure()

    def test_fe_plotting(self):
        bt.tl.fluxmap(
            sdata=self.data,
            points_key="transcripts",
            instance_key="cell_boundaries",
            res=1,
            train_size=1,
            n_clusters=3,
            plot_error=False
        )
        bt.tl.fe_fazal2019(self.data)
        self.data = bt.geo.sjoin_shapes(self.data, instance_key="cell_boundaries", shape_keys=["fluxmap1_boundaries"])
        bt.pl.fe(
            self.data, 
            "flux_OMM", 
            res=1, 
            shapes=["cell_boundaries", "fluxmap1_boundaries"], 
            fname=f"{self.imgdir}flux/fe_flux_OMM_fluxmap1"
        )
        plt.figure()
