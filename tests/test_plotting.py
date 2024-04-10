import unittest
import bento as bt
import spatialdata as sd
import matplotlib.pyplot as plt


# Test if plotting functions run without error
class TestPlotting(unittest.TestCase):
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

    def test_points_plotting(self):
        bt.pl.points(
            self.data, 
            fname=f"{self.imgdir}points/points_uncolored_synced"
        )
        plt.figure()

        bt.pl.points(
            self.data, 
            sync_points=False, 
            fname=f"{self.imgdir}points/points_uncolored_unsynced"
        )
        plt.figure()

        bt.pl.points(
            self.data, 
            hue="feature_name", 
            legend=False, 
            fname=f"{self.imgdir}points/points_colored_synced"
        )
        plt.figure()

        bt.pl.points(
            self.data, 
            hue="feature_name", 
            legend=False, 
            sync_points=False, 
            fname=f"{self.imgdir}points/points_colored_unsynced"
        )
        plt.figure()

        genes = ["LUM", "POSTN", "CCDC80"]
        bt.pl.points(
            self.data,
            hue="feature_name", 
            hue_order=genes, 
            legend=False, 
            fname=f"{self.imgdir}points/points_subseted_genes_synced"
        )
        plt.figure()

        bt.pl.points(
            self.data, 
            hue="feature_name", 
            hue_order=genes, 
            legend=False, 
            sync_points=False, 
            fname=f"{self.imgdir}points/points_subseted_genes_unsynced"
        )
        plt.figure()

    def test_density_plotting(self):
        bt.pl.density(
            self.data, 
            fname=f"{self.imgdir}density/density_hist"
        )
        plt.figure()

        bt.pl.density(
            self.data, 
            kind="kde", 
            fname=f"{self.imgdir}density/density_kde"
        )
        plt.figure()

        bt.pl.density(
            self.data, 
            hue="feature_name", 
            legend=False, 
            fname=f"{self.imgdir}density/density_hist_gene"
        )
        plt.figure()

        bt.pl.density(
            self.data, 
            hue="feature_name", 
            legend=False, 
            kind="kde", 
            fname=f"{self.imgdir}density/density_kde_gene"
        )
        plt.figure()

        genes = ["LUM", "POSTN", "CCDC80"]
        bt.pl.density(
            self.data, 
            hue="feature_name", 
            hue_order=genes, 
            legend=False, 
            fname=f"{self.imgdir}density/density_hist_subsetted_genes"
        )
        plt.figure()
        
        bt.pl.density(
            self.data, 
            hue="feature_name", 
            hue_order=genes, 
            legend=False, 
            kind="kde", 
            fname=f"{self.imgdir}density/density_kde_subsetted_genes"
        )
        plt.figure()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        bt.pl.density(self.data, ax=axes[0], title="default styling")
        bt.pl.density(
            self.data,
            ax=axes[1],
            axis_visible=True,
            frame_visible=True,
            square=True,
            title="square plot + axis",
            fname=f"{self.imgdir}density/density_square",
        )
        plt.tight_layout()
        plt.figure()

    def test_shapes_plotting(self):
        bt.pl.shapes(
            self.data, 
            fname=f"{self.imgdir}shapes/shapes_uncolored_synced"
        )
        plt.figure()

        bt.pl.shapes(
            self.data,
            sync_shapes=False, 
            fname=f"{self.imgdir}shapes/shapes_uncolored_unsynced"
        )
        plt.figure()

        bt.pl.shapes(
            self.data, 
            color_style="fill", 
            fname=f"{self.imgdir}shapes/shapes_colored_synced"
        )
        plt.figure()

        bt.pl.shapes(
            self.data, 
            color_style="fill",
            sync_shapes=False, 
            fname=f"{self.imgdir}shapes/shapes_colored_unsynced"
        )
        plt.figure()

        bt.pl.shapes(
            self.data, 
            hue="cell",
            color_style="fill",
            fname=f"{self.imgdir}shapes/shapes_cell_colored_synced"
        )
        plt.figure()

        bt.pl.shapes(
            self.data, 
            hue="cell",
            color_style="fill",
            sync_shapes=False, 
            fname=f"{self.imgdir}shapes/shapes_cell_colored_unsynced"
        )
        plt.figure()
        
        fig, ax = plt.subplots()
        bt.pl.shapes(self.data, shapes="cell_boundaries", linestyle="--", ax=ax)
        bt.pl.shapes(
            self.data,
            shapes="nucleus_boundaries",
            edgecolor="black",
            facecolor="lightseagreen",
            ax=ax,
            fname=f"{self.imgdir}shapes/shapes_nucleus_colored_synced"
        )
    
