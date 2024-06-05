import unittest
import bento as bt
import spatialdata as sd
import matplotlib.pyplot as plt
import os
from unittest.mock import patch

# Test if plotting functions run without error

# class TestPlotting(unittest.TestCase):

#     @classmethod
#     def setUpClass(self):
#         datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
#         self.imgdir = "/".join(bt.__file__.split("/")[:-2]) + "/tests/img/plotting"
#         os.makedirs(self.imgdir, exist_ok=True)
#         self.data = sd.read_zarr(f"{datadir}/merfish_sample.zarr")
#         self.data = bt.io.prep(
#             sdata=self.data,
#             points_key="transcripts",
#             feature_key="feature_name",
#             instance_key="cell_boundaries",
#             shape_keys=["cell_boundaries", "nucleus_boundaries"],
#         )

#     @patch("matplotlib.pyplot.savefig")
#     def test_points_plotting(self, mock_savefig):
#         plt.figure()
#         bt.pl.points(
#             self.data, 
#             fname=f"{self.imgdir}/points_uncolored_synced"
#         )

#         plt.figure()
#         bt.pl.points(
#             self.data, 
#             hide_outside=False, 
#             fname=f"{self.imgdir}/points_uncolored_unsynced.png"
#         )

#         plt.figure()
#         bt.pl.points(
#             self.data, 
#             hue="feature_name", 
#             legend=False, 
#             fname=f"{self.imgdir}/points_colored_synced.png"
#         )

#         plt.figure()
#         bt.pl.points(
#             self.data, 
#             hue="feature_name", 
#             legend=False, 
#             hide_outside=False, 
#             fname=f"{self.imgdir}/points_colored_unsynced.png"
#         )

#         genes = ["LUM", "POSTN", "CCDC80"]
#         plt.figure()
#         bt.pl.points(
#             self.data,
#             hue="feature_name", 
#             hue_order=genes, 
#             legend=False, 
#             fname=f"{self.imgdir}/points_subseted_genes_synced.png"
#         )

#         plt.figure()
#         bt.pl.points(
#             self.data, 
#             hue="feature_name", 
#             hue_order=genes, 
#             legend=False, 
#             hide_outside=False, 
#             fname=f"{self.imgdir}/points_subseted_genes_unsynced.png"
#         )
#         mock_savefig.assert_called()

#     @patch("matplotlib.pyplot.savefig")
#     def test_density_plotting(self, mock_savefig):
#         plt.figure()
#         bt.pl.density(
#             self.data, 
#             fname=f"{self.imgdir}/density_hist.png"
#         )

#         plt.figure()
#         bt.pl.density(
#             self.data, 
#             kind="kde", 
#             fname=f"{self.imgdir}/density_kde.png"
#         )

#         plt.figure()
#         bt.pl.density(
#             self.data, 
#             hue="feature_name", 
#             legend=False, 
#             fname=f"{self.imgdir}/density_hist_gene.png"
#         )

#         plt.figure()
#         bt.pl.density(
#             self.data, 
#             hue="feature_name", 
#             legend=False, 
#             kind="kde", 
#             fname=f"{self.imgdir}/density_kde_gene.png"
#         )

#         genes = ["LUM", "POSTN", "CCDC80"]
#         plt.figure()
#         bt.pl.density(
#             self.data, 
#             hue="feature_name", 
#             hue_order=genes, 
#             legend=False, 
#             fname=f"{self.imgdir}/density_hist_subsetted_genes.png"
#         )
        
#         plt.figure()
#         bt.pl.density(
#             self.data, 
#             hue="feature_name", 
#             hue_order=genes, 
#             legend=False, 
#             kind="kde", 
#             fname=f"{self.imgdir}/density_kde_subsetted_genes.png"
#         )

#         fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#         bt.pl.density(self.data, ax=axes[0], title="default styling")
#         bt.pl.density(
#             self.data,
#             ax=axes[1],
#             axis_visible=True,
#             frame_visible=True,
#             square=True,
#             title="square plot + axis",
#             fname=f"{self.imgdir}/density_square.png"
#         )
#         plt.tight_layout()
#         mock_savefig.assert_called()

#     @patch("matplotlib.pyplot.savefig")
#     def test_shapes_plotting(self, mock_savefig):
#         plt.figure()
#         bt.pl.shapes(
#             self.data, 
#             fname=f"{self.imgdir}/shapes_uncolored_synced.png"
#         )

#         plt.figure()
#         bt.pl.shapes(
#             self.data,
#             hide_outside=False, 
#             fname=f"{self.imgdir}/shapes_uncolored_unsynced.png"
#         )

#         plt.figure()
#         bt.pl.shapes(
#             self.data, 
#             color_style="fill", 
#             fname=f"{self.imgdir}/shapes_colored_synced.png"
#         )

#         plt.figure()
#         bt.pl.shapes(
#             self.data, 
#             color_style="fill",
#             hide_outside=False, 
#             fname=f"{self.imgdir}/shapes_colored_unsynced.png"
#         )

#         plt.figure()
#         bt.pl.shapes(
#             self.data, 
#             hue="cell",
#             color_style="fill",
#             fname=f"{self.imgdir}/shapes_cell_colored_synced.png"
#         )

#         plt.figure()
#         bt.pl.shapes(
#             self.data, 
#             hue="cell",
#             color_style="fill",
#             hide_outside=False, 
#             fname=f"{self.imgdir}/shapes_cell_colored_unsynced.png"
#         )
        
#         fig, ax = plt.subplots()
#         bt.pl.shapes(self.data, shapes="cell_boundaries", linestyle="--", ax=ax)
#         bt.pl.shapes(
#             self.data,
#             shapes="nucleus_boundaries",
#             edgecolor="black",
#             facecolor="lightseagreen",
#             ax=ax,
#             fname=f"{self.imgdir}/shapes_nucleus_colored_synced.png"
#         )
    
#         mock_savefig.assert_called()
