# import unittest
# import bento as bt
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# adata = bt.ds.sample_data()


# # Test if plotting functions run without error
# class TestPlotting(unittest.TestCase):
#     def test_analyze(self):
#         bt.pl.points(adata)

#         bt.pl.points(adata, hue="gene", legend=False)

#         genes = ["MALAT1", "TLN1", "SPTBN1"]
#         bt.pl.points(adata[:, genes], hue="gene")

#         bt.pl.density(adata)

#         bt.pl.density(adata, kind="kde")

#         bt.pl.shapes(adata)

#         bt.pl.shapes(adata, color_style="fill")

#         bt.pl.shapes(adata, hue="cell", color_style="fill")

#         fig, ax = plt.subplots()
#         bt.pl.shapes(adata, shapes="cell", linestyle="--", ax=ax)
#         bt.pl.shapes(
#             adata,
#             shapes="nucleus",
#             edgecolor="black",
#             facecolor="lightseagreen",
#             ax=ax,
#         )
#         fig, axes = plt.subplots(1, 2, figsize=(8, 4))

#         bt.pl.density(adata, ax=axes[0], title="default styling")

#         bt.pl.density(
#             adata,
#             ax=axes[1],
#             axis_visible=True,
#             frame_visible=True,
#             square=True,
#             title="square plot + axis",
#         )
#         plt.tight_layout()
#         with mpl.style.context("dark_background"):
#             fig, ax = plt.subplots()
#             bt.pl.shapes(adata, shapes="cell", linestyle="--", ax=ax)
#             bt.pl.shapes(
#                 adata,
#                 shapes="nucleus",
#                 edgecolor="black",
#                 facecolor="lightseagreen",
#                 ax=ax,
#             )
#         cells = adata.obs_names[:8]  # get some cells
#         ncells = len(cells)

#         ncols = 4
#         nrows = 2
#         ax_height = 1.5
#         fig, axes = plt.subplots(
#             nrows, ncols, figsize=(ncols * ax_height, nrows * ax_height)
#         )  # instantiate

#         for c, ax in zip(cells, axes.flat):
#             bt.pl.density(
#                 adata[c],
#                 ax=ax,
#                 square=True,
#                 title="",
#             )

#         plt.subplots_adjust(wspace=0, hspace=0, bottom=0, top=1, left=0, right=1)
#         batches = adata.obs["batch"].unique()[:6]  # get 6 batches
#         nbatches = len(batches)

#         ncols = 3
#         nrows = 2
#         ax_height = 3
#         fig, axes = plt.subplots(
#             nrows, ncols, figsize=(ncols * ax_height, nrows * ax_height)
#         )  # instantiate

#         for b, ax in zip(batches, axes.flat):
#             bt.pl.density(
#                 adata,
#                 batch=b,
#                 ax=ax,
#                 square=True,
#                 title="",
#             )

#         # remove empty axes
#         for ax in axes.flat[nbatches:]:
#             ax.remove()

#         plt.subplots_adjust(wspace=0, hspace=0, bottom=0, top=1, left=0, right=1)

#         self.assertTrue(True)
