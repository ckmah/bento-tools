from typing import List, Tuple, Union
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from spatialdata._core.spatialdata import SpatialData
from upsetplot import UpSet, from_indicators

from .._constants import PATTERN_COLORS, PATTERN_NAMES
from ..tools import lp_stats
from .._utils import get_points
from ._utils import savefig
from ._multidimensional import _radviz


@savefig
def lp_dist(sdata, show_counts="", show_percentages="{:.1%}", scale=1, fname=None):
    """Plot pattern combination frequencies as an UpSet plot.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    show_counts : str, optional
        Format count labels with this string, do not show counts if empty, by default ""
    show_percentages : str, optional
        Format count labels with this string, do not show counts if empty, by default ""
    scale : int, optional
        scale > 1 scales the plot larger, scale < 1 scales. the plot smaller, by default 1
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    sample_labels = sdata.tables["table"].uns["lp"]
    sample_labels = sample_labels == 1

    # Sort by degree, then pattern name
    sample_labels["degree"] = -sample_labels[PATTERN_NAMES].sum(axis=1)
    sample_labels = (
        sample_labels.reset_index()
        .sort_values(["degree"] + PATTERN_NAMES, ascending=False)
        .drop("degree", axis=1)
    )

    upset = UpSet(
        from_indicators(PATTERN_NAMES, data=sample_labels),
        element_size=scale * 40,
        min_subset_size=sample_labels.shape[0] * 0.001,
        facecolor="lightgray",
        sort_by=None,
        show_counts=show_counts,
        show_percentages=show_percentages,
    )

    for p, color in zip(PATTERN_NAMES, PATTERN_COLORS):
        if sample_labels[p].sum() > 0:
            upset.style_subsets(present=p, max_degree=1, facecolor=color)

    upset.plot()
    plt.suptitle(f"Localization Patterns\n{sample_labels.shape[0]} samples")


@savefig
def lp_genes(
    sdata: SpatialData,
    groupby: str = "feature_name",
    points_key="transcripts",
    instance_key="cell_boundaries",
    annotate: Union[int, List[str], None] = None,
    sizes: Tuple[int] = (2, 100),
    size_norm: Tuple[int] = (0, 100),
    ax: plt.Axes = None,
    fname: str = None,
    **kwargs,
):
    """
    Plot the pattern distribution of each group in a RadViz plot. RadViz projects
    an N-dimensional data set into a 2D space where the influence of each dimension
    can be interpreted as a balance between the influence of all dimensions.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    groupby : str
        Grouping variable, default "gene"
    annotate : int, list of str, optional
        Annotate the top n genes or a list of genes, by default None
    sizes : tuple
        Minimum and maximum point size to scale points, default (2, 100)
    size_norm : tuple
        Minimum and maximum data values to scale point size, default (0, 100)
    ax : matplotlib.Axes, optional
        Axis to plot on, by default None
    fname : str, optional
        Save the figure to specified filename, by default None
    **kwargs
        Options to pass to matplotlib plotting method.
    """
    lp_stats(sdata)

    palette = dict(zip(PATTERN_NAMES, PATTERN_COLORS))

    n_cells = sdata.tables["table"].n_obs
    gene_frac = sdata.tables["table"].uns["lp_stats"][PATTERN_NAMES] / n_cells
    genes = gene_frac.index
    gene_expression_array = sdata.tables["table"][:, genes].X.toarray()
    gene_logcount = gene_expression_array.mean(axis=0, where=gene_expression_array > 0)
    gene_logcount = np.log2(gene_logcount + 1)
    gene_frac["logcounts"] = gene_logcount

    cell_fraction = (
        100
        * get_points(sdata, points_key, astype="pandas", sync=True)
        .groupby(groupby, observed=True)[instance_key]
        .nunique()
        / n_cells
    )
    gene_frac["cell_fraction"] = cell_fraction

    scatter_kws = dict(sizes=sizes, size_norm=size_norm)
    scatter_kws.update(kwargs)
    _radviz(gene_frac, annotate=annotate, ax=ax, **scatter_kws)


@savefig
def lp_diff_discrete(sdata: SpatialData, phenotype: str, fname: str = None):
    """Visualize gene pattern frequencies between groups of cells. Plots the
    log2 fold change and -log10 p-value of each gene, similar to volcano plot. Run after :func:`bento.tl.lp_diff_discrete()`

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    phenotype : str
        Variable used to group cells when calling :func:`bento.tl.lp_diff()`.
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    diff_stats = sdata.tables["table"].uns[f"diff_{phenotype}"]

    palette = dict(zip(PATTERN_NAMES, PATTERN_COLORS))
    g = sns.relplot(
        data=diff_stats,
        x="log2fc",
        y="-log10padj",
        size=4,
        hue="pattern",
        col="phenotype",
        col_wrap=3,
        height=2.5,
        palette=palette,
        s=20,
        linewidth=0,
    )

    g.set_titles(col_template="{col_name}")

    for ax in g.axes:
        ax.axvline(0, lw=0.5, c="grey")  # -log2fc = 0
        ax.axvline(-2, lw=1, c="pink", ls="dotted")  # log2fc = -2
        ax.axvline(2, lw=1, c="pink", ls="dotted")  # log2fc = 2
        ax.axhline(
            -np.log10(0.05), c="pink", ls="dotted", zorder=0
        )  # line where FDR = 0.05
        sns.despine()

    return g
