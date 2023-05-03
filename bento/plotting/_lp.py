from typing import List, Tuple, Union
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from upsetplot import UpSet, from_indicators

from .._constants import PATTERN_COLORS, PATTERN_NAMES
from ..tools import lp_stats
from ._utils import savefig
from ._multidimensional import _radviz


@savefig
def lp_dist(data, percentage=False, scale=1, fname=None):
    """Plot pattern combination frequencies as an UpSet plot.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    percentage : bool, optional
        If True, label each bar as a percentage else label as a count, by default False
    scale : int, optional
        scale > 1 scales the plot larger, scale < 1 scales. the plot smaller, by default 1
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    sample_labels = data.uns["lp"]
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
        show_counts=(not percentage),
        show_percentages=percentage,
    )

    for p, color in zip(PATTERN_NAMES, PATTERN_COLORS):
        if sample_labels[p].sum() > 0:
            upset.style_subsets(present=p, max_degree=1, facecolor=color)

    upset.plot()
    plt.suptitle(f"Localization Patterns\n{sample_labels.shape[0]} samples")


@savefig
def lp_gene_dist(data, fname=None):
    """Plot the cell fraction distribution of each pattern as a density plot.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    lp_stats(data)

    col_names = [f"{p}_fraction" for p in PATTERN_NAMES]
    gene_frac = data.var[col_names]
    gene_frac.columns = PATTERN_NAMES
    # Plot frequency distributions
    sns.displot(
        data=gene_frac,
        kind="kde",
        multiple="layer",
        height=3,
        palette=PATTERN_COLORS,
    )
    plt.xlim(0, 1)
    sns.despine()


@savefig
def lp_genes(
    data: AnnData,
    groupby: str = "gene",
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
    data : AnnData
        Spatial formatted AnnData
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
    lp_stats(data, groupby)

    palette = dict(zip(PATTERN_NAMES, PATTERN_COLORS))

    n_cells = data.n_obs
    gene_frac = data.uns["lp_stats"][PATTERN_NAMES] / n_cells

    gene_logcount = data.X.mean(axis=0, where=data.X > 0)
    gene_logcount = np.log2(gene_logcount + 1)
    gene_frac["logcounts"] = gene_logcount

    cell_fraction = (
        100
        * data.uns["points"].groupby("gene", observed=True)["cell"].nunique()
        / n_cells
    )
    gene_frac["cell_fraction"] = cell_fraction

    scatter_kws = dict(sizes=sizes, size_norm=size_norm)
    scatter_kws.update(kwargs)
    _radviz(gene_frac, annotate=annotate, ax=ax, **scatter_kws)


@savefig
def lp_diff(data: AnnData, phenotype: str, fname: str = None):
    """Visualize gene pattern frequencies between groups of cells by plotting
    log2 fold change and -log10p, similar to volcano plot. Run after :func:`bento.tl.lp_diff()`

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    phenotype : str
        Variable used to group cells when calling :func:`bento.tl.lp_diff()`.
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    diff_stats = data.uns[f"diff_{phenotype}"]

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
