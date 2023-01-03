import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import radviz
from upsetplot import UpSet, from_indicators

from .._utils import PATTERN_COLORS, PATTERN_NAMES
from ..tools._lp import lp_stats
from ._utils import savefig


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


def lp_genes(
    data,
    groupby="gene",
    highlight_groups=None,
    kind="scatter",
    hue="Pattern",
    sizes=(2, 100),
    gridsize=20,
    ax=None,
    fname=None,
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
    kind : str
        'Scatter' for scatter plot, 'hex' for hex plot, default "scatter"
    hue : str
        Name of columns in data.obs to color points, default "Pattern"
    sizes : tuple
        Minimum and maximum point size to scale points, default (2, 100)
    gridsize : int
        Number of hex bins along each axis, default 20
    fname : str, optional
        Save the figure to specified filename, by default None
    **kwargs
        Options to pass to matplotlib plotting method.
    """
    lp_stats(data, groupby)

    palette = dict(zip(PATTERN_NAMES, PATTERN_COLORS))

    gene_frac = data.uns["lp_stats"][PATTERN_NAMES] / data.n_obs
    gene_frac["Pattern"] = gene_frac.idxmax(axis=1)
    gene_frac_copy = gene_frac.copy()
    gene_frac_copy["Pattern"] = ""
    _radviz(
        gene_frac, highlight_groups, kind, hue, sizes, gridsize, ax, kwargs, palette
    )


@savefig
def _radviz(
    df,
    accent=None,
    kind="scatter",
    hue=None,
    palette=None,
    sizes=(2, 100),
    gridsize=20,
    ax=None,
    fname=None,
    **kwargs,
):
    with plt.rc_context({"font.size": 14}):
        # RADVIZ plot
        if not ax:
            figsize = (6, 6)
            fig = plt.figure(figsize=figsize)

        # Plot the "circular" axis, labels and point positions
        if not ax:
            ax = radviz(df, hue, s=0)
        else:
            radviz(df, hue, s=0, ax=ax)

        ax.get_legend().remove()
        circle = plt.Circle((0, 0), radius=1, color="black", fill=False)
        ax.add_patch(circle)

        # Hide 2D axes
        ax.axis(False)

        # Get points
        pts = []
        for c in ax.collections:
            pts.extend(c.get_offsets().data)

        pts = np.array(pts).reshape(-1, 2)
        xy = pd.DataFrame(pts, index=df.index)
        xy[hue] = df[hue]

        # Point size ~ row total
        xy["Total"] = df.sum(axis=1)
        size_norm = (0, 1)

        # Plot points as scatter or hex
        if kind == "scatter":
            del ax.collections[0]

            # Plot points
            sns.scatterplot(
                data=xy,
                x=0,
                y=1,
                size="Total",
                hue=hue,
                sizes=sizes,
                size_norm=size_norm,
                linewidth=0,
                palette=palette,
                ax=ax,
                **kwargs,
            )
            plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=False)

        elif kind == "hex":
            # Hexbin
            xy.plot.hexbin(
                x=0,
                y=1,
                gridsize=gridsize,
                extent=(-1, 1, -1, 1),
                cmap=sns.light_palette("gray", as_cmap=True),
                mincnt=1,
                colorbar=False,
                ax=ax,
                **kwargs,
            )
            # [left, bottom, width, height]
            plt.colorbar(
                ax.collections[-1],
                cax=fig.add_axes([1, 0.4, 0.05, 0.3]),
                label="genes",
            )

        if isinstance(accent, list):
            sns.scatterplot(
                data=xy.loc[accent],
                x=0,
                y=1,
                hue=hue,
                size="Total",
                sizes=sizes,
                size_norm=size_norm,
                linewidth=2,
                palette=palette,
                edgecolor="gray",
                legend=False,
                ax=ax,
            )
            # plt.legend(
            #     bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=False
            # )
            texts = [
                ax.text(row[[0]] + 0.03, row[[1]] + 0.03, group)
                for group, row in xy.loc[accent].iterrows()
            ]


@savefig
def lp_diff(data, phenotype, fname=None):
    """Visualize gene pattern frequencies between groups of cells by plotting
    log2 fold change and -log10p, similar to volcano plot. Run after `bento.tl.lp_diff()`

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    phenotype : str
        Variable used to group cells when calling `bento.tl.lp_diff()`.
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
