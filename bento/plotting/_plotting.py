import warnings

import altair as alt
import descartes
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, is_color_like
from shapely import geometry
from shapely.affinity import translate

from ..tools._tools import subsample

alt.themes.enable("opaque")
alt.renderers.enable("default")
alt.data_transformers.enable("json")

# Masala color palette by Noor
# Note: tested colorblind friendliness, did not do so well
masala_palette = sns.color_palette(
    [
        (1.0, 0.78, 0.27),
        (1.0, 0.33, 0.22),
        (0.95, 0.6, 0.71),
        (0, 0.78, 0.86),
        (0.39, 0.81, 0.63),
        (0.25, 0.25, 0.25),
    ]
)


def quality_metrics(data, width=900, height=250):
    # Count points per cell
    n_points = data.obs["cell"].value_counts().to_frame().reset_index()
    n_points.columns = ["Cell ID", "Transcript Count"]
    # Plot points per cell
    cell_count_chart = (
        alt.Chart(n_points)
        .mark_area(opacity=0.5, interpolate="step")
        .encode(
            alt.X(
                "Transcript Count:Q",
                bin=alt.Bin(maxbins=25),
                axis=alt.Axis(title="Transcript Count"),
            ),
            alt.Y("count()", stack=None, axis=alt.Axis(title="Number of Cells")),
        )
        .properties(title="Transcripts / cell", width=width / 3, height=height)
    )

    # Count genes per cell
    n_genes = (
        data.obs.groupby("cell")
        .apply(lambda df: len(df["gene"].unique()))
        .reset_index()
    )
    n_genes.columns = ["Cell ID", "Gene Count"]
    # Plot genes per cell
    gene_count_chart = (
        alt.Chart(n_genes)
        .mark_area(opacity=0.5, interpolate="step")
        .encode(
            alt.X(
                "Gene Count", bin=alt.Bin(maxbins=25), axis=alt.Axis(title="Gene Count")
            ),
            alt.Y("count()", stack=None, axis=alt.Axis(title="Number of Cells")),
        )
        .properties(title="Genes / cell", width=width / 3, height=height)
    )

    # Count points per gene per cell
    n_points_per_gc = (
        data.obs.groupby(["cell", "gene"]).apply(len).to_frame().reset_index()
    )
    n_points_per_gc.columns = ["Cell", "Gene", "Transcript Count"]
    # Plot points per gene per cell
    gene_by_cell_count_chart = (
        alt.Chart(n_points_per_gc)
        .mark_area(opacity=0.5, interpolate="step")
        .encode(
            alt.X(
                "Transcript Count:Q",
                bin=alt.Bin(maxbins=50),
                axis=alt.Axis(title="Transcript Count"),
            ),
            alt.Y(
                "count()",
                stack=None,
                scale=alt.Scale(type="log"),
                axis=alt.Axis(title="Samples"),
            ),
        )
        .properties(
            title="Gene expression distribution", width=width / 3, height=height
        )
    )

    chart = (
        cell_count_chart | gene_count_chart | gene_by_cell_count_chart
    ).configure_view(strokeWidth=0)

    return chart


def spots_diff(data, groups):
    dl_results = data.uns["sample_data"][f"dl_{groups}"]
    # with sns.axes_style("whitegrid"):
    g = sns.relplot(
        data=dl_results.sort_values("pattern"),
        x="dy/dx",
        y="-log10padj",
        hue="pattern",
        size="-log10padj",
        sizes=(10, 200),
        col="phenotype",
        s=4,
        height=4,
        linewidth=0,
        palette="tab10",
    )
    for ax in g.axes.flatten():
        ax.grid(False, axis="y")
        ax.axhline(-np.log10(0.05), color="gray", lw=1, linestyle="--")
        ax.axvline(0, color="black", lw=1, linestyle="-")
        sns.despine(top=False, right=False, bottom=False, left=False)
        plt.xlim(-1, 1)
    plt.ylim(-0.05, 3)


def spots_freq(data, gene, groups=None, relative=True, stacked=False):
    """Plot SPOTS localization pattern frequencies for a given gene.

    Parameters
    ----------
    data : [type]
        [description]
    gene : str
        Gene name to show frequencies.
    groups : str, optional
        Sample category to stratify frequencies, by default None.
    relative : bool, optional
        Whether to calculate relative fractions or absolute number of cells, by default True.
    stacked : bool, optional
        Whether to use a single plot or multiple plots for each category, by default False.
    """
    gene_mask = (data.uns["sample_index"]["gene"] == gene).values
    classes = data.uns["sample_data"]["patterns"].columns.tolist()

    if groups:
        pattern_fracs = pd.concat(
            [
                data.uns["sample_data"]["patterns"].reset_index(drop=True),
                data.uns["sample_data"][groups].reset_index(drop=True),
            ],
            axis=1,
        ).loc[gene_mask]

        if relative:
            pattern_fracs = pattern_fracs.groupby(groups).apply(
                lambda df: df[classes].sum() / df.shape[0]
            )
        else:
            pattern_fracs = pattern_fracs.groupby(groups).apply(
                lambda df: df[classes].sum()
            )

        pattern_fracs = pattern_fracs.reset_index()

    else:
        pattern_fracs = (
            data.uns["sample_data"]["patterns"].reset_index(drop=True).loc[gene_mask]
        )

        if relative:
            pattern_fracs = (
                (pattern_fracs[classes].sum() / pattern_fracs.shape[0]).to_frame().T
            )

        else:
            pattern_fracs = pattern_fracs[classes].sum().to_frame().T

    # Calculate polar angles
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    stats = pd.concat((pattern_fracs[classes], pattern_fracs[[classes[0]]]), axis=1)

    if groups:
        stats = pd.concat([stats, pattern_fracs[groups]], axis=1)

    def polar_gene(row, ax, color=None):
        with sns.axes_style("whitegrid"):

            # Plot gene trace
            ax.plot(angles, row, color=color)
            ax.fill(angles, row, facecolor=color, alpha=0.2)

            # Set theta axis labels
            ax.set_thetagrids(angles * 180 / np.pi, classes + [classes[0]])
            ax.tick_params(axis="x", labelsize="small")

            # Format r ticks
            ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune='lower'))
            ax.tick_params(axis="y", labelsize="x-small")

            ax.grid(True, linestyle=":")

    if groups:
        ngroups = stats[groups].nunique()
        if stacked:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)

            for name, row in stats.iterrows():
                row = row[row.index != groups]
                polar_gene(row, ax)

            labels = stats[groups].values.tolist()
            ax.legend(
                labels,
                loc=(0.9, 0.95),
                labelspacing=0.1,
                fontsize="small",
                frameon=False,
            )

        else:
            fig, axs = plt.subplots(
                1,
                ngroups,
                sharex=True,
                subplot_kw=dict(polar=True),
                figsize=(4 * ngroups, 3),
            )
            fig.subplots_adjust(wspace=0.3)

            for ax, (name, row) in zip(axs, stats.iterrows()):
                ax.set_title(row[groups])
                row = row[row.index != groups]
                polar_gene(row, ax)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        polar_gene(stats.iloc[0], ax)


def plot_cells(
    data,
    style="points",
    cells=None,
    genes=None,
    fraction=0.1,
    scatter_hue=None,
    scatter_palette=["red", "grey"],
    s=3,
    alpha=1,
    power=1,
    heatmap_cmap="mako",
    draw_masks=["cell"],
    width=10,
    height=10,
):
    """
    Visualize distribution of variable in spatial coordinates.
    Parameters
    ----------
    data : dict
    type : str
        Options include 'points' and 'heatmap'
    cells: None, list
        Select specified list of cells. Default value of None selects all cells.
    genes: None, list
        Select specified list of genes. Default value of None selects all genes.
    fraction: float (0,1]
        Fraction to fraction when plotting. Useful when dealing with large datasets.
    scatter_hue: str
        Name of column in data.obs to interpet for scatter plot color. First tries to interpret if values are matplotlib color-like. For categorical data, tries to plot a different color for each label. For numerical, scales color by numerical values (need to be between [0,1]). Returns error if outside range.
    draw_masks: list
       masks to draw outlines for. Will always include outline for union of `masks`.
    """

    points = data

    # Format cells input
    if cells is None:
        cells = set(data.obs_vector("cell"))
    else:
        # print('Subsetting cells...')
        if type(cells) != list:
            cells = [cells]

    if "-1" in cells:
        warnings.warn(
            "Detected points outside of cells. TODO write clean fx that drops these points"
        )

    # Format genes input
    if genes is None:
        genes = data.obs_vector("gene")
    else:
        # print('Subsetting genes...')
        if type(genes) != list:
            genes = [genes]

    # Subset points to specified cells and genes
    points_in_cells = data.obs["cell"].isin(cells)
    points_in_genes = data.obs["gene"].isin(genes)
    points = data[points_in_cells & points_in_genes, :]

    # * fraction points per cell
    if fraction < 1:
        # print('Downsampling points...')
        points = subsample(points, fraction)

    # Format as long table format with points and annotations
    points_df = pd.DataFrame(points.X, columns=["x", "y"])
    points_df = pd.concat([points_df, points.obs.reset_index(drop=True)], axis=1)

    # Initialize figure
    fig = plt.figure(figsize=(width, height), dpi=100)
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    # Subset masks to specified cells
    mask_outlines = {}
    for mask in draw_masks:
        if mask == "cell":
            mask_outlines[mask] = data.uns["masks"]["cell"].loc[cells]
        else:
            mask_index = data.uns["mask_index"][mask]
            n_indices = mask_index[mask_index["cell"].isin(cells)].index
            mask_outlines[mask] = data.uns["masks"][mask].loc[n_indices]

    # Plot clipping mask
    clip = mask_outlines["cell"].unary_union
    clip = clip.envelope.symmetric_difference(clip)
    # clip_patch = descartes.PolygonPatch(clip, color="white")
    # mask_outlines["cell"].plot(color="white", ax=ax)
    # ax.add_patch(clip_patch)

    bounds = clip.bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # * Plot mask faces
    for mask in draw_masks:
        if mask == "cell":
            mask_outlines[mask].plot(ax=ax, facecolor='black', lw=0, alpha=0.1)
        else:
            mask_outlines[mask].plot(ax=ax, facecolor='tab:blue', lw=0, alpha=0.1)


    # * Plot raw points
    if style == "points":
        # print('Plotting points...')

        # Default point color is teal
        if scatter_hue is None:
            color = "tab:red"
            scatter_palette = None
            ax.scatter(data=points_df, x="x", y="y", c=color, s=s, alpha=alpha)

        # Handle coloring by variable
        else:
            hue_values = points_df[scatter_hue]

            # Color by [0,1] range quantitative variable
            if np.issubdtype(hue_values, np.number):
                if (hue_values >= 0).all() and (hue_values <= 1).all():
                    color = np.expand_dims(points_df[scatter_hue].values, 1) ** (
                        1.0 / power
                    )
                    color = [(max(0.3, x), 0.3, 0.3, max(0.3, x)) for x in color]
                    scatter_cmap = ListedColormap(scatter_palette)
                else:
                    return ValueError(
                        "Numeric values for 'scatter_hue' parameter must be in range [0,1]."
                    )

                ax.scatter(
                    data=points_df,
                    x="x",
                    y="y",
                    c=scatter_hue,
                    s=s,
                    alpha=alpha,
                    cmap=scatter_cmap,
                )

            # Color as qualitative variable
            else:
                # Try to interpret as matplotlib color
                if is_color_like(hue_values.iloc[0]):
                    color = hue_values.apply(is_color_like)
                # Color points by category (custom _determinisitic_ color mapping)
                else:
                    phenomap, color = pheno_to_color(
                        points_df[scatter_hue], palette=scatter_palette
                    )

                ax.scatter(data=points_df, x="x", y="y", c=color, s=s, alpha=alpha)

    # * Plot heatmap
    elif style == "heatmap":
        sns.kdeplot(
            data=points_df,
            x="x",
            y="y",
            cmap=sns.color_palette("light:darkred", as_cmap=True),
            ax=ax,
            bw_adjust=0.15,
            shade_lowest=False,
            levels=50,
            fill=True,
            alpha=alpha
        )

    # Plot mask outlines
    for mask in draw_masks:
        if mask == "cell":
            mask_outlines[mask].boundary.plot(ax=ax, lw=0.5, edgecolor=(0,0,0,1))
        else:
            mask_outlines[mask].boundary.plot(ax=ax, lw=0.5, edgecolor='tab:blue')


    return fig


def pca(
    data, c1=0, c2=1, hue="gene", huetype="nominal", width=400, height=400, path=""
):
    return _plot_dim(
        data,
        "pca",
        hue=hue,
        huetype=huetype,
        c1=c1,
        c2=c2,
        width=width,
        height=height,
        path=path,
    )


def umap(
    data, c1=0, c2=1, hue="gene", huetype="nominal", width=400, height=400, path=""
):
    return _plot_dim(
        data,
        "umap",
        hue=hue,
        huetype=huetype,
        c1=c1,
        c2=c2,
        width=width,
        height=height,
        path=path,
    )


def _plot_dim(data, dim_type, **kwargs):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    dim_type : str
        'pca' or 'umap'
    """

    df = data.uns[f"sample_{dim_type}"]

    sns.scatter(df[:, 0], df[:, 1])

    # Map labels to points
    if hue in kwargs["hue"] == "label":
        df["label"] = data.uns["labels"]["label"]

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.8, size=10)
        .encode(
            x=f'{kwargs["c1"]}:Q',
            y=f'{kwargs["c2"]}:Q',
            color=alt.Color(
                kwargs["hue"], type=kwargs["huetype"], scale=alt.Scale(scheme="dark2")
            ),
        )
        .properties(width=kwargs["width"], height=kwargs["height"])
    )

    if kwargs["path"]:
        chart.save(kwargs["path"], scale_factor=2)

    return chart


# plot = Petal(bounds=[np.repeat(0,6), np.repeat(1.,6)], figsize=(30,5), cmap='Set1', labels=sf_pred_prob.columns.sort_values().tolist())
# plot.add(sf_pred_prob[sf_pred_prob.columns.sort_values()].iloc[:6].values)
# plot.show()


def pheno_to_color(pheno):
    """
    Maps list of categorical labels to a color palette.
    Input values are first sorted alphanumerically least to greatest before mapping to colors. This ensures consistent colors regardless of input value order.

    Parameters
    ----------
    pheno : pd.Series
        Categorical labels to map
    palette : list
        list of RGB tuples

    Returns
    -------
    dict
        Mapping of label to color in RGBA
    tuples
        List of converted colors for each sample, formatted as RGBA tuples.


    list of RGBA tuples

    """
    if type(palette) is str:
        palette = sns.color_palette(palette)
    else:
        palette = palette

    values = pheno.unique()
    values.sort()
    n_colors = len(values)
    palette = sns.color_palette(palette, n_colors=n_colors)
    study2color = dict(zip(values, palette))
    sample_colors = list(pheno.map(study2color))
    return study2color, sample_colors
