import warnings
import geopandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like, ListedColormap

import seaborn as sns
from .._utils import _quantify_variable

from shapely.affinity import translate
from shapely import geometry

import altair as alt
import descartes


alt.themes.enable('opaque')
alt.renderers.enable('default')
alt.data_transformers.enable('json')

# Masala color palette by Noor
# Note: tested colorblind friendliness, did not do so well
masala_palette = sns.color_palette([(1.0, .78, .27), (1.0, .33, .22), (.95, .6, .71), (0, .78, .86), (.39, .81, .63), (.25, .25, .25)])

def quality_metrics(data, width=900, height=250):
    # Count points per cell
    n_points = data.obs['cell'].value_counts().to_frame().reset_index()
    n_points.columns = ['Cell ID', 'Transcript Count']
    # Plot points per cell
    cell_count_chart = alt.Chart(n_points).mark_area(
        opacity=0.5,
        interpolate='step').encode(
        alt.X("Transcript Count:Q", bin=alt.Bin(maxbins=25), axis=alt.Axis(title='Transcript Count')),
        alt.Y('count()', stack=None, axis=alt.Axis(title='Number of Cells'))
    ).properties(
        title='Transcripts / cell',
        width=width/3,
        height=height
    )

    # Count genes per cell
    n_genes = data.obs.groupby('cell').apply(lambda df: len(df['gene'].unique())).reset_index()
    n_genes.columns = ['Cell ID', 'Gene Count']
    # Plot genes per cell
    gene_count_chart = alt.Chart(n_genes).mark_area(
        opacity=0.5,
        interpolate='step'
        ).encode(
        alt.X("Gene Count", bin=alt.Bin(maxbins=25), axis=alt.Axis(title='Gene Count')),
        alt.Y('count()', stack=None, axis=alt.Axis(title='Number of Cells'))
    ).properties(
        title='Genes / cell',
        width=width/3,
        height=height
    )

    # Count points per gene per cell
    n_points_per_gc = data.obs.groupby(['cell', 'gene']).apply(len).to_frame().reset_index()
    n_points_per_gc.columns = ['Cell', 'Gene', 'Transcript Count']
    # Plot points per gene per cell
    gene_by_cell_count_chart = alt.Chart(n_points_per_gc).mark_area(
        opacity=0.5,
        interpolate='step'
    ).encode(
        alt.X("Transcript Count:Q", bin=alt.Bin(maxbins=50), axis=alt.Axis(title='Transcript Count')),
        alt.Y('count()', stack=None, scale=alt.Scale(type='log'), axis=alt.Axis(title='Samples'))
    ).properties(
        title='Gene expression distribution',
        width=width/3,
        height=height
    )


    chart = (cell_count_chart | gene_count_chart | gene_by_cell_count_chart).configure_view(
        strokeWidth=0
    )

    return chart


def plot_cells(data, style='points', cells=None, genes=None, downsample=0.1, scatter_hue=None, scatter_palette='colorblind', power=1, heatmap_cmap='mako', draw_masks=['cell'], width=10, height=10):
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
    downsample: float (0,1]
        Fraction to downsample when plotting. Useful when dealing with large datasets.
    scatter_hue: str
        Name of column in data.obs to interpet for scatter plot color. First tries to interpret if values are matplotlib color-like. For categorical data, tries to plot a different color for each label. For numerical, scales color by numerical values (need to be between [0,1]). Returns error if outside range.
    draw_masks: list
       masks to draw outlines for. Will always include outline for union of `masks`.
    """

    points = data

    # Format cells input
    if cells is None:
        cells =  set(data.obs_vector('cell'))
    else:
        print('Subsetting cells...')
        if type(cells) != list:
            cells = [cells]

    cells = [str.upper(str(i)) for i in cells]

    if -1 in cells:
        warnings.warn('Detected points outside of cells. TODO write clean fx that drops these points')
        
    # Format genes input
    if genes is None:
        genes = data.obs_vector('gene')
    else:
        print('Subsetting genes...')
        if type(genes) != list:
            genes = [genes]

    genes = [str.upper(str(i)) for i in genes]
        
    # Subset points to specified cells and genes
    points_in_cells = data.obs['cell'].astype(str).str.upper().isin(cells)
    points_in_genes = data.obs['gene'].astype(str).str.upper().isin(genes)
    points = data[points_in_cells & points_in_genes,:]

    # * Downsample points per cell
    if downsample < 1:
        print('Downsampling points...')
        downsample_mask = points.obs.groupby('cell').apply(lambda df: df.sample(frac=downsample).index.tolist())
        downsample_mask = downsample_mask.explode().dropna().tolist()
        points = points[downsample_mask,:]
        
    # Format as long table format with points and annotations
    points_df = pd.DataFrame(points.X, columns=['x', 'y'])
    points_df = pd.concat([points_df, points.obs.reset_index(drop=True)], axis=1)

    # Initialize figure
    fig = plt.figure(figsize=(width,height), dpi=100)
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    
    # Plot clipping mask
    clip = data.uns['masks']['cell'].unary_union
    clip = clip.envelope.symmetric_difference(clip)
    clip_patch = descartes.PolygonPatch(clip, color='black')
    data.uns['masks']['cell'].plot(color='black', ax=ax)
    ax.add_patch(clip_patch)
    
    bounds = clip.bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # * Plot raw points
    if style == 'points':
        print('Plotting points...')
        
        # Default point color is teal
        if scatter_hue is None: 
            color = 'teal'
            scatter_palette = None
        else:
            hue_values = points_df[scatter_hue]
            if np.issubdtype(hue_values, np.number):
            
                # Scale point color by obs variable
                if (hue_values >= 0).all() and (hue_values <= 1).all():
                    color = np.expand_dims(points_df[scatter_hue].values, 1)**(1./power)
                    color = [(max(0.3, x), 0.3, 0.3, max(0.3, x)) for x in color]
                    scatter_cmap = ListedColormap(scatter_palette)
                    ax.scatter(data=points_df, x='x', y='y', c=scatter_hue, s=3, cmap=scatter_cmap)
                else:
                    return ValueError('Numeric values for \'scatter_hue\' parameter must be in range [0,1].')
                
            else: 
                # Try to interpret as matplotlib color
                if is_color_like(hue_values.iloc[0]):
                    color = hue_values.apply(is_color_like)
                else:
                    # Color points by category (custom _determinisitic_ color mapping)
                    phenomap, color = pheno_to_color(points_df[scatter_hue], palette=scatter_palette)
                
                ax.scatter(data=points_df, x='x', y='y', c=color, s=3)

        

    # * Plot heatmap
    elif style == 'heatmap':
        print('Plotting heatmap...')
        sns.kdeplot(data=points_df, x='x', y='y', cmap=heatmap_cmap,
                    ax=ax, bw_adjust=0.1, shade_lowest=False, 
                    levels=100, fill=True)

    # * Plot mask outlines
    for mask in draw_masks:
        mask_outline = data.uns['masks'][mask].boundary
        mask_outline.plot(ax=ax, lw=1, color=(1,1,1,0.5))
    
    ax.set_facecolor('black')

    return fig

def pca(data, c1=0, c2=1, hue='gene', huetype='nominal', width=400, height=400, path=''):
    return _plot_dim(data, 'pca', hue=hue, huetype=huetype, c1=c1, c2=c2, width=width, height=height, path=path)



def umap(data, c1=0, c2=1, hue='gene', huetype='nominal', width=400, height=400, path=''):
    return _plot_dim(data, 'umap', hue=hue, huetype=huetype, c1=c1, c2=c2, width=width, height=height, path=path)


def _plot_dim(data, dim_type, **kwargs):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    dim_type : str
        'pca' or 'umap'
    """

    df = data.uns[f'{dim_type}_components']

    # Map labels to points
    if kwargs["hue"] == 'label':
        df['label'] = data.uns['labels']['label']

    chart = alt.Chart(df).mark_circle(
        opacity=0.8,
        size=10
    ).encode(
        x=f'{kwargs["c1"]}:Q',
        y=f'{kwargs["c2"]}:Q',
        color=alt.Color(kwargs["hue"], type=kwargs['huetype'], scale=alt.Scale(scheme='dark2'))
    ).properties(
        width=kwargs["width"],
        height=kwargs["height"]
    )

    if kwargs["path"]:
        chart.save(kwargs["path"], scale_factor=2)

    return chart

def pheno_to_color(pheno, palette=masala_palette):
    '''
    Maps list of categorical labels to a color palette.
    Input values are first sorted alphanumerically least to greatest before mapping to colors. This ensures consistent colors regardless of input value order.
    
    Parameters
    ----------
    pheno : pd.Series
        Categorical labels to map
    palette : 
    
    Returns
    -------
    dict
        Mapping of label to color in RGBA
    tuples
        List of converted colors for each sample, formatted as RGBA tuples.
    
    
    list of RGBA tuples
    
    '''
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


