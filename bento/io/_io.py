import warnings
warnings.filterwarnings("ignore")

anndata = None
import geopandas
import numpy as np
import pandas as pd
from shapely import geometry, wkt


def read_h5ad(filename, backed=None):
    """Load bento processed AnnData object from h5ad.

    Parameters
    ----------
    filename : str
        File name to load data file.
    backed : 'r', 'r+', True, False, None
        If 'r', load AnnData in backed mode instead of fully loading it into memory (memory mode).
        If you want to modify backed attributes of the AnnData object, you need to choose 'r+'.
        By default None.
    Returns
    -------
    AnnData
        AnnData data object.
    """
    global anndata
    if anndata is None:
        import anndata

    adata = anndata.read_h5ad(filename, backed=backed)

    
    
    # Load obs columns that are shapely geometries
    adata.obs = adata.obs.apply(
        lambda col: geopandas.GeoSeries(
            col.astype(str).apply(lambda val: wkt.loads(val) if val != "None" else None)
        )
        if col.astype(str).str.startswith("POLYGON").any()
        else geopandas.GeoSeries(col)
    )
    
    adata.obs.index.name = 'cell'
    adata.var.index.name = 'gene'

    return adata


def write_h5ad(data, filename):
    """Write AnnData to h5ad. Casts each GeoDataFrame in adata.uns['masks'] for h5ad compatibility.

    Parameters
    ----------
    adata : AnnData
        bento loaded AnnData
    filename : str
        File name to write data file.
    """
    # Convert geometry from GeoSeries to list for h5ad serialization compatibility
    adata = data.copy()

    adata.obs = adata.obs.apply(
        lambda col: col.apply(lambda val: val.wkt if val is not None else val).astype(
            str
        )
        if col.astype(str).str.startswith("POLYGON").any()
        else col
    )
    
    adata.uns['points'] = adata.uns['points'].drop('geometry', axis=1, errors='ignore')

    # Write to h5ad
    adata.write(filename)


def read_geodata(points, cell, other={}):
    """Load spots and masks for many cells.

    Parameters
    ----------
    points : str
        Filepath to spots .shp file. Expects GeoDataFrame with geometry of Points, and 'gene' column at minimum.
    cell : str
        Filepath to cell segmentation masks .shp file. Expects GeoDataFrame with geometry of Polygons.
    other : dict(str)
        Filepaths to all other segmentation masks .shp files; expects GeoDataFrames of same format.
        Use keys of dict to access corresponding outputs.
    Returns
    -------
        AnnData object
    """
    print("Loading points...")
    points = geopandas.read_file(points)

    # Load masks
    print("Loading masks...")
    mask_paths = {"cell": cell, **other}
    masks = pd.Series(mask_paths).apply(_load_masks)

    # Index points for all masks
    print("Indexing points...")
    point_index = masks.apply(lambda mask: _index_points(points[["geometry"]], mask)).T

    # Index masks to cell
    print("Indexing masks...")
    mask_geoms = _index_masks(masks)

    # Main long dataframe for reformatting
    uns_points = pd.concat(
        [
            points[["x", "y", "gene"]].reset_index(drop=True),
            point_index.reset_index(drop=True),
        ],
        axis=1,
    )

    # Remove extracellular points
    uns_points = uns_points.loc[uns_points["cell"] != "-1"]

    # Aggregate points to counts
    print("Formatting AnnData object...")
    if "nucleus" not in uns_points.columns:
        expression = (
            uns_points[["cell", "gene"]]
            .groupby(["cell", "gene"])
            .apply(lambda x: x.shape[0])
            .reset_index()
        )
    else:
        # Use nuclear inclusion for splicing/unspliced layers
        expression = (
            uns_points[["cell", "gene", "nucleus"]]
            .groupby(["cell", "gene", "nucleus"])
            .apply(lambda x: x.shape[0])
            .to_frame()
            .reset_index()
        )

    # Create cell x gene matrix
    print("Processing expression...")
    cellxgene = expression.pivot_table(
        index=["cell"], columns=["gene"], aggfunc="sum"
    ).fillna(0)
    cellxgene.columns = cellxgene.columns.get_level_values("gene")

    # Add splice data
    if "nucleus" in uns_points.columns:
        print("Processing splicing...")
        spliced, unspliced = _to_spliced_expression(expression)

    print("Processing point coordinates...")

    # Create scanpy anndata object
    adata = anndata.AnnData(X=cellxgene)
    mask_geoms = mask_geoms.reindex(adata.obs.index)
    adata.obs = pd.concat([adata.obs, mask_geoms], axis=1)
    adata.obs.index = adata.obs.index.astype(str)

    # Save spliced/unspliced counts to layers
    if "nucleus" in uns_points.columns:
        adata.layers["spliced"] = spliced
        adata.layers["unspliced"] = unspliced

    # Save cell, gene, and nucleus as categorical type to save memory
    uns_points["cell"] = uns_points["cell"].astype("category")
    uns_points["gene"] = uns_points["gene"].astype("category")

    if "nucleus" in uns_points.columns:
        uns_points["nucleus"] = uns_points["nucleus"].astype("category")

    adata.uns = {"points": uns_points}

    print("Done.")
    return adata


def _load_masks(path):
    """Load GeoDataFrame from path.

    Parameters
    ----------
    path : str
        Path to .shp file.

    Returns
    -------
    GeoDataFrame
        Contains masks as Polygons.
    """
    mask = geopandas.read_file(path)
    mask.index = mask.index.astype(str)

    for i, poly in enumerate(mask["geometry"]):
        if type(poly) == geometry.MultiPolygon:
            print(f"Object at index={i} is a MultiPolygon.")
            print(poly)
            return

    # Cleanup polygons
    # mask.geometry = mask.geometry.buffer(2).buffer(-2)
    # mask.geometry = mask.geometry.apply(unary_union)

    return mask


def _index_masks(masks):
    """Spatially index other masks to cell mask.

    Parameters
    ----------
    masks : dict
        Dictionary of mask GeoDataFrames.

    Returns
    -------
    GeoDataFrame
        [description]
    """
    cell_mask = masks["cell"]

    shapes = cell_mask.copy()
    if "FID" in shapes.columns:
        shapes = shapes.drop("FID", axis=1)

    shapes = shapes.rename(columns={"geometry": "cell_shape"})

    for m, mask in masks.items():
        if m != "cell":
            geometry = (
                geopandas.sjoin(
                    mask, cell_mask, how="left", op="within", rsuffix="cell"
                )
                .dropna()
                .drop_duplicates("index_cell")
                .set_index("index_cell")
                .reindex(cell_mask.index)["geometry"]
            )
            geometry.name = f"{m}_shape"
            shapes[f"{m}_shape"] = geometry

    return shapes


def _index_points(points, mask):
    """Index points to each mask item and save. Assumes non-overlapping masks.

    Parameters
    ----------
    points : GeoDataFrame
        Point coordinates.
    mask : GeoDataFrame
        Mask polygons.
    Returns
    -------
    Series
        Return list of mask indices corresponding to each point.
    """
    index = geopandas.sjoin(points.reset_index(), mask, how="left", op="intersects")

    # remove multiple cells assigned to same point
    index = index.drop_duplicates(subset="index", keep="first")
    index = index.sort_index()
    index = index.reset_index()["index_right"]
    index = index.fillna(-1).astype(str)

    return pd.Series(index)


def concatenate(adatas):
    uns_points = []
    for i, adata in enumerate(adatas):
        points = adata.uns["points"].copy()
        points["cell"] = points["cell"].astype(str) + f"-{i}"

        points["batch"] = i
        uns_points.append(points)

    new_adata = adatas[0].concatenate(adatas[1:])

    uns_points = pd.concat(uns_points)
    points["cell"] = points["cell"].astype("category")
    points["batch"] = points["batch"].astype("category")

    new_adata.uns["points"] = uns_points

    return new_adata


def _to_spliced_expression(expression):
    cell_nucleus = expression.pivot(index=["cell", "nucleus"], columns="gene")
    unspliced = []
    spliced = []
    idx = pd.IndexSlice
    spliced_index = "-1"

    def to_splice_layers(cell_df):
        unspliced_index = (
            cell_df.index.get_level_values("nucleus")
            .drop(spliced_index, errors="ignore")
            .tolist()
        )

        unspliced.append(
            cell_df.loc[idx[:, unspliced_index], :]
            .sum()
            .to_frame()
            .T.reset_index(drop=True)
        )

        # Extract spliced counts for this gene if there are any.
        if spliced_index in cell_df.index.get_level_values("nucleus"):
            spliced.append(
                cell_df.xs(spliced_index, level="nucleus").reset_index(drop=True)
            )
        else:
            # Initialize empty zeros
            spliced.append(
                pd.DataFrame(np.zeros((1, cell_df.shape[1])), columns=cell_df.columns)
            )

    cell_nucleus.groupby("cell").apply(to_splice_layers)

    cells = cell_nucleus.index.get_level_values("cell").unique()

    unspliced = pd.concat(unspliced)
    unspliced.index = cells
    spliced = pd.concat(spliced)
    spliced.index = cells

    spliced = spliced.fillna(0)
    unspliced = unspliced.fillna(0)

    return spliced, unspliced


def to_scanpy(data):
    # Extract points
    expression = pd.DataFrame(
        data.X, index=pd.MultiIndex.from_frame(data.obs[["cell", "gene"]])
    )

    # Aggregate points to counts
    expression = (
        data.obs[["cell", "gene"]]
        .groupby(["cell", "gene"])
        .apply(lambda x: x.shape[0])
        .to_frame()
    )
    expression = expression.reset_index()

    # Remove extracellular points
    expression = expression.loc[expression["cell"] != "-1"]

    # Format as dense cell x gene counts matrix
    expression = expression.pivot(index="cell", columns="gene").fillna(0)
    expression.columns = expression.columns.droplevel(0)
    expression.columns = expression.columns.str.upper()

    # Create anndata object
    sc_data = anndata.AnnData(expression)

    return sc_data
