import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import geometry, wkt
from shapely.geometry import Polygon
from tqdm.auto import tqdm
import anndata
# import rasterio
# import rasterio.features
import emoji


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

    adata = anndata.read_h5ad(filename, backed=backed)

    # Load obs columns that are shapely geometries
    adata.obs = adata.obs.apply(
        lambda col: gpd.GeoSeries(
            col.astype(str).apply(lambda val: wkt.loads(val) if val != "None" else None)
        )
        if col.astype(str).str.startswith("POLYGON").any()
        else gpd.GeoSeries(col)
    )

    adata.obs.index.name = "cell"
    adata.var.index.name = "gene"

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

    adata.uns["points"] = adata.uns["points"].drop("geometry", axis=1, errors="ignore")

    # Write to h5ad
    adata.write(filename)


# def prepare(
#     molecules,
#     cell_seg,
#     x="x",
#     y="y",
#     gene="gene",
#     other_seg=dict(),
# ):
#     """Prepare AnnData with molecule-level spatial data.

#     Parameters
#     ----------
#     molecules : DataFrame
#         Molecule coordinates and annotations.
#     cell_seg : np.array
#         Cell segmentation masks represented as 2D numpy array where 1st and 2nd
#         dimensions correspond to x and y respectively. Connected regions must
#         have same value to be considered a valid shape. Data type must be one
#         of rasterio.int16, rasterio.int32, rasterio.uint8, rasterio.uint16, or
#         rasterio.float32. See rasterio.features.shapes for more details.
#     x : str
#         Column name for x coordinates, by default 'x'.
#     y : str
#         Column name for x coordinates, by default 'y'.
#     gene : str
#         Column name for gene name, by default 'gene'.
#     other_seg
#         Additional keyword arguments are interpreted as additional segmentation
#         masks. The user specified parameter name is used to store these masks as
#         {name}_shape in adata.obs.
#     Returns
#     -------
#         AnnData object
#     """
#     for var in [x, y, gene]:
#         if var not in molecules.columns:
#             return

#     pbar = tqdm(total=6)
#     pbar.set_description(emoji.emojize(":test_tube: Loading inputs"))
#     points = molecules[[x, y, gene]]
#     points.columns = ["x", "y", "gene"]
#     points = gpd.GeoDataFrame(
#         points, geometry=gpd.points_from_xy(x=points.x, y=points.y)
#     )
#     points["gene"] = points["gene"].astype("category")  # Save memory
#     pbar.update()

#     # Load each set of masks as GeoDataFrame
#     # shapes = Series where index = segs.keys() and values = GeoDataFrames
#     segs_dict = {"cell": cell_seg, **other_seg}
#     # Already formatted, select geometry column already
#     if isinstance(cell_seg, gpd.GeoDataFrame):
#         shapes_dict = {
#             shape_name: shape_seg[["geometry"]]
#             for shape_name, shape_seg in segs_dict.items()
#         }
#     # Load shapes from numpy array image
#     elif isinstance(cell_seg, np.array):
#         shapes_dict = {
#             shape_name: _load_shapes_np(shape_seg)
#             for shape_name, shape_seg in segs_dict.items()
#         }
#     else:
#         print("Segmentation mask format not recognized.")
#         pbar.close()
#         return
#     pbar.update()

#     # Index shapes to cell
#     pbar.set_description(emoji.emojize(":open_book: Indexing"))
#     obs_shapes = _index_shapes(shapes_dict, "cell")
#     pbar.update()

#     # Index points for all shapes
#     point_index = dict()
#     for col in obs_shapes.columns:
#         shp_gdf = gpd.GeoDataFrame(geometry=obs_shapes[col])
#         shp_name = '_'.join(str(col).split('_')[:-1])
#         point_index[shp_name] = _index_points(points, shp_gdf)
#     point_index = pd.DataFrame.from_dict(point_index)
#     pbar.update()

#     # Main long dataframe for reformatting
#     pbar.set_description(emoji.emojize(":computer_disk: Formatting"))
#     uns_points = pd.concat(
#         [
#             points[["x", "y", "gene"]].reset_index(drop=True),
#             point_index.reset_index(drop=True),
#         ],
#         axis=1,
#     )

#     # Remove extracellular points
#     uns_points = uns_points.loc[uns_points["cell"] != "-1"]
#     if len(uns_points) == 0:
#         print("No molecules found within cells. Data not processed.")
#         pbar.close()
#         return
#     uns_points[["cell", "gene"]] = uns_points[["cell", "gene"]].astype('category')

#     # Aggregate points to counts
#     expression = (
#         uns_points[["cell", "gene"]]
#         .groupby(["cell", "gene"])
#         .apply(lambda x: x.shape[0])
#         .reset_index()
#     )

#     # Create cell x gene matrix
#     cellxgene = expression.pivot_table(
#         index="cell", columns="gene", aggfunc="sum"
#     ).fillna(0)
#     cellxgene.columns = cellxgene.columns.get_level_values("gene")
#     pbar.update()

#     # Create scanpy anndata object
#     pbar.set_description(emoji.emojize(":package: Create AnnData"))
#     adata = anndata.AnnData(X=cellxgene)
#     obs_shapes = obs_shapes.reindex(index=adata.obs.index)
#     adata.obs = pd.concat([adata.obs, obs_shapes], axis=1)
#     adata.obs.index = adata.obs.index.astype(str)

#     # Save cell, gene, batch, and other shapes as categorical type to save memory
#     uns_points["cell"] = uns_points["cell"].astype("category")
#     uns_points["gene"] = uns_points["gene"].astype("category")
#     for shape_name in list(other_seg.keys()):
#         uns_points[shape_name] = uns_points[shape_name].astype('category')

#     adata.uns = {"points": uns_points}

#     pbar.set_description(emoji.emojize(":bento_box: Finished!"))
#     pbar.update()
#     pbar.close()
#     return adata


# def _load_shapes_np(seg_img):
#     """Extract shapes from segmentation image.

#     Parameters
#     ----------
#     seg_img : np.array
#         Segmentation masks represented as 2D numpy array where 1st and 2nd dimensions correspond to x and y respectively.

#     Returns
#     -------
#     GeoDataFrame
#         Single column GeoDataFrame where each row is a single Polygon.
#     """
#     seg_img = seg_img.astype("uint16")
#     contours = rasterio.features.shapes(seg_img)  # rasterio to generate contours
#     # Convert to shapely Polygons
#     polygons = [Polygon(p["coordinates"][0]) for p, v in contours]
#     shapes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))  # Cast to GeoDataFrame
#     shapes.drop(
#         shapes.area.sort_values().tail(1).index, inplace=True
#     )  # Remove extraneous shape
#     shapes = shapes[shapes.geom_type != "MultiPolygon"]

#     shapes.index = shapes.index.astype(str)

#     # Cleanup polygons
#     # mask.geometry = mask.geometry.buffer(2).buffer(-2)
#     # mask.geometry = mask.geometry.apply(unary_union)

#     return shapes


# def _load_shapes_json(seg_json):
#     """Extract shapes from python object loaded with json.

#     Parameters
#     ----------
#     seg_json : list
#         list loaded by json.load(file)

#     Returns
#     -------
#     GeoDataFrame
#         Each row represents a single shape,
#     """
#     polys = []
#     for i in range(len(seg_json)):
#         polys.append(Polygon(seg_json[i]["coordinates"][0]))

#     shapes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
#     shapes = shapes[shapes.geom_type != "MultiPolygon"]

#     shapes.index = shapes.index.astype(str)

#     # Cleanup polygons
#     # mask.geometry = mask.geometry.buffer(2).buffer(-2)
#     # mask.geometry = mask.geometry.apply(unary_union)

#     return shapes


def _index_shapes(shapes, cell_key):
    """Spatially index other masks to cell mask.

    Parameters
    ----------
    shapes : dict
        Dictionary of GeoDataFrames.

    Returns
    -------
    indexed_shapes : GeoDataFrame
        Each column is
    """
    cell_shapes = shapes[cell_key]

    indexed_shapes = cell_shapes.copy()
    for shape_name, shape in shapes.items():

        # Don't index cell to itself
        if shape_name == "cell":
            continue

        # For each cell, get all overlapping shapes
        geometry = gpd.sjoin(
            shape, cell_shapes, how="left", op="intersects", rsuffix="cell"
        ).dropna()

        # Calculate fraction overlap for each pair SLOW
        geometry["fraction_overlap"] = (
            geometry.intersection(
                cell_shapes.loc[geometry["index_cell"]], align=False
            ).area
            / geometry.area
        )

        # Keep shape that overlaps with cell_shapes the most
        geometry = (
            geometry.sort_values("fraction_overlap", ascending=False)
            .drop_duplicates("index_cell")
            .set_index("index_cell")
            .reindex(cell_shapes.index)["geometry"]
        )
        geometry.name = f"{shape_name}_shape"

        # Add indexed shapes as new column in GeoDataFrame
        indexed_shapes[f"{shape_name}_shape"] = geometry


    # Cells are rows, intersecting shape sets are columns
    indexed_shapes = indexed_shapes.rename(columns={"geometry": "cell_shape"})
    indexed_shapes.index = indexed_shapes.index.astype(str)
    return indexed_shapes


def _index_points(points, shapes):
    """Index points to each set of shapes item and save. Assumes non-overlapping shapes.

    Parameters
    ----------
    points : GeoDataFrame
        Point coordinates.
    shapes : GeoDataFrame
        Single column of Polygons.
    Returns
    -------
    Series
        Return list of mask indices corresponding to each point.
    """
    index = gpd.sjoin(points.reset_index(), shapes, how="left", op="intersects")

    # remove multiple cells assigned to same point
    index = (
        index.drop_duplicates(subset="index", keep="first")
        .sort_index()
        .reset_index()["index_right"]
        .fillna(-1)
        .astype(str)
    )

    return pd.Series(index)


def concatenate(adatas):
    uns_points = []
    for i, adata in enumerate(adatas):
        points = adata.uns["points"].copy()

        if "batch" not in points.columns:
            points["batch"] = i

        points["cell"] = points["cell"].astype(str) + "-" + points["batch"].astype(str)

        uns_points.append(points)

    new_adata = adatas[0].concatenate(adatas[1:])

    uns_points = pd.concat(uns_points)
    uns_points["cell"] = uns_points["cell"].astype("category")
    uns_points["gene"] = uns_points["gene"].astype("category")
    uns_points["batch"] = uns_points["batch"].astype("category")

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
