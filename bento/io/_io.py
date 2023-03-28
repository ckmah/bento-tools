import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import geometry, wkt
from shapely.geometry import Polygon
from tqdm.auto import tqdm
import anndata
import rasterio
import rasterio.features
import emoji
import cv2
from skimage.measure import regionprops
from scipy.ndimage import binary_erosion, label
import alphashape

from .._utils import sc_format


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
        else pd.Series(col)
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

    sc_format(adata)

    adata.uns["points"] = adata.uns["points"].drop("geometry", axis=1, errors="ignore")

    # Write to h5ad
    adata.write(filename)


def prepare(
    molecules,
    cell_seg,
    x="x",
    y="y",
    gene="gene",
    other_seg=dict(),
):
    """Prepare AnnData with molecule-level spatial data.

    Parameters
    ----------
    molecules : DataFrame
        Molecule coordinates and annotations.
    cell_seg : np.array or gpd.GeoDataFrame
        Cell segmentation masks represented as 2D numpy array where 1st and 2nd
        dimensions correspond to x and y respectively. Connected regions must
        have same value to be considered a valid shape. Data type must be one
        of rasterio.int16, rasterio.int32, rasterio.uint8, rasterio.uint16, or
        rasterio.float32. See rasterio.features.shapes for more details.
        
        Alternatively, if cell segmentations are already shapely objects, they can
        be appended to one single GeoDataFrame as and used as the input.
    x : str
        Column name for x coordinates, by default 'x'.
    y : str
        Column name for x coordinates, by default 'y'.
    gene : str
        Column name for gene name, by default 'gene'.
    other_seg
        Additional keyword arguments are interpreted as additional segmentation
        masks. The user specified parameter name is used to store these masks as
        {name}_shape in adata.obs.
    Returns
    -------
        AnnData object
    """
    for var in [x, y, gene]:
        if var not in molecules.columns:
            return

    pbar = tqdm(total=6)
    pbar.set_description(emoji.emojize(":test_tube: Loading inputs"))
    points = molecules[[x, y, gene]]
    points.columns = ["x", "y", "gene"]
    points = gpd.GeoDataFrame(
        points, geometry=gpd.points_from_xy(x=points.x, y=points.y)
    )
    points["gene"] = points["gene"].astype("category")  # Save memory
    pbar.update()

    # Load each set of masks as GeoDataFrame
    # shapes = Series where index = segs.keys() and values = GeoDataFrames
    segs_dict = {"cell": cell_seg, **other_seg}
    # Already formatted, select geometry column already
    if isinstance(cell_seg, gpd.GeoDataFrame):
        shapes_dict = {
            shape_name: shape_seg[["geometry"]]
            for shape_name, shape_seg in segs_dict.items()
        }
    # Load shapes from numpy array image
    elif isinstance(cell_seg, np.ndarray):
        shapes_dict = {
            shape_name: _load_shapes_np(shape_seg)
            for shape_name, shape_seg in segs_dict.items()
        }
    else:
        print("Segmentation mask format not recognized.")
        pbar.close()
        return
    pbar.update()

    # Index shapes to cell
    pbar.set_description(emoji.emojize(":open_book: Indexing"))
    obs_shapes = _index_shapes(shapes_dict, "cell")
    pbar.update()

    # Index points for all shapes
    # TODO: refactor to use geometry.sindex_points
    point_index = dict()
    for col in obs_shapes.columns:
        shp_gdf = gpd.GeoDataFrame(geometry=obs_shapes[col])
        shp_name = "_".join(str(col).split("_")[:-1])
        point_index[shp_name] = _index_points(points, shp_gdf)
    point_index = pd.DataFrame.from_dict(point_index)
    pbar.update()

    # Main long dataframe for reformatting
    pbar.set_description(emoji.emojize(":computer_disk: Formatting"))
    uns_points = pd.concat(
        [
            points[["x", "y", "gene"]].reset_index(drop=True),
            point_index.reset_index(drop=True),
        ],
        axis=1,
    )

    # Remove extracellular points
    uns_points = uns_points.loc[uns_points["cell"] != "-1"]
    if len(uns_points) == 0:
        print("No molecules found within cells. Data not processed.")
        pbar.close()
        return
    uns_points[["cell", "gene"]] = uns_points[["cell", "gene"]].astype("category")

    # Aggregate points to counts
    expression = (
        uns_points[["cell", "gene"]]
        .groupby(["cell", "gene"])
        .apply(lambda x: x.shape[0])
        .reset_index()
    )

    # Create cell x gene matrix
    cellxgene = expression.pivot_table(
        index="cell", columns="gene", aggfunc="sum"
    ).fillna(0)
    cellxgene.columns = cellxgene.columns.get_level_values("gene")
    pbar.update()

    # Create scanpy anndata object
    pbar.set_description(emoji.emojize(":package: Create AnnData"))
    adata = anndata.AnnData(X=cellxgene)
    obs_shapes = obs_shapes.reindex(index=adata.obs.index)
    adata.obs = pd.concat([adata.obs, obs_shapes], axis=1)
    adata.obs.index = adata.obs.index.astype(str)

    # Save cell, gene, batch, and other shapes as categorical type to save memory
    uns_points["cell"] = uns_points["cell"].astype("category")
    uns_points["gene"] = uns_points["gene"].astype("category")
    for shape_name in list(other_seg.keys()):
        uns_points[shape_name] = uns_points[shape_name].astype("category")

    adata.uns = {"points": uns_points}
    adata.obs['batch'] = [0]*len(adata.obs) #initialize batch column in case it's needed

    pbar.set_description(emoji.emojize(":bento_box: Finished!"))
    pbar.update()
    pbar.close()
    return adata
    

def _load_shapes_np(seg_img):
    """Extract shapes from segmentation image.

    Parameters
    ----------
    seg_img : np.array
        Segmentation masks represented as 2D numpy array where 1st and 2nd dimensions correspond to x and y respectively.

    Returns
    -------
    GeoDataFrame
        Single column GeoDataFrame where each row is a single Polygon.
    """
    seg_img = seg_img.astype("uint16")
    contours = rasterio.features.shapes(seg_img)  # rasterio to generate contours
    # Convert to shapely Polygons
    polygons = [Polygon(p["coordinates"][0]) for p, v in contours]
    shapes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))  # Cast to GeoDataFrame
    shapes.drop(
        shapes.area.sort_values().tail(1).index, inplace=True
    )  # Remove extraneous shape
    shapes = shapes[shapes.geom_type != "MultiPolygon"]

    shapes.index = shapes.index.astype(str)

    # Cleanup polygons
    # mask.geometry = mask.geometry.buffer(2).buffer(-2)
    # mask.geometry = mask.geometry.apply(unary_union)

    return shapes


def _load_shapes_json(seg_json):
    """Extract shapes from python object loaded with json.

    Parameters
    ----------
    seg_json : list
        list loaded by json.load(file)

    Returns
    -------
    GeoDataFrame
        Each row represents a single shape,
    """
    polys = []
    for i in range(len(seg_json)):
        polys.append(Polygon(seg_json[i]["coordinates"][0]))

    shapes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
    shapes = shapes[shapes.geom_type != "MultiPolygon"]

    shapes.index = shapes.index.astype(str)

    # Cleanup polygons
    # mask.geometry = mask.geometry.buffer(2).buffer(-2)
    # mask.geometry = mask.geometry.apply(unary_union)

    return shapes


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
    # TODO: #1. Parallelize across shapes. #2. crop points to each shape's min/max x,y vals to speed up indexing
    
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
    # Read point registry to identify point sets to concatenate
    # TODO

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

def _alphashape_poly_generate(molecules,x_label='x',y_label='y',alpha=0.05):
    points =  np.array([molecules[x_label],molecules[y_label]]).T
    poly = alphashape.alphashape(points, alpha).buffer(0)
    poly = poly.buffer(1).buffer(-1).buffer(0) # get rid of weird self-intersections


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

def read_xenium(
    data_dir,
    data_prefix
):
    """Prepare AnnData from Xenium data. Wrapper around prepare()

    Parameters
    ----------
    data_dir : String
        Directory containing Xenium generated files (parquet files used).
    data_prefix : String
        Prefix of all file names.
    Returns
    -------
        AnnData object
    """
    molecules = pd.read_parquet(data_dir + data_prefix + '_transcripts.parquet',
                                engine='fastparquet')
    def convert_to_shapely(vertex_df):
        return Polygon(zip(vertex_df['vertex_x'],vertex_df['vertex_y'])).buffer(0)
    all_cell_coords = pd.read_parquet(data_dir + data_prefix + '_cell_boundaries.parquet',
                                engine='fastparquet')
    all_cell_ids = list(np.unique(all_cell_coords['cell_id']))
    cell_polys = []
    print("Converting cell boundaries to polygons")
    # TODO: Parallelize this loop
    for i in tqdm(all_cell_ids):
        df = all_cell_coords[all_cell_coords['cell_id']==i]
        cell_polys.append(convert_to_shapely(df))
    # convert to GeoDataFrame
    cell_gdf = gpd.GeoDataFrame(geometry=cell_polys)
    
    # Do same thing for nuclei
    all_nuc_coords = pd.read_parquet(data_dir + data_prefix + '_nucleus_boundaries.parquet',
                                engine='fastparquet')
    all_nuc_ids = list(np.unique(all_nuc_coords['cell_id']))
    nuc_polys = []
    print("Converting nuclear boundaries to polygons")
    # TODO: Parallelize this loop
    for i in tqdm(all_nuc_ids):
        df = all_nuc_coords[all_nuc_coords['cell_id']==i]
        nuc_polys.append(convert_to_shapely(df))
    # convert to GeoDataFrame
    nuc_gdf = gpd.GeoDataFrame(geometry=nuc_polys)
    adata =  prepare(molecules,
                   cell_seg=cell_gdf,
                   x='x_location',
                   y='y_location',
                   gene='feature_name',
                   other_seg=dict(nucleus=nuc_gdf))
    return adata

def read_cosmx_smi(
    data_dir,
    data_prefix,
    nucleus_label=1,
    save=None
):
    """Prepare AnnData from Xenium data. Wrapper around prepare()

    Parameters
    ----------
    data_dir : String
        Directory containing Xenium generated files (parquet files used).
    data_prefix : String
        Prefix of all file names.
    nucleus_label: int
        label value for nuclei in CompartmentLabels image to use. Default = 1
    save: String
        Optional. path and filename to save anndata once written
    Returns
    -------
        AnnData object
    """
    fov_positions = pd.read_csv(data_dir + data_prefix + '_fov_positions_file.csv',
                                index_col='fov')
                                
    molecules = pd.read_csv(data_dir + data_prefix + '_tx_file.csv')
    
    num_fovs = len(fov_positions)

    # TODO: parallelize this
    print("Converting cells to polygons for each FOV")
    all_cell_polys = []
    for fov in tqdm(range(1,num_fovs+1)):
        fov_str = str(fov)
        fov_str = '0'*(3-len(fov_str)) + fov_str
        cell_file = data_dir + 'CellLabels/CellLabels_F' + fov_str + '.tif'
        cell_labels = cv2.imread(cell_file,-1)
        cell_props = regionprops(cell_labels)
        cell_polys = []
        x_adjust = fov_positions.loc[fov]['x_global_px']
        y_adjust = fov_positions.loc[fov]['y_global_px']
        for prop in cell_props:
            if prop.coords.shape[0] > 3 and prop.area > 50:
                cell_poly = Polygon(prop.coords + np.array([x_adjust,y_adjust])).buffer(1).buffer(-1).buffer(0)
                cell_polys.append(cell_poly)
            else:
                pass
        all_cell_polys += cell_polys
    cell_gdf = gpd.GeoDataFrame(geometry=all_cell_polys)

    print("Converting nuclei to polygons for each FOV")
    all_nuc_polys = []
    for fov in tqdm(range(1,num_fovs+1)):
        fov_str = str(fov)
        fov_str = '0'*(3-len(fov_str)) + fov_str
        overlay_file = data_dir + 'CellOverlay/CellOverlay_F' + fov_str + '.jpg'
        overlay = cv2.imread(overlay_file,-1)
        color1 = np.asarray([0,0,0])
        color2 = np.asarray([200,200,200])
        mask = cv2.inRange(overlay,color1,color2)
        compartment_labeled_file = data_dir + 'CompartmentLabels/CompartmentLabels_F' + fov_str + '.tif'
        compartment_labeled = cv2.imread(compartment_labeled_file,-1)
        nuc_comp = np.where(compartment_labeled==nucleus_label,1,0)
        nuc_divided = cv2.bitwise_and(nuc_comp,nuc_comp,mask=mask)
        nuc_divided = binary_erosion(nuc_divided,structure=np.ones((5,5))).astype(int)
        nuc_labeled, num_nucs = label(nuc_divided)
        nuc_props = regionprops(nuc_labeled)
        nuc_polys = []
        x_adjust = fov_positions.loc[fov]['x_global_px']
        y_adjust = fov_positions.loc[fov]['y_global_px']
        for prop in nuc_props:
            if prop.coords.shape[0] > 3 and prop.area > 50:
                nuc_poly = Polygon(prop.coords + np.array([x_adjust,y_adjust])).buffer(1).buffer(-1).buffer(0)
                nuc_polys.append(nuc_poly)
            else:
                pass
        all_nuc_polys += nuc_polys
    nuc_gdf = gpd.GeoDataFrame(geometry=all_nuc_polys)
    
    adata = prepare(molecules,
                    cell_seg=cell_gdf,
                    x='x_global_px',
                    y='y_global_px',
                    gene='target',
                    other_seg=dict(nucleus=nuc_gdf))
    if save is not None:
        write_h5ad(adata,save)
    else:
        pass
        
    return adata
    
def read_clustermap(
    clustermap_path,
    nuclear_path,
    save_name=None,
):
    """Prepare AnnData from Xenium data. Wrapper around prepare()

    Parameters
    ----------
    clustermap_path : String
        Path to clustermap results.
    nuclear_path : String
        path to nuclear segmentations as a labelled 2D numpy array where 1st and 2nd dimensions
        correspond to x and y, and pixel value are labels for each unique nuclei.
    save_name : String
        Optional; path to save anndata object
    Returns
    -------
        AnnData object
    """
    clustermap_results = pd.read_csv(clustermap_path)
    nuc_segs = cv2.imread(nuclear_path,-1)
    props = regionprops(nuc_segs)
    nuc_polys = []
    print("Converting nuclei to polygons")
    for prop in tqdm(props):
        nuc_poly = Polygon(prop.coords).buffer(0)
        nuc_polys.append(nuc_poly)
    nuc_gdf = gpd.GeoDataFrame(geometry=nuc_polys)
    cell_polys = []
    print("Converting clustermap results to alphashape polygons")
    cell_idxs = list(np.unique(clustermap_results['clustermap']))
    # TODO: parallelize this
    for cell in tqdm(cell_idxs[1:]): # ignore -1 label as that means unclustered
        df = clustermap_results[clustermap_results['clustermap']==cell]
        cell_poly = _alphashape_poly_generate(df,
                                              x_label='spot_location_1',
                                              y_label='spot_location_2')
        cell_polys.append(cell_poly)
    cell_gdf = gpd.GeoDataFrame(geometry=cell_polys)
    adata = prepare(clustermap_results,
                    cell_seg=cell_gdf,
                    x='spot_location_1',
                    y='spot_location_2',
                    gene='gene_name',
                    other_seg=dict(nucleus=nuc_gdf))
    if save_name is not None:
        write_h5ad(adata,save_name)
    else:
        pass
        
    return adata