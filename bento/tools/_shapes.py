import geopandas as gpd

from .._utils import track


@track
def outer_edge(data, shape_name, distance=30, copy=False):

    adata = data.copy() if copy else data

    if distance <= 0:
        print("Distance must be positive.")
        return

    # Get region that is distance outside of specified shape but within the cell
    cell_shapes = gpd.GeoSeries(adata.obs["cell_shape"])
    query_shapes = gpd.GeoSeries(adata.obs[shape_name])
    outer_shapes = cell_shapes & (query_shapes.buffer(distance) - query_shapes)

    if outer_shapes[0].type == "MultiLineString":
        print("Invalid region, no area within cell.")
        return

    name_prefix = shape_name.split("_shape")[0]
    adata.obs[f"{name_prefix}_outer_edge_shape"] = outer_shapes

    return adata if copy else None


@track
def inner_edge(data, shape_name, distance=30, copy=False):

    adata = data.copy() if copy else data

    if distance <= 0:
        print("Distance must be positive.")
        return

    # Get region that is distance inside of specified shape but within the cell
    cell_shapes = gpd.GeoSeries(adata.obs["cell_shape"])
    query_shapes = gpd.GeoSeries(adata.obs[shape_name])
    inner_shapes = cell_shapes & (query_shapes - query_shapes.buffer(-distance))

    if inner_shapes[0].type == "MultiLineString":
        print("Invalid region, no area within cell.")
        return

    name_prefix = shape_name.split("_shape")[0]
    adata.obs[f"{name_prefix}_inner_edge_shape"] = inner_shapes

    return adata if copy else None
