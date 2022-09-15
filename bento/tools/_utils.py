import geopandas as gpd
from shapely import wkt


def get_shape(adata, shape_name):
    """Get a GeoSeries of Polygon objects from an AnnData object."""
    if shape_name not in adata.obs.columns:
        raise ValueError(f"Shape {shape_name} not found in adata.obs.")

    if adata.obs[shape_name].astype(str).str.startswith("POLYGON").any():
        return gpd.GeoSeries(
            adata.obs[shape_name]
            .astype(str)
            .apply(lambda val: wkt.loads(val) if val != "None" else None)
        )

    else:
        return gpd.GeoSeries(adata.obs[shape_name])
