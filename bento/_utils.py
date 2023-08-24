from spatialdata._core.spatialdata import SpatialData
from dask.dataframe import from_pandas

def sync_points(sdata: SpatialData) -> SpatialData:
    """
    Sync existing point sets and associated metadata with sdata.table.obs_names and sdata.table.var_names

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    """

    # Iterate over point sets
    for point_key in sdata.points:
        points = sdata.points[point_key].compute()

        # Subset for cells
        cells = sdata.table.obs_names.tolist()
        in_cells = points["cell"].isin(cells)

        # Subset for genes
        in_genes = [True] * points.shape[0]
        if "gene" in points.columns:
            genes = sdata.table.var_names.tolist()
            in_genes = points["gene"].isin(genes)
        
        # Combine boolean masks
        valid_mask = (in_cells & in_genes).values

        # Sync points using mask
        points = points.loc[valid_mask]

        # Remove unused categories for categorical columns
        for col in points.columns:
            if points[col].dtype == "category":
                points[col].cat.remove_unused_categories(inplace=True)

        sdata.points[point_key] = from_pandas(points.reset_index(drop=True), npartitions=sdata.points[point_key].npartitions)

        return sdata
