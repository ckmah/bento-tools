from abc import ABCMeta, abstractmethod

from astropy.stats.spatial import RipleysKEstimator
from scipy.stats.stats import spearmanr
from scipy.spatial import distance

import dask_geopandas
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm.rich import tqdm
from tqdm.dask import TqdmCallback

from .. import tools as tl
from ..preprocessing import get_points


def analyze_points(
    data,
    shape_names,
    feature_names,
    groupby=None,
    progress=True,
    chunksize=100,
    copy=False,
):
    """Calculate the set of specified `features` for every sample, defined as the set of
    molecules corresponding to every cell-gene pair.

    Parameters
    ----------
    data : AnnData
        Spatially formatted AnnData
    shape_names : str or list of str
        Names of the shapes to analyze.
    feature_names : str or list of str
        Names of the features to analyze.
    groupby : str or list of str, optional (default: None)
        Key in `data.uns['points'] to groupby, by default None. Always treats each cell separately
    chunksize : int, optional
        Number of cells to process in each chunk, passed to `dask`, by default 100.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:
        `.layers[`keys`]` if `groupby` == "gene"
            See the output of each :class:`SampleFeature` in `features` for keys added.
        `.obsm[`point_features`]` if `groupby` != "gene"
            DataFrame with rows aligned to `adata.obs_names` and `features` as columns.

    """
    adata = data.copy() if copy else data

    
    # Cast to list if not already
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    # Cast to list if not already
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Make sure groupby is a list
    if isinstance(groupby, str):
        groupby = ["cell", groupby]
    else:
        groupby = ["cell"]

    # Make sure all groupby keys are in point columns
    for g in groupby:
        if g not in get_points(adata).columns:
            raise ValueError(f"Groupby key {g} not found in point columns.")
    
    pbar = tqdm(desc="Cell features", total=3)

    # Generate feature x shape combinations
    feature_combos = [
        point_features[f](s) for f in feature_names for s in shape_names
    ]

    # Compile dependency set of features and attributes
    cell_features = set()
    obs_attrs = set()
    for f in feature_combos:
        cell_features.update(f.cell_features)
        obs_attrs.update(f.attributes)

    cell_features = list(cell_features)
    obs_attrs = list(obs_attrs)

    tl.analyze_shapes(adata, "cell_shape", cell_features, progress=False)

    # Make sure attributes are present
    attrs_found = set(obs_attrs).intersection(set(adata.obs.columns.tolist()))
    if len(attrs_found) != len(obs_attrs):
        raise KeyError(f"df does not have all columns: {obs_attrs}.")

    pbar.update()

    pbar.set_description("Sample features")
    # extract cell attributes
    points_df = (
        get_points(adata, asgeo=True)
        .set_index("cell")
        .join(data.obs[obs_attrs])
        .reset_index()
    )

    # Handle categories as strings to avoid ambiguous cat types
    for col in points_df.loc[:, (points_df.dtypes == "category").values]:
        points_df[col] = points_df[col].astype(str)

    # Handle shape indexes as strings to avoid ambiguous types
    for shape_name in adata.obs.columns[adata.obs.columns.str.endswith("_shape")]:
        shape_prefix = "_".join(shape_name.split("_")[:-1])
        if shape_prefix in points_df.columns:
            points_df[shape_prefix] = points_df[shape_prefix].astype(str)

    # Calculate features for a sample
    def process_sample(df):
        sample_output = {}
        for f in feature_combos:
            sample_output.update(f.extract(df))
        return sample_output

    # Process all samples in a partition
    def process_partition(partition_df):
        # Groupby by cell and groupby keys and process each sample
        return partition_df.groupby(groupby, observed=True).apply(process_sample)

    # Cast to dask dataframe
    ddf = dask_geopandas.from_geopandas(points_df, npartitions=1)

    # Create chunks with chunksize cells
    groups, group_loc = np.unique(
        points_df["cell"],
        return_index=True,
    )
    if len(groups) > chunksize:
        divisions = [group_loc[loc] for loc in range(0, len(group_loc), chunksize)]
        divisions.append(len(points_df) - 1)
        ddf = ddf.repartition(divisions=divisions)

    # Run on a single sample to get output metadata
    meta_output = process_partition(points_df.head())
    meta = pd.DataFrame(meta_output.tolist(), index=meta_output.index)

    # Parallel process each partition
    if progress:
        with TqdmCallback():
            output = ddf.map_partitions(process_partition, meta=meta.dtypes).compute()
    else:
        output = ddf.map_partitions(process_partition, meta=meta.dtypes).compute()
        
    pbar.update()
    pbar.set_description("Saving to AnnData")

    # Format from Series of dicts to DataFrame
    output = pd.DataFrame(output.tolist(), index=output.index).reset_index()

    # Save results to data layers
    feature_names = output.columns[~output.columns.isin(groupby)]

    if groupby == ["cell", "gene"]:
        for feature_name in feature_names:
            adata.layers[feature_name] = (
                output.pivot(index="cell", columns="gene", values=feature_name)
                .reindex(index=adata.obs_names, columns=adata.var_names)
                .astype(float)
            )
    else:
        adata.obsm["point_features"] = output.set_index("cell").reindex(index=adata.obs_names).astype(float)

    pbar.update()
    pbar.set_description("Done!")
    pbar.close()

    return adata if copy else None


class PointFeature(metaclass=ABCMeta):
    """Abstract class for calculating sample features. A sample is defined as the set of
    molecules corresponding to a single cell-gene pair.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features.
    attributes : int
        Names (keys) used to store computed cell-level features.
    """

    def __init__(self, shape_name):
        self.cell_features = set()
        self.attributes = set()

        if shape_name:
            self.attributes.add(shape_name)
            self.shape_name = shape_name
            self.shape_prefix = "_".join(shape_name.split("_")[:-1])

    @abstractmethod
    def extract(self, df):
        """Calculates this feature for a given sample.

        Parameters
        ----------
        df : DataFrame
            Assumes each row is a molecule and that columns `x`, `y`, `cell`, and `gene` are present.
        """
        return df


class ShapeDistance(PointFeature):
    """For a set of points, computes the distance of points within `shape_name`
    as well as the distance of points outside `shape_name`. 

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"{shape_prefix}_inner_distance"`: distance of points inside `shape_name`
        `"{shape_prefix}_outer_distance"`: distance of points outside `shape_name`
    """

    # Cell-level features needed for computing sample-level features
    def __init__(self, shape_name):
        super().__init__(shape_name)

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get points
        points_geo = df["geometry"].values

        # Check for points within shape, assume all are intracellular
        if self.shape_prefix == "cell":
            inner = np.array([True] * len(df))
        else:
            inner = df[self.shape_prefix] != "-1"
        outer = ~inner

        if inner.sum() > 0:
            inner_dist = points_geo[inner].distance(shape.boundary).mean()
        else:
            inner_dist = np.nan

        if outer.sum() > 0:
            outer_dist = points_geo[outer].distance(shape.boundary).mean()
        else:
            outer_dist = np.nan

        return {
            f"{self.shape_prefix}_inner_distance": inner_dist,
            f"{self.shape_prefix}_outer_distance": outer_dist,
        }


class ShapeAsymmetry(PointFeature):
    """For a set of points, computes the asymmetry of points within `shape_name`
    as well as the asymmetry of points outside `shape_name`. Asymmetry is defined as
    the offset between the centroid of points to the centroid of the specified
    `shape_name`.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features
    shape_name : str
        Name of shape to use, must be column name in input DataFrame

    Returns
    -------
    dict
        `"{shape_prefix}_inner_asymmetry"`: asymmetry of points inside `shape_name`
        `"{shape_prefix}_outer_asymmetry"`: asymmetry of points outside `shape_name`
    """
    def __init__(self, shape_name):
        super().__init__(shape_name)


    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get points
        points_geo = df["geometry"].values

        # Check for points within shape, assume all are intracellular
        if self.shape_prefix == "cell":
            inner = np.array([True] * len(df))
        else:
            inner = df[self.shape_prefix] != "-1"
        outer = ~inner

        if inner.sum() > 0:
            inner_to_centroid = points_geo[inner].distance(shape.centroid).mean()
        else:
            inner_to_centroid = np.nan

        if outer.sum() > 0:
            outer_to_centroid = points_geo[outer].distance(shape.centroid).mean()
        else:
            outer_to_centroid = np.nan

        return {
            f"{self.shape_prefix}_inner_asymmetry": inner_to_centroid,
            f"{self.shape_prefix}_outer_asymmetry": outer_to_centroid,
        }


class PointDispersion(PointFeature):
    """For a set of points, calculates the second moment of all points in a cell
    relative to the centroid of the total RNA signal.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"point_dispersion"`: measure of point dispersion
    """

    # shape_name set to None to follow the same convention as other shape features
    def __init__(self, shape_name=None):
        super().__init__(shape_name)

    def extract(self, df):
        df = super().extract(df)

        # Get precomputed cell centroid and raster
        pt_centroid = df[["x", "y"]].values.mean(axis=0).reshape(1, 2)

        # calculate points moment
        point_moment = _second_moment(pt_centroid, df[["x", "y"]].values)

        return {"point_dispersion": point_moment}


class ShapeDispersion(PointFeature):
    """For a set of points, calculates the second moment of all points in a cell relative to the
    centroid of `shape_name`.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"{shape_prefix}_dispersion"`: measure of point dispersion relative to `shape_name`
    """

    def __init__(self, shape_name):
        super().__init__(shape_name)

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # calculate points moment
        point_moment = _second_moment(shape.centroid, df[["x", "y"]].values)

        return {f"{self.shape_prefix}_dispersion": point_moment}


class RipleyStats(PointFeature):
    """For a set of points, calculates properties of the L-function. The L-function
    measures spatial clustering of a point pattern over the area of the cell.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"l_max": The max value of the L-function evaluated at r=[1,d], where d is half the cell’s maximum diameter.
        `"l_max_gradient"`: The max value of the gradient of the above L-function.
        `"l_min_gradient"`: The min value of the gradient of the above L-function.
        `"l_monotony"`: The correlation of the L-function and r=[1,d].
        `"l_half_radius"`: The value of the L-function evaluated at 1/4 of the maximum cell diameter.

    """

    def __init__(self, shape_name=None):
        super().__init__(shape_name)
        self.cell_features.update(["span", "bounds", "area"])

        self.attributes.update(
            [
                "cell_span",
                "cell_minx",
                "cell_miny",
                "cell_maxx",
                "cell_maxy",
                "cell_area",
            ]
        )

    def extract(self, df):
        df = super().extract(df)

        # Get precomputed centroid and cell moment
        cell_span = df["cell_span"].values[0]
        cell_minx = df["cell_minx"].values[0]
        cell_miny = df["cell_miny"].values[0]
        cell_maxx = df["cell_maxx"].values[0]
        cell_maxy = df["cell_maxy"].values[0]
        cell_area = df["cell_area"].values[0]

        estimator = RipleysKEstimator(
            area=cell_area,
            x_min=cell_minx,
            y_min=cell_miny,
            x_max=cell_maxx,
            y_max=cell_maxy,
        )

        quarter_span = cell_span / 4
        radii = np.linspace(1, quarter_span * 2, num=int(quarter_span * 2))

        # Get points
        points_geo = df["geometry"].values
        points_geo = np.array([points_geo.x, points_geo.y]).T

        # Compute ripley function stats
        stats = estimator.Hfunction(data=points_geo, radii=radii, mode="none")

        # Max value of the L-function
        l_max = max(stats)

        # Max and min value of the gradient of L
        ripley_smooth = pd.Series(stats).rolling(5).mean()
        ripley_smooth.dropna(inplace=True)

        # Can't take gradient of single number
        if len(ripley_smooth) < 2:
            ripley_smooth = np.array([0, 0])

        ripley_gradient = np.gradient(ripley_smooth)
        l_max_gradient = ripley_gradient.max()
        l_min_gradient = ripley_gradient.min()

        # Monotony of L-function in the interval
        l_monotony = spearmanr(radii, stats)[0]

        # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
        l_half_radius = estimator.Hfunction(
            data=points_geo, radii=[quarter_span], mode="none"
        )[0]

        result = {
            "l_max": l_max,
            "l_max_gradient": l_max_gradient,
            "l_min_gradient": l_min_gradient,
            "l_monotony": l_monotony,
            "l_half_radius": l_half_radius,
        }

        return result


class ShapeEnrichment(PointFeature):
    """For a set of points, calculates the fraction of points within `shape_name`
    out of all points in the cell.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features
    shape_name : str
        Name of shape to use, must be column name in input DataFrame

    Returns
    -------
    dict
        `"{shape_prefix}_enrichment"`: enrichment fraction of points in `shape_name`
    """

    def __init__(self, shape_name):
        super().__init__(shape_name)

    def extract(self, df):
        df = super().extract(df)

        # Get points outside shape
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if self.shape_prefix == "cell":
            enrichment = 1.0
        else:
            inner_count = (df[self.shape_prefix] != "-1").sum()
            enrichment = inner_count / float(len(points_geo))

        return {f"{self.shape_prefix}_enrichment": enrichment}


def _second_moment(centroid, pts):
    """
    Calculate second moment of points with centroid as reference.

    Parameters
    ----------
    centroid : [1 x 2] float
    pts : [n x 2] float
    """
    centroid = np.array(centroid).reshape(1, 2)
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


point_features = dict(
    distance=ShapeDistance,
    asymmetry=ShapeAsymmetry,
    point_dispersion=PointDispersion,
    shape_dispersion=ShapeDispersion,
    ripley=RipleyStats,
    shape_enrichment=ShapeEnrichment,
    )


def register_point_feature(name, func):
    """Register a new point feature function.

    Parameters
    ----------
    name : str
        Name of feature function
    func : class
        Function that takes a DataFrame of points and return a dictionary of features.
    """

    new_feature = PointFeature()

    point_features[name] = func