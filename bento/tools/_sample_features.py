from abc import ABCMeta, abstractmethod

from astropy.stats.spatial import RipleysKEstimator
from scipy.stats.stats import spearmanr
from scipy.spatial import distance

import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

from .. import tools as tl
from .._utils import track
from ..preprocessing import get_points


def analyze_samples(data, features, chunks=None, chunksize=None, copy=False):
    """Calculate the set of specified `features` for every sample, defined as the set of
    molecules corresponding to every cell-gene pair.

    Parameters
    ----------
    data : AnnData
        Spatially formatted AnnData
    features : list of :class:`SampleFeature`
        List of :class:`SampleFeature` to compute.
    chunks : int, optional
        Number of partitions to use, passed to `dask`, by default None.
    chunksize : int, optional
        Size of partitions, passed to `dask`, by default None.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:
        `.layers[`keys`]`
            See the output of each :class:`SampleFeature` in `features` for keys added.
    """
    adata = data.copy() if copy else data

    # Cast features to type list
    if not isinstance(features, list):
        features = [features]

    features = [sample_features[f] for f in features]
        
    cell_features = set()  # Cell-level fns to run
    cell_attributes = set()  # Cell-level attributes needed to compute sample features
    for f in features:
        cell_features.update(f.cell_features)
        cell_attributes.update(f.cell_attributes)

    cell_features = list(cell_features)
    cell_attributes = list(cell_attributes)

    # compute cell features
    for cf in cell_features:
        cf.__wrapped__(adata)

    # Make sure attributes are present
    attrs_found = set(cell_attributes).intersection(set(adata.obs.columns.tolist()))
    if len(attrs_found) != len(cell_attributes):
        raise KeyError(f"df does not have all columns: {cell_attributes}.")

    # extract cell attributes
    points_df = (
        get_points(adata, asgeo=True)
        .set_index("cell")
        .join(data.obs[cell_attributes])
        .reset_index()
        .set_index("cell")
        .sort_index()
    )

    # Calculate features for a sample
    def process_sample(df, features):
        sample_output = {}
        for f in features:
            sample_output.update(f.extract(df))

        return sample_output

    # Process all samples in a partition
    def process_partition(partition_df, features):
        # TODO update dask progress bar somehow?
        result = partition_df.groupby(["cell", "gene"], observed=True).apply(
            lambda sample_df: process_sample(sample_df, features)
        )

        return result

    # Run on a single sample to get output metadata
    meta_output = process_partition(
        points_df.reset_index().set_index(["cell", "gene"]).head(1), features
    )
    meta = pd.DataFrame(meta_output.tolist(), index=meta_output.index)

    # Cast to dask dataframe
    if not chunks and not chunksize:
        chunks = 1
    ddf = dask_geopandas.from_geopandas(
        points_df, npartitions=chunks, chunksize=chunksize
    )

    # Parallel process each partition
    with ProgressBar():
        task = ddf.map_partitions(
            lambda partition: process_partition(partition, features), meta=meta.dtypes
        )
        output = task.compute()

    # Format from Series of dicts to DataFrame
    output_index = output.index
    output = pd.DataFrame(output.tolist(), index=output_index).reset_index()

    # Save results to data layers
    feature_names = output.columns[~output.columns.isin(["cell", "gene"])]
    for feature_name in feature_names:
        adata.layers[feature_name] = (
            output.pivot(index="cell", columns="gene", values=feature_name)
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )


class SampleFeature(metaclass=ABCMeta):
    """Abstract class for calculating sample features. A sample is defined as the set of
    molecules corresponding to a single cell-gene pair.

    Parameters
    ----------
    metaclass : _type_, optional
        _description_, by default ABCMeta

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features.
    cell_attributes : int
        Names (keys) used to store computed cell-level features.
    """

    @abstractmethod
    def __init__(self):
        self.cell_features = set()
        self.cell_attributes = set()

    @abstractmethod
    def extract(self, df):
        """Calculates this feature for a given sample.

        Parameters
        ----------
        df : DataFrame
            Assumes each row is a molecule and that columns `x`, `y`, `cell`, and `gene` are present.
        """

        return df


class ShapeProximity(SampleFeature):
    """For a set of points, computes the proximity of points within `shape_name`
    as well as the proximity of points outside `shape_name`. Proximity is defined as
    the average absolute distance to the specified `shape_name` normalized by cell
    radius. Values closer to 0 denote farther from the `shape_name`, values closer
    to 1 denote closer to the `shape_name`.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    cell_attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"{shape_prefix}_inner_proximity"`: proximity of points inside `shape_name`
        `"{shape_prefix}_outer_proximity"`: proximity of points outside `shape_name`
    """

    def __init__(self, shape_name):
        super().__init__()
        self.cell_features.add(tl.cell_radius)

        attrs = [shape_name, "cell_radius"]
        self.cell_attributes.update(attrs)

        self.shape_name = shape_name

    def extract(self, df):
        df = super().extract(df)

        shape_prefix = self.shape_name.split("_shape")[0]

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get points
        points_geo = df["geometry"].values

        # Check for points within shape, assume all are intracellular
        if shape_prefix == "cell":
            inner = np.array([True] * len(df))
        else:
            inner = df[shape_prefix] != -1
        outer = ~inner

        inner_dist = np.nan
        outer_dist = np.nan

        if inner.sum() > 0:
            inner_dist = points_geo[inner].distance(shape.boundary).mean()

        if outer.sum() > 0:
            outer_dist = points_geo[outer].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        cell_radius = df["cell_radius"].values[0]
        inner_proximity = (cell_radius - inner_dist) / cell_radius
        outer_proximity = (cell_radius - outer_dist) / cell_radius

        if np.isnan(inner_proximity):
            inner_proximity = 0

        if np.isnan(outer_proximity):
            outer_proximity = 0

        return {
            f"{shape_prefix}_inner_proximity": inner_proximity,
            f"{shape_prefix}_outer_proximity": outer_proximity,
        }


class ShapeAsymmetry(SampleFeature):
    """For a set of points, computes the asymmetry of points within `shape_name`
    as well as the asymmetry of points outside `shape_name`. Asymmetry is defined as
    the offset between the centroid of points to the centroid of the specified
    `shape_name`, normalized by cell radius. Values closer to 0 denote symmetry,
    values closer to 1 denote asymmetry.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    cell_attributes : int
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
        super().__init__()
        self.cell_features.add(tl.cell_radius)

        attrs = [shape_name, "cell_radius"]
        self.cell_attributes.update(attrs)

        self.shape_name = shape_name

    def extract(self, df):
        df = super().extract(df)

        shape_prefix = self.shape_name.split("_shape")[0]

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get points
        points_geo = df["geometry"].values

        # Check for points within shape, assume all are intracellular
        if shape_prefix == "cell":
            inner = np.array([True] * len(df))
        else:
            inner = df[shape_prefix] != -1
        outer = ~inner

        inner_to_centroid = np.nan
        outer_to_centroid = np.nan

        if inner.sum() > 0:
            inner_to_centroid = points_geo[inner].distance(shape.centroid).mean()

        if outer.sum() > 0:
            outer_to_centroid = points_geo[outer].distance(shape.centroid).mean()

        # Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        cell_radius = df["cell_radius"].values[0]
        inner_asymmetry = inner_to_centroid / cell_radius
        outer_asymmetry = outer_to_centroid / cell_radius

        if np.isnan(inner_asymmetry):
            inner_asymmetry = 0

        if np.isnan(outer_asymmetry):
            outer_asymmetry = 0

        return {
            f"{shape_prefix}_inner_asymmetry": inner_asymmetry,
            f"{shape_prefix}_outer_asymmetry": outer_asymmetry,
        }


class PointDispersion(SampleFeature):
    """For a set of points, calculates the second moment of all points in a cell
    relative to the centroid of the total RNA signal. This value is normalized by
    the second moment of a uniform distribution within the cell boundary.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    cell_attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"point_dispersion"`: measure of point dispersion
    """

    def __init__(self):
        super().__init__()
        self.cell_features.add(tl.raster_cell)

        attrs = ["cell_raster"]
        self.cell_attributes.update(attrs)

    def extract(self, df):
        df = super().extract(df)

        # Get precomputed cell centroid and raster
        pt_centroid = df[["x", "y"]].values.mean(axis=0).reshape(1, 2)
        cell_raster = df["cell_raster"].values[0]

        # calculate points moment
        point_moment = _second_moment(pt_centroid, df[["x", "y"]].values)
        cell_moment = _second_moment(pt_centroid, cell_raster)

        # Normalize by cell moment
        norm_moment = point_moment / cell_moment

        return {"point_dispersion": norm_moment}


class ShapeDispersion(SampleFeature):
    """For a set of points, calculates the second moment of all points in a cell relative to the
    centroid of `shape_name`. This value is normalized by the second moment of a uniform
    distribution within the cell boundary.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    cell_attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"{shape_prefix}_dispersion"`: measure of point dispersion relative to `shape_name`
    """

    def __init__(self, shape_name):
        super().__init__()

        self.cell_features.add(tl.raster_cell)
        attrs = [shape_name, "cell_raster"]
        self.cell_attributes.update(attrs)

        self.shape_name = shape_name

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get precomputed shape centroid and raster
        cell_raster = df["cell_raster"].values[0]

        # calculate points moment
        point_moment = _second_moment(shape.centroid, df[["x", "y"]].values)
        cell_moment = _second_moment(shape.centroid, cell_raster)

        # Normalize by cell moment
        norm_moment = point_moment / cell_moment

        shape_prefix = self.shape_name.split("_shape")[0]

        return {f"{shape_prefix}_dispersion": norm_moment}


class RipleyStats(SampleFeature):
    """For a set of points, calculates properties of the L-function. The L-function
    measures spatial clustering of a point pattern over the area of the cell.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    cell_attributes : int
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

    def __init__(self):
        super().__init__()
        self.cell_features.update([tl.cell_span, tl.cell_bounds, tl.cell_area])

        self.cell_attributes.update(
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
        cell_span = df["cell_span"][0]
        cell_minx = df["cell_minx"][0]
        cell_miny = df["cell_miny"][0]
        cell_maxx = df["cell_maxx"][0]
        cell_maxy = df["cell_maxy"][0]
        cell_area = df["cell_area"][0]

        estimator = RipleysKEstimator(
            area=cell_area,
            x_min=cell_minx,
            y_min=cell_miny,
            x_max=cell_maxx,
            y_max=cell_maxy,
        )

        half_span = cell_span / 2
        radii = np.linspace(1, half_span * 2, num=int(half_span * 2))

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
            data=points_geo, radii=[half_span], mode="none"
        )[0]

        return {
            "l_max": l_max,
            "l_max_gradient": l_max_gradient,
            "l_min_gradient": l_min_gradient,
            "l_monotony": l_monotony,
            "l_half_radius": l_half_radius,
        }


class ShapeEnrichment(SampleFeature):
    """For a set of points, calculates the fraction of points within `shape_name`
    out of all points in the cell.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    cell_attributes : int
        Names (keys) used to store computed cell-level features
    shape_name : str
        Name of shape to use, must be column name in input DataFrame

    Returns
    -------
    dict
        `"{shape_prefix}_enrichment"`: enrichment fraction of points in `shape_name`
    """

    def __init__(self, shape_name):
        super().__init__()

        attrs = [shape_name]
        self.cell_attributes.update(attrs)

        self.shape_name = shape_name

    def extract(self, df):
        df = super().extract(df)

        shape_prefix = self.shape_name.split("_shape")[0]

        # Get points outside shape
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if shape_prefix == "cell":
            enrichment = 1.0
        else:
            inner_count = (df[shape_prefix] != -1).sum()
            enrichment = inner_count / float(len(points_geo))

        return {f"{shape_prefix}_enrichment": enrichment}


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


sample_features = dict(
    cell_proximity=ShapeProximity("cell_shape"),
    nucleus_proximity=ShapeProximity("nucleus_shape"),
    cell_asymmetry=ShapeAsymmetry("cell_shape"),
    nucleus_asymmetry=ShapeAsymmetry("nucleus_shape"),
    point_dispersion=PointDispersion(),
    nucleus_dispersion=ShapeDispersion("nucleus_shape"),
    ripley_stats=RipleyStats(),
    nucleus_enrichment=ShapeEnrichment("nucleus_shape"),
)
