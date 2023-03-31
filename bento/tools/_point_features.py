from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from astropy.stats.spatial import RipleysKEstimator
from scipy.spatial import distance
from scipy.stats.stats import spearmanr
from tqdm.auto import tqdm
import re

from .. import tools as tl
from .._utils import track
from ..geometry import get_points


@track
def analyze_points(
    data: AnnData,
    shape_names: List[str],
    feature_names: List[str],
    groupby: Optional[Union[str, List[str]]] = None,
    copy: bool = False,
):
    """Calculate the set of specified `features` for each point group. Groups are within each cell.

    Parameters
    ----------
    data : AnnData
        Spatially formatted AnnData
    shape_names : str or list of str
        Names of the shapes to analyze.
    feature_names : str or list of str
        Names of the features to analyze.
    groupby : str or list of str, optional
        Key(s) in `data.uns['points'] to groupby, by default None. Always treats each cell separately
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
    elif isinstance(groupby, list):
        groupby = ["cell"] + groupby
    else:
        groupby = ["cell"]

    # Make sure all groupby keys are in point columns
    for g in groupby:
        if g not in get_points(adata).columns:
            raise ValueError(f"Groupby key {g} not found in point columns.")

    # Generate feature x shape combinations
    feature_combos = [point_features[f](s) for f in feature_names for s in shape_names]

    # Compile dependency set of features and attributes
    cell_features = set()
    obs_attrs = set()
    for f in feature_combos:
        cell_features.update(f.cell_features)
        obs_attrs.update(f.attributes)

    cell_features = list(cell_features)
    obs_attrs = list(obs_attrs)

    print("Calculating cell features...")
    tl.analyze_shapes(adata, "cell_shape", cell_features, progress=True)

    # Make sure attributes are present
    attrs_found = set(obs_attrs).intersection(set(adata.obs.columns.tolist()))
    if len(attrs_found) != len(obs_attrs):
        raise KeyError(f"df does not have all columns: {obs_attrs}.")

    # extract cell attributes
    points_df = (
        get_points(adata, asgeo=True)
        .set_index("cell")
        .join(data.obs[obs_attrs])
        .reset_index()
    )

    for g in groupby:
        points_df[g] = points_df[g].astype("category")
    # Handle categories as strings to avoid ambiguous cat types
    # for col in points_df.loc[:, (points_df.dtypes == "category").values]:
    #     points_df[col] = points_df[col].astype(str)

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
        out = partition_df.groupby(groupby, observed=True).apply(process_sample)
        return pd.DataFrame.from_records(out.values, index=out.index)

    # Process points of each cell separately
    cells, group_loc = np.unique(
        points_df["cell"],
        return_index=True,
    )

    end_loc = np.append(group_loc[1:], points_df.shape[0])

    output = []
    print("Processing point features...")
    for start, end in tqdm(zip(group_loc, end_loc), total=len(cells)):
        cell_points = points_df.iloc[start:end]
        output.append(process_partition(cell_points))
    output = pd.concat(output)

    # Save and overwrite existing
    print("Saving results...")
    output_key = "_".join([*groupby, "features"])
    if output_key in adata.uns:
        adata.uns[output_key][output.columns] = output.reset_index(drop=True)
    else:
        adata.uns[output_key] = output.reset_index()

    print("Done.")
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


class ShapeProximity(PointFeature):
    """For a set of points, computes the proximity of points within `shape_name`
    as well as the proximity of points outside `shape_name`. Proximity is defined as
    the average absolute distance to the specified `shape_name` normalized by cell
    radius. Values closer to 0 denote farther from the `shape_name`, values closer
    to 1 denote closer to the `shape_name`.

    Attributes
    ----------
    cell_features : int
        Set of cell-level features needed for computing sample-level features
    attributes : int
        Names (keys) used to store computed cell-level features

    Returns
    -------
    dict
        `"{shape_prefix}_inner_proximity"`: proximity of points inside `shape_name`
        `"{shape_prefix}_outer_proximity"`: proximity of points outside `shape_name`
    """

    def __init__(self, shape_name):
        super().__init__(shape_name)
        self.cell_features.add("radius")
        self.attributes.add("cell_radius")
        self.shape_name = shape_name

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get points
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if self.shape_prefix == "cell":
            inner = np.array([True] * len(df))
        else:
            inner = df[self.shape_prefix] != "-1"
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
            f"{self.shape_prefix}_inner_proximity": inner_proximity,
            f"{self.shape_prefix}_outer_proximity": outer_proximity,
        }


class ShapeAsymmetry(PointFeature):
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
        super().__init__(shape_name)
        self.cell_features.add("radius")
        self.attributes.add("cell_radius")
        self.shape_name = shape_name

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_name].values[0]

        # Get points
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if self.shape_prefix == "cell":
            inner = np.array([True] * len(df))
        else:
            inner = df[self.shape_prefix] != "-1"
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
            f"{self.shape_prefix}_inner_asymmetry": inner_asymmetry,
            f"{self.shape_prefix}_outer_asymmetry": outer_asymmetry,
        }


class PointDispersionNorm(PointFeature):
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

    def __init__(self, shape_name):
        super().__init__(shape_name)
        self.cell_features.add("raster")

        attrs = ["cell_raster"]
        self.attributes.update(attrs)

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

        return {"point_dispersion_norm": norm_moment}


class ShapeDispersionNorm(PointFeature):
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
        super().__init__(shape_name)

        self.cell_features.add("raster")
        attrs = ["cell_raster"]
        self.attributes.update(attrs)

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

        return {f"{self.shape_prefix}_dispersion_norm": norm_moment}


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


class ShapeOffset(PointFeature):
    """For a set of points, computes the offset of points within `shape_name`
    as well as the offset of points outside `shape_name`. Offset is defined as
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
        `"{shape_prefix}_inner_offset"`: offset of points inside `shape_name`
        `"{shape_prefix}_outer_offset"`: offset of points outside `shape_name`
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
            f"{self.shape_prefix}_inner_offset": inner_to_centroid,
            f"{self.shape_prefix}_outer_offset": outer_to_centroid,
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


def list_point_features():
    """Return a DataFrame of available point features. Pulls descriptions from function docstrings.

    Returns
    -------
    list
        List of available point features.
    """

    # Get point feature descriptions from docstrings
    df = dict()
    for k, v in point_features.items():
        description = v.__doc__.split("Attributes")[0].strip()
        description = re.sub("\s +", " ", description)
        df[k] = description

    return df


point_features = dict(
    proximity=ShapeProximity,
    asymmetry=ShapeAsymmetry,
    point_dispersion_norm=PointDispersionNorm,
    shape_dispersion_norm=ShapeDispersionNorm,
    distance=ShapeDistance,
    offset=ShapeOffset,
    point_dispersion=PointDispersion,
    shape_dispersion=ShapeDispersion,
    ripley=RipleyStats,
    shape_enrichment=ShapeEnrichment,
)


def register_point_feature(name: str, FeatureClass: PointFeature):
    """Register a new point feature function.

    Parameters
    ----------
    name : str
        Name of feature function
    func : class
        Class that extends PointFeature. Needs to override abstract functions.
    """

    point_features[name] = FeatureClass

    print(f"Registered point feature '{name}' to `bento.tl.shape_features`.")
