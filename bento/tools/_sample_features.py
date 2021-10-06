from abc import ABCMeta, abstractmethod
from shapely.geometry import Point, LineString
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np


def extract_many():
    pass

def extract(data, feature_name, n_jobs=1, copy=False):
    adata = data.copy() if copy else data
    if feature_name == "cyto_distance_to_cell":
        CytoDistanceToCell().transform(adata, ["cell_shape"], feature_name, n_jobs)
    elif feature_name == "cyto_distance_to_nucleus":
        CytoDistanceToNucleus().transform(
            adata, ["nucleus_shape"], feature_name, n_jobs
        )
    elif feature_name == "intranuclear_distance_to_nucleus":
        IntranuclearDistanceToNucleus().transform(
            adata, ["nucleus_shape"], feature_name, n_jobs
        )
    elif feature_name == "zone_enrichment":
        for zone in adata.uns['zones']:
            print("Processing enrichment in " + zone)
            ZoneEnrichment().transform(
                adata, [zone], zone+'_enrichment', n_jobs
            )
    elif feature_name == "zone_polarization":
        for zone in adata.uns['zones']:
            print("Computing polarization in " + zone)
            ZonePolarization().transform(
                adata, [zone], zone+'_polarization', n_jobs
            )
    else:
        raise ValueError("Not a valid 'feature_name'.")

    return adata if copy else None


class AbstractFeature(metaclass=ABCMeta):

    __point_metadata = pd.Series(["x", "y", "cell", "gene", "nucleus"])

    @abstractmethod
    def extract(self, points, shapes):
        """Given a set of points, extract and return a single feature value.

        Parameters
        ----------
        points : DataFrame
            Point coordinates.
        shapes : list of Polygon objects (Shapely)

        """
        return points, shapes

    @classmethod
    def transform(self, data, shape_names, layer, n_jobs):
        """Applies self.extract() to all points grouped by cell and gene.

        Parameters
        ----------
        data : spatial formatted AnnData
            data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
        shape_names : list of str
            Column names in data.obs referring to shapes to use in calculation.
        """

        points = data.uns["points"]

        # Check points DataFrame for missing columns
        if not self.__point_metadata.isin(points.columns).all():
            raise KeyError(
                f"'points' DataFrame needs to have all columns: {self.__point_metadata.tolist()}."
            )

        # Group points by cell and gene
        points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y))
        points[['cell', 'gene', 'nucleus']] = points[['cell', 'gene', 'nucleus']].astype('category')
        points_groupby = points.groupby(["cell", "gene"])

        # Extract feature
        values = Parallel(n_jobs=n_jobs)(
            delayed(self.extract)(self, p, data.obs.loc[name[0], shape_names])
            for name, p in tqdm(points_groupby, total=points_groupby.ngroups, desc="Analyzing")
        )
        
        # Save results to data layer
        feature_df = pd.DataFrame(points_groupby.groups.keys(), columns=["cell", "gene"])
        feature_df[layer] = values

        data.layers[layer] = (
            feature_df.pivot(index="cell", columns="gene", values=layer)
            .reindex(index=data.obs_names, columns=data.var_names)
            .astype(float)
        )

        print('Done.')

        return


class CytoDistanceToCell(AbstractFeature):
    def extract(self, points, shapes):
        """Given a set of points, calculate and return the average distance between cytoplasmic points to the cell membrane.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shapes : list of Polygon objects (Shapely)
            Assumes first element is cell membrane shape.
        """
        points, shapes = super().extract(self, points, shapes)
        cell_shape = shapes[0]
        cytoplasmic = points["nucleus"].astype(str) == "-1"

        return points[cytoplasmic].distance(cell_shape.boundary).mean()


class CytoDistanceToNucleus(AbstractFeature):
    def extract(self, points, shapes):
        """Given a set of points, calculate and return the average distance between cytoplasmic points to the nuclear membrane.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shapes : list of Polygon objects (Shapely)
            Assumes first element is nuclear membrane shape.
        """
        points, shapes = super().extract(self, points, shapes)
        nuclear_shape = shapes[0]
        cytoplasmic = points["nucleus"].astype(str) == "-1"

        return points[cytoplasmic].distance(nuclear_shape.boundary).mean()


class IntranuclearDistanceToNucleus(AbstractFeature):
    def extract(self, points, shapes):
        """Given a set of points, calculate and return the average distance between intranuclear points to the nuclear membrane.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shapes : list of Polygon objects (Shapely)
            Assumes first element is nuclear membrane shape.
        """
        points, shapes = super().extract(self, points, shapes)
        nuclear_shape = shapes[0]
        nuclear = points["nucleus"].astype(str) != "-1"

        return points[nuclear].distance(nuclear_shape.boundary).mean()

class ZoneEnrichment(AbstractFeature):
    def extract(self, points, shapes):
        """Given a set of points and a subcellular region, calculate the proportion of points in said region.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shapes : list of Polygon objects (Shapely)
            Assumes first element is zone shape.
        """
        points, shapes = super().extract(self, points, shapes)
        zone_shape = shapes[0]

        return list(points.within(zone_shape)).count(True)/len(points)

class ZonePolarization(AbstractFeature):
    def extract(self, points, shapes):
        """Given a set of points and zone shape, compute how distant the centroid of the point cloud is from the centroid of the shape normalized to the major axis of the shape.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shapes : list of Polygon objects (Shapely)
            Assumes first element is zone shape.
        """
        points, shapes = super().extract(self, points, shapes)
        shape = shapes[0]
        # define major axis of shape
        # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
        mbr_points = list(zip(*shape.minimum_rotated_rectangle.exterior.coords.xy))
        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]
        # get major/minor axis measurements
        #minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)
        shape_centroid = shape.centroid
        pnt_centroid = Point(np.mean(points.x),np.mean(points.y))
        centroid_diff = shape_centroid.distance(pnt_centroid)

        return abs(centroid_diff/major_axis)