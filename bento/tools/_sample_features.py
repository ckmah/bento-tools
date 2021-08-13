from abc import ABCMeta, abstractmethod

import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
import geopandas as gpd


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
