from abc import ABCMeta, abstractmethod

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import geopandas as gpd


def extract(data, feature_name, n_cores=1, copy=False):
    adata = data.copy() if copy else data
    if feature_name == "cyto_distance_to_cell":
        CytoDistanceToCell().transform(adata, ["cell_shape"], feature_name, n_cores)
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
        pass

    @classmethod
    def transform(self, data, shape_names, layer, n_cores):
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
                f"'points' DataFrame does not have columns {self.__point_metadata.tolist()}."
            )

        # Group points by cell and gene
        group_names = []
        group_points = []
        for name, p in points.groupby(["cell", "gene"]):
            group_names.append(name)
            group_points.append(
                gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p.x, p.y))
            )

        # Get sample cell and gene names
        group_cells = [name[0] for name in group_names]
        group_genes = [name[1] for name in group_names]

        # Enumerate shapes associated with each sample
        group_shapes = data.obs.loc[group_cells, shape_names].to_numpy()

        # Extract feature
        values = Parallel(n_jobs=n_cores)(
            delayed(self.extract)(self, p, s)
            for p, s in tqdm(zip(group_points, group_shapes), total=len(group_points))
        )
        # Save results to data layer
        feature_df = pd.DataFrame(
            [group_cells, group_genes, values], index=["cell", "gene", layer]
        ).T

        data.layers[layer] = (
            feature_df.pivot(index="cell", columns="gene", values=layer)
            .reindex(index=data.obs_names, columns=data.var_names)
            .astype(float)
        )

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
        cell_shape = shapes[0]
        cytoplasmic = points["nucleus"].astype(str) == "-1"

        return points[cytoplasmic].distance(cell_shape.boundary).mean()
