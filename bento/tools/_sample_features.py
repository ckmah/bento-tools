from abc import ABCMeta, abstractmethod

import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
import geopandas as gpd


def proximity(data, shape_name, position='inner', n_jobs=1, copy=False):
    """
    Compute proximity of points to shape.

    Parameters
    ----------
    data : spatial formatted AnnData
        data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
    shape_name : str
        Column name in data.obs referring to shape to use in calculation.
    position : str
        Expect one of 'inner', or 'outer', by default 'inner'. 'Outer' refers to points outside the shape but within the same overall cellular compartment; 'inner' refers to points inside shape.
    n_jobs : int
        Number of jobs to run in parallel.
    copy : bool
        Whether to return a copy of data or to modify in place, by default False.

    """
    adata = data.copy() if copy else data
    if shape_name not in data.obs.columns:
        raise ValueError("Not a valid shape")

    layer = f'{shape_name.split(sep="_shape")[0]}_{position}_proximity'
        
    if position == 'inner':
        # TODO switch to indexed shapes
        InnerToShape().transform(data, shape_name, layer, n_jobs)
    elif position == 'outer':
        if shape_name == 'cell_shape':
            raise ValueError("Extracellular points not supported")
        OuterToShape().transform(data, shape_name, layer, n_jobs)
    else:
        raise ValueError("Not a valid position.")

    return adata if copy else None


class AbstractFeature(metaclass=ABCMeta):

    __point_metadata = pd.Series(["x", "y", "cell", "gene", "nucleus"])

    @abstractmethod
    def extract(self, points, shape):
        """Given a set of points, extract and return a single feature value.

        Parameters
        ----------
        points : DataFrame
            Point coordinates.
        shape : Polygon (Shapely)

        """
        return points, shape

    @classmethod
    def transform(self, data, shape_name, layer, n_jobs):
        """Applies self.extract() to all points grouped by cell and gene.

        Parameters
        ----------
        data : spatial formatted AnnData
            data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
        shape_name : str
            Column name in data.obs referring to shape to use in calculation.
        """

        points = data.uns["points"]

        # Check points DataFrame for missing columns
        if not self.__point_metadata.isin(points.columns).all():
            raise KeyError(
                f"'points' DataFrame needs to have all columns: {self.__point_metadata.tolist()}."
            )

        # GeoDataFrame for spatial operations
        points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y))
        
        # Precompute max cell radius
        max_dist = data.obs['cell_shape'].apply(lambda c: c.boundary.hausdorff_distance(c.centroid))
        max_dist.index.name = 'cell'
        max_dist.name = 'max_dist'

        # Precompute if shape is in nucleus
#         shape_geo = gpd.GeoSeries(data.obs[shape_name])
        nucleus_geo = gpd.GeoSeries(data.obs['nucleus_shape'])
#         shape_in_nucleus = shape_geo.within(nucleus_geo)
#         shape_in_nucleus.index.name = 'cell'
#         shape_in_nucleus.name = 'shape_in_nucleus'
        
        # Join attributes to points
        input_data = points.set_index('cell').join(data.obs[shape_name]).join(max_dist)
        
        if shape_name != 'nucleus_shape':
            input_data = input_data.join(nucleus_geo)
            
        input_data = input_data.reset_index()
        
        # Precompute if points are within respective shape
#         print('Warning: compatible with non-indexed shapes for now (4x slower)')
#         input_data['point_in_shape'] = input_data.within(input_data[shape_name])
        
        # Cast categorical type to save memory
        cat_vars = ['cell', 'gene', 'nucleus']
        input_data[cat_vars] = input_data[cat_vars].astype('category')

        # Group points by cell and gene
        input_grouped = input_data.groupby(["cell", "gene"])
        
        # Extract feature
        values = Parallel(n_jobs=n_jobs)(
            delayed(self.extract)(self, inp, inp[shape_name].iloc[0])
            for name, inp in tqdm(input_grouped, total=len(input_grouped), desc="Measure proximity")
        )
        
        # Save results to data layer
        feature_df = pd.DataFrame(input_grouped.groups.keys(), columns=["cell", "gene"])
        feature_df[layer] = values

        data.layers[layer] = (
            feature_df.pivot(index="cell", columns="gene", values=layer)
            .reindex(index=data.obs_names, columns=data.var_names)
            .astype(float)
        )

        print('Done.')

        return

    
class OuterToShape(AbstractFeature):
    def extract(self, points, shape):
        """Given a set of points, calculate and return the average proximity between points outside to the shape boundary.
        
        Only considers points inside the same major subcellular compartment (cytoplasm or nucleus).

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shape : Polygon (Shapely)
        """
        points, shape = super().extract(self, points, shape)
        nucleus_shape = points['nucleus_shape'].values[0]
        
        # Only look at points in the same major compartment
        if shape.within(nucleus_shape):
            in_compartment = points["nucleus"] != "-1"
        else:
            in_compartment = points["nucleus"] == "-1"

        outer = ~points.within(shape)
        
            
        dist = points[in_compartment & outer].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        max_dist = points['max_dist'].values[0]
        proximity = (max_dist - dist) / max_dist
        return proximity
    
class InnerToShape(AbstractFeature):
    def extract(self, points, shape):
        """Given a set of points, calculate and return the average proximity between points inside to the shape boundary.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.
        shape : Polygon (Shapely)
            """
        points, shape = super().extract(self, points, shape)
        inner = points.within(shape)

        dist = points[inner].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        max_dist = points['max_dist'].values[0]
        proximity = (max_dist - dist) / max_dist
        return proximity