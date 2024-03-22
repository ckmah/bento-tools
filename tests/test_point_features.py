import unittest
import bento as bt
import spatialdata as sd


class TestPointFeatures(unittest.TestCase):
    def setUp(self):
        self.point_features = bt.tl.list_point_features().keys()
        self.instance_key = ["cell_boundaries"]
        self.feature_key = ["feature_name"]
        self.indpendent_features = [
            'point_dispersion_norm', 
            'point_dispersion',
            'l_max', 
            'l_max_gradient', 
            'l_min_gradient',
            'l_monotony',
            'l_half_radius'
        ]
        self.cell_features = [
            'cell_boundaries_inner_proximity',
            'cell_boundaries_outer_proximity',
            'cell_boundaries_inner_asymmetry',
            'cell_boundaries_outer_asymmetry',
            'cell_boundaries_dispersion_norm',
            'cell_boundaries_inner_distance',
            'cell_boundaries_outer_distance',
            'cell_boundaries_inner_offset',
            'cell_boundaries_outer_offset',
            'cell_boundaries_dispersion',
            'cell_boundaries_enrichment'
        ]
        self.nucleus_features = [
            'nucleus_boundaries_inner_proximity',
            'nucleus_boundaries_outer_proximity',
            'nucleus_boundaries_inner_asymmetry',
            'nucleus_boundaries_outer_asymmetry',
            'nucleus_boundaries_dispersion_norm',
            'nucleus_boundaries_inner_distance',
            'nucleus_boundaries_outer_distance',
            'nucleus_boundaries_inner_offset',
            'nucleus_boundaries_outer_offset',
            'nucleus_boundaries_dispersion',
            'nucleus_boundaries_enrichment'
        ]


    # Test case to check if features are calculated for a single shape and a single group
    def test_single_shape_single_group(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_points(
            sdata=data,
            shape_keys=["cell_boundaries"],
            feature_names=self.point_features,
            groupby=None,
            recompute=False,
            progress=True,
        )

        point_features = self.instance_key + self.indpendent_features + self.cell_features

        # Check if cell_boundaries point features are calculated
        for feature in point_features:
            self.assertTrue(feature in data.table.uns["cell_boundaries_features"].columns)


    # Test case to check if features are calculated for a single shape and multiple groups
    def test_single_shape_multiple_groups(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_points(
            sdata=data,
            shape_keys=["cell_boundaries"],
            feature_names=self.point_features,
            groupby=["feature_name"],
            recompute=False,
            progress=True,
        )

        point_features = self.instance_key + self.feature_key+ self.indpendent_features + self.cell_features

        # Check if cell_boundaries and gene point features are calculated
        for feature in point_features:
            self.assertTrue(feature in data.table.uns["cell_boundaries_feature_name_features"].columns)

    # Test case to check if point features are calculated for multiple shapes and a single group
    def test_multiple_shapes_single_group(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_points(
            sdata=data,
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
            feature_names=self.point_features,
            groupby=None,
            recompute=False,
            progress=True,
        )

        point_features = self.instance_key + self.indpendent_features + self.cell_features + self.nucleus_features

        # Check if cell_boundaries and nucleus_boundaries point features are calculated
        for feature in point_features:
            self.assertTrue(feature in data.table.uns["cell_boundaries_features"].columns)

    # Test case to check if multiple shape features are calculated for multiple shapes
    def test_multiple_shapes_multiple_groups(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_points(
            sdata=data,
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
            feature_names=self.point_features,
            groupby=["feature_name"],
            recompute=False,
            progress=True,
        )

        point_features = self.instance_key + self.feature_key + self.indpendent_features + self.cell_features + self.nucleus_features

        # Check if cell_boundaries and nucleus_boundaries point features are calculated
        for feature in point_features:
            self.assertTrue(feature in data.table.uns["cell_boundaries_feature_name_features"].columns)
