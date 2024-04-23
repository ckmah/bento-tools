import unittest
import bento as bt
import spatialdata as sd


class TestPointFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.data = sd.read_zarr(f"{datadir}/small_data.zarr")
        self.data = bt.io.prep(
            sdata=self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        self.point_features = bt.tl.list_point_features().keys()
        self.instance_key = ["cell_boundaries"]
        self.feature_key = ["feature_name"]
        self.indpendent_features = [
            "point_dispersion_norm",
            "point_dispersion",
            "l_max",
            "l_max_gradient",
            "l_min_gradient",
            "l_monotony",
            "l_half_radius",
        ]

        feature_names = [
            "inner_proximity",
            "outer_proximity",
            "inner_asymmetry",
            "outer_asymmetry",
            "dispersion_norm",
            "inner_distance",
            "outer_distance",
            "inner_offset",
            "outer_offset",
            "dispersion",
            "enrichment",
        ]

        self.cell_features = [f"cell_boundaries_{x}" for x in feature_names]
        self.nucleus_features = [f"nucleus_boundaries_{x}" for x in feature_names]

    # Test case to check if features are calculated for a single shape and a single group
    def test_single_shape_single_group(self):
        bt.tl.analyze_points(
            sdata=self.data,
            shape_keys=["cell_boundaries"],
            feature_names=self.point_features,
            groupby=None,
            recompute=False,
            progress=True,
        )

        point_features = (
            self.instance_key + self.indpendent_features + self.cell_features
        )

        # Check if cell_boundaries point features are calculated
        for feature in point_features:
            self.assertTrue(
                feature in self.data.table.uns["cell_boundaries_features"].columns
            )

    # Test case to check if features are calculated for a single shape and multiple groups
    def test_single_shape_multiple_groups(self):
        bt.tl.analyze_points(
            sdata=self.data,
            shape_keys=["cell_boundaries"],
            feature_names=self.point_features,
            groupby=["feature_name"],
            recompute=False,
            progress=True,
        )

        point_features = (
            self.instance_key
            + self.feature_key
            + self.indpendent_features
            + self.cell_features
        )

        # Check if cell_boundaries and gene point features are calculated
        for feature in point_features:
            self.assertTrue(
                feature
                in self.data.table.uns["cell_boundaries_feature_name_features"].columns
            )

    # Test case to check if point features are calculated for multiple shapes and a single group
    def test_multiple_shapes_single_group(self):
        bt.tl.analyze_points(
            sdata=self.data,
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
            feature_names=self.point_features,
            groupby=None,
            recompute=False,
            progress=True,
        )

        point_features = (
            self.instance_key
            + self.indpendent_features
            + self.cell_features
            + self.nucleus_features
        )

        # Check if cell_boundaries and nucleus_boundaries point features are calculated
        for feature in point_features:
            self.assertTrue(
                feature in self.data.table.uns["cell_boundaries_features"].columns
            )

    # Test case to check if multiple shape features are calculated for multiple shapes
    def test_multiple_shapes_multiple_groups(self):
        bt.tl.analyze_points(
            sdata=self.data,
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
            feature_names=self.point_features,
            groupby=["feature_name"],
            recompute=False,
            progress=True,
        )

        point_features = (
            self.instance_key
            + self.feature_key
            + self.indpendent_features
            + self.cell_features
            + self.nucleus_features
        )

        # Check if cell_boundaries and nucleus_boundaries point features are calculated
        for feature in point_features:
            self.assertTrue(
                feature
                in self.data.table.uns["cell_boundaries_feature_name_features"].columns
            )
