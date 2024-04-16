import unittest
import bento as bt
import spatialdata as sd


class TestShapeFeatures(unittest.TestCase):
    def setUp(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.data = sd.read_zarr(f"{datadir}/small_data.zarr")
        self.data = bt.io.format_sdata(
            sdata=self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        self.shape_features = bt.tl.list_shape_features().keys()

        feature_names = [
            "area",
            "aspect_ratio",
            "minx",
            "miny",
            "maxx",
            "maxy",
            "density",
            "open_0.5_shape",
            "perimeter",
            "radius",
            "raster",
            "moment",
            "span",
        ]
        self.cell_features = [f"cell_boundaries_{x}" for x in feature_names]
        self.nucleus_features = [f"nucleus_boundaries_{x}" for x in feature_names]

    # Simplest test to check if a single shape feature is calculated for a single shape
    def test_single_shape_single_feature(self):
        self.data = bt.tl.analyze_shapes(
            sdata=self.data,
            shape_keys="cell_boundaries",
            feature_names="area",
            progress=True,
        )

        # Check if cell_boundaries shape features are calculated
        self.assertTrue(
            "cell_boundaries_area" in self.data.shapes["cell_boundaries"].columns
        )

        # Check shapes attrs
        self.assertTrue("transform" in self.data.shapes["cell_boundaries"].attrs.keys())

    # Test case to check if multiple shape features are calculated for a single shape
    def test_single_shape_multiple_features(self):
        self.data = bt.tl.analyze_shapes(
            sdata=self.data,
            shape_keys="cell_boundaries",
            feature_names=self.shape_features,
            feature_kws={"opening": {"proportion": 0.5}},
            progress=True,
        )

        # Check if cell_boundaries shape features are calculated
        for feature in self.cell_features:
            self.assertTrue(feature in self.data.shapes["cell_boundaries"].columns)

        # Check that raster is a points element
        self.assertTrue("cell_boundaries_raster" in self.data.points.keys())

        # Check points attrs
        self.assertTrue(
            "transform" in self.data.points["cell_boundaries_raster"].attrs.keys()
        )
        self.assertTrue(
            self.data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"][
                "feature_key"
            ]
            == "feature_name"
        )
        self.assertTrue(
            self.data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"][
                "instance_key"
            ]
            == "cell_boundaries"
        )

        # Check shapes attrs
        self.assertTrue("transform" in self.data.shapes["cell_boundaries"].attrs.keys())

    # Test case to check if a single shape feature is calculated for multiple shapes
    def test_multiple_shapes_single_feature(self):
        self.data = bt.tl.analyze_shapes(
            sdata=self.data,
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
            feature_names="area",
            progress=True,
        )

        # Check if cell_boundaries and nucleus_boundaries shape features are calculated
        self.assertTrue(
            "cell_boundaries_area" in self.data.shapes["cell_boundaries"].columns
        )
        self.assertTrue(
            "nucleus_boundaries_area" in self.data.shapes["nucleus_boundaries"].columns
        )

        # Check shapes attrs
        self.assertTrue("transform" in self.data.shapes["cell_boundaries"].attrs.keys())
        self.assertTrue(
            "transform" in self.data.shapes["nucleus_boundaries"].attrs.keys()
        )

    # Test case to check if multiple shape features are calculated for multiple shapes
    def test_multiple_shapes_multiple_features(self):
        self.data = bt.tl.analyze_shapes(
            sdata=self.data,
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
            feature_names=self.shape_features,
            feature_kws={"opening": {"proportion": 0.5}},
            progress=True,
        )

        # Check if cell_boundaries shape features are calculated
        for feature in self.cell_features:
            self.assertTrue(feature in self.data.shapes["cell_boundaries"].columns)

        # Check that raster is a points element
        self.assertTrue("cell_boundaries_raster" in self.data.points.keys())

        # Check points attrs
        self.assertTrue(
            "transform" in self.data.points["cell_boundaries_raster"].attrs.keys()
        )
        self.assertTrue(
            self.data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"][
                "feature_key"
            ]
            == "feature_name"
        )
        self.assertTrue(
            self.data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"][
                "instance_key"
            ]
            == "cell_boundaries"
        )

        # Check shapes attrs
        self.assertTrue("transform" in self.data.shapes["cell_boundaries"].attrs.keys())

        # Check if nucleus_boundaries shape features are calculated
        for feature in self.nucleus_features:
            self.assertTrue(feature in self.data.shapes["nucleus_boundaries"].columns)

        # Check that raster is a points element
        self.assertTrue("nucleus_boundaries_raster" in self.data.points.keys())

        # Check points attrs
        self.assertTrue(
            "transform" in self.data.points["nucleus_boundaries_raster"].attrs.keys()
        )
        self.assertTrue(
            self.data.points["nucleus_boundaries_raster"].attrs["spatialdata_attrs"][
                "feature_key"
            ]
            == "feature_name"
        )
        self.assertTrue(
            self.data.points["nucleus_boundaries_raster"].attrs["spatialdata_attrs"][
                "instance_key"
            ]
            == "cell_boundaries"
        )

        # Check shapes attrs
        self.assertTrue(
            "transform" in self.data.shapes["nucleus_boundaries"].attrs.keys()
        )

    # Test case to check if shape_stats function calculates area, aspect_ratio and density for both cell_boundaries and nucleus_boundaries
    def test_shape_stats(self):
        bt.tl.shape_stats(sdata=self.data)

        # Check if cell_boundaries and nucleus_boundaries shape features are calculated
        self.assertTrue(
            "cell_boundaries_area" in self.data.shapes["cell_boundaries"].columns
        )
        self.assertTrue(
            "cell_boundaries_aspect_ratio"
            in self.data.shapes["cell_boundaries"].columns
        )
        self.assertTrue(
            "cell_boundaries_density" in self.data.shapes["cell_boundaries"].columns
        )
        self.assertTrue(
            "nucleus_boundaries_area" in self.data.shapes["nucleus_boundaries"].columns
        )
        self.assertTrue(
            "nucleus_boundaries_aspect_ratio"
            in self.data.shapes["nucleus_boundaries"].columns
        )
        self.assertTrue(
            "nucleus_boundaries_density"
            in self.data.shapes["nucleus_boundaries"].columns
        )

        # Check shapes attrs
        self.assertTrue("transform" in self.data.shapes["cell_boundaries"].attrs.keys())
        self.assertTrue(
            "transform" in self.data.shapes["nucleus_boundaries"].attrs.keys()
        )
