import unittest
import bento as bt
import spatialdata as sd


class TestShapeFeatures(unittest.TestCase):
    def setUp(self):
        self.shape_features = bt.tl.list_shape_features().keys()

    # Simplest test to check if a single shape feature is calculated for a single shape
    def test_single_shape_single_feature(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_shapes(
            sdata=data, 
            shape_keys="cell_boundaries", 
            feature_names="area", 
            progress=True
        )
        
        # Check if cell_boundaries shape features are calculated
        self.assertTrue("cell_boundaries_area" in data.shapes["cell_boundaries"].columns)

        # Check shapes attrs
        self.assertTrue("transform" in data.shapes["cell_boundaries"].attrs.keys())

    # Test case to check if multiple shape features are calculated for a single shape
    def test_single_shape_multiple_features(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_shapes(
            sdata=data, 
            shape_keys="cell_boundaries", 
            feature_names=self.shape_features,
            feature_kws={"opening": {"proportion": 0.5}},
            progress=True
        )
        
        # Check if cell_boundaries shape features are calculated
        self.assertTrue("cell_boundaries_area" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_aspect_ratio" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_density" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_minx" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_miny" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_maxx" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_maxy" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_radius" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_open_0.5_shape" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_perimeter" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_raster" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_raster" in data.points.keys())
        self.assertTrue("cell_boundaries_moment" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_span" in data.shapes["cell_boundaries"].columns)

        # Check points attrs
        self.assertTrue("transform" in data.points["cell_boundaries_raster"].attrs.keys())
        self.assertTrue(data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"]["feature_key"] == "feature_name")
        self.assertTrue(data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"]["instance_key"] == "cell_boundaries")

        # Check shapes attrs
        self.assertTrue("transform" in data.shapes["cell_boundaries"].attrs.keys())

    # Test case to check if a single shape feature is calculated for multiple shapes
    def test_multiple_shapes_single_feature(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_shapes(
            sdata=data, 
            shape_keys=["cell_boundaries", "nucleus_boundaries"], 
            feature_names="area", 
            progress=True
        )
        
        # Check if cell_boundaries and nucleus_boundaries shape features are calculated
        self.assertTrue("cell_boundaries_area" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_area" in data.shapes["nucleus_boundaries"].columns)

        # Check shapes attrs
        self.assertTrue("transform" in data.shapes["cell_boundaries"].attrs.keys())
        self.assertTrue("transform" in data.shapes["nucleus_boundaries"].attrs.keys())

    # Test case to check if multiple shape features are calculated for multiple shapes
    def test_multiple_shapes_multiple_features(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        data = bt.tl.analyze_shapes(
            sdata=data, 
            shape_keys=["cell_boundaries", "nucleus_boundaries"], 
            feature_names=self.shape_features,
            feature_kws={"opening": {"proportion": 0.5}},
            progress=True
        )
        
        # Check if cell_boundaries shape features are calculated
        self.assertTrue("cell_boundaries_area" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_aspect_ratio" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_density" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_minx" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_miny" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_maxx" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_maxy" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_radius" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_open_0.5_shape" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_perimeter" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_raster" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_raster" in data.points.keys())
        self.assertTrue("cell_boundaries_moment" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_span" in data.shapes["cell_boundaries"].columns)

        # Check points attrs
        self.assertTrue("transform" in data.points["cell_boundaries_raster"].attrs.keys())
        self.assertTrue(data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"]["feature_key"] == "feature_name")
        self.assertTrue(data.points["cell_boundaries_raster"].attrs["spatialdata_attrs"]["instance_key"] == "cell_boundaries")

        # Check shapes attrs
        self.assertTrue("transform" in data.shapes["cell_boundaries"].attrs.keys())

        # Check if nucleus_boundaries shape features are calculated
        self.assertTrue("nucleus_boundaries_area" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_aspect_ratio" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_density" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_minx" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_miny" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_maxx" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_maxy" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_radius" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_open_0.5_shape" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_perimeter" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_raster" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_raster" in data.points.keys())
        self.assertTrue("nucleus_boundaries_moment" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_span" in data.shapes["nucleus_boundaries"].columns)

        # Check points attrs
        self.assertTrue("transform" in data.points["nucleus_boundaries_raster"].attrs.keys())
        self.assertTrue(data.points["nucleus_boundaries_raster"].attrs["spatialdata_attrs"]["feature_key"] == "feature_name")
        self.assertTrue(data.points["nucleus_boundaries_raster"].attrs["spatialdata_attrs"]["instance_key"] == "cell_boundaries")

        # Check shapes attrs
        self.assertTrue("transform" in data.shapes["nucleus_boundaries"].attrs.keys())
    
    # Test case to check if obs_stats function calculates area, aspect_ratio and density for both cell_boundaries and nucleus_boundaries
    def test_obs_stats(self):
        data = sd.read_zarr("/mnt/d/spatial_datasets/small_data.zarr")
        data = bt.io.format_sdata(
            sdata=data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        bt.tl.obs_stats(sdata=data)
        
        # Check if cell_boundaries and nucleus_boundaries shape features are calculated
        self.assertTrue("cell_boundaries_area" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_aspect_ratio" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("cell_boundaries_density" in data.shapes["cell_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_area" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_aspect_ratio" in data.shapes["nucleus_boundaries"].columns)
        self.assertTrue("nucleus_boundaries_density" in data.shapes["nucleus_boundaries"].columns)

        # Check shapes attrs
        self.assertTrue("transform" in data.shapes["cell_boundaries"].attrs.keys())
        self.assertTrue("transform" in data.shapes["nucleus_boundaries"].attrs.keys())
