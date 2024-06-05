```{toctree}
:hidden: true
```

# {octicon}`gear` How it Works

## Data Format

Under the hood, we use the [SpatialData](https://spatialdata.scverse.org/en/latest/) framework to manage `SpatialData` objects in Python, allowing us to store and manipulate spatial data in a standardized format. Briefly, `SpatialData` objects are stored on-disk in the Zarr storage format. We aim to be fully compatible with SpatialData, so you can use the same objects in both Bento and SpatialData.

To enable scalable and performant operation with Bento, we perform spatial indexing on the data upfront and store these indices as metadata. This allows us to quickly query points within shapes, and shapes that contain points. Bento adopts a cell-centric approach, where each cell is treated as an independent unit of analysis. This allows us to perform subcellular spatial analysis within individual cells, and aggregate results across cells.

```{eval-rst}
.. code-block:: python
    :caption: Example SpatialData object

    from spatialdata.datasets import blobs
    sdata = blobs()
    print(sdata)
```

```
SpatialData object
├── Images
│     ├── 'blobs_image': SpatialImage[cyx] (3, 512, 512)
│     └── 'blobs_multiscale_image': MultiscaleSpatialImage[cyx] (3, 512, 512), (3, 256, 256), (3, 128, 128)
├── Labels
│     ├── 'blobs_labels': SpatialImage[yx] (512, 512)
│     └── 'blobs_multiscale_labels': MultiscaleSpatialImage[yx] (512, 512), (256, 256), (128, 128)
├── Points
│     └── 'blobs_points': DataFrame with shape: (<Delayed>, 4) (2D points)
├── Shapes
│     ├── 'blobs_circles': GeoDataFrame shape: (5, 2) (2D shapes)
│     ├── 'blobs_multipolygons': GeoDataFrame shape: (2, 1) (2D shapes)
│     └── 'blobs_polygons': GeoDataFrame shape: (5, 1) (2D shapes)
└── Tables
      └── 'table': AnnData (26, 3)
with coordinate systems:
    ▸ 'global', with elements:
        blobs_image (Images), blobs_multiscale_image (Images), blobs_labels (Labels), blobs_multiscale_labels (Labels), blobs_points (Points), blobs_circles (Shapes), blobs_multipolygons (Shapes), blobs_polygons (Shapes)
```

The `SpatialData` object is a container for the following elements:
- `Images`: raw images, segmented images
- `Labels`: cell masks, nucleus masks
- `Points`: transcript coordinates, cell coordinates, landmarks
- `Shapes`: boundaries, circles, polygons
- `Tables`: annotations, count matrices

See the [Data Prep Guide](tutorial_gallery/Data_Prep_Guide.html) for more information on how to prepare `SpatialData` objects for Bento and official [SpatialData documentation](https://spatialdata.scverse.org) for more info.


## RNAflux

RNAflux is a method for quantifying spatial composition gradients in the cell.


## RNAforest input features
    
The following describes input features of the RNAforest model.

| **Categories** | **Features** |
| -------------- | ------------ |
| Distance       | **Cell inner proximity**: The average distance between all points within the cell to the cell boundary normalized by cell radius. Values closer to 0 denote farther from the cell boundary, values closer to 1 denote closer to the cell boundary.<br>**Nucleus inner proximity**: The average distance between all points within the nucleus to the nucleus boundary normalized by cell radius. Values closer to 0 denote farther from the nucleus boundary, values closer to 1 denote closer to the nucleus boundary.<br>**Nucleus outer proximity**: The average distance between all points within the cell and outside the nucleus to the nucleus boundary normalized by cell radius. Values closer to 0 denote farther from the nucleus boundary, values closer to 1 denote closer to the nucleus boundary. |
| Symmetry       | **Cell inner asymmetry**: The offset between the centroid of all points within the cell to the centroid of the cell boundary, normalized by cell radius. Values closer to 0 denote symmetry, values closer to 1 denote asymmetry.<br>**Nucleus inner asymmetry**: The offset between the centroid of all points within the nucleus to the centroid of the nucleus boundary, normalized by cell radius. Values closer to 0 denote symmetry, values closer to 1 denote asymmetry.<br>**Nucleus outer asymmetry**: The offset between the centroid of all points within the cell and outside the nucleus to the centroid of the nucleus boundary, normalized by cell radius. Values closer to 0 denote symmetry, values closer to 1 denote asymmetry.                                                                |
| Dispersion     | **Point dispersion**: The second moment of all points in a cell relative to the centroid of the total RNA signal. This value is normalized by the second moment of a uniform distribution within the cell boundary.<br>**Nucleus dispersion**: The second moment of all points in a cell relative to the centroid of the nucleus boundary. This value is normalized by the second moment of a uniform distribution within the cell boundary.                                                                                                                                                                                                                                                                                                                                                                    |
| Density        | **L-function max**: The max value of the L-function evaluated at r=[1,d], where d is half the cell’s maximum diameter.<br>**L-function max gradient**: The max value of the gradient of the above L-function.<br>**L-function min gradient**: The min value of the gradient of the above L-function.<br>**L monotony**: The correlation of the L-function and r=[1,d].<br>**L-function at d/2**: The value of the L-function evaluated at ¼ of the maximum cell diameter.   *The L-function measures spatial clustering of a point pattern over an area of interest.*                                                                                                                                   
</details>
<br>
