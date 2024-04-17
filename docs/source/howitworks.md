```{toctree}
:hidden: true
```

# {octicon}`gear` How it Works

## Data Structure

Datasets are stored as `AnnData` objects, where observations are cells, variables are genes, and the X is the count matrix. Bento additionally stores molecular coordinates in `uns['points']` and polygons as columns in `obs`. 

```{figure}  _static/tutorial_img/bento_data_structure.png
:class: p-2

AnnData adapted to hold spatial data
``` 

### Shapes

Currently, shapes are stored as `GeoSeries` columns according to which cell they belong to. These columns are identified with the suffix `"_shape"`. Each element in the `GeoSeries` is either a shapely `Polygon` or `MultiPolygon`. Shape properties are also stored as columns and identified with the corresponding shape name as the prefix e.g. `"cell"`, `"nucleus"`, etc.

### Points
For fast spatial queries, Bento indexes points to shape layers upfront, and saves them as columns `points`, denoted as `"shape index"` above. For example, `"cell"` and `"nucleus"` columns are added to indicate whether points are within the shape.

Metadata for points are stored as matrices `uns`. These metadata matrices are the same length as `points`, which makes it easy to query points and associated metadata. All metadata keys are registered to `uns['points_metadata']`, which is used to keep them in sync.

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
