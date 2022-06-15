# How it Works

## Data Structure

More info coming soon!

## Localization Patterns

More info coming soon!

### Spatial Features
    
| **Categories** | **Features** |
| -------------- | ------------ |
| Distance       | **Cell inner proximity**: The average distance between all points within the cell to the cell boundary normalized by cell radius. Values closer to 0 denote farther from the cell boundary, values closer to 1 denote closer to the cell boundary.<br>**Nucleus inner proximity**: The average distance between all points within the nucleus to the nucleus boundary normalized by cell radius. Values closer to 0 denote farther from the nucleus boundary, values closer to 1 denote closer to the nucleus boundary.<br>**Nucleus outer proximity**: The average distance between all points within the cell and outside the nucleus to the nucleus boundary normalized by cell radius. Values closer to 0 denote farther from the nucleus boundary, values closer to 1 denote closer to the nucleus boundary. |
| Symmetry       | **Cell inner asymmetry**: The offset between the centroid of all points within the cell to the centroid of the cell boundary, normalized by cell radius. Values closer to 0 denote symmetry, values closer to 1 denote asymmetry.<br>**Nucleus inner asymmetry**: The offset between the centroid of all points within the nucleus to the centroid of the nucleus boundary, normalized by cell radius. Values closer to 0 denote symmetry, values closer to 1 denote asymmetry.<br>**Nucleus outer asymmetry**: The offset between the centroid of all points within the cell and outside the nucleus to the centroid of the nucleus boundary, normalized by cell radius. Values closer to 0 denote symmetry, values closer to 1 denote asymmetry.                                                                |
| Dispersion     | **Point dispersion**: The second moment of all points in a cell relative to the centroid of the total RNA signal. This value is normalized by the second moment of a uniform distribution within the cell boundary.<br>**Nucleus dispersion**: The second moment of all points in a cell relative to the centroid of the nucleus boundary. This value is normalized by the second moment of a uniform distribution within the cell boundary.                                                                                                                                                                                                                                                                                                                                                                    |
| Density        | **L-function max**: The max value of the L-function evaluated at r=[1,d], where d is half the cell’s maximum diameter.<br>**L-function max gradient**: The max value of the gradient of the above L-function.<br>**L-function min gradient**: The min value of the gradient of the above L-function.<br>**L monotony**: The correlation of the L-function and r=[1,d].<br>**L-function at d/2**: The value of the L-function evaluated at ¼ of the maximum cell diameter.   *The L-function measures spatial clustering of a point pattern over an area of interest.*                                                                                                                                   
</details>
<br>