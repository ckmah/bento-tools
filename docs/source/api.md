```{eval-rst}
.. module:: bento

.. automodule:: bento
   :noindex:
```

# API

Import Bento with:

```python
import bento
```

Bento takes inspiration from the success of other tools in the python single-cell omics ecosystem. The API is organized under a set of modules including:

- `bento.pp`: preprocessing data
- `bento.tl`: subcellular spatial analyses
- `bento.pl`: visualizing molecule-resolved spatial data, localization pattern statistics, and more
- `bento.io`: reading and writing data stored in the `AnnData` format, customized to hold molecular coordinates and segmentation masks

# Preprocessing
```{eval-rst}
.. module:: bento.pp
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
    
    pp.get_points
    pp.set_points
    pp.get_layers
```

# Sample Features
Compute spatial properties of samples. A sample is the set of points for a given gene-pair. These include properties of shape(s) and points associated with each cell.

A list of available cell features and their names is stored in the dict `bento.tl.sample_features`.


```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.analyze_samples
    tl.PointDispersion
    tl.RipleyStats
    tl.ShapeAsymmetry
    tl.ShapeDispersion
    tl.ShapeEnrichment
    tl.ShapeProximity    
```


# Cell Features
Compute spatial properties of cells. These include properties of shape(s) and points associated with each cell.

A list of available cell features and their names is stored in the dict `bento.tl.cell_features`.

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.analyze_cells
    tl.cell_area
    tl.cell_aspect_ratio
    tl.cell_bounds
    tl.cell_density
    tl.cell_moments
    tl.cell_morph_open
    tl.cell_perimeter
    tl.cell_radius
    tl.cell_span
    tl.is_nuclear
    tl.nucleus_area
    tl.nucleus_area_ratio
    tl.nucleus_aspect_ratio
    tl.nucleus_offset
    tl.raster_cell

```

# Localization Patterns
```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
    
    tl.lp
    tl.lp_stats
    tl.lp_diff
    tl.lp_signatures
    tl.decompose_tensor
    tl.select_tensor_rank   
```

# Colocalization
```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
    
    tl.coloc_quotient
```



# Plotting
```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    pl.qc_metrics
    pl.cellplot
    pl.lp_genes
    pl.lp_gene_dist
    pl.lp_signatures
    pl.lp_dist
    pl.lp_diff
```


# Read/Write
```{eval-rst}
.. module:: bento.io
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
   
    io.read_h5ad
    io.write_h5ad
    io.concatenate
```

# Datasets
```{eval-rst}
.. module:: bento.datasets
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
   
    datasets.load_dataset
```