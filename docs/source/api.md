
.. module:: bento

.. automodule:: bento
   :noindex:

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

# Cell Features
Spatial properties of cells. These include properties of shape(s) and points within each cell.
A list of available functions, call bento.tl.cell_features.

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.cell_features

```

# Sample Features
```{eval-rst}
.. autosummary::
    :toctree: api/

    tl.analyze
    tl.PointDispersion
    tl.RipleyStats
    tl.ShapeAsymmetry
    tl.ShapeDispersion
    tl.ShapeEnrichment
    tl.ShapeProximity    
```

# Localization Patterns
```{eval-rst}

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

    pl.lp_dist
    pl.lp_genes
    pl.lp_gene_dist
    pl.lp_diff
    pl.plot_cells
    pl.qc_metrics
    pl.lp_signatures
```


Read/Write
-----------

.. module:: bento.io
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
   
    io.read_h5ad
    io.write_h5ad
    io.concatenate