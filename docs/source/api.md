
# API Reference
 
Import Bento with:

```python
import bento
```

## Subcellular Localization

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api
    
    tl.lp
    tl.lp_stats
    tl.lp_diff
    tl.loc_signatures
    tl.decompose_tensor
    tl.select_tensor_rank   
```

## Colocalization

```{eval-rst}
.. autosummary::
    :toctree: api
    
    tl.coloc_quotient
```

## Sample features

```{eval-rst}
.. autosummary::
    :toctree: api

    tl.analyze
    tl.PointDispersion
    tl.RipleyStats
    tl.ShapeAsymmetry
    tl.ShapeDispersion
    tl.ShapeEnrichment
    tl.ShapeProximity    
```

## Plotting

```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api

    pl.lp_dist
    pl.lp_genes
    pl.lp_gene_dist
    pl.lp_diff
    pl.plot_cells
    pl.qc_metrics
    pl.lp_signatures

```


## Read/Write

```{eval-rst}
.. module:: bento.io
.. currentmodule:: bento

.. autosummary::
    :toctree: api
    
    io.read_h5ad
    io.write_h5ad
    io.concatenate
```