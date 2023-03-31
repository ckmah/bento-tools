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

Bento's API structure takes inspiration from other libraries in the Scverse ecosystem. It is organized under a set of modules including:

- `bento.tl`: subcellular spatial analyses
- `bento.pl`: conveniently plot spatial data and embeddings
- `bento.io`: reading and writing spatial data to `AnnData` as `h5ad` files
- `bento.geo`: functions for manipulating spatial data
- `bento.datasets`: included spatial transcriptomics datasets
- `bento.ut`: utility functions

# Tools

## Point features

Compute spatial summary statistics describing groups of molecules e.g. distance to the cell membrane, relative symmetry, dispersion, etc.

A list of available cell features and their names is stored in the dict `bento.tl.point_features`.

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.analyze_points
    tl.register_point_feature
```

## Shape features

Compute spatial properties of shape features e.g. area, aspect ratio, etc. of the cell, nucleus, or other region of interest.

A list of available cell features and their names is stored in the dict `bento.tl.shape_features`.

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.analyze_shapes
    tl.register_shape_feature
    tl.obs_stats
```

## RNAflux: Subcellular RNA embeddings and domains

Methods for computing RNAflux embeddings and semantic segmentation of subcellular domains.

```{eval-rst}
.. module: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.flux
    tl.fluxmap
    tl.fe
    tl.fe_fazal2019
    tl.load_gene_sets
```

## RNAforest: Predict RNA localization patterns

Perform multilabel classification of RNA localization patterns using spatial summary statistics as features.

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.lp
    tl.lp_stats
    tl.lp_diff
```

## Colocalization analysis

Methods for colocalization analyses of gene pairs.

```{eval-rst}
.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    tl.colocation
    tl.coloc_quotient
```

# Plotting

## Spatial plots

These are convenient functions for quick 2D visualizations of cells, molecules, and embeddings. We generate `matplotlib` style figures for accessible publication quality plots.

```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    pl.points
    pl.density
    pl.shapes
    pl.flux
    pl.fluxmap

```

## Shape features

```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    pl.obs_stats
```

## RNAflux

```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    pl.flux_summary
    pl.fe
```

## RNAforest

```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    pl.lp_genes
    pl.lp_gene_dist
    pl.lp_dist
    pl.lp_diff

```

## Colocalization analysis

```{eval-rst}
.. module:: bento.pl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    pl.signatures
    pl.signatures_error
    pl.factor
    pl.colocation

```

# Manipulating spatial data

Convenient methods for setting, getting, and reformatting data for `bento-tools`.

```{eval-rst}
.. module:: bento.geo
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    geo.count_points
    geo.crop
    geo.get_points
    geo.get_points_metadata
    geo.get_shape
    geo.rename_shapes
    geo.sindex_points
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
    io.prepare
```

# Datasets

```{eval-rst}
.. module:: bento.ds
.. currentmodule:: bento

.. autosummary::
    :toctree: api/

    ds.sample_data
    ds.load_dataset
    ds.get_dataset_info

```
