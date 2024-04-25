```{toctree}
:hidden: true
:maxdepth: 2
```

```{eval-rst}
.. module:: bento
```


# {octicon}`code-square` API

Import Bento with:

```{eval-rst}
.. code-block:: python

    import bento as bt
```


Bento's API structure takes inspiration from other libraries in the Scverse ecosystem. It is organized under a set of modules including:

- `bt.io`: provides out of the box compatibility with `SpatialData` objects
- `bt.tl`: subcellular analysis tools
- `bt.pl`: conveniently plot spatial data and embeddings
- `bt.geo`: manipulating data structures
- `bt.datasets`: included spatial transcriptomics datasets `WIP`

## Read and Write

Bento is designed to work with [`SpatialData`](https://spatialdata.scverse.org/en/latest/) objects out of the box! Check out [SpatialData documentation](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks.html) to learn how to bring your own data, whether it is from commercial platforms or a custom data format. 

```{eval-rst}
.. currentmodule:: bento.io

.. autosummary::
    :toctree: api
    :nosignatures:

    prep
```

## Tools

Bento provides a set of tools to analyze the spatial composition of cells and molecules. These tools are designed to work with `SpatialData` objects and can be used to compute spatial summary statistics, shape features, and RNA localization patterns.

### Composition


```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    comp
    comp_diff
```

### Point Features

Compute spatial summary statistics describing groups of molecules e.g. distance to the cell membrane, relative symmetry, dispersion, etc. The set of available point features is described in the Point Feature Catalog. Use the function `bt.tl.analyze_points()` to compute features and add your own custom calculation. See the [tutorial](https://bento-tools.github.io/bento/tutorials/TBD.html) for more information.

```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    analyze_points
    list_point_features
    register_point_feature
```

#### Point Feature Catalog

The set of implemented point features is described below. Each feature is computed using the `bt.tl.analyze_points()` function.

TODO

### Shape Features

Compute spatial properties of shape features e.g. area, aspect ratio, etc. of the cell, nucleus, or other region of interest. The set of available shape features is described in the Shape Feature Catalog. Use the function `bt.analyze_points()` to compute features and add your own custom calculation. See the [tutorial](https://bento-tools.github.io/bento/tutorials/TBD.html) for more information.

```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    shape_stats
    analyze_shapes
    list_shape_features
    register_shape_feature
```

#### Shape Feature Catalog

The set of implemented shape features is described below. Each feature is computed using the `bt.analyze_shapes()` function.

```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    area
    aspect_ratio
    bounds
    density
    opening
    perimeter
    radius
    raster
    second_moment
    span

```

### RNAflux: Subcellular RNA embeddings and domains

Methods for computing RNAflux embeddings and semantic segmentation of subcellular domains.

```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    flux
    fluxmap
    fe
    fe_fazal2019
    fe_xia2019
    load_gene_sets
```

### RNAforest: Predict RNA localization patterns

Perform multilabel classification of RNA localization patterns using spatial summary statistics as features.

```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    lp
    lp_stats
    lp_diff_discrete
    lp_diff_continuous
```

### RNAcoloc: Colocalization analysis

Methods for compartments-ecific gene-gene colocalization analyses.

```{eval-rst}
.. currentmodule:: bento.tl

.. autosummary::
    :toctree: api
    :nosignatures:

    colocation
    coloc_quotient
```

## Plotting

These are convenient functions for static 2D plots of cells, molecules, and embeddings. We generate `matplotlib` style figures for accessible publication quality plots. There are a couple additional functions summarizing results from `bt.tl` analysis.

### Spatial plots

```{eval-rst}
.. currentmodule:: bento.pl

.. autosummary::
    :toctree: api
    :nosignatures:

    points
    density
    shapes
```

### Composition

```{eval-rst}
.. currentmodule:: bento.pl

.. autosummary::
    :toctree: api
    :nosignatures:

    comp
```


### Shape features

```{eval-rst}
.. currentmodule:: bento.pl

.. autosummary::
    :toctree: api
    :nosignatures:

    shape_stats
```

### RNAflux

```{eval-rst}
.. currentmodule:: bento.pl

.. autosummary::
    :toctree: api
    :nosignatures:

    flux
    fluxmap
    fe
```

### RNAforest

```{eval-rst}
.. currentmodule:: bento.pl

.. autosummary::
    :toctree: api
    :nosignatures:

    lp_dist
    lp_genes
    lp_diff_discrete
```

### Colocalization analysis

```{eval-rst}
.. currentmodule:: bento.pl

.. autosummary::
    :toctree: api
    :nosignatures:

    factor
    colocation
```

## Utilities for data manipulation

Convenient methods for setting, getting, and reformatting data. These functions are used internally by other functions in Bento.

```{eval-rst}
.. currentmodule:: bento.ut

.. autosummary::
    :toctree: api
    :nosignatures:

    get_points
    get_points_metadata
    set_points_metadata
    
    get_shape
    get_shape_metadata
    set_shape_metadata

```

## Geometric operations

```{eval-rst}
.. currentmodule:: bento.geo

.. autosummary::
    :toctree: api
    :nosignatures:
    
    overlay