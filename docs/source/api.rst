
.. module:: bento

.. automodule:: bento
   :noindex:

API
===

Import Bento with:

.. code:: python

    import bento

Localization Patterns
---------------------

.. module:: bento.tl
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
    
    tl.lp
    tl.lp_stats
    tl.lp_diff
    tl.loc_signatures
    tl.decompose_tensor
    tl.select_tensor_rank   

Colocalization
--------------

.. autosummary::
    :toctree: api/
    
    tl.coloc_quotient

Sample Features
----------------

.. autosummary::
    :toctree: api/

    tl.analyze
    tl.PointDispersion
    tl.RipleyStats
    tl.ShapeAsymmetry
    tl.ShapeDispersion
    tl.ShapeEnrichment
    tl.ShapeProximity    

Plotting
---------
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



Read/Write
-----------

.. module:: bento.io
.. currentmodule:: bento

.. autosummary::
    :toctree: api/
   
    io.read_h5ad
    io.write_h5ad
    io.concatenate