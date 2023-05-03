from ._colocation import coloc_quotient, colocation
from ._composition import comp_diff
from ._flux import flux, fluxmap
from ._flux_enrichment import fe, fe_fazal2019, gene_sets, load_gene_sets
from ._lp import lp, lp_diff, lp_stats
from ._point_features import analyze_points, list_point_features, register_point_feature
from ._shape_features import (
    analyze_shapes,
    obs_stats,
    register_shape_feature,
    list_shape_features,
)
from ._decomposition import decompose, to_tensor
