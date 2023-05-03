PATTERN_COLORS = ["#17becf", "#1f77b4", "#7f7f7f", "#ff7f0e", "#d62728"]
PATTERN_NAMES = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]
PATTERN_PROBS = [f"{p}_p" for p in PATTERN_NAMES]
PATTERN_FEATURES = [
    "cell_inner_proximity",
    "nucleus_inner_proximity",
    "nucleus_outer_proximity",
    "cell_inner_asymmetry",
    "nucleus_inner_asymmetry",
    "nucleus_outer_asymmetry",
    "l_max",
    "l_max_gradient",
    "l_min_gradient",
    "l_monotony",
    "l_half_radius",
    "point_dispersion_norm",
    "nucleus_dispersion_norm",
]
