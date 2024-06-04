import pytest
import spatialdata as sd

import bento as bt

TEST_ZARR = "small_data.zarr"
SIX_CELL_ZARR = "six_cell_data.zarr"
CELL_TO_NUCLEUS_MAP = {
    "c0": "",
    "c1": "n4",
    "c2": "n6",
    "c3": "",
    "c4": "n0",
    "c5": "",
}
NUCLEUS_TO_CELL_MAP = {"n0": "c4", "n4": "c1", "n6": "c2"}

FLUX_RES = 0.1
FLUX_RADIUS = 0.5
FLUXMAP_MIN_COUNT = 5
FLUXMAP_TRAIN_SIZE = 1
FLUXMAP_N_CLUSTERS = 3
FAZAL2019_FEATURES = [
    "Cytosol",
    "ER Lumen",
    "ERM",
    "Lamina",
    "Nuclear Pore",
    "Nucleolus",
    "Nucleus",
    "OMM",
]
XIA2019_FEATURES = ["ER", "Nucleus"]

SHAPE_FEATURES = bt.tl.list_shape_features().keys()
SHAPE_FEATURE_NAMES = [
    "area",
    "aspect_ratio",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "density",
    "open_0.5_shape",
    "perimeter",
    "radius",
    "raster",
    "moment",
    "span",
]
CELL_FEATURES = [f"cell_boundaries_{x}" for x in SHAPE_FEATURE_NAMES]
NUCLEUS_FEATURES = [f"nucleus_boundaries_{x}" for x in SHAPE_FEATURE_NAMES]
OPENING_PARAMS = {"opening": {"proportion": 0.5}}

POINT_FEATURES = bt.tl.list_point_features().keys()
POINT_ONLY_FEATURE_NAMES = [
    "point_dispersion_norm",
    "point_dispersion",
    "l_max",
    "l_max_gradient",
    "l_min_gradient",
    "l_monotony",
    "l_half_radius",
]
POINT_FEATURE_NAMES = [
    "inner_proximity",
    "outer_proximity",
    "inner_asymmetry",
    "outer_asymmetry",
    "dispersion_norm",
    "inner_distance",
    "outer_distance",
    "inner_offset",
    "outer_offset",
    "dispersion",
    "enrichment",
]
POINT_CELL_FEATURE_NAMES = [f"cell_boundaries_{x}" for x in POINT_FEATURE_NAMES]
POINT_NUCLEUS_FEATURE_NAMES = [f"nucleus_boundaries_{x}" for x in POINT_FEATURE_NAMES]

LP_COLUMNS = [
    "cell_boundaries",
    "feature_name",
    "cell_edge",
    "cytoplasmic",
    "none",
    "nuclear",
    "nuclear_edge",
]
LP_STATS_COLUMNS = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]

LP_DIFF_DISCRETE_COLUMNS = [
    "feature_name",
    "pattern",
    "phenotype",
    "dy/dx",
    "std_err",
    "z",
    "pvalue",
    "ci_low",
    "ci_high",
    "padj",
    "-log10p",
    "-log10padj",
    "log2fc",  
]

LP_DIFF_CONTINUOUS_COLUMNS = [
    "feature_name",
    "pattern",
    "pearson_correlation",
]

@pytest.fixture(scope="session")
def small_data():
    data = sd.read_zarr(bt.__file__.rsplit("/", 1)[0] + "/datasets/" + TEST_ZARR)
    data = bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
    )
    return data
