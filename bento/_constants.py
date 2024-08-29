from enum import Enum

PATTERN_COLORS = ["#17becf", "#1f77b4", "#7f7f7f", "#ff7f0e", "#d62728"]
PATTERN_NAMES = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]
PATTERN_PROBS = [f"{p}_p" for p in PATTERN_NAMES]
PATTERN_THRESHOLDS_CALIB = [0.45300, 0.43400, 0.37900, 0.43700, 0.50500]


class CosMx(Enum):
    """CosMx microscope constants"""

    # https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/SMI-ReadMe.html
    PIXEL_MICRONS = 0.18


class Merscope(Enum):
    """Merscope microscope constants"""

    # https://vizgen.com/wp-content/uploads/2023/06/91600001_MERSCOPE-Instrument-User-Guide_Rev-G.pdf
    PIXEL_MICRONS = 1


class Xenium(Enum):
    """Xenium microscope constants"""

    # https://www.10xgenomics.com/instruments/xenium-analyzer
    PIXEL_MICRONS = 0.2125
