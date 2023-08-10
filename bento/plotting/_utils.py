import inspect
from functools import wraps
from typing import Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, quantile_transform


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def savefig(plot_fn):
    """
    Save figure from plotting function.
    """

    @wraps(plot_fn)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(plot_fn)
        kwargs.update(kwds)

        plot_fn(*args, **kwds)

        fname = kwargs["fname"]
        rc = {
            "image.interpolation": "none",
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Verdana", "DejaVu Sans"],
            "pdf.fonttype": 42,
        }
        if fname:
            with plt.rc_context(rc):
                plt.savefig(fname, dpi=400, pad_inches=0, bbox_inches="tight")

            print(f"Saved to {fname}")

    return wrapper


def vec2color(
    vec: np.ndarray,
    alpha_vec: Optional[np.ndarray] = None,
    fmt: Literal[
        "rgb",
        "hex",
    ] = "hex",
    vmin: float = 0,
    vmax: float = 1,
):
    """Convert vector to color."""

    # Grab the first 3 channels
    color = vec[:, :3]
    color = quantile_transform(color[:,:3])
    color = minmax_scale(color, feature_range=(vmin, vmax))

    # If vec has fewer than 3 channels, fill empty channels with 0
    if color.shape[1] < 3:
        color = np.pad(color, ((0, 0), (0, 3 - color.shape[1])), constant_values=0)

    
    # Add alpha channel
    if alpha_vec is not None:
        alpha = alpha_vec.reshape(-1, 1)
        # alpha = quantile_transform(alpha)
        alpha = alpha / alpha.max()
        color = np.c_[color, alpha]

    if fmt == "rgb":
        pass
    elif fmt == "hex":
        color = np.apply_along_axis(mpl.colors.to_hex, 1, color, keep_alpha=True)
    return color