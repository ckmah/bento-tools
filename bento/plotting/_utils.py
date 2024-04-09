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
