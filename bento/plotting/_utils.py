import inspect
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import (
    Collection,
)
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib_scalebar.scalebar import ScaleBar


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def savefig(plot_func):
    """
    Save figure from plotting function.
    """

    @wraps(plot_func)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(plot_func)
        kwargs.update(kwds)

        plot_func(*args, **kwds)

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


def setup_ax(plot_func):
    @wraps(plot_func)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(plot_func)
        kwargs.update(kwds)
        ax = kwargs.get("ax")

        if ax is None:
            ax = plt.gca()

        plot_func(*args, **kwds)

        # Infer font color from theme
        edgecolor = sns.axes_style()["axes.edgecolor"]

        scalebar = ScaleBar(
            dx=kwargs.get("dx"),
            units=kwargs.get("units"),
            location="lower right",
            box_alpha=0,
            color=edgecolor,
            frameon=False,
            scale_loc="top",
        )
        ax.add_artist(scalebar)

        ax_kws = dict(aspect=1, box_aspect=None)

        if not kwargs.get("axis_visible"):
            ax_kws.update(
                dict(
                    xticks=[],
                    yticks=[],
                    xticklabels=[],
                    yticklabels=[],
                    ylabel=None,
                    xlabel=None,
                    xmargin=0.01,
                    ymargin=0.01,
                )
            )

        if kwargs["square"]:
            ax_kws["box_aspect"] = 1

        # Update ax_kws with keys in kwds only if they exist in ax_kws
        ax_kws.update((k, v) for k, v in kwds.items() if k in ax_kws)

        plt.setp(ax, **ax_kws)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(
            kwargs.get("frame_visible")
        )

        ax.set_title(kwargs.get("title", ""), color=edgecolor)

        return ax

    return wrapper


def polytopatch(poly):
    exterior = Path(np.asarray(poly.exterior.coords)[:, :2])
    interiors = [Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
    path = Path.make_compound_path(exterior, *interiors)
    return PathPatch(path)
