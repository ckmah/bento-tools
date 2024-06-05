import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

blue_rgb = np.array([30, 136, 229]) / 255
red_rgb = np.array([255, 13, 87]) / 255
red2blue = LinearSegmentedColormap.from_list("red2blue", [blue_rgb, "white", red_rgb])
red2blue_dark = LinearSegmentedColormap.from_list(
    "red2blue_dark", [blue_rgb, "black", red_rgb]
)

mpl.colormaps.register(red2blue, name="red2blue")
mpl.colormaps.register(red2blue_dark, name="red2blue_dark")


red_light = sns.light_palette(red_rgb, as_cmap=True)
blue_light = sns.light_palette(blue_rgb, as_cmap=True)

red_dark = sns.dark_palette(red_rgb, as_cmap=True)
red_dark = LinearSegmentedColormap.from_list("red_dark", [[0, 0, 0], red_rgb])
blue_dark = LinearSegmentedColormap.from_list("blue_dark", [[0, 0, 0], blue_rgb])

bento6 = sns.color_palette(
    [
        "#FFD166",
        "#06D6A0",
        "#118ab2",
        "#396270",
        "#f78c6b",
        "#ef476f",
    ],
    as_cmap=True,
)
