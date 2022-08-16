from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

blue_rgb = np.array([30, 136, 229]) / 255
red_rgb = np.array([255, 13, 87]) / 255
red2blue = LinearSegmentedColormap.from_list("red2blue", [blue_rgb, red_rgb])

red_light = sns.light_palette(red_rgb, as_cmap=True)
blue_light = sns.light_palette(blue_rgb, as_cmap=True)
