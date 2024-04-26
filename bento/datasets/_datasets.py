from importlib.resources import as_file, files

import spatialdata as sd
from ..io import prep

# def get_dataset_info():
#     """Return DataFrame with info about builtin datasets.

#     Returns
#     -------
#     DataFrame
#         Info about builtin datasets indexed by dataset name.
#     """
#     global pkg_resources
#     if pkg_resources is None:
#         import pkg_resources

#     stream = pkg_resources.resource_stream(__name__, "datasets.csv")
#     return pd.read_csv(stream, index_col=0)


# def load_dataset(name, cache=True, data_home="~/bento-data", **kws):
#     """Load a builtin dataset.

#     Parameters
#     ----------
#     name : str
#         Name of the dataset to load.
#     cache : bool (default: True)
#         If True, try to load from local cache first, download as needed.
#     data_home : str (default: "~/bento-data")
#         Path to directory where datasets are stored.
#     **kws
#         Keyword arguments passed to :func:`bento.io.read_h5ad`.
#     """
#     datainfo = get_dataset_info()

#     # Check if dataset name exists
#     if name not in datainfo.index:
#         raise KeyError(
#             f"No builtin dataset named '{name}'. Use :func:`bento.ds.get_dataset_info` to list info about available datasets."
#         )

#     # Sanitize user path
#     data_home = os.path.expanduser(data_home)

#     # Make data folder if it doesn't exist
#     if not os.path.exists(data_home):
#         os.makedirs(data_home)

#     # Try to load from local cache first, download as needed
#     url = datainfo.loc[name, "url"]
#     cache_path = os.path.join(data_home, os.path.basename(url))
#     if cache:
#         if not os.path.exists(cache_path):
#             _download(url, cache_path)
#     else:
#         _download(url, cache_path)

#     adata = read_h5ad(cache_path, **kws)

#     return adata


# # Taken from https://github.com/theislab/scanpy/blob/master/scanpy/readwrite.py
# def _download(url, path):
#     try:
#         import ipywidgets
#         from tqdm.auto import tqdm
#     except ImportError:
#         from tqdm import tqdm

#     from pathlib import Path
#     from urllib.request import Request, urlopen

#     blocksize = 1024 * 8
#     blocknum = 0

#     path = Path(path)

#     try:
#         with urlopen(Request(url, headers={"User-agent": "bento"})) as resp:
#             total = resp.info().get("content-length", None)
#             with tqdm(
#                 unit="B",
#                 unit_scale=True,
#                 miniters=1,
#                 unit_divisor=1024,
#                 total=total if total is None else int(total),
#             ) as t, path.open("wb") as f:
#                 block = resp.read(blocksize)
#                 while block:
#                     f.write(block)
#                     blocknum += 1
#                     t.update(len(block))
#                     block = resp.read(blocksize)

#     except (KeyboardInterrupt, Exception):
#         # Make sure file doesn’t exist half-downloaded
#         if path.is_file():
#             path.unlink()
#         raise


def sample_data():
    ref = files(__package__) / "merfish_sample.zarr"
    with as_file(ref) as path:
        sdata = sd.read_zarr(path)
        sdata = prep(
            sdata,
            points_key="transcripts",
            instance_key="cell_boundaries",
            feature_key="feature_name",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )
        return sdata
