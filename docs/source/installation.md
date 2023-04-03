# {octicon}`terminal` Installation 
Bento requires Python version 3.8 or 3.9. Only Python 3.8 is supported for Windows.


## Setup a virtual environment
We highly recommend using a virtual environment to install Bento to avoid conflicting dependencies with other packages. If you are unfamiliar with virtual environments, we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To setup a virtual environment with `Miniconda`, run the following. This will create and activate a new environment called `bento` with Python 3.8.

```bash
VERSION=3.8 # or 3.9
conda create -n bento python=$VERSION

# set channel priorities
conda config --env --add channels defaults
conda config --env --add channels bioconda
conda config --env --add channels conda-forge

conda activate bento
```
## 2. Install Bento

We recommend installing the Bento conda package from the [TBD channel](TBD). This will install all dependencies and Bento.

### Install with conda

```bash
conda install -c TBD bento-tools
```

### Install with pip

You can also install Bento with pip but you will need to install dependencies manually. See the [dependencies](#Dependencies) section for more details.

```bash
pip install bento-tools
```

## Dependencies

Bento makes use of several packages for spatial analyses that require addtional non-Python dependencies. These dependencies are automatically installed when using the conda package. If you are installing with pip, you will need to install these dependencies manually.

1. [gdal](https://gdal.org/) (see docs for [geopandas dependencies](https://geopandas.org/en/stable/getting_started/install.html#dependencies))
2. [cmake](https://cmake.org/) ([xgboost](https://xgboost.readthedocs.io/en/latest/install.html))
3. [pandoc](https://pandoc.org/installing.html) (for sphinx documentation)

---
## GPU Support (Optional)
Bento currently only uses GPUs to accelerate tensor decomposition (via [Tensorly](https://tensorly.org/stable/index.html)) GPU support can be enabled by installing PyTorch. We recommend the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for more details as installation varies by platform.


## Development
The package and its dependencies are built using [Poetry](https://python-poetry.org/).

1. Install [Poetry](https://python-poetry.org/).
2. Clone the `bento-tools` GitHub repository.
3. Use poetry to manage dependencies and pip to install the package in editable mode.

    ```bash
    cd bento-tools
    poetry install
    pip install -e .
    ```

    For updating documentation locally, install extra dependencies and launch a live server to preview doc builds:

    ```bash
    poetry install --extras "docs"
    cd docs
    make livehtml # See output for URL
    ```
