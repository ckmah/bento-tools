# {octicon}`terminal` Installation

Bento requires Python version 3.8 or 3.9.

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

## 2. Dependencies

Bento makes use of several packages for spatial analyses that require addtional non-Python dependencies.

```bash
conda install -c conda-forge gdal cmake
```

For developing docs, you will also need pandoc:

```bash
conda install -c conda-forge pandoc
```

## 3. Install Bento

All that's left is the package itself. Install with pip:

```bash
pip install bento-tools
```

---

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

   First install pandoc (see [dependencies](#Dependencies) section for more details).

   ```bash
   poetry install --extras "docs"
   pip install -e .\[docs\]
   cd docs
   make livehtml # See output for URL
   ```

---

## GPU Support (Optional)

Bento currently only uses GPUs to accelerate tensor decomposition (via [Tensorly](https://tensorly.org/stable/index.html)) GPU support can be enabled by installing PyTorch. We recommend the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for more details as installation varies by platform.
