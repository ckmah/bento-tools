```{toctree}
:hidden: true
```

# {octicon}`terminal` Installation

Bento requires Python version 3.9 or higher. We recommend using a virtual environment to install Bento to avoid conflicting dependencies with other packages. If you are unfamiliar with virtual environments, we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## PyPI

Bento is available on PyPI:

```console
$ pip install bento-tools
```

## Troubleshooting
If you encounter installation errors, you may need double check that package dependencies are installed correctly.

- [GeoPandas](https://geopandas.org/en/stable/getting_started/install.html#installing-with-pip) install instructions
## Development

We currently use [Rye](https://rye-up.com/) for package management. 

1. Install [Rye](https://rye-up.com/guide/installation/).
2. Clone the repository and navigate to the root directory.
   ```console
   gh repo clone ckmah/bento-tools
   cd bento-tools
   ```

3. Install the package in editable mode:

   ```console
      pip install -e .
   ```

Run tests:
   ```console
      python -m unittest tests

### Documentation

   ```console
      pip install -e ".[docs]" # install optional deps for docs
      cd docs
      make livehtml
   ```


## GPU Support (Optional)

   ```console
      pip install -e ".[torch]"
   ```


Bento currently only uses GPUs to accelerate tensor decomposition (via [Tensorly](https://tensorly.org/stable/index.html)). PyTorch can be installed with the `torch` extra, but users should consult [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for enabling GPU support, as installation varies by platform.
