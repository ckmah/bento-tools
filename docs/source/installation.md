# Installation
Bento requires Python version 3.8 or 3.9. Only Python 3.8 is supported for installation Windows.

Install Bento with pip.

```bash
pip install bento-tools
```

To enable GPU usage, run the following:

```bash
pip install bento-tools[torch]
```

## Development
The package and its dependencies are built using [Poetry](https://python-poetry.org/).

1. Install [Poetry](https://python-poetry.org/).
2. Clone the `bento-tools` GitHub repository.
3. Use poetry to setup the virtual environment.

    ```bash
    cd bento-tools
    poetry shell
    poetry install
    ```
4. You have a couple options on what dependencies to install depending on what you want to modify.
    - For developing Bento, install normal package dependencies:

        ```bash
        poetry install
        
        # or with gpu enabled
        poetry install --extras "torch"
        ```
    - To modify and build documentation, install extra dependencies:
    
        ```bash
        poetry install --extras "docs"
        ```

5. Launch a local live server to preview doc builds:

    ```bash
    cd docs
    make livehtml # See output for URL
    ```
