# codif-ape-llm-synthetic-data
Testing a ML model using synthetic data generated with LLMs.

## Setup

Install [`uv`](https://github.com/astral-sh/uv) using for instance

```bash
pip install uv
```

Then install the dependencies using:

```bash
uv sync
```

The script `.setup.sh` does exactly that so feel free to run
```bash
. .setup.sh
```

from the root of the project.

## Launch

This project uses [`hydra`](https://hydra.cc/docs/intro/). To run use

```bash
uv run main.py bias_type='"Genre & Nombre"' temperature=0.1,0.8
```

Take care of the quoting for the `bias_type` parameter to avoid any error.
