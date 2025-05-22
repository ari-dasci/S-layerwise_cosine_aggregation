# Layerwise Cosine Aggregation Schema

This repository includes the source code for the paper "Improving alpha-f
Byzantine Resilience in Federated Learning via layerwise aggregation and cosine
distance".

Find preprint [here](https://arxiv.org/abs/2503.21244).

## Installation

This project uses Python dependencies. To install them using
[uv](https://github.com/astral-sh/uv), run:

```bash
uv sync
```

Now, you can execute the code with

```bash
uv run layerwise_krum.py
```

Run the following command for getting more information about how to configure
the experiments:

```bash
uv run layerwise_krum.py --help
```

**Note**: The code requires you to have a CUDA device. Feel free to change the
assert at the top of the file (after imports).
