# Installation and Compute

## Python

Python 3.10 or newer is required. Create an isolated environment and install
the repository in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
```

PyTorch wheels are platform-specific. For GPU experiments, install the PyTorch
build appropriate for the local CUDA driver before the editable install. The
tested package versions are recorded in `requirements/constraints-tested.txt`.

## R

The statistical analyses require R and the packages `lme4`, `lmerTest`,
`emmeans`, `nlme`, and `jsonlite`. The analyses were rerun with R 4.5.3;
exact package versions are listed in `requirements/r-tested.txt`.

## Model Access

By default, the model IDs in `configs/models.yaml` are passed to Transformers.
For an existing local download, set the corresponding variable from
`.env.example`, for example:

```bash
export VAA_MODEL_LLAMA3_8B=/models/Llama-3.1-8B-Instruct
```

The runners do not contain authentication tokens. Access-controlled models
must be obtained separately under the provider's terms.

## Hardware

All configuration validation, statistics, result checks, and figure generation
run on CPU. Model inference requires accelerator memory for the model weights,
activations, and generation cache. As a rough lower bound, bfloat16 weights
alone require about 2 GB per billion parameters; runtime memory is higher.
Models up to 9B were designed for a 24 GB single-GPU environment. The 14B,
32B, and 72B registrations normally require larger-memory or multi-GPU setups
with `device_map=auto`.

Runtime depends strongly on model size, sequence length, hardware, and task.
The largest generation workload is the Preference-Induced Sycophancy intervention
run: 100 arguments x 3 preference conditions x 11 alpha levels, or 3,300
generations per model. Use the documented `--max-items` and `--alpha-values`
options for an environment check before launching a complete run.
