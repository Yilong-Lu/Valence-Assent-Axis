# Figure Notebooks

Each notebook is generated from the corresponding Python script
under `analysis/figures/`. Edit the script for reproducible changes, then run:

```bash
python -m analysis.figures.sync_notebooks
```

The notebooks intentionally contain no cached outputs, absolute paths, API
calls, model loading, or model-generation steps.
