# Contributing

This repository is primarily a reproducibility record for a published study.
Bug reports, portability fixes, and documentation improvements are welcome.

Before opening a pull request:

1. Install the editable package with `python -m pip install -e '.[analysis,test]'`.
2. Run `python analysis/python/release_audit.py` and `pytest -q`.
3. Keep prompts, stimulus IDs, target layers, alpha grids, and estimands unchanged
   unless the proposed change explicitly creates a new version.
4. Do not commit model weights, API credentials, generated response archives,
   machine-specific paths, or third-party argument text.
5. Regenerate figure notebooks with `python -m analysis.figures.sync_notebooks`
   after changing a figure script.

Scientific changes should state which frozen result is affected and include a
small regression fixture. Code-only changes should remain compatible with the
repository-relative configuration and all eight model registrations.
