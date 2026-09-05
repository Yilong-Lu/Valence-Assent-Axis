"""Synchronize percent-format figure scripts to editable Jupyter notebooks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import nbformat


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
MAIN = ("figure2", "figure3", "figure4", "figure5", "figure6")
SUPPLEMENT = ("feedback_sycophancy", "decoding_robustness")
MARKER = re.compile(r"^# %%(?:\s+\[(markdown)\])?\s*$")


def _markdown(lines: list[str]) -> str:
    return "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in lines).rstrip()


def convert(source: Path, destination: Path) -> None:
    bootstrap = """from pathlib import Path\nimport sys\n\nfor candidate in (Path.cwd(), *Path.cwd().parents):\n    if (candidate / \"pyproject.toml\").exists():\n        sys.path.insert(0, str(candidate))\n        break"""
    bootstrap_cell = nbformat.v4.new_code_cell(bootstrap)
    bootstrap_cell.id = hashlib.sha1(f"{source.name}:bootstrap".encode()).hexdigest()[:8]
    cells = [bootstrap_cell]
    kind = "code"
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        text = _markdown(lines) if kind == "markdown" else "\n".join(lines).rstrip()
        cell = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
        cell.id = hashlib.sha1(f"{source.name}:{len(cells)}:{text}".encode()).hexdigest()[:8]
        cells.append(cell)
        lines = []

    for line in source.read_text(encoding="utf-8").splitlines():
        match = MARKER.match(line)
        if match:
            if lines:
                flush()
            kind = "markdown" if match.group(1) else "code"
        else:
            lines.append(line)
    if lines:
        flush()
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "VAA", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)


def main() -> None:
    for stem in MAIN:
        convert(MODULE_DIR / f"{stem}.py", ROOT / "notebooks" / "main_figures" / f"{stem}.ipynb")
    for stem in SUPPLEMENT:
        convert(MODULE_DIR / f"{stem}.py", ROOT / "notebooks" / "supplementary_figures" / f"{stem}.ipynb")


if __name__ == "__main__":
    main()
