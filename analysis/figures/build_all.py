"""Build all reproducible main and Supplementary figure outputs."""

from __future__ import annotations

from analysis.figures import (
    decoding_robustness,
    feedback_sycophancy,
    figure2,
    figure3,
    figure4,
    figure5,
    figure6,
)


def main() -> None:
    for module in (figure2, figure3, figure4, figure5, figure6, feedback_sycophancy, decoding_robustness):
        module.main()


if __name__ == "__main__":
    main()
