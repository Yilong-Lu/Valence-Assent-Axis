# Manuscript Figures

`main/` contains the complete manuscript figures. `supplementary/` contains the
Supplementary figures retained by this repository. Running the figure
scripts also creates editable PDF/SVG and review PNG files under `generated/`.

`figure1.jpg` is the conceptual overview distributed with the original public
release; it has no numerical Source Data or programmatic plotting script.

Figures 2, 3, and 6 can be regenerated as complete quantitative figures. In
Figures 4 and 5, response examples and the compact task schematic were arranged
in a vector editor after generating the quantitative panels. Their complete
composites are therefore tracked alongside the reproducible panel builders:

- `analysis/figures/figure4.py` builds Figure 4a/b/d/e;
- `analysis/figures/figure5.py` builds Figure 5b/c/e/f/g;
- `figures/main/figure4.pdf` and `figures/main/figure5.pdf` are the complete
  composites used by the manuscript.

The Figure 4 composite source is retained as
`source/figure4_complete.tex`, together with its quantitative base PDF. Figure
5 remains editable in draw.io because `main/figure5_editable.drawio.pdf`
contains the embedded diagram XML.

The `submitted_*.pdf` files in `supplementary/` preserve the originally
submitted Supplement layouts. Compact source tables for the cross-domain,
Alphabetical Order, Factual Judgment, and Stance Taking figures are available
under `data/source_data/`. The exploratory t-SNE panel is intentionally marked
as a provenance artifact: its historical intermediate activations and
fitting parameters could not be recovered reliably, so this repository does
not claim to regenerate that panel.

TIFF copies and exploratory working layouts are not included.
