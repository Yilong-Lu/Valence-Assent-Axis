# Data

The repository separates scientific inputs and outputs by their role:

- `stimuli/`: author-created or licensed task inputs and text-free selection
  manifests;
- `judge_validation/`: archived expert and model-judge ratings;
- `processed/`: compact item-level analysis tables used by the statistical
  scripts;
- `source_data/`: stable plotting inputs for the manuscript figures;
- `raw_external/`: complete generations, token diagnostics, and locally
  prepared third-party text. This directory is excluded from Git.

The files under `value/` and `sentiment/`, together with
`statement_general.csv` and `TruthfulQA_30.csv`, retain the paths used by the
original public release for backward compatibility. Current experiment runners
use the versioned materials under `stimuli/`.

`manifest/stimuli.yaml` records row counts and checksums for tracked scientific
inputs. `manifest/results.yaml` records the processed tables and the
external-output boundary. Field definitions are summarized in
`docs/data_dictionary.md`.

The Factual Judgment questions are derived from Apache-2.0-licensed TruthfulQA.
The tracked SycophancyEval stimulus directory contains only item identifiers
because its upstream repository did not declare a dataset license. The complete
archival deposit retains executed prompts in the model-output records; see
`THIRD_PARTY_NOTICES.md` and `docs/reproduction.md` for details and the
deterministic local preparation command.

The software license does not supersede third-party data or model licenses.
Complete generated outputs are included in the associated archival data package
rather than stored as ordinary Git objects.
