# Map Stylization Experiments

Reference-guided road-map stylization with a frozen diffusion backbone: code,
fixed configurations, recorded results, and reproducibility assets.

**The complete Release archives are the authoritative distribution.** Files
browsable on the main branch are a core-source preview, not a complete runnable
checkout. Download the following three assets from [Release v1.0.0](https://github.com/Changkio/geometry-guided-map-stylization/releases/tag/v1.0.0):

- `geometry-guided-map-stylization-code-v1.0.0.zip` — complete public code package,
  configurations, tests, metadata, and compact result summaries.
- `geometry-guided-map-stylization-experiments-v1.0.0.zip` — complete experiment
  package, including code, archived inputs, outputs, logs, and original result paths.
- `EXPERIMENTS_SHA256SUMS.txt` — full SHA256 checksums for both archives.

For reproduction, verify the checksums and extract the **experiments** archive into
a new directory. Run from that extracted root, not the preview checkout. Model
weights are downloaded separately. Follow [REPRODUCTION.md](REPRODUCTION.md);
software and data terms are summarized in [LICENSING.md](LICENSING.md).

Only software, data, and experimental records are released. No manuscript,
submission files, manuscript author list, or publication claim is included.

## Inside the complete experiment package

- `project/src/` and `project/third_party/AttentionDistillation/`: implementation,
  metrics, statistics, tests, and pinned upstream source.
- `configs/` and `revision_20260925/configs/`: fixed original and exploratory settings.
- `project/data/`: archived source data and rendered inputs.
- `project/results/` and `revision_20260925/results/`: complete recorded runs,
  output images, metrics, diagnostics, and analysis.
- `provenance/`, `tools/`, and `tests/`: historical records, safe weight-download
  helper, and offline verification tests.

These paths describe the downloaded package; they need not all be present on the
main branch.

## Recorded measurements

Test means cover 12 geographic windows and 3 fixed references. Boundary F1 is a
local road-neighborhood edge measure; Gram and LPIPS are separate appearance proxies.

| Configuration | Boundary F1 | Gram discrepancy | LPIPS |
|---|---:|---:|---:|
| Uniform Q (B0/B1) | 0.8548 | 35.23 | 0.2734 |
| Geometry Q (A1/AT) | 0.8686 | 24.43 | 0.2895 |
| QK (B2) | 0.4684 | 1.91 | 0.7182 |
| Shifted prior, exploratory | 0.8546 | 30.08 | 0.2833 |
| Road-only, exploratory | 0.8700 | 27.37 | 0.2696 |

The A1−B0 F1 difference is +0.0138; its paired 95% map-bootstrap interval
[−0.0003, 0.0265] includes zero. These proxies do not establish topology,
aesthetics, or navigation quality. Shifted-prior and road-only controls were
registered after inspection of the original test outputs and remain exploratory.

The archive retains 192 development runs, 180 original test runs, 36 FP32 controls,
72 exploratory runs, and 3 implementation checks, including poor-quality outputs.
The checks are not independent test observations. B0/B1 and A1/AT selected identical
configurations.

## Provenance and license scope

Attention Distillation (`xugao97/AttentionDistillation`) is pinned to commit
`142800c389bb4dff41e7f79cb9c2e88c981bbec2`; SD1.5 is pinned to revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. Recorded execution used Windows,
Python 3.10.11, RTX 5070 Ti 16 GB, 32 GB RAM, PyTorch 2.8.0+cu128, and Diffusers
0.32.2. Other systems are not promised to reproduce image bytes.

Road data © OpenStreetMap contributors, ODbL. Reference artworks are CC0 Open Access
images from The Metropolitan Museum of Art. The MIT license for original software
does not replace upstream code, data, or model terms; see [LICENSING.md](LICENSING.md).
OpenAI Codex assisted with code, experiment execution, and analysis. No GPU experiment
was rerun during packaging.
