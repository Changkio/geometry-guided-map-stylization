# Map Stylization

Experimental code, fixed configurations, recorded metrics, and reproducibility assets
for reference-guided road-map stylization with a frozen diffusion backbone.

This repository releases software and experimental records only. No manuscript,
submission files, publication claim, or manuscript author list is supplied.

## Contents

- `project/src/`: generation, geometry utilities, metrics, statistics, and tests.
- `project/third_party/AttentionDistillation/`: pinned upstream implementation.
- `configs/` and `revision_20260925/`: fixed original and exploratory experiment settings.
- `results/`: small CSV/JSON summaries with flattened paths; see `results/INDEX.md`.
- `tools/download_weights_repro.py`: pinned downloads with historical full-hash verification.

The complete experiment asset is
`geometry-guided-map-stylization-experiments-v1.0.0.zip`. Extract it into a new
directory before data-dependent execution. Model weights are not included.
The smaller code archive is `geometry-guided-map-stylization-code-v1.0.0.zip`.
See [REPRODUCTION.md](REPRODUCTION.md) and [LICENSING.md](LICENSING.md).

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

The A1−B0 F1 difference is +0.0138, with paired 95% map-bootstrap interval
[−0.0003, 0.0265], which includes zero. These metrics do not establish topology,
aesthetics, or navigation quality. The shifted-prior and road-only studies were
registered after the original test outputs were inspected and remain exploratory.

Records cover 192 development runs, 180 original test runs, 36 FP32 controls,
72 exploratory runs, and 3 implementation checks. The checks are not independent
test observations. B0/B1 and A1/AT selected identical configurations. All measured
outputs, including poor-quality cases, remain in the experiment asset.

## Provenance

Upstream [Attention Distillation](https://github.com/xugao97/AttentionDistillation)
is pinned to commit `142800c389bb4dff41e7f79cb9c2e88c981bbec2`; SD1.5 is pinned to
revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. The recorded environment uses
Windows, Python 3.10.11, RTX 5070 Ti 16 GB, 32 GB RAM, PyTorch 2.8.0+cu128, and
Diffusers 0.32.2. Other environments are not promised to reproduce image bytes.

Road data © OpenStreetMap contributors, ODbL. The three reference images are CC0
Open Access artworks from The Metropolitan Museum of Art. Pretrained weights
and dependencies retain their upstream terms. OpenAI Codex assisted with code,
experiment execution and analysis. No GPU experiment was rerun during packaging.
