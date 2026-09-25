# Reproduce the experimental records

The main branch is a **core-source preview**, not the complete runtime tree.
The Release archives are the authoritative code and experiment distribution.
Run the commands below only inside a fresh extraction of the complete experiment
package; do not try to fill gaps in the preview checkout manually.

## Download and restore the complete package

From [Release v1.0.0](https://github.com/Changkio/geometry-guided-map-stylization/releases/tag/v1.0.0), download:

- `geometry-guided-map-stylization-code-v1.0.0.zip` — complete public source package.
- `geometry-guided-map-stylization-experiments-v1.0.0.zip` — code plus archived inputs,
  all recorded outputs and logs, and original experiment paths; use this for reproduction.
- `EXPERIMENTS_SHA256SUMS.txt` — full SHA256 checksums for both packages.

In the download directory, compute both archive hashes and compare them with the
checksum file. Continue only if they match. Choose a new extraction directory:

```powershell
Get-FileHash -Algorithm SHA256 .\geometry-guided-map-stylization-code-v1.0.0.zip, .\geometry-guided-map-stylization-experiments-v1.0.0.zip
Get-Content .\EXPERIMENTS_SHA256SUMS.txt
Expand-Archive -LiteralPath .\geometry-guided-map-stylization-experiments-v1.0.0.zip -DestinationPath .\map-stylization-experiments-v1.0.0
Set-Location .\map-stylization-experiments-v1.0.0
```

The extracted root contains `project/`, `configs/`, `provenance/`, `tools/`,
`tests/`, and `revision_20260925/`. All paths below refer to that root, not to the
current main-branch preview. Keep the downloaded archives and an untouched evidence
copy. Model weights are not included. See [LICENSING.md](LICENSING.md) for the
software and data terms. The packages contain no manuscript or submission files.

## Environment and weights

The recorded environment is Windows, Python 3.10.11, RTX 5070 Ti 16 GB, 32 GB host RAM,
PyTorch 2.8.0+cu128 and Diffusers 0.32.2. Exact package versions are in the two
`configs/requirements-*.lock.txt` files. Bitwise equality on other systems is not promised.

```powershell
py -3.10 -m venv .venv
& .venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r configs/requirements-model.lock.txt
& .venv\Scripts\python.exe -m pip check
& .venv\Scripts\python.exe tools/download_weights_repro.py --models all
& .venv\Scripts\python.exe tools/download_weights_repro.py --models all --verify-only
```

Use the new helper for downloads. It checks complete historical hashes and writes
local verification under `provenance/local/`, preserving the frozen model record.
Historical downloaders are retained as evidence and should not be used for a new run.
Existing wrong-hash files are rejected. If a terminated process leaves a stale `.part` file,
inspect it before retrying. Use archived OSM responses, not new live queries.

## Original suites

Keep an untouched copy of the supplied evidence. Successful matching runs are
resumed/skipped by the generator. In a separate rerun copy, omit the delivered
`project/results/` only when regenerating all original suites. Preserve configurations,
data, and provenance; do not tune on test results.

```powershell
& .venv\Scripts\python.exe -m unittest discover -s project/src -p test_correctness.py
& .venv\Scripts\python.exe -m unittest discover -s project/src -p test_metrics.py
& .venv\Scripts\python.exe -m unittest discover -s project/src -p test_analysis.py
& .venv\Scripts\python.exe -m unittest discover -s project/src -p test_precision_control.py
& .venv\Scripts\python.exe project/src/run_experiments.py --config configs/development_search.json
& .venv\Scripts\python.exe project/src/evaluate.py --result-root project/results/development_search
& .venv\Scripts\python.exe project/src/run_experiments.py --config configs/formal_test.json
& .venv\Scripts\python.exe project/src/evaluate.py --result-root project/results/formal_test
& .venv\Scripts\python.exe project/src/precision_control.py
& .venv\Scripts\python.exe project/src/evaluate.py --result-root project/results/precision_control
```

Run one GPU workload at a time. Selection and protocol are already frozen;
`analyze.py select-freeze` is a historical registration step, not a command to
overwrite the delivered test protocol. Full artifact checks require the complete
recorded directory, not a partially regenerated copy.

## Exploratory controls

For a separate control rerun, retain original `project/results/` and the frozen
records. Omit only the delivered `revision_20260925/results/implementation_checks/`,
`shifted_prior/`, and `road_only/` output directories. Keep `results/prior_checks/`
under the revision directory. These studies remain exploratory.

```powershell
& .venv\Scripts\python.exe revision_20260925/code/run_revision.py --config revision_20260925/configs/implementation_checks.json
& .venv\Scripts\python.exe revision_20260925/code/run_revision.py --config revision_20260925/configs/shifted_prior.json
& .venv\Scripts\python.exe revision_20260925/code/run_revision.py --config revision_20260925/configs/road_only.json
& .venv\Scripts\python.exe project/src/evaluate.py --result-root revision_20260925/results/shifted_prior
& .venv\Scripts\python.exe project/src/evaluate.py --result-root revision_20260925/results/road_only
```

Do not rewrite a freeze to bypass a mismatch. The implementation checks contain
one 200-step replay and two 32-step checks; they are not new statistical replicates.

## Statistics and integrity

From the complete experiment directory, a document environment with pandas, NumPy,
Pillow and Matplotlib can run `revision_20260925/code/reanalyze.py`. It aggregates
archived measurements and performs source/output checks. Resampling uses 12 maps,
10,000 paired resamples and seed 20260925; the three fixed styles stay together.
`revision_20260925/code/boundary_controls.py` evaluates local-boundary diagnostics.

The download-helper tests use no GPU or network:

```powershell
python -m unittest discover -s tests -p test_download_weights_repro.py -v
```

`PUBLIC_RELEASE_MANIFEST.json` verifies this experiment-only payload. The 27 frozen
scientific files and all 483 declared output hashes were retained and checked.
Publishing/building/writing utilities and submission material are deliberately absent.
No GPU experiment was rerun during this packaging operation.
