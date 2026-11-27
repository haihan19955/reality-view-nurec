# Contributing

Thanks for helping. This is a research codebase: scripts are run by hand on one GPU, and results are measured, not
assumed. The rules below keep it reproducible and safe to publish.

## Ground rules

1. **Never commit data, renders or weights.** NuRec scenes are licensed for AV development only, and renders made
   from them are covered too. Everything generated goes to `work/` (gitignored), and `*.pt`, `*.npy`, `*.safetensors`
   and `*.log` are ignored as well. If you need a figure in the docs, raise it in an issue first.
2. **No machine-specific paths.** Every location comes from `realityview/paths.py` (environment variables
   `RV_WORK`, `RV_MODELS`, …). Scripts take paths as arguments with those defaults.
3. **Back claims with numbers.** A change that affects quality or speed reports the before / after metric and the
   command that produced it (validation loss, `benchmarks/student/evaluate.py`, viewer `chunk_ms`, …) in the PR.
4. **Keep both READMEs in sync.** `README.md` (English) and `README.zh-CN.md` (简体中文) have the same sections.
   Change both in the same PR. Detailed guides in `docs/` are English only.

## Development setup

```bash
git clone https://github.com/haihan19955/reality-view-nurec && cd reality-view-nurec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[nurec,student,dev]"
```

Linux or WSL2 with a CUDA GPU is needed for anything beyond linting. Third-party models are listed in
[docs/video-student.md](docs/video-student.md#1-models-and-third-party-code).

Before pushing, run what CI runs:

```bash
ruff check .
python -m compileall -q realityview scripts viewer benchmarks tools
bash -n scripts/*.sh scripts/*/*.sh viewer/*.sh
```

For training-code changes, also run the script with `--smoke` (train.py, distill.py) on real data. CI has no GPU.

## Code layout and conventions

- `realityview/` is the importable library. `scripts/` are command-line entry points: one file per step, a module
  docstring with the usage line, `argparse` for every option.
- Code and comments are in English. Comments explain *why* (a finding, a constraint), not what the line does.
- Match the surrounding style: compact research code, line length 130, no type-annotation churn.
- New options default to the current behaviour, so old commands and checkpoints keep working. Checkpoints carry what
  is needed to rebuild the model (for example, the ego input width decides free view vs chase-only, and
  `feat.txt` sits next to the checkpoint).

## Branches, commits, pull requests

- Branch from `main`: `feat/…`, `fix/…`, `docs/…`, `exp/…` (experiments that may not be merged).
- Commit messages: `<area>: <imperative summary>`, where the area is `nurec`, `student`, `viewer`, `realityview`,
  `benchmarks`, `docs` or `build`. Put the reason and the measured effect in the body.
- One topic per PR. Fill in the template: what changed, how it was tested, the metrics.
- `main` must always pass CI. Releases are tagged `vX.Y.Z` and listed in [CHANGELOG.md](CHANGELOG.md).

## Reporting issues

Use the issue templates. Include the command, the full traceback or log tail, GPU / driver / CUDA versions and
whether you are on Linux or WSL2. Do not attach NuRec renders.
