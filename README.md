# reward-identified-set

Feature-space vulnerability auditing for RLHF reward models.

This repository implements the experimental pipeline for the NeurIPS 2027
submission *"The Identified Set of Reward: Feature-Space Vulnerability and
the Neural Manipulation Cost in RLHF"*. It audits how transformer-based
reward models implicitly couple structurally disconnected comparison-graph
components through low-rank feature manifolds, and quantifies the resulting
reward-hacking surface via a Fisher-metric manipulation cost.

## Paper Anchor

| Quantity | Value | Source |
|---|---|---|
| Tabular identified-set dimension | 160,055 | E5 audit, HH-RLHF |
| Neural effective rank (95% var.) | 36 | E16, DeBERTa-v3-base |
| Within-component cosine | 0.960 ± 0.012 | E16 |
| Cross-component cosine | 0.773 ± 0.028 | E16 |
| LLF manipulation cost (m=0.1) | 0.0129 | E17 |
| Probe accuracy (pooled hidden) | 0.472 ± 0.018 | E20 |
| TF-IDF baseline accuracy | 0.920 ± 0.014 | E20 |
| MC_RRF / MC_LLF (E21) | 0.2645 / 0.2647 | E21 |
| Vulnerability ratio | 14.6 | E88 |
| Geometric asymmetry | ≈ 68 | E88 |

All experiments use `seed = 42` and are deterministic.

## Experiment Map

| ID | Name | Config | Module |
|---|---|---|---|
| E16 | Feature-Space Audit (PCA rank + cosine) | `configs/e16.yaml` | `experiments/e16_feature_space_audit.py` |
| E17 | Neural MC Audit (LLF + CG) | `configs/e17.yaml` | `experiments/e17_neural_mc_audit.py` |
| E20 | Three-Level Pattern (linear probe) | `configs/e20.yaml` | `experiments/e20_three_level_pattern.py` |
| E21 | VJP Range Finder (25 prompts) | `configs/e21.yaml` | `experiments/e21_vjp_range_finder.py` |
| E22 | Regime Hunt at Scale (200 prompts, TPU) | `configs/e22.yaml` | `experiments/e22_regime_hunt.py` |
| E88 | Real-World Feature-Space Audit (flip cost) | `configs/e88.yaml` | `experiments/e88_real_world_audit.py` |

## Installation

```bash
# Python >= 3.10 required
pip install -e ".[dev]"
```

For GPU/TPU support, install the appropriate JAX backend:

```bash
# GPU (CUDA 12)
pip install "jax[cuda12]"
# CPU only
pip install "jax[cpu]"
```

## Usage

Run the full pipeline:

```bash
python run_all.py
```

Run a single experiment:

```bash
python run_all.py --experiment e16
```

Verify results against archived baselines:

```bash
python run_all.py --verify
```

## Repository Structure

```
reward-identified-set/
├── README.md
├── LICENSE                          # MIT
├── pyproject.toml                   # Single source of truth for deps
├── run_all.py                       # Single entry point
├── configs/                         # One YAML per experiment
│   ├── base.yaml
│   ├── e16.yaml
│   └── ...
├── src/
│   ├── models/                      # RewardModelAdapter protocol + implementations
│   ├── fisher/                      # Fisher matrix construction, LLF, Schur
│   ├── metrics/                     # Manipulation cost, vulnerability ratio
│   ├── analysis/                    # PCA, cosine, linear probe
│   └── utils/                       # Seed, config, logging
├── experiments/                     # Thin orchestration (one file per experiment)
├── data/
│   ├── raw/                         # Downloaded datasets (gitignored)
│   ├── processed/                   # Cached features (gitignored)
│   └── download_hh_rlhf.py
├── tests/                           # pytest suite
├── results/
│   ├── baseline/                    # Archived JSON baselines
│   └── diff_tool.py                 # JSON diff, exit code 0/1
├── paper/
│   ├── main_neurips.tex
│   ├── references_neurips.bib
│   └── figures/
└── .github/
    └── workflows/ci.yaml            # pytest + mypy on push
```

## Development Workflow

Experiments are added incrementally. Each experiment requires:

1. A config file in `configs/` (extends `base.yaml`).
2. A module in `experiments/` exposing `run(config: ExperimentConfig) -> ExperimentResult`.
3. Unit tests in `tests/`.
4. A baseline JSON in `results/baseline/`.

`run_all.py` auto-discovers experiments by scanning `configs/e*.yaml`.

## Reproducibility Contract

- All RNG state is set via `src/utils/seed.py::seed_everything(42)`.
- All code is type-hinted; `mypy --strict` must pass.
- `results/diff_tool.py` compares output JSONs against baselines with
  `atol=0.0` (exact match required).
- CI runs `pytest` and `mypy` on every push.

## Scope Limitation

All empirical results are scoped to a single architecture
(DeBERTa-v3-base), a single checkpoint, and a single seed (42).
Multi-architecture and multi-seed extensions are planned but not yet
implemented. Until such replication is performed, the results should be
interpreted as a single-model structural diagnostic, not as a universal
property of all deep reward models.

## License

MIT
