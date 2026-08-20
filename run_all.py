#!/usr/bin/env python3
"""Single entry point for the reward-identified-set pipeline.

Usage:
    python run_all.py                  # Run all experiments
    python run_all.py --experiment e16 # Run a single experiment
    python run_all.py --verify         # Verify results against baselines

This script auto-discovers experiments by scanning configs/e*.yaml and
invoking the corresponding experiments/<name>.py::run() function.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS_DIR = Path("configs")
EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")
BASELINE_DIR = RESULTS_DIR / "baseline"

logger = logging.getLogger("run_all")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_base_config() -> dict[str, Any]:
    """Load the shared base configuration."""
    base_path = CONFIGS_DIR / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(name: str) -> dict[str, Any]:
    """Load an experiment-specific config, merging with base."""
    base = load_base_config()
    exp_path = CONFIGS_DIR / f"{name}.yaml"
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f) or {}
    # Shallow merge: experiment values override base
    merged = {**base, **exp_cfg}
    merged["experiment_name"] = name
    return merged


def discover_experiments() -> list[str]:
    """Find all experiment configs matching configs/e*.yaml."""
    if not CONFIGS_DIR.exists():
        return []
    return sorted(
        p.stem for p in CONFIGS_DIR.glob("e*.yaml")
    )


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
def run_experiment(name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Import and execute a single experiment module.

    Each experiment module must expose:
        run(config: dict) -> dict
    returning a JSON-serializable results dictionary.
    """
    module_name = f"experiments.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.warning(
            "Experiment module '%s' not found. Skipping.", module_name
        )
        return {"status": "skipped", "reason": "module not found"}

    if not hasattr(module, "run"):
        logger.warning(
            "Module '%s' has no run() function. Skipping.", module_name
        )
        return {"status": "skipped", "reason": "no run() function"}

    logger.info("Running experiment: %s", name)
    try:
        results = module.run(config)
    except Exception as exc:
        logger.error("Experiment %s failed: %s", name, exc, exc_info=True)
        return {"status": "error", "error": str(exc)}

    return {"status": "completed", **results}


def save_results(name: str, results: dict[str, Any]) -> Path:
    """Write experiment results to results/<name>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info("Results written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Baseline verification
# ---------------------------------------------------------------------------
def verify_against_baseline(name: str, results: dict[str, Any]) -> bool:
    """Compare results against the archived baseline JSON.

    Returns True if results match exactly (atol=0.0, rtol=0.0).
    """
    baseline_path = BASELINE_DIR / f"{name}.json"
    if not baseline_path.exists():
        logger.warning("No baseline for %s. Skipping verification.", name)
        return True  # No baseline yet; not a failure

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # Remove non-comparable keys
    for key in ("status", "experiment_name"):
        baseline.pop(key, None)
        results_copy = {k: v for k, v in results.items()
                        if k not in ("status", "experiment_name")}

    if results_copy != baseline:
        logger.error(
            "MISMATCH for %s. See results/diff_tool.py for details.", name
        )
        return False
    logger.info("Baseline verification PASSED for %s.", name)
    return True


# ---------------------------------------------------------------------------
# Seed enforcement (K4)
# ---------------------------------------------------------------------------
def enforce_seed(seed: int) -> None:
    """Set global RNG state for reproducibility.

    Delegates to src.utils.seed.seed_everything once implemented.
    Falls back to manual seeding until the module exists.
    """
    try:
        from src.utils.seed import seed_everything
        seed_everything(seed)
    except ImportError:
        # Fallback: seed core libraries directly
        import random
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        logger.info(
            "Seeded with fallback (seed=%d). "
            "Install src.utils.seed for full coverage.", seed
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the reward-identified-set pipeline."
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Run a single experiment by name (e.g., 'e16').",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify results against archived baselines.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered experiments and exit.",
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Discover experiments
    all_experiments = discover_experiments()
    if args.list:
        print("Discovered experiments:")
        for name in all_experiments:
            print(f"  - {name}")
        return 0

    if not all_experiments:
        logger.warning(
            "No experiment configs found in %s. "
            "Create configs/e*.yaml to get started.", CONFIGS_DIR
        )
        return 0

    # Select experiments to run
    if args.experiment:
        if args.experiment not in all_experiments:
            logger.error(
                "Experiment '%s' not found. Available: %s",
                args.experiment, all_experiments,
            )
            return 1
        selected = [args.experiment]
    else:
        selected = all_experiments

    # Load base config for seed
    base_config = load_base_config()
    seed = base_config.get("seed", 42)
    enforce_seed(seed)

    # Run experiments
    all_passed = True
    for name in selected:
        config = load_experiment_config(name)
        results = run_experiment(name, config)

        if results.get("status") == "skipped":
            continue

        if results.get("status") == "error":
            all_passed = False
            continue

        save_results(name, results)

        if args.verify:
            passed = verify_against_baseline(name, results)
            if not passed:
                all_passed = False

    if all_passed:
        logger.info("All experiments completed successfully.")
        return 0
    else:
        logger.error("One or more experiments failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
