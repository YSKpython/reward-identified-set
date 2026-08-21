"""CLI tool for comparing actual results JSON against baseline JSON.

This module provides functionality to recursively compare two dictionaries
representing experimental results, with support for numerical tolerance
checking via absolute and relative tolerances.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DiffResult:
    """Result of comparing actual results against baseline.

    Attributes:
        all_passed: True if all checks passed, False otherwise.
        missing_keys: List of keys present in baseline but not in actual.
        extra_keys: List of keys present in actual but not in baseline.
        type_mismatches: List of (key_path, actual_type, baseline_type) tuples.
        numerical_mismatches: List of (key_path, actual_value, baseline_value) tuples.
        string_mismatches: List of (key_path, actual_value, baseline_value) tuples.
        n_checks: Total number of checks performed.
        n_passed: Number of checks that passed.
    """

    all_passed: bool
    missing_keys: list[str]
    extra_keys: list[str]
    type_mismatches: list[tuple[str, str, str]]
    numerical_mismatches: list[tuple[str, int | float, int | float]]
    string_mismatches: list[tuple[str, str, str]]
    n_checks: int
    n_passed: int


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file from the given path.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from None


def _compare_values(
    actual: Any,
    baseline: Any,
    key_path: str,
    atol: float,
    rtol: float,
) -> DiffResult:
    """Compare two values recursively.

    Args:
        actual: The actual value.
        baseline: The baseline value.
        key_path: Dot-separated path to the current key.
        atol: Absolute tolerance for numerical comparisons.
        rtol: Relative tolerance for numerical comparisons.

    Returns:
        DiffResult containing comparison results.
    """
    missing_keys: list[str] = []
    extra_keys: list[str] = []
    type_mismatches: list[tuple[str, str, str]] = []
    numerical_mismatches: list[tuple[str, int | float, int | float]] = []
    string_mismatches: list[tuple[str, str, str]] = []
    n_checks = 0
    n_passed = 0

    # Handle dict comparison
    if isinstance(actual, dict) and isinstance(baseline, dict):
        actual_keys = set(actual.keys())
        baseline_keys = set(baseline.keys())

        missing_keys.extend(f"{key_path}.{k}" for k in baseline_keys - actual_keys)
        extra_keys.extend(f"{key_path}.{k}" for k in actual_keys - baseline_keys)

        for key in actual_keys & baseline_keys:
            new_path = f"{key_path}.{key}" if key_path else key
            result = _compare_values(actual[key], baseline[key], new_path, atol, rtol)
            missing_keys.extend(result.missing_keys)
            extra_keys.extend(result.extra_keys)
            type_mismatches.extend(result.type_mismatches)
            numerical_mismatches.extend(result.numerical_mismatches)
            string_mismatches.extend(result.string_mismatches)
            n_checks += result.n_checks
            n_passed += result.n_passed

    # Handle list comparison
    elif isinstance(actual, list) and isinstance(baseline, list):
        min_len = min(len(actual), len(baseline))
        for i in range(min_len):
            new_path = f"{key_path}[{i}]"
            result = _compare_values(actual[i], baseline[i], new_path, atol, rtol)
            missing_keys.extend(result.missing_keys)
            extra_keys.extend(result.extra_keys)
            type_mismatches.extend(result.type_mismatches)
            numerical_mismatches.extend(result.numerical_mismatches)
            string_mismatches.extend(result.string_mismatches)
            n_checks += result.n_checks
            n_passed += result.n_passed

        if len(actual) != len(baseline):
            n_checks += 1
            if len(actual) < len(baseline):
                for i in range(min_len, len(baseline)):
                    missing_keys.append(f"{key_path}[{i}]")
            else:
                for i in range(min_len, len(actual)):
                    extra_keys.append(f"{key_path}[{i}]")

    # Handle numerical comparison
    elif isinstance(actual, (int, float)) and isinstance(baseline, (int, float)):
        n_checks += 1
        if not isinstance(actual, bool) and not isinstance(baseline, bool):
            tolerance = atol + rtol * abs(baseline)
            if abs(actual - baseline) <= tolerance:
                n_passed += 1
            else:
                numerical_mismatches.append((key_path, actual, baseline))
        else:
            # Booleans should be compared exactly
            n_checks -= 1  # Will be handled in type mismatch or exact match
            if actual == baseline:
                n_checks = 1
                n_passed = 1
            else:
                n_checks = 1
                numerical_mismatches.append((key_path, actual, baseline))

    # Handle string comparison
    elif isinstance(actual, str) and isinstance(baseline, str):
        n_checks += 1
        if actual == baseline:
            n_passed += 1
        else:
            string_mismatches.append((key_path, actual, baseline))

    # Handle boolean comparison
    elif isinstance(actual, bool) and isinstance(baseline, bool):
        n_checks += 1
        if actual == baseline:
            n_passed += 1
        else:
            numerical_mismatches.append((key_path, actual, baseline))

    # Handle type mismatch
    elif type(actual) is not type(baseline):
        n_checks += 1
        type_mismatches.append(
            (key_path, type(actual).__name__, type(baseline).__name__)
        )

    # Same type but not covered above (e.g., None, etc.)
    elif actual == baseline:
        n_checks += 1
        n_passed += 1
    else:
        n_checks += 1
        # For other types, treat as mismatch
        if isinstance(actual, str) or isinstance(baseline, str):
            string_mismatches.append((key_path, str(actual), str(baseline)))

    return DiffResult(
        all_passed=True,  # Will be updated by caller
        missing_keys=missing_keys,
        extra_keys=extra_keys,
        type_mismatches=type_mismatches,
        numerical_mismatches=numerical_mismatches,
        string_mismatches=string_mismatches,
        n_checks=n_checks,
        n_passed=n_passed,
    )


def compare_results(
    actual: dict[str, Any],
    baseline: dict[str, Any],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> DiffResult:
    """Recursively compare two dictionaries representing results.

    Numeric values pass if abs(actual - baseline) <= atol + rtol * abs(baseline).
    Strings must match exactly. Nested dicts recurse. Lists compare element-wise.

    Args:
        actual: The actual results dictionary.
        baseline: The baseline results dictionary.
        atol: Absolute tolerance for numerical comparisons. Default 0.0.
        rtol: Relative tolerance for numerical comparisons. Default 0.0.

    Returns:
        DiffResult containing all comparison details.
    """
    result = _compare_values(actual, baseline, "", atol, rtol)
    all_passed = (
        len(result.missing_keys) == 0
        and len(result.extra_keys) == 0
        and len(result.type_mismatches) == 0
        and len(result.numerical_mismatches) == 0
        and len(result.string_mismatches) == 0
        and result.n_checks == result.n_passed
    )
    return DiffResult(
        all_passed=all_passed,
        missing_keys=result.missing_keys,
        extra_keys=result.extra_keys,
        type_mismatches=result.type_mismatches,
        numerical_mismatches=result.numerical_mismatches,
        string_mismatches=result.string_mismatches,
        n_checks=result.n_checks,
        n_passed=result.n_passed,
    )


def main() -> int:
    """CLI entry point for the diff tool.

    Returns:
        Exit code: 0 if all checks pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Compare actual results JSON against baseline JSON."
    )
    parser.add_argument(
        "--actual",
        required=True,
        help="Path to the actual results JSON file.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline JSON file.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for numerical comparisons. Default 0.0.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for numerical comparisons. Default 0.0.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Force atol=0 and rtol=0 for strict comparison.",
    )
    parser.add_argument(
        "--tolerant",
        action="store_true",
        help="Use atol=1e-6 and rtol=1e-4 for tolerant comparison.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison results.",
    )

    args = parser.parse_args()

    # Handle mutually exclusive flags
    if args.strict and args.tolerant:
        parser.error("--strict and --tolerant are mutually exclusive.")

    # Apply flag overrides
    if args.strict:
        args.atol = 0.0
        args.rtol = 0.0
    elif args.tolerant:
        args.atol = 1e-6
        args.rtol = 1e-4

    try:
        actual = load_json(args.actual)
        baseline = load_json(args.baseline)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = compare_results(actual, baseline, atol=args.atol, rtol=args.rtol)

    if args.verbose:
        print(f"Checks: {result.n_checks}, Passed: {result.n_passed}")
        if result.missing_keys:
            print(f"Missing keys: {result.missing_keys}")
        if result.extra_keys:
            print(f"Extra keys: {result.extra_keys}")
        if result.type_mismatches:
            print(f"Type mismatches: {result.type_mismatches}")
        if result.numerical_mismatches:
            print(f"Numerical mismatches: {result.numerical_mismatches}")
        if result.string_mismatches:
            print(f"String mismatches: {result.string_mismatches}")

    if result.all_passed:
        print("All checks passed.")
        return 0
    else:
        print("Some checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
