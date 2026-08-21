"""Test suite for the diff_tool module.

This module contains pytest tests for the results comparison tool,
covering dict comparison, numerical tolerance, string matching,
type checking, nested structures, list comparison, and CLI behavior.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Import the diff_tool module from results directory
sys.path.insert(0, str(Path(__file__).parent.parent / "results"))
from diff_tool import (
    DiffResult,
    compare_results,
    load_json,
    main,
)


BASELINE_DIR = Path(__file__).parent.parent / "results" / "baseline"
BASELINE_FILES = ["e16.json", "e17.json", "e20.json", "e21.json", "e22.json", "e88.json"]


class TestCompareResults:
    """Tests for the compare_results function."""

    def test_identical_dicts_pass(self):
        """Identical dicts produce all_passed == True."""
        actual = {"a": 1, "b": 2}
        baseline = {"a": 1, "b": 2}
        result = compare_results(actual, baseline)
        assert result.all_passed is True
        assert result.n_checks == result.n_passed

    def test_missing_key_detected(self):
        """Key in baseline but not actual is reported."""
        actual = {"a": 1}
        baseline = {"a": 1, "b": 2}
        result = compare_results(actual, baseline)
        assert result.all_passed is False
        assert ".b" in result.missing_keys

    def test_extra_key_reported(self):
        """Key in actual but not baseline listed in extra_keys."""
        actual = {"a": 1, "c": 3}
        baseline = {"a": 1}
        result = compare_results(actual, baseline)
        assert result.all_passed is False
        assert ".c" in result.extra_keys

    def test_numeric_exact_match(self):
        """Identical numbers pass with atol=0, rtol=0."""
        actual = {"value": 1.5}
        baseline = {"value": 1.5}
        result = compare_results(actual, baseline, atol=0.0, rtol=0.0)
        assert result.all_passed is True

    def test_numeric_within_atol(self):
        """Numbers differing by less than atol pass."""
        actual = {"value": 1.5001}
        baseline = {"value": 1.5}
        result = compare_results(actual, baseline, atol=0.001, rtol=0.0)
        assert result.all_passed is True

    def test_numeric_beyond_atol(self):
        """Numbers differing by more than atol fail."""
        actual = {"value": 1.51}
        baseline = {"value": 1.5}
        result = compare_results(actual, baseline, atol=0.001, rtol=0.0)
        assert result.all_passed is False
        assert len(result.numerical_mismatches) > 0

    def test_numeric_within_rtol(self):
        """Numbers differing by less than rtol pass."""
        actual = {"value": 100.05}
        baseline = {"value": 100.0}
        # 0.05 / 100.0 = 0.0005, so rtol=0.001 should pass
        result = compare_results(actual, baseline, atol=0.0, rtol=0.001)
        assert result.all_passed is True

    def test_string_match(self):
        """Identical strings pass."""
        actual = {"name": "test"}
        baseline = {"name": "test"}
        result = compare_results(actual, baseline)
        assert result.all_passed is True

    def test_string_mismatch(self):
        """Different strings fail."""
        actual = {"name": "test1"}
        baseline = {"name": "test2"}
        result = compare_results(actual, baseline)
        assert result.all_passed is False
        assert len(result.string_mismatches) > 0

    def test_type_mismatch(self):
        """Int vs str for same key is reported."""
        actual = {"value": 1}
        baseline = {"value": "1"}
        result = compare_results(actual, baseline)
        assert result.all_passed is False
        assert len(result.type_mismatches) > 0

    def test_nested_dict_recursion(self):
        """Nested dicts compared recursively."""
        actual = {"outer": {"inner": 1}}
        baseline = {"outer": {"inner": 1}}
        result = compare_results(actual, baseline)
        assert result.all_passed is True

        actual_fail = {"outer": {"inner": 2}}
        baseline_fail = {"outer": {"inner": 1}}
        result_fail = compare_results(actual_fail, baseline_fail)
        assert result_fail.all_passed is False

    def test_list_comparison(self):
        """Lists compared element-wise."""
        actual = {"items": [1, 2, 3]}
        baseline = {"items": [1, 2, 3]}
        result = compare_results(actual, baseline)
        assert result.all_passed is True

    def test_list_length_mismatch(self):
        """Lists of different length fail."""
        actual = {"items": [1, 2]}
        baseline = {"items": [1, 2, 3]}
        result = compare_results(actual, baseline)
        assert result.all_passed is False

    def test_boolean_comparison(self):
        """Booleans compared exactly."""
        actual = {"flag": True}
        baseline = {"flag": True}
        result = compare_results(actual, baseline)
        assert result.all_passed is True

        actual_false = {"flag": False}
        baseline_true = {"flag": True}
        result_false = compare_results(actual_false, baseline_true)
        assert result_false.all_passed is False

    def test_empty_dicts_pass(self):
        """Two empty dicts pass."""
        actual = {}
        baseline = {}
        result = compare_results(actual, baseline)
        assert result.all_passed is True


class TestBaselineFiles:
    """Tests for baseline JSON files."""

    def test_baseline_files_exist(self):
        """All six baseline JSON files exist."""
        for filename in BASELINE_FILES:
            file_path = BASELINE_DIR / filename
            assert file_path.exists(), f"Missing baseline file: {filename}"

    def test_baseline_files_valid_json(self):
        """Each baseline is valid JSON."""
        for filename in BASELINE_FILES:
            file_path = BASELINE_DIR / filename
            with open(file_path, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{filename} is not a valid JSON object"

    def test_baseline_files_have_required_keys(self):
        """Each has experiment_id, experiment_name, expected_values, metadata."""
        required_keys = {"experiment_id", "experiment_name", "expected_values", "metadata"}
        for filename in BASELINE_FILES:
            file_path = BASELINE_DIR / filename
            with open(file_path, "r") as f:
                data = json.load(f)
            assert required_keys.issubset(data.keys()), (
                f"{filename} missing required keys"
            )

    def test_baseline_experiment_ids_match(self):
        """experiment_id matches filename stem."""
        for filename in BASELINE_FILES:
            file_path = BASELINE_DIR / filename
            expected_id = Path(filename).stem
            with open(file_path, "r") as f:
                data = json.load(f)
            assert data["experiment_id"] == expected_id, (
                f"{filename}: experiment_id mismatch"
            )

    def test_baseline_seed_is_42(self):
        """Each metadata.seed is 42."""
        for filename in BASELINE_FILES:
            file_path = BASELINE_DIR / filename
            with open(file_path, "r") as f:
                data = json.load(f)
            assert data["metadata"]["seed"] == 42, f"{filename}: seed is not 42"


class TestCLI:
    """Tests for the CLI interface."""

    @pytest.fixture
    def temp_files(self, tmp_path):
        """Create temporary JSON files for CLI testing."""
        actual_file = tmp_path / "actual.json"
        baseline_file = tmp_path / "baseline.json"

        actual_data = {"value": 1.0, "name": "test"}
        baseline_data = {"value": 1.0, "name": "test"}

        with open(actual_file, "w") as f:
            json.dump(actual_data, f)
        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f)

        return actual_file, baseline_file

    def run_cli(self, args):
        """Run the CLI and return the result."""
        cmd = [sys.executable, "-m", "diff_tool"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "results"),
        )
        return result

    def test_cli_strict_mode(self, temp_files):
        """--strict sets atol=0, rtol=0."""
        actual_file, baseline_file = temp_files
        result = self.run_cli(
            ["--actual", str(actual_file), "--baseline", str(baseline_file), "--strict"]
        )
        assert result.returncode == 0

    def test_cli_tolerant_mode(self, temp_files):
        """--tolerant sets atol=1e-6, rtol=1e-4."""
        actual_file, baseline_file = temp_files
        result = self.run_cli(
            ["--actual", str(actual_file), "--baseline", str(baseline_file), "--tolerant"]
        )
        assert result.returncode == 0

    def test_cli_strict_and_tolerant_conflict(self, temp_files):
        """Both flags raise error."""
        actual_file, baseline_file = temp_files
        result = self.run_cli(
            [
                "--actual",
                str(actual_file),
                "--baseline",
                str(baseline_file),
                "--strict",
                "--tolerant",
            ]
        )
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_cli_exit_code_zero_on_pass(self, temp_files):
        """Exit 0 when results match."""
        actual_file, baseline_file = temp_files
        result = self.run_cli(
            ["--actual", str(actual_file), "--baseline", str(baseline_file)]
        )
        assert result.returncode == 0

    def test_cli_exit_code_one_on_fail(self, tmp_path):
        """Exit 1 on mismatch."""
        actual_file = tmp_path / "actual.json"
        baseline_file = tmp_path / "baseline.json"

        actual_data = {"value": 1.5}
        baseline_data = {"value": 2.5}

        with open(actual_file, "w") as f:
            json.dump(actual_data, f)
        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f)

        result = self.run_cli(
            [
                "--actual",
                str(actual_file),
                "--baseline",
                str(baseline_file),
                "--strict",
            ]
        )
        assert result.returncode == 1


class TestLoadJson:
    """Tests for the load_json function."""

    def test_load_valid_json(self, tmp_path):
        """Load valid JSON file successfully."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value"}
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        loaded = load_json(str(test_file))
        assert loaded == test_data

    def test_load_missing_file(self, tmp_path):
        """FileNotFoundError raised for missing file."""
        missing_file = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            load_json(str(missing_file))

    def test_load_invalid_json(self, tmp_path):
        """ValueError raised for invalid JSON."""
        test_file = tmp_path / "invalid.json"
        with open(test_file, "w") as f:
            f.write("not valid json")

        with pytest.raises(ValueError):
            load_json(str(test_file))


@pytest.mark.parametrize(
    "actual,baseline,atol,rtol,expected_pass",
    [
        # Exact match
        ({"v": 1.0}, {"v": 1.0}, 0.0, 0.0, True),
        # Within atol
        ({"v": 1.001}, {"v": 1.0}, 0.01, 0.0, True),
        # Beyond atol
        ({"v": 1.1}, {"v": 1.0}, 0.01, 0.0, False),
        # Within rtol (1% of 100 = 1)
        ({"v": 100.5}, {"v": 100.0}, 0.0, 0.01, True),
        # Beyond rtol
        ({"v": 105.0}, {"v": 100.0}, 0.0, 0.01, False),
    ],
)
def test_numeric_tolerance_parametrized(actual, baseline, atol, rtol, expected_pass):
    """Parametrized test for numerical tolerance."""
    result = compare_results(actual, baseline, atol=atol, rtol=rtol)
    assert result.all_passed == expected_pass
