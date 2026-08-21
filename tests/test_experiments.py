"""Test suite for the experiment orchestration layer.

This module tests the experiment modules (e16, e17, e20, e21, e22, e88)
and config files WITHOUT downloading the real model. It uses unittest.mock
to patch the adapter with deterministic mock outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Import experiment modules
import experiments.e16 as e16
import experiments.e17 as e17
import experiments.e20 as e20
import experiments.e21 as e21
import experiments.e22 as e22
import experiments.e88 as e88


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock reward model adapter with deterministic outputs.

    Returns:
        MagicMock object implementing the RewardModelAdapter interface with:
        - extract_features returns np.random.RandomState(42).normal(0, 1, (n, 1024))
        - predict_rewards returns np.random.RandomState(42).normal(0, 1, n)
        - compute_jacobian_rows returns np.random.RandomState(42).normal(0, 1, (n, 1024))
        - Properties return expected DeBERTa-v3 values.
    """
    adapter = MagicMock()
    adapter.model_name = "OpenAssistant/reward-model-deberta-v3-base"
    adapter.architecture = "deberta-v3-base"
    adapter.feature_dim = 1024

    # Create deterministic RNG for reproducibility
    rng = np.random.RandomState(42)

    def extract_features(prompts: list[str], layer: str = "penultimate") -> np.ndarray:
        n = len(prompts)
        return rng.randn(n, 1024).astype(np.float32)

    def predict_rewards(prompts: list[str]) -> np.ndarray:
        n = len(prompts)
        return rng.randn(n).astype(np.float32)

    def compute_jacobian_rows(prompts: list[str]) -> np.ndarray:
        n = len(prompts)
        return rng.randn(n, 1024).astype(np.float32)

    adapter.extract_features = extract_features
    adapter.predict_rewards = predict_rewards
    adapter.compute_jacobian_rows = compute_jacobian_rows

    return adapter


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Load the base configuration."""
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "seed": 42,
        "device": "cpu",
        "model": {
            "name": "OpenAssistant/reward-model-deberta-v3-base",
            "architecture": "deberta-v3-base",
            "feature_dim": 1024,
        },
        "data": {
            "cache_dir": "data/processed",
        },
    }


def _merge_config(base: dict[str, Any], exp_config: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge experiment config over base config."""
    merged = {**base, **exp_config}
    return merged


class TestE16:
    """Tests for Experiment E16: Feature-Space Audit."""

    @pytest.mark.slow
    def test_e16_run_with_mock(
        self, mock_adapter: MagicMock, base_config: dict[str, Any]
    ) -> None:
        """Test E16 run function with mocked adapter.

        Verifies that the output dict has all expected keys and status == "completed".
        """
        # Load experiment config
        with open("configs/e16.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(e16.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = e16.run(config)

        # Verify output structure
        assert result["status"] == "completed"
        assert result["experiment_id"] == "e16"
        assert result["experiment_name"] == "Feature-Space Audit"
        assert "results" in result
        assert "expected_values" in result

        # Verify results keys
        results = result["results"]
        required_keys = [
            "effective_rank",
            "within_cosine_mean",
            "within_cosine_std",
            "cross_cosine_mean",
            "cross_cosine_std",
            "cosine_gap",
            "pca_explained_variance_at_rank",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"


class TestE17:
    """Tests for Experiment E17: Neural Manipulation Cost Audit."""

    @pytest.mark.slow
    def test_e17_run_with_mock(
        self, mock_adapter: MagicMock, base_config: dict[str, Any]
    ) -> None:
        """Test E17 run function with mocked adapter."""
        with open("configs/e17.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(e17.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = e17.run(config)

        assert result["status"] == "completed"
        assert result["experiment_id"] == "e17"
        assert "results" in result
        assert "expected_values" in result

        results = result["results"]
        required_keys = [
            "mc_llf",
            "vulnerability_score",
            "kl_budget",
            "fisher_rank",
            "null_space_dim",
            "cg_converged",
            "cg_iterations",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"


class TestE20:
    """Tests for Experiment E20: Three-Level Pattern."""

    @pytest.mark.slow
    def test_e20_run_with_mock(
        self, mock_adapter: MagicMock, base_config: dict[str, Any]
    ) -> None:
        """Test E20 run function with mocked adapter."""
        with open("configs/e20.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(e20.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = e20.run(config)

        assert result["status"] == "completed"
        assert result["experiment_id"] == "e20"
        assert "results" in result
        assert "expected_values" in result

        results = result["results"]
        required_keys = [
            "pooled_accuracy",
            "pooled_std",
            "tfidf_accuracy",
            "tfidf_std",
            "random_projection_accuracy",
            "permuted_label_accuracy",
            "three_level_pattern_confirmed",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"


class TestE21:
    """Tests for Experiment E21: VJP Range Finder."""

    @pytest.mark.slow
    def test_e21_run_with_mock(
        self, mock_adapter: MagicMock, base_config: dict[str, Any]
    ) -> None:
        """Test E21 run function with mocked adapter."""
        with open("configs/e21.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(e21.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = e21.run(config)

        assert result["status"] == "completed"
        assert result["experiment_id"] == "e21"
        assert "results" in result
        assert "expected_values" in result

        results = result["results"]
        required_keys = [
            "mc_rrf",
            "mc_llf",
            "relative_gap",
            "rank_jw",
            "range_inclusion_holds",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"


class TestE22:
    """Tests for Experiment E22: Regime Hunt at Scale."""

    @pytest.mark.slow
    def test_e22_run_with_mock(
        self, mock_adapter: MagicMock, base_config: dict[str, Any]
    ) -> None:
        """Test E22 run function with mocked adapter."""
        with open("configs/e22.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(e22.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = e22.run(config)

        assert result["status"] == "completed"
        assert result["experiment_id"] == "e22"
        assert "results" in result
        assert "expected_values" in result

        results = result["results"]
        required_keys = [
            "mc_full",
            "mc_llf",
            "divergence_coefficient",
            "range_inclusion_holds",
            "n_audit_prompts",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"


class TestE88:
    """Tests for Experiment E88: Real-World Feature-Space Audit."""

    @pytest.mark.slow
    def test_e88_run_with_mock(
        self, mock_adapter: MagicMock, base_config: dict[str, Any]
    ) -> None:
        """Test E88 run function with mocked adapter."""
        with open("configs/e88.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(e88.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = e88.run(config)

        assert result["status"] == "completed"
        assert result["experiment_id"] == "e88"
        assert "results" in result
        assert "expected_values" in result

        results = result["results"]
        required_keys = [
            "vulnerability_ratio",
            "geometric_asymmetry",
            "cross_median",
            "within_median",
            "fisher_rank",
            "null_space_dim",
            "feature_dim",
            "n_cross",
            "n_within",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"


class TestConfigLoading:
    """Tests for config file loading and validation."""

    def test_config_loading(self) -> None:
        """Verify each configs/eNN.yaml loads as valid YAML and contains required keys."""
        experiment_ids = ["e16", "e17", "e20", "e21", "e22", "e88"]

        for exp_id in experiment_ids:
            config_path = Path(f"configs/{exp_id}.yaml")
            assert config_path.exists(), f"Config file not found: {config_path}"

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            assert "experiment_id" in config, f"Missing 'experiment_id' in {config_path}"
            assert "name" in config, f"Missing 'name' in {config_path}"
            assert config["experiment_id"] == exp_id, (
                f"Mismatched experiment_id in {config_path}"
            )


class TestModuleNamingConvention:
    """Tests for module naming convention."""

    def test_module_naming_convention(self) -> None:
        """Verify each experiment module file name matches its config stem."""
        experiment_ids = ["e16", "e17", "e20", "e21", "e22", "e88"]

        for exp_id in experiment_ids:
            config_path = Path(f"configs/{exp_id}.yaml")
            module_path = Path(f"experiments/{exp_id}.py")

            assert config_path.exists(), f"Config not found: {config_path}"
            assert module_path.exists(), f"Module not found: {module_path}"


class TestExpectedValuesPresent:
    """Tests for expected_values in experiment outputs."""

    @pytest.mark.parametrize("exp_module,exp_id,exp_name", [
        (e16, "e16", "Feature-Space Audit"),
        (e17, "e17", "Neural Manipulation Cost Audit"),
        (e20, "e20", "Three-Level Pattern"),
        (e21, "e21", "VJP Range Finder"),
        (e22, "e22", "Regime Hunt at Scale"),
        (e88, "e88", "Real-World Feature-Space Audit"),
    ])
    @pytest.mark.slow
    def test_expected_values_present(
        self,
        exp_module: Any,
        exp_id: str,
        exp_name: str,
        mock_adapter: MagicMock,
        base_config: dict[str, Any],
    ) -> None:
        """Verify each experiment's output contains expected_values dict."""
        with open(f"configs/{exp_id}.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(exp_module.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = exp_module.run(config)

        assert "expected_values" in result
        assert isinstance(result["expected_values"], dict)
        assert len(result["expected_values"]) > 0, (
            f"expected_values is empty for {exp_id}"
        )


class TestJsonSerializable:
    """Tests for JSON serializability of experiment outputs."""

    @pytest.mark.parametrize("exp_module,exp_id", [
        (e16, "e16"),
        (e17, "e17"),
        (e20, "e20"),
        (e21, "e21"),
        (e22, "e22"),
        (e88, "e88"),
    ])
    @pytest.mark.slow
    def test_json_serializable(
        self,
        exp_module: Any,
        exp_id: str,
        mock_adapter: MagicMock,
        base_config: dict[str, Any],
    ) -> None:
        """Verify each experiment's output can be serialized via json.dumps."""
        with open(f"configs/{exp_id}.yaml", "r", encoding="utf-8") as f:
            exp_config = yaml.safe_load(f)

        config = _merge_config(base_config, exp_config)

        with patch.object(exp_module.DeBERTaV3Adapter, "__new__", return_value=mock_adapter):
            result = exp_module.run(config)

        # This should not raise any exception
        json_str = json.dumps(result)
        assert len(json_str) > 0


class TestUnsupportedModel:
    """Tests for unsupported model handling."""

    def test_unsupported_model_raises(self, base_config: dict[str, Any]) -> None:
        """Verify that a config with an unsupported model name raises NotImplementedError."""
        # Modify config to use unsupported model
        unsupported_config = base_config.copy()
        unsupported_config["model"] = {
            "name": "unsupported/model-name",
            "architecture": "unknown",
        }

        with pytest.raises(NotImplementedError, match="Unsupported model architecture"):
            e16.run(unsupported_config)
