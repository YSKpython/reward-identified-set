"""Tests for the HH-RLHF download pipeline.

These tests verify the download pipeline functionality WITHOUT network access.
The datasets.load_dataset call is mocked to avoid actual downloads.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data.download_hh_rlhf import (
    compute_checksum,
    download_hh_rlhf,
    extract_prompts_and_responses,
    save_subset,
    select_prompt_subset,
)


def create_mock_dataset(size: int = 1000) -> MagicMock:
    """Create a mock dataset for testing.

    Args:
        size: Number of items in the mock dataset.

    Returns:
        A MagicMock object simulating a datasets.Dataset.
    """
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=size)

    def get_item(idx: int) -> Dict[str, str]:
        return {
            "chosen": f"\n\nHuman: Test prompt {idx}\n\nAssistant: Chosen response {idx}",
            "rejected": f"\n\nHuman: Test prompt {idx}\n\nAssistant: Rejected response {idx}",
        }

    mock_dataset.__getitem__ = MagicMock(side_effect=get_item)
    return mock_dataset


class TestSelectPromptSubset:
    """Tests for select_prompt_subset function."""

    def test_select_subset_deterministic(self) -> None:
        """Calling select_prompt_subset twice with the same seed produces identical indices."""
        mock_dataset = create_mock_dataset(1000)
        n_prompts = 100
        seed = 42

        indices1 = select_prompt_subset(mock_dataset, n_prompts, seed=seed)
        indices2 = select_prompt_subset(mock_dataset, n_prompts, seed=seed)

        assert indices1 == indices2, "Indices should be identical with same seed"

    def test_select_subset_different_seeds_differ(self) -> None:
        """Different seeds produce different subsets."""
        mock_dataset = create_mock_dataset(1000)
        n_prompts = 100

        indices1 = select_prompt_subset(mock_dataset, n_prompts, seed=42)
        indices2 = select_prompt_subset(mock_dataset, n_prompts, seed=123)

        assert indices1 != indices2, "Indices should differ with different seeds"

    def test_select_subset_sorted(self) -> None:
        """Selected indices are sorted in ascending order."""
        mock_dataset = create_mock_dataset(1000)
        n_prompts = 100

        indices = select_prompt_subset(mock_dataset, n_prompts, seed=42)

        assert indices == sorted(indices), "Indices should be sorted"

    def test_select_subset_size(self) -> None:
        """The number of selected indices equals n_prompts."""
        mock_dataset = create_mock_dataset(1000)
        n_prompts = 100

        indices = select_prompt_subset(mock_dataset, n_prompts, seed=42)

        assert len(indices) == n_prompts, f"Should have {n_prompts} indices"

    def test_select_subset_too_large_raises(self) -> None:
        """Requesting more prompts than available raises ValueError."""
        mock_dataset = create_mock_dataset(100)

        with pytest.raises(ValueError, match="exceeds dataset size"):
            select_prompt_subset(mock_dataset, n_prompts=101, seed=42)

    def test_select_subset_zero_raises(self) -> None:
        """n_prompts=0 raises ValueError."""
        mock_dataset = create_mock_dataset(100)

        with pytest.raises(ValueError, match="must be positive"):
            select_prompt_subset(mock_dataset, n_prompts=0, seed=42)


class TestExtractPromptsAndResponses:
    """Tests for extract_prompts_and_responses function."""

    def test_extract_prompts_structure(self) -> None:
        """Mock a dataset with known chosen/rejected fields. Verify extracted records have expected keys."""
        mock_dataset = create_mock_dataset(100)
        indices = [0, 5, 10]

        records = extract_prompts_and_responses(mock_dataset, indices)

        assert len(records) == len(indices)
        for record in records:
            assert "index" in record
            assert "prompt" in record
            assert "chosen" in record
            assert "rejected" in record
            assert isinstance(record["index"], int)
            assert isinstance(record["prompt"], str)
            assert isinstance(record["chosen"], str)
            assert isinstance(record["rejected"], str)


class TestSaveSubset:
    """Tests for save_subset function."""

    def test_save_subset_creates_file(self, tmp_path: Path) -> None:
        """Verify save_subset creates a JSON file at the specified path."""
        records: List[Dict[str, Any]] = [
            {"index": 0, "prompt": "test", "chosen": "c", "rejected": "r"}
        ]
        output_path = tmp_path / "test_output.json"

        save_subset(records, str(output_path))

        assert output_path.exists(), "Output file should exist"
        assert output_path.is_file(), "Output should be a file"

    def test_save_subset_roundtrip(self, tmp_path: Path) -> None:
        """Save records, load them back, verify they match."""
        records: List[Dict[str, Any]] = [
            {"index": 0, "prompt": "prompt 0", "chosen": "chosen 0", "rejected": "rejected 0"},
            {"index": 5, "prompt": "prompt 5", "chosen": "chosen 5", "rejected": "rejected 5"},
        ]
        output_path = tmp_path / "roundtrip.json"

        save_subset(records, str(output_path))

        with open(output_path, "r", encoding="utf-8") as f:
            loaded_records = json.load(f)

        assert loaded_records == records, "Loaded records should match original"


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    def test_compute_checksum_deterministic(self, tmp_path: Path) -> None:
        """Checksum of the same content is identical across calls."""
        content = b"test content for checksum"
        file_path = tmp_path / "checksum_test.bin"

        file_path.write_bytes(content)

        checksum1 = compute_checksum(str(file_path))
        checksum2 = compute_checksum(str(file_path))

        assert checksum1 == checksum2, "Checksums should be identical"

    def test_compute_checksum_different_content(self, tmp_path: Path) -> None:
        """Different content produces different checksums."""
        file1 = tmp_path / "file1.bin"
        file2 = tmp_path / "file2.bin"

        file1.write_bytes(b"content one")
        file2.write_bytes(b"content two")

        checksum1 = compute_checksum(str(file1))
        checksum2 = compute_checksum(str(file2))

        assert checksum1 != checksum2, "Different content should have different checksums"


class TestDownloadHhRlhf:
    """Tests for download_hh_rlhf function."""

    @patch("data.download_hh_rlhf.load_dataset")
    def test_download_invalid_split_raises(self, mock_load: MagicMock) -> None:
        """split='validation' raises ValueError."""
        # Note: The function validates split before calling load_dataset
        with pytest.raises(ValueError, match="Invalid split"):
            download_hh_rlhf(split="validation", cache_dir="data/raw")


class TestCliDryRun:
    """Tests for CLI dry_run functionality."""

    @patch("data.download_hh_rlhf.download_hh_rlhf")
    def test_cli_dry_run(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Running with --dry_run does not create output files."""
        from data.download_hh_rlhf import main

        # Create mock dataset
        mock_dataset = create_mock_dataset(1000)
        mock_download.return_value = mock_dataset

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        import sys
        original_argv = sys.argv
        sys.argv = [
            "download_hh_rlhf.py",
            "--n_prompts", "10",
            "--seed", "42",
            "--split", "test",
            "--output_dir", str(output_dir),
            "--cache_dir", str(tmp_path / "cache"),
            "--dry_run",
        ]

        try:
            exit_code = main()
            assert exit_code == 0, "CLI should succeed in dry run mode"
        finally:
            sys.argv = original_argv

        # Verify no files were created in output directory
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) == 0, "No output files should be created in dry run mode"


@pytest.mark.parametrize("n_prompts,expected_count", [
    (10, 10),
    (50, 50),
    (100, 100),
])
def test_select_subset_parametrized(n_prompts: int, expected_count: int) -> None:
    """Parametrized test for subset size verification."""
    mock_dataset = create_mock_dataset(1000)

    indices = select_prompt_subset(mock_dataset, n_prompts, seed=42)

    assert len(indices) == expected_count
