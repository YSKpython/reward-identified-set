"""Reproducibly download and subset the Anthropic HH-RLHF dataset.

This script downloads the Anthropic HH-RLHF dataset from HuggingFace,
selects a deterministic subset of prompts, and saves them as JSON.

Usage:
    python data/download_hh_rlhf.py --n_prompts 500 --seed 42 --output_dir data/processed
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, load_dataset


def download_hh_rlhf(
    split: str = "test",
    cache_dir: str = "data/raw"
) -> Dataset:
    """Download the Anthropic HH-RLHF dataset.

    Args:
        split: Dataset split to download ("train" or "test").
        cache_dir: Directory to cache the downloaded dataset.

    Returns:
        The downloaded dataset object.

    Raises:
        ValueError: If `split` is not "train" or "test".
        RuntimeError: If the download fails (network error, dataset unavailable).
    """
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")

    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}") from e

    return dataset


def select_prompt_subset(
    dataset: Dataset,
    n_prompts: int,
    seed: int = 42
) -> List[int]:
    """Select a deterministic subset of prompt indices from the dataset.

    Args:
        dataset: The dataset to select from.
        n_prompts: Number of prompts to select.
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of selected indices.

    Raises:
        ValueError: If `n_prompts > len(dataset)` or `n_prompts <= 0`.
    """
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive, got {n_prompts}.")

    if n_prompts > len(dataset):
        raise ValueError(
            f"n_prompts ({n_prompts}) exceeds dataset size ({len(dataset)})."
        )

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n_prompts, replace=False)
    return sorted(indices.tolist())


def extract_prompts_and_responses(
    dataset: Dataset,
    indices: List[int]
) -> List[Dict[str, Any]]:
    """Extract prompt, chosen response, and rejected response for selected indices.

    The HH-RLHF dataset has fields `chosen` and `rejected`, each containing a
    conversation string. The prompt is extracted as the last human turn.

    Args:
        dataset: The dataset to extract from.
        indices: List of indices to extract.

    Returns:
        List of dicts with keys: "index", "prompt", "chosen", "rejected".
    """
    records: List[Dict[str, Any]] = []

    for idx in indices:
        item = dataset[idx]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Extract prompt from conversation (last human turn)
        # Conversations are formatted as "\n\nHuman: ...\n\nAssistant: ..."
        # We find the last "\n\nHuman: " occurrence and extract until the next "\n\nAssistant:"
        def extract_prompt(conversation: str) -> str:
            # Split by human/assistant markers
            parts = conversation.split("\n\nHuman: ")
            if len(parts) < 2:
                # Fallback: return the whole conversation
                return conversation.strip()

            # Get the last human part
            last_human_part = parts[-1]
            # Find where the assistant response starts
            assistant_idx = last_human_part.find("\n\nAssistant: ")
            if assistant_idx != -1:
                prompt = last_human_part[:assistant_idx].strip()
            else:
                prompt = last_human_part.strip()
            return prompt

        prompt = extract_prompt(chosen)
        # Verify prompt matches in rejected (should be the same)
        prompt_rejected = extract_prompt(rejected)
        if prompt != prompt_rejected:
            # Use the chosen prompt as canonical
            pass

        records.append({
            "index": int(idx),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return records


def save_subset(records: List[Dict[str, Any]], output_path: str) -> None:
    """Save the subset as a JSON file.

    Args:
        records: List of records to save.
        output_path: Path to the output JSON file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def compute_checksum(path: str) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the SHA-256 checksum.
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def main() -> int:
    """CLI entry point for downloading and subsetting the HH-RLHF dataset.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Download and subset the Anthropic HH-RLHF dataset."
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=500,
        help="Number of prompts to select (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection (default: 42)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for the subset (default: data/processed)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/raw",
        help="HuggingFace cache directory (default: data/raw)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected indices without downloading or saving",
    )

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Split: {args.split}")
    print(f"  Number of prompts: {args.n_prompts}")
    print(f"  Seed: {args.seed}")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Dry run: {args.dry_run}")

    try:
        # Download dataset
        print(f"\nDownloading {args.split} split...")
        dataset = download_hh_rlhf(split=args.split, cache_dir=args.cache_dir)
        print(f"Dataset size: {len(dataset)}")

        # Select subset
        print(f"\nSelecting {args.n_prompts} prompts with seed {args.seed}...")
        indices = select_prompt_subset(dataset, args.n_prompts, seed=args.seed)
        print(f"Selected indices (first 10): {indices[:10]}")

        if args.dry_run:
            print("\n[Dry run] Not downloading or saving files.")
            return 0

        # Extract prompts and responses
        print("\nExtracting prompts and responses...")
        records = extract_prompts_and_responses(dataset, indices)

        # Save subset
        output_filename = f"hh_rlhf_{args.split}_{args.n_prompts}_seed{args.seed}.json"
        output_path = Path(args.output_dir) / output_filename
        print(f"\nSaving subset to {output_path}...")
        save_subset(records, str(output_path))

        # Compute checksum
        checksum = compute_checksum(str(output_path))
        print(f"Checksum (SHA-256): {checksum}")

        # Print summary
        print("\n=== Summary ===")
        print(f"Dataset size: {len(dataset)}")
        print(f"Subset size: {len(records)}")
        print(f"Output path: {output_path}")
        print(f"Checksum: {checksum}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
