# Data Directory

This directory contains scripts and documentation for data acquisition and processing.

## Dataset: Anthropic HH-RLHF

### Source

- **Name**: Anthropic HH-RLHF (Helpful and Harmless Reinforcement Learning from Human Feedback)
- **HuggingFace**: `Anthropic/hh-rlhf`
- **License**: CC BY 4.0
- **Citation**: 
  ```
  @article{bai2022training,
    title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
    author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Nova and Drain, Dawn and Fort, Stanislav and Ganguli, Deep and Henighan, Tom and Joseph, Nicholas and Kadavath, Saurav and Kernion, Jackson and Conerly, Connor and El-Showk, Sheer and Elhage, Nelson and Hatfield-Dodds, Zac and Hernndez, David and Hume, Tristan and Johnston, Scott and Kravec, Shauna and Lovitt, Liane and Nanda, Neel and Olsson, Catherine and Amodei, Dario and Brown, Tom and Clark, Jack and McCandlish, Sam and Olah, Christopher and Mann, Ben and Kaplan, Jared},
    journal={arXiv preprint arXiv:2204.05862},
    year={2022}
  }
  ```

### Dataset Structure

The dataset contains conversation pairs with the following fields:
- `chosen`: The preferred response conversation (string)
- `rejected`: The non-preferred response conversation (string)

Each conversation is formatted as alternating human/assistant turns:
```
\n\nHuman: <prompt>\n\nAssistant: <response>...
```

### Dataset Size

- **Training split**: ~170,000 prompt/response pairs
- **Test split**: ~8,544 prompt/response pairs

Only the test split is used in the paper's experiments.

## Subset Selection

For the feature-space vulnerability audits (E16, E88), we use a deterministic subset of 500 prompts from the test split:

- **Selection method**: Random sampling without replacement
- **Random seed**: 42
- **Implementation**: `np.random.RandomState(42).choice(len(dataset), size=500, replace=False)`
- **Indices**: Sorted in ascending order for reproducibility

For the neural MC audit (E17), a 100-prompt subset is used.

## Output Format

The processed data is saved as JSON with the following structure:

```json
[
  {
    "index": 123,
    "prompt": "<extracted prompt text>",
    "chosen": "<full chosen conversation>",
    "rejected": "<full rejected conversation>"
  },
  ...
]
```

### Fields

- `index`: Original index in the test split (integer)
- `prompt`: Extracted prompt text (last human turn)
- `chosen`: Full chosen conversation string
- `rejected`: Full rejected conversation string

## Usage

Download and process the dataset:

```bash
python data/download_hh_rlhf.py --n_prompts 500 --seed 42 --output_dir data/processed
```

### Arguments

- `--n_prompts`: Number of prompts to select (default: 500)
- `--seed`: Random seed for reproducibility (default: 42)
- `--split`: Dataset split, "train" or "test" (default: "test")
- `--output_dir`: Output directory for processed data (default: "data/processed")
- `--cache_dir`: HuggingFace cache directory (default: "data/raw")
- `--dry_run`: Print selected indices without downloading

## Data Integrity

After running the download script, record the SHA-256 checksum for verification:

```
Checksum (SHA-256): <to be filled after first run>
```

The script is idempotent: running it twice with the same arguments produces identical output.

## File Locations

- `data/raw/`: HuggingFace cache (automatically managed)
- `data/processed/`: Processed JSON subsets
- `data/download_hh_rlhf.py`: Download and subset script
- `data/README.md`: This documentation file
