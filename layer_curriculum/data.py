import os
from itertools import chain

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


def load_tokenized_dataset(path: str, holdout_percentage: int = 0) -> Dataset:
    """Load a tokenized dataset produced by ``scripts/prepare_slimpajama.py`` (or any ``save_to_disk`` output).

    ``path`` may be a directory of ``*.arrow`` shards, a ``save_to_disk`` directory or a single ``.arrow`` file.
    With ``holdout_percentage > 0`` the first ``holdout_percentage`` % of the blocks are held out (reserved for
    validation) and only the remaining blocks are returned.
    """
    if os.path.isfile(path):
        files = [path]
    else:
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".arrow")]
        if not files:
            dataset = load_from_disk(path)
            return dataset.select(range(round(len(dataset) * holdout_percentage / 100), len(dataset)))
    if holdout_percentage > 0:
        return load_dataset("arrow", data_files=files, split=f"train[{holdout_percentage}%:]")
    return concatenate_datasets([Dataset.from_file(f) for f in files])


def load_raw_dataset(name_or_path: str, split: str = "train") -> Dataset:
    """Load a raw text dataset from the HF Hub (e.g. ``DKYoon/SlimPajama-6B``) or a local directory."""
    if os.path.isdir(name_or_path):
        return load_tokenized_dataset(name_or_path)
    return load_dataset(name_or_path, split=split)


def group_texts(examples, block_size: int):
    """Concatenate all texts and split them into chunks of ``block_size`` tokens (remainder dropped)."""
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result
