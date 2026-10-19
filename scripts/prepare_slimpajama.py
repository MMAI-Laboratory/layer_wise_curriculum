"""Sample and tokenize a training subset of SlimPajama-6B (``DKYoon/SlimPajama-6B``).

Documents are drawn per RedPajama domain with the proportions below (the domain mixture of Sheared-LLaMA,
Xia et al., 2024), tokenized with the teacher's tokenizer,
concatenated and split into blocks of ``--block_size`` tokens. The result is saved with ``save_to_disk`` to
``<out_dir>/random_<num_data>-<block_size>``.

    python scripts/prepare_slimpajama.py --model meta-llama/Llama-3.2-3B --num_data 250000 \
        --out_dir data/slimpajama/llama3

CommonCrawl, ArXiv and Book documents are much longer than the others, so their proportions are divided by
a normalization value to keep the number of *tokens* per domain balanced.
"""
import argparse
import os
from functools import partial

from datasets import concatenate_datasets
from transformers import AutoTokenizer

from layer_curriculum.data import group_texts, load_raw_dataset

PROPORTIONS = {
    "RedPajamaC4": 0.492,
    "RedPajamaStackExchange": 0.01,
    "RedPajamaCommonCrawl": 0.361 / 3,
    "RedPajamaGithub": 0.008,
    "RedPajamaWikipedia": 0.031,
    "RedPajamaArXiv": 0.007 / 20,
    "RedPajamaBook": 0.091 / 200,
}
NUM_HELD_OUT = 3000  # documents excluded from sampling


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model whose tokenizer is used")
    parser.add_argument("--dataset", default="DKYoon/SlimPajama-6B", help="HF hub id or local copy of the train split")
    parser.add_argument("--num_data", type=int, default=250000, help="Number of sampled documents")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Seed of the random document sampling")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dataset = load_raw_dataset(args.dataset, split="train")

    sampled = []
    for name, proportion in PROPORTIONS.items():
        domain = dataset.filter(lambda x: x["meta"] == {"redpajama_set_name": name}, keep_in_memory=True,
                                num_proc=args.num_proc)
        rest = domain.train_test_split(test_size=(NUM_HELD_OUT * proportion) / len(domain), seed=args.seed)["train"]
        sampled.append(rest.train_test_split(test_size=1 - (args.num_data * proportion) / len(rest),
                                             seed=args.seed)["train"])
    dataset = concatenate_datasets(sampled)

    column_names = dataset.column_names
    text_column = "text" if "text" in column_names else column_names[0]
    dataset = dataset.map(lambda examples: tokenizer(examples[text_column]), batched=True, keep_in_memory=True,
                          remove_columns=column_names, num_proc=args.num_proc, desc="Tokenizing")
    dataset = dataset.map(partial(group_texts, block_size=args.block_size), batched=True, keep_in_memory=True,
                          num_proc=args.num_proc, desc=f"Grouping texts in chunks of {args.block_size}")

    save_dir = os.path.join(args.out_dir, f"random_{args.num_data}-{args.block_size}")
    dataset.save_to_disk(save_dir)
    print(f"Saved {len(dataset)} blocks of {args.block_size} tokens to {save_dir}")


if __name__ == "__main__":
    main()
