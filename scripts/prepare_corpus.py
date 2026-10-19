"""Sample and tokenize the training data of BERT and GPT-2: 120M tokens each (Appendix E).

* ``wikibook``     English Wikipedia (``wikimedia/wikipedia``, 20231101.en) and BookCorpus (``bookcorpus/bookcorpus``)
                   for BERT, mixed 3:1 by tokens (the proportions of the BERT pre-training corpus). Every block is
                   ``[CLS] <block_size - 2 tokens> [SEP]``.
* ``openwebtext``  OpenWebText (``Skylion007/openwebtext``) for GPT-2, documents separated by ``<|endoftext|>``.

Documents are tokenized with the teacher's tokenizer, concatenated and split into blocks of ``--block_size`` tokens.
The result is saved with ``save_to_disk`` to ``<out_dir>/<corpus>_<num_tokens / 1e6>M-<block_size>``:

    python scripts/prepare_corpus.py --corpus wikibook --model google-bert/bert-base-uncased --block_size 512 \
        --out_dir data/bert
    python scripts/prepare_corpus.py --corpus openwebtext --model gpt2 --block_size 1024 --out_dir data/gpt2

Wikipedia and OpenWebText are streamed and shuffled with a buffer of documents. BookCorpus (one sentence per line)
is downloaded completely and sampled as passages of consecutive lines.
"""
import argparse
import os
import random

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

CORPORA = {
    "wikibook": [("wikimedia/wikipedia", "20231101.en", 0.75), ("bookcorpus/bookcorpus", None, 0.25)],
    "openwebtext": [("Skylion007/openwebtext", None, 1.0)],
}
BOOK_PASSAGE_LINES = 256  # consecutive BookCorpus lines (sentences) per passage


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    parser.add_argument("--model", required=True, help="Model whose tokenizer is used")
    parser.add_argument("--num_tokens", type=int, default=120_000_000)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_buffer", type=int, default=10000, help="Documents in the streaming shuffle buffer")
    return parser.parse_args()


def documents(name, config, seed, shuffle_buffer):
    """Texts of a corpus in random order."""
    if name == "bookcorpus/bookcorpus":
        lines = load_dataset(name, split="train", trust_remote_code=True)["text"]
        starts = list(range(0, len(lines), BOOK_PASSAGE_LINES))
        random.Random(seed).shuffle(starts)
        for start in starts:
            yield " ".join(lines[start: start + BOOK_PASSAGE_LINES])
    else:
        stream = load_dataset(name, config, split="train", streaming=True, trust_remote_code=True)
        for example in stream.shuffle(seed=seed, buffer_size=shuffle_buffer):
            yield example["text"]


def blocks(tokenizer, texts, num_tokens, block_size, masked_lm):
    """Token blocks of ``block_size`` tokens from ``texts`` until ``num_tokens`` tokens are collected."""
    if masked_lm:  # [CLS] ... [SEP], documents separated by [SEP]
        body, prefix, suffix, separator = block_size - 2, [tokenizer.cls_token_id], [tokenizer.sep_token_id], \
            [tokenizer.sep_token_id]
    else:          # documents separated by <|endoftext|>
        body, prefix, suffix, separator = block_size, [], [], [tokenizer.eos_token_id]
    buffer, collected, batch = [], 0, []

    def flush(batch):
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            buffer.extend(ids + separator)

    for text in texts:
        batch.append(text)
        if len(batch) < 1000:
            continue
        flush(batch)
        batch = []
        while len(buffer) >= body:
            ids = prefix + buffer[:body] + suffix
            del buffer[:body]
            yield ids
            collected += len(ids)
            if collected >= num_tokens:
                return


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    masked_lm = args.corpus == "wikibook"

    def generate():
        for name, config, share in CORPORA[args.corpus]:
            texts = documents(name, config, args.seed, args.shuffle_buffer)
            for ids in blocks(tokenizer, texts, round(args.num_tokens * share), args.block_size, masked_lm):
                example = {"input_ids": ids, "attention_mask": [1] * len(ids)}
                if masked_lm:
                    example["token_type_ids"] = [0] * len(ids)
                else:
                    example["labels"] = ids
                yield example

    dataset = Dataset.from_generator(generate).shuffle(seed=args.seed).flatten_indices()
    save_dir = os.path.join(args.out_dir, f"{args.corpus}_{args.num_tokens // 1_000_000}M-{args.block_size}")
    dataset.save_to_disk(save_dir)
    print(f"Saved {len(dataset)} blocks of {args.block_size} tokens to {save_dir}")


if __name__ == "__main__":
    main()
