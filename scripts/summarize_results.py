"""Print lm-eval results in the column order and metrics of the paper tables (Tables 3 and 4).

    python scripts/summarize_results.py outputs/llama3.2-3b_28to19/eval_post_train [more result dirs ...]

Each argument is a ``results_*.json`` file written by ``lm_eval --output_path`` or a directory containing one
(searched recursively; the newest file is used). Scores are in %, ``AVG`` is the mean over the ten tasks.
"""
import argparse
import glob
import json
import os

# (column, lm-eval task, metric)
COLUMNS = [
    ("ARC-E", "arc_easy", "acc_norm"),
    ("ARC-C", "arc_challenge", "acc_norm"),
    ("RACE", "race", "acc"),
    ("BoolQ", "boolq", "acc"),
    ("WnGd", "winogrande", "acc"),
    ("HeSW", "hellaswag", "acc_norm"),
    ("PIQA", "piqa", "acc_norm"),
    ("MathQA", "mathqa", "acc_norm"),
    ("COQA", "coqa", "f1"),
    ("MMLU", "mmlu", "acc"),
]
def find_results(path):
    if os.path.isfile(path):
        return path
    files = glob.glob(os.path.join(path, "**", "results*.json"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no results*.json under {path}")
    return max(files, key=os.path.getmtime)


def scores(results_file):
    results = json.load(open(results_file))["results"]
    row = {}
    for column, task, metric in COLUMNS:
        entry = results.get(task, {})
        value = entry.get(f"{metric},none", entry.get(metric))
        row[column] = None if value is None else 100 * value
    valid = [v for v in row.values() if v is not None]
    row["AVG"] = sum(valid) / len(valid) if len(valid) == len(COLUMNS) else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--names", nargs="*", help="Row names (default: the given paths)")
    args = parser.parse_args()
    names = args.names or args.results
    header = [c for c, _, _ in COLUMNS] + ["AVG"]
    width = max(len(n) for n in names)
    print(f"{'':{width}}  " + " ".join(f"{h:>6}" for h in header))
    for name, path in zip(names, args.results):
        row = scores(find_results(path))
        print(f"{name:{width}}  " + " ".join("     -" if row[h] is None else f"{row[h]:6.1f}" for h in header))


if __name__ == "__main__":
    main()
