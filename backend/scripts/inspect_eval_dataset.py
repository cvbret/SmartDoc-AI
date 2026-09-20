"""Inspect all fixed answers with supporting texts; no model or metric calls."""

import argparse
import json

from evaluation.dataset import dataset_statistics, load_evaluation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs="+", help="Optional sample IDs; by default show all")
    args = parser.parse_args()
    data, contexts = load_evaluation_dataset()
    if args.sample and set(args.sample) - {sample["id"] for sample in data["samples"]}:
        parser.error("Unknown sample ID")
    print(json.dumps(dataset_statistics(data, contexts), ensure_ascii=False, indent=2))
    for sample in data["samples"]:
        if args.sample and sample["id"] not in args.sample:
            continue
        print(f"\n{sample['id']} [{sample['type']}, {sample['difficulty']}]")
        print(f"Question: {sample['question']}")
        print(f"Reference Answer: {sample['reference_answer']}")
        for context_id in sample["reference_context_ids"]:
            print(f"Reference Context [{context_id}]:\n{contexts[context_id]}")


if __name__ == "__main__":
    main()
