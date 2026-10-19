"""Run offline RAGAS evaluation. No result file is written unless --output is set."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from evaluation.metrics import safe_error
from evaluation.runner import run_evaluation


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "reranker"), default="reranker")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--sample", nargs="+", help="Optional stable sample IDs, for targeted smoke runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_evaluation(args.mode, args.limit, args.sample))
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(json.dumps({"metadata": report["metadata"], "summary": report["summary"]},
                         ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["metadata"]["completed_sample_count"] == report["metadata"]["configured_sample_count"] else 1
    except Exception as exc:
        # Includes setup/index creation and output failures; no raw traceback/secret.
        print(json.dumps(safe_error("run_setup_or_output", exc)), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
