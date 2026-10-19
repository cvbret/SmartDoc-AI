"""Compare saved runs without model imports or API calls."""

import argparse
import json
from pathlib import Path

from evaluation.comparison import ComparisonError, compare_evaluations, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--reranker", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    paths = [args.baseline.resolve(), args.reranker.resolve(), args.output.resolve()]
    if args.report:
        paths.append(args.report.resolve())
    if len(paths) != len(set(paths)):
        parser.error("Input and output paths must be distinct")
    try:
        result = compare_evaluations(json.loads(args.baseline.read_text(encoding="utf-8")),
                                     json.loads(args.reranker.read_text(encoding="utf-8")))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.report:
            args.report.write_text(render_report(result), encoding="utf-8")
    except ComparisonError as exc:
        parser.error(str(exc))
    print(json.dumps(result["retrieval_summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
