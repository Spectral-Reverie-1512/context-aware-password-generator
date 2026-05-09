from __future__ import annotations

import argparse
import json
from pathlib import Path

from .processor import ContextProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2.5 context encoder")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw context text file (plain text).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the 128-d context vector as JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device (e.g., cpu, cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = ContextProcessor(device=args.device)
    vector = processor.encode_file(Path(args.input)).tolist()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(vector), encoding="utf-8")
    else:
        print(json.dumps(vector))


if __name__ == "__main__":
    main()

