#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

from stage1.segmenter import Segmenter
from stage1.vocab import VocabBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Segmenter + raw vocab")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to RockYou plaintext file (one password per line).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write outputs (default: output).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=12,
        help="Max token length for vocab filtering (default: 12).",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Min frequency threshold for vocab filtering (default: 5).",
    )
    parser.add_argument(
        "--wordlist",
        default=None,
        help="Optional wordlist path for word detection.",
    )
    return parser.parse_args()


def write_segmentations(output_path: Path, rows: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row)
            f.write("\n")


def write_vocab(output_path: Path, vocab_counts: Counter, min_freq: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for token, count in vocab_counts.most_common():
            if count < min_freq:
                break
            f.write(f"{token}\t{count}\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    segmentations_path = output_dir / "segmentations.tsv"
    vocab_path = output_dir / "vocab_raw.tsv"

    segmenter = Segmenter(wordlist_path=args.wordlist)
    vocab_builder = VocabBuilder(max_len=args.max_len)

    segmentation_rows: list[str] = []

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            password = line.rstrip("\r\n")
            if not password:
                continue

            segments = segmenter.segment(password)
            row = [password, "1"]
            for segment, tag in segments:
                row.append(segment)
                row.append(tag)
            segmentation_rows.append("\t".join(row))

            vocab_builder.add_segments(segments)

    write_segmentations(segmentations_path, segmentation_rows)
    write_vocab(vocab_path, vocab_builder.counts, args.min_freq)


if __name__ == "__main__":
    main()

