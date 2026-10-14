#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def sample_rows(input_path: Path, output_path: Path, sample_size: int, seed: int | None) -> None:
    with input_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()

    total = len(lines)
    if sample_size > total:
        raise ValueError(
            f"Requested sample_size={sample_size}, but file only has {total} rows."
        )

    rng = random.Random(seed)
    sampled_lines = rng.sample(lines, sample_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.writelines(sampled_lines)

    print(
        f"Wrote {sample_size} randomly selected rows from {input_path} "
        f"(total {total}) to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample rows from a text file.")
    parser.add_argument("input_path", type=Path, help="Input text file path")
    parser.add_argument("output_path", type=Path, help="Output text file path")
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=20000,
        help="Number of rows to sample (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    args = parser.parse_args()

    sample_rows(args.input_path, args.output_path, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
