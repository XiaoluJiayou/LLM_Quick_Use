"""Reproducible CSV reservoir sampling with bounded memory."""
from __future__ import annotations

import argparse
import random
try:
    from ._csv_tools import atomic_csv_writer, csv_header, csv_rows, validate_output
except ImportError:
    from _csv_tools import atomic_csv_writer, csv_header, csv_rows, validate_output


def random_sample_csv(input_file, output_file, total_sample_size=1000, *, seed=42, columns=None, overwrite=False):
    if total_sample_size < 0:
        raise ValueError("抽样数量不能为负数")
    output = validate_output([input_file], output_file, overwrite)
    header = csv_header(input_file)
    selected = list(columns) if columns else header
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(header):
        raise ValueError("抽样列名必须存在且不能重复")
    rng = random.Random(seed)
    reservoir = []
    for index, row in enumerate(csv_rows(input_file)):
        if index < total_sample_size:
            reservoir.append(row)
        elif total_sample_size:
            position = rng.randrange(index + 1)
            if position < total_sample_size:
                reservoir[position] = row
    rng.shuffle(reservoir)
    with atomic_csv_writer(output, selected) as writer:
        writer.writerows({key: row[key] for key in selected} for row in reservoir)
    print(f"已抽样 {len(reservoir)} 行，保存至 {output}")
    return len(reservoir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return random_sample_csv(args.input, args.output, args.count, seed=args.seed, columns=args.columns, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
