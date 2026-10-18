"""Two-pass proportional stratified CSV sampling; memory is O(sample size)."""
from __future__ import annotations

import argparse
import random
from collections import Counter
try:
    from ._csv_tools import atomic_csv_writer, csv_header, csv_rows, validate_output
except ImportError:
    from _csv_tools import atomic_csv_writer, csv_header, csv_rows, validate_output


def stratified_sample_csv(input_file, output_file, total_sample_size, *, seed=42, category_column="一级领域", overwrite=False):
    if total_sample_size < 0:
        raise ValueError("抽样数量不能为负数")
    output = validate_output([input_file], output_file, overwrite)
    header = csv_header(input_file)
    if category_column not in header:
        raise ValueError(f"CSV 缺少分层列：{category_column}")
    counts = Counter(row[category_column] for row in csv_rows(input_file))
    total_rows = sum(counts.values())
    target = min(total_sample_size, total_rows)
    allocations = {key: (count * target // total_rows if total_rows else 0) for key, count in counts.items()}
    deficit = target - sum(allocations.values())
    # Integer remainders avoid floating point rounding, retaining input order for ties.
    order = sorted(counts, key=lambda key: -(counts[key] * target % total_rows)) if total_rows else []
    for key in order[:deficit]:
        allocations[key] += 1
    rng = random.Random(seed)
    reservoirs = {key: [] for key in counts}
    seen = Counter()
    for row in csv_rows(input_file):
        key = row[category_column]
        seen[key] += 1
        limit = allocations[key]
        bucket = reservoirs[key]
        if len(bucket) < limit:
            bucket.append(row)
        elif limit:
            position = rng.randrange(seen[key])
            if position < limit:
                bucket[position] = row
    if seen != counts:
        raise ValueError("输入 CSV 在两遍读取间发生变化，请重新运行")
    sampled = [row for bucket in reservoirs.values() for row in bucket]
    rng.shuffle(sampled)
    with atomic_csv_writer(output, header) as writer:
        writer.writerows(sampled)
    print(f"已按 {category_column} 分层抽样 {len(sampled)}/{total_rows} 行，保存至 {output}")
    return len(sampled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category-column", default="一级领域")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return stratified_sample_csv(args.input, args.output, args.count, seed=args.seed, category_column=args.category_column, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
