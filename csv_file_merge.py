"""Merge CSV files deterministically without including a previous output."""
from __future__ import annotations

import argparse
from pathlib import Path
from data_generate._csv_tools import atomic_csv_writer, csv_header, csv_rows, validate_output


def merge_multiple_csv(input_dir, output_file, *, allow_schema_union=False, overwrite=False):
    output = Path(output_file).resolve()
    all_files = sorted(path for path in Path(input_dir).glob("*.csv") if path.resolve() != output)
    if not all_files:
        raise FileNotFoundError(f"目录中未找到输入 CSV 文件：{input_dir}")
    output = validate_output(all_files, output, overwrite)
    header = []
    for path in all_files:
        columns = csv_header(path)
        if not header:
            header = columns
        elif set(columns) != set(header):
            if not allow_schema_union:
                raise ValueError(f"CSV 列名不一致：{path}；可用 --allow-schema-union 显式取并集")
            header.extend(column for column in columns if column not in header)
    total = 0
    with atomic_csv_writer(output, header) as writer:
        for path in all_files:
            for row in csv_rows(path):
                writer.writerow(row)
                total += 1
    print(f"已合并 {len(all_files)} 个文件、{total} 行，保存至 {output}")
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-schema-union", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return merge_multiple_csv(args.input_dir, args.output, allow_schema_union=args.allow_schema_union, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
