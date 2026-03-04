import argparse
import csv
import json
import os
from typing import Any, Iterable, List

import megfile


def _iter_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                yield float(item)
            elif isinstance(item, dict) and "score" in item and isinstance(item["score"], (int, float)):
                yield float(item["score"])
        return
    if isinstance(value, dict):
        if "score" in value and isinstance(value["score"], (int, float)):
            yield float(value["score"])
            return
        for v in value.values():
            if isinstance(v, (int, float)):
                yield float(v)
            elif isinstance(v, dict) and "score" in v and isinstance(v["score"], (int, float)):
                yield float(v["score"])
        return


def _load_json(path: str) -> Any:
    with megfile.smart_open(path, "r") as f:
        return json.load(f)


def _mean_numbers(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.jsons:
        name = os.path.splitext(os.path.basename(path))[0]
        data = _load_json(path)
        values: List[float] = []
        if isinstance(data, dict):
            for v in data.values():
                values.extend(list(_iter_numbers(v)))
        elif isinstance(data, list):
            for v in data:
                values.extend(list(_iter_numbers(v)))
        mean_value = _mean_numbers(values)
        rows.append((name, mean_value))

    out_dir = os.path.dirname(args.out_csv) or "."
    if args.out_csv.startswith(("s3://", "oss://")):
        megfile.smart_makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    with megfile.smart_open(args.out_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "mean"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
