#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("汇总style低频消融的rho指标")
    p.add_argument("--root_dir", required=True, help="消融输出根目录，包含多个case子目录")
    p.add_argument("--out_csv", required=True, help="输出汇总csv路径")
    return p.parse_args()


def safe_mean(xs):
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def main():
    args = parse_args()
    root = Path(args.root_dir)
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    rows = []

    for case_dir in case_dirs:
        metrics_path = case_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        by_key_all = defaultdict(list)
        by_key_early = defaultdict(list)
        by_key_mid = defaultdict(list)
        by_key_late = defaultdict(list)

        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                rec = json.loads(s)
                key = str(rec.get("key", ""))
                rho = float(rec.get("rho_lp", 0.0))
                prog = float(rec.get("progress", 0.0))
                by_key_all[key].append(rho)
                if prog < 0.35:
                    by_key_early[key].append(rho)
                elif prog < 0.7:
                    by_key_mid[key].append(rho)
                else:
                    by_key_late[key].append(rho)

        all_keys = sorted(by_key_all.keys())
        case_all = []
        case_early = []
        case_mid = []
        case_late = []
        for k in all_keys:
            case_all.append(safe_mean(by_key_all[k]))
            case_early.append(safe_mean(by_key_early[k]))
            case_mid.append(safe_mean(by_key_mid[k]))
            case_late.append(safe_mean(by_key_late[k]))

        rows.append(
            {
                "case": case_dir.name,
                "num_keys": len(all_keys),
                "rho_mean_all": safe_mean(case_all),
                "rho_mean_early": safe_mean(case_early),
                "rho_mean_mid": safe_mean(case_mid),
                "rho_mean_late": safe_mean(case_late),
                "metrics_jsonl": str(metrics_path),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "num_keys",
                "rho_mean_all",
                "rho_mean_early",
                "rho_mean_mid",
                "rho_mean_late",
                "metrics_jsonl",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[DONE] rows={len(rows)} csv={out_csv}")


if __name__ == "__main__":
    main()
