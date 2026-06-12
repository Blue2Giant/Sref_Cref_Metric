#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_input_rows(path: Path) -> Tuple[List[Tuple[str, List[str]]], Dict[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    mapping: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                continue
            for key, value in obj.items():
                key = str(key).strip()
                if not key:
                    continue
                if isinstance(value, list):
                    paths = [x for x in value if isinstance(x, str)]
                elif isinstance(value, str):
                    paths = [value]
                else:
                    paths = []
                rows.append((key, paths))
                mapping[key] = paths
    return rows, mapping


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_per_image_paths(rec: Dict) -> List[str]:
    detail = rec.get("detail", {}) if isinstance(rec.get("detail"), dict) else {}
    per_image = detail.get("per_image", []) if isinstance(detail.get("per_image"), list) else []
    out: List[str] = []
    for item in per_image:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if path:
            out.append(path)
    return out


def extract_path_buckets(rec: Dict) -> Tuple[str, List[str], List[str], List[str]]:
    key = str(rec.get("key", "")).strip()
    detail = rec.get("detail", {}) if isinstance(rec.get("detail"), dict) else {}
    if detail.get("mode") == "jsonl_key_aggregate":
        true_paths = list(detail.get("true_paths", []) or [])
        false_paths = list(detail.get("false_paths", []) or [])
        error_paths = list(detail.get("error_paths", []) or [])
        return key, true_paths, false_paths, error_paths

    path = str(rec.get("path", "")).strip()
    status = str(rec.get("status") or detail.get("status") or ("true" if rec.get("bad_similar_people") else "false"))
    true_paths = [path] if path and status == "true" else []
    false_paths = [path] if path and status == "false" else []
    error_paths = [path] if path and status == "error" else []
    return key, true_paths, false_paths, error_paths


def verify_detail_coverage(
    input_map: Dict[str, List[str]],
    detail_results: List[Dict],
) -> Dict:
    detail_map: Dict[str, List[str]] = {}
    for rec in detail_results:
        key = str(rec.get("key", "")).strip()
        if not key:
            continue
        detail_map[key] = extract_per_image_paths(rec)

    input_keys = set(input_map)
    detail_keys = set(detail_map)
    missing_keys = sorted(input_keys - detail_keys)
    extra_keys = sorted(detail_keys - input_keys)
    count_mismatch = []
    path_mismatch = []

    for key in sorted(input_keys & detail_keys):
        input_paths = input_map[key]
        detail_paths = detail_map[key]
        if len(input_paths) != len(detail_paths):
            count_mismatch.append(
                {
                    "key": key,
                    "input_count": len(input_paths),
                    "detail_count": len(detail_paths),
                }
            )
        if Counter(input_paths) != Counter(detail_paths):
            missing_paths = list((Counter(input_paths) - Counter(detail_paths)).elements())
            extra_paths = list((Counter(detail_paths) - Counter(input_paths)).elements())
            path_mismatch.append(
                {
                    "key": key,
                    "missing_count": len(missing_paths),
                    "extra_count": len(extra_paths),
                    "missing_examples": missing_paths[:10],
                    "extra_examples": extra_paths[:10],
                }
            )

    return {
        "input_key_count": len(input_map),
        "detail_key_count": len(detail_map),
        "input_total_images": sum(len(v) for v in input_map.values()),
        "detail_total_images": sum(len(v) for v in detail_map.values()),
        "missing_key_count": len(missing_keys),
        "extra_key_count": len(extra_keys),
        "count_mismatch_key_count": len(count_mismatch),
        "path_mismatch_key_count": len(path_mismatch),
        "missing_key_examples": missing_keys[:10],
        "extra_key_examples": extra_keys[:10],
        "count_mismatch_examples": count_mismatch[:10],
        "path_mismatch_examples": path_mismatch[:10],
    }


def key_hits_whitelist(key: str, whitelist_ids: List[str]) -> bool:
    parts = str(key).split("__")
    return any(item in parts for item in whitelist_ids)


def write_subset_jsonl(
    rows: List[Tuple[str, List[str]]],
    selected_keys: List[str],
    out_path: Path,
) -> None:
    selected_set = set(selected_keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for key, paths in rows:
            if key in selected_set:
                f.write(json.dumps({key: paths}, ensure_ascii=False) + "\n")


def run_rerun(
    judge_script: Path,
    subset_jsonl: Path,
    rerun_dir: Path,
    judge_times: int,
    min_true: int,
    min_similar_people: int,
    num_procs: int,
    probe_timeout: float,
    endpoints: List[str],
) -> None:
    if rerun_dir.exists():
        shutil.rmtree(rerun_dir)
    rerun_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(judge_script),
        "--input-jsonl",
        str(subset_jsonl),
        "--jsonl-task-mode",
        "aggregate_key",
        "--judge_times",
        str(judge_times),
        "--min_true",
        str(min_true),
        "--min_similar_people",
        str(min_similar_people),
        "--num_procs",
        str(num_procs),
        "--probe-timeout",
        str(probe_timeout),
        "--out_all",
        str(rerun_dir / "similar_people_all.json"),
        "--out_pos",
        str(rerun_dir / "similar_people_bad.json"),
        "--out_neg",
        str(rerun_dir / "similar_people_good.json"),
        "--out_true_jsonl",
        str(rerun_dir / "similar_people_true.jsonl"),
        "--out_false_jsonl",
        str(rerun_dir / "similar_people_false.jsonl"),
        "--out_error_jsonl",
        str(rerun_dir / "similar_people_error.jsonl"),
        "--out_detail",
        str(rerun_dir / "similar_people_detail.json"),
        "--keep_empty_keys",
    ]
    for endpoint in endpoints:
        cmd.extend(["--endpoint", endpoint])
    subprocess.run(cmd, check=True)


def build_outputs_from_results(results: List[Dict]) -> Dict:
    all_obj = {}
    bad_obj = {}
    good_obj = {}
    true_rows = []
    false_rows = []
    error_rows = []
    error_key_count = 0
    error_image_count = 0

    for rec in results:
        key, true_paths, false_paths, error_paths = extract_path_buckets(rec)
        if not key:
            continue
        value = 1 if true_paths else 0
        all_obj[key] = value
        if value == 1:
            bad_obj[key] = 1
        else:
            good_obj[key] = 0
        true_rows.append({key: true_paths})
        false_rows.append({key: false_paths})
        error_rows.append({key: error_paths})
        if error_paths:
            error_key_count += 1
            error_image_count += len(error_paths)

    return {
        "all_obj": all_obj,
        "bad_obj": bad_obj,
        "good_obj": good_obj,
        "true_rows": true_rows,
        "false_rows": false_rows,
        "error_rows": error_rows,
        "error_key_count": error_key_count,
        "error_image_count": error_image_count,
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--source-output-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rerun-dir", required=True)
    ap.add_argument("--subset-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--judge-script", required=True)
    ap.add_argument("--whitelist-ids", required=True, help="comma separated model ids")
    ap.add_argument("--judge-times", type=int, default=3)
    ap.add_argument("--min-true", type=int, default=2)
    ap.add_argument("--min-similar-people", type=int, default=3)
    ap.add_argument("--num-procs", type=int, default=32)
    ap.add_argument("--probe-timeout", type=float, default=3.0)
    ap.add_argument("--endpoint", action="append", default=[])
    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl)
    source_output_dir = Path(args.source_output_dir)
    output_dir = Path(args.output_dir)
    rerun_dir = Path(args.rerun_dir)
    subset_jsonl = Path(args.subset_jsonl)
    summary_json = Path(args.summary_json)
    judge_script = Path(args.judge_script)
    whitelist_ids = [x.strip() for x in args.whitelist_ids.split(",") if x.strip()]

    if not whitelist_ids:
        raise RuntimeError("whitelist ids is empty")

    input_rows, input_map = load_input_rows(input_jsonl)
    orig_detail = load_json(source_output_dir / "similar_people_detail.json")
    orig_bad = load_json(source_output_dir / "similar_people_bad.json")
    orig_results = orig_detail.get("results", [])
    coverage = verify_detail_coverage(input_map, orig_results)

    mismatch_found = any(
        coverage[k] != 0
        for k in [
            "missing_key_count",
            "extra_key_count",
            "count_mismatch_key_count",
            "path_mismatch_key_count",
        ]
    )
    if mismatch_found:
        raise RuntimeError(f"detail coverage mismatch: {json.dumps(coverage, ensure_ascii=False)}")

    rerun_keys = sorted(k for k in orig_bad.keys() if key_hits_whitelist(k, whitelist_ids))
    write_subset_jsonl(input_rows, rerun_keys, subset_jsonl)

    rerun_bad = {}
    rerun_results_map = {}
    rerun_summary = {}
    if rerun_keys:
        run_rerun(
            judge_script=judge_script,
            subset_jsonl=subset_jsonl,
            rerun_dir=rerun_dir,
            judge_times=args.judge_times,
            min_true=args.min_true,
            min_similar_people=args.min_similar_people,
            num_procs=args.num_procs,
            probe_timeout=args.probe_timeout,
            endpoints=args.endpoint,
        )
        rerun_bad = load_json(rerun_dir / "similar_people_bad.json")
        rerun_detail = load_json(rerun_dir / "similar_people_detail.json")
        rerun_summary = rerun_detail.get("summary", {})
        rerun_results_map = {str(rec.get("key", "")).strip(): rec for rec in rerun_detail.get("results", [])}
        missing_rerun_keys = sorted(set(rerun_keys) - set(rerun_results_map))
        if missing_rerun_keys:
            raise RuntimeError(f"rerun missing keys: {missing_rerun_keys[:20]}")

    merged_results = []
    for rec in orig_results:
        key = str(rec.get("key", "")).strip()
        if key in rerun_results_map:
            merged_results.append(rerun_results_map[key])
        else:
            merged_results.append(rec)

    outputs = build_outputs_from_results(merged_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "similar_people_all.json", outputs["all_obj"])
    write_json(output_dir / "similar_people_bad.json", outputs["bad_obj"])
    write_json(output_dir / "similar_people_good.json", outputs["good_obj"])
    write_jsonl(output_dir / "similar_people_true.jsonl", outputs["true_rows"])
    write_jsonl(output_dir / "similar_people_false.jsonl", outputs["false_rows"])
    write_jsonl(output_dir / "similar_people_error.jsonl", outputs["error_rows"])

    orig_summary = orig_detail.get("summary", {})
    merged_summary = dict(orig_summary)
    merged_summary.update(
        {
            "input_jsonl": str(input_jsonl),
            "picked": len(outputs["all_obj"]),
            "bad_count": len(outputs["bad_obj"]),
            "bad_ratio": len(outputs["bad_obj"]) / float(len(outputs["all_obj"])) if outputs["all_obj"] else 0.0,
            "error_key_count": outputs["error_key_count"],
            "error_image_count": outputs["error_image_count"],
            "out_true_jsonl": str(output_dir / "similar_people_true.jsonl"),
            "out_false_jsonl": str(output_dir / "similar_people_false.jsonl"),
            "out_error_jsonl": str(output_dir / "similar_people_error.jsonl"),
            "whitelist_rejudge": {
                "whitelist_ids": whitelist_ids,
                "rejudged_key_count": len(rerun_keys),
                "rejudge_min_similar_people": args.min_similar_people,
                "rerun_dir": str(rerun_dir),
                "subset_jsonl": str(subset_jsonl),
                "rerun_model": rerun_summary.get("model"),
                "rerun_base_url": rerun_summary.get("base_url"),
                "rerun_bad_count": len(rerun_bad),
                "rerun_good_count": len(rerun_keys) - len(rerun_bad),
                "removed_bad_keys": sorted(set(rerun_keys) - set(rerun_bad)),
                "remaining_bad_keys": sorted(rerun_bad.keys()),
            },
            "detail_coverage_check": coverage,
        }
    )
    write_json(output_dir / "similar_people_detail.json", {"summary": merged_summary, "results": merged_results})

    summary = {
        "input_jsonl": str(input_jsonl),
        "source_output_dir": str(source_output_dir),
        "output_dir": str(output_dir),
        "rerun_dir": str(rerun_dir),
        "subset_jsonl": str(subset_jsonl),
        "whitelist_ids": whitelist_ids,
        "detail_coverage_check": coverage,
        "original_bad_count": len(orig_bad),
        "new_bad_count": len(outputs["bad_obj"]),
        "original_key_count": len(orig_results),
        "new_key_count": len(outputs["all_obj"]),
        "rejudged_key_count": len(rerun_keys),
        "whitelist_bad_count_before": len(rerun_keys),
        "whitelist_bad_count_after": len(rerun_bad),
        "removed_bad_keys": sorted(set(rerun_keys) - set(rerun_bad)),
        "remaining_bad_keys_after_rejudge": sorted(rerun_bad.keys()),
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
