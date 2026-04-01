#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python insight/attention_metrics_demo.py /mnt/jfs/qwen-edit-attn-fullmap-keycolor-save/20241009--1377--fbb336bbe394b62e92dc1f748c3193a92a2dd937__low_poly_attn/attention_step_block_grid.pt --device cuda --dtype float32
"""
import argparse
from pathlib import Path

import torch

try:
    from insight.attention_metrics import (
        DEFAULT_GROUP_NAMES,
        compute_metrics_for_named_groups_from_pt,
        get_k_slice_from_meta,
        summarize_group_metrics,
    )
except ModuleNotFoundError:
    from attention_metrics import (
        DEFAULT_GROUP_NAMES,
        compute_metrics_for_named_groups_from_pt,
        get_k_slice_from_meta,
        summarize_group_metrics,
    )


def parse_args():
    parser = argparse.ArgumentParser("Attention metrics demo for qwen attention_step_block_grid.pt")
    parser.add_argument("pt_path", help="Path to attention_step_block_grid.pt")
    parser.add_argument("--device", default="auto", help='Device such as "auto", "cpu", "cuda", or "cuda:0"')
    parser.add_argument("--dtype", default="float32", help='Compute dtype such as "float32", "float16", or "bfloat16"')
    parser.add_argument("--groups", default="text,cref,sref", help="Comma-separated named groups")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--reduction", choices=["mean", "median", "max", "min"], default="mean")
    return parser.parse_args()


def main():
    args = parse_args()
    group_names = tuple(x.strip() for x in str(args.groups).split(",") if x.strip())
    if not group_names:
        group_names = DEFAULT_GROUP_NAMES

    batch, metrics = compute_metrics_for_named_groups_from_pt(
        pt_path=args.pt_path,
        device=args.device,
        dtype=args.dtype,
        group_names=group_names,
        use_sample=True,
        topk=int(args.topk),
    )
    summary = summarize_group_metrics(metrics, reduction=args.reduction)

    print(f"pt_path={Path(args.pt_path)}")
    print(f"tensor_key={batch.tensor_key}")
    print(f"device={batch.attn.device}")
    print(f"attn_shape={tuple(batch.attn.shape)} dtype={batch.attn.dtype}")
    if batch.attn_mask is not None:
        valid = int(batch.attn_mask.sum().item())
        total = int(batch.attn_mask.numel())
        print(f"attn_mask_shape={tuple(batch.attn_mask.shape)} valid_panels={valid}/{total}")
    else:
        print("attn_mask_shape=None")

    for group_name in group_names:
        k_start, k_end = get_k_slice_from_meta(batch.k_range_metadata_by_name, group_name, use_sample=True)
        print(f"group={group_name} sampled_k=[{k_start},{k_end})")
        print(
            "  "
            f"mass_ratio={summary[group_name]['mass_ratio']:.6f} "
            f"enrichment={summary[group_name]['enrichment']:.6f} "
            f"k_entropy={summary[group_name]['k_entropy']:.6f} "
            f"q_entropy={summary[group_name]['q_entropy']:.6f} "
            f"qk_nmi={summary[group_name]['qk_normalized_mutual_information']:.6f}"
        )
        print(
            "  "
            f"metric_shape={tuple(metrics[group_name]['mass_ratio'].shape)} "
            f"k_topk_mass={summary[group_name]['k_topk_mass']:.6f} "
            f"high_response_query_ratio={summary[group_name]['high_response_query_ratio']:.6f}"
        )

    if batch.attn.is_cuda:
        torch.cuda.synchronize(batch.attn.device)
        alloc_mb = torch.cuda.memory_allocated(batch.attn.device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(batch.attn.device) / (1024 ** 2)
        print(f"cuda_memory_allocated_mb={alloc_mb:.2f}")
        print(f"cuda_memory_reserved_mb={reserved_mb:.2f}")


if __name__ == "__main__":
    main()
