#!/usr/bin/env bash
# Metric run for:
# /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/5030_omni_compare_checkpoint-26000_fixed_transfer_sref_1024x1024
# Uses a snapshot of currently existing target images, so metrics are internally consistent even if inference continues writing new images.
set -u -o pipefail

SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"
RESULT_DIR="$SREF_ROOT/5030_omni_compare_checkpoint-26000_fixed_transfer_sref_1024x1024"
CONTENT_DIR="$SREF_ROOT/cref"
STYLE_DIR="$SREF_ROOT/sref"
SREF_PROMPT="$SREF_ROOT/prompts.json"
RUNNER_PY="/data/benchmark_metrics/benchmark_metrics/encoder_batch_runner.py"
GPUS="${GPUS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_PROCS_VLM="${NUM_PROCS_VLM:-64}"
OVERWRITE_ENCODER="${OVERWRITE_ENCODER:-1}"
OVERWRITE_VLM="${OVERWRITE_VLM:-1}"

# Model weights
DINOV2_MODEL="/mnt/jfs/model_zoo/dinov2-with-registers-large"
CAS_MODEL="/mnt/jfs/model_zoo/dinov2-base"
ONEIG_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder"
CSD_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder/csd.pth"
CSD_MODEL_ONLY="/mnt/jfs/model_zoo/OneIG-StyleEncoder/vit-b-300ep.pth.tar"
VIT_L="/mnt/jfs/model_zoo/OneIG-StyleEncoder/ViT-L-14.pt"
ONEALIGN_MODEL="/mnt/jfs/model_zoo/one-align"
ONEALIGN_TASK="aesthetics"

# Outputs in target dir
OUT_DINOV2_JSON="$RESULT_DIR/dinov2_out.json"
OUT_CAS_JSON="$RESULT_DIR/cas_out.json"
OUT_ONEIG_JSON="$RESULT_DIR/oneig_out.json"
OUT_CLIPCAP_JSON="$RESULT_DIR/clipcap_out.json"
OUT_ONEALIGN_JSON="$RESULT_DIR/onealign_out.json"
OUT_CSD_JSON="$RESULT_DIR/csd_out.json"
OUT_LAION_JSON="$RESULT_DIR/laion_scores.json"
OUT_V25_AESTHETIC="$RESULT_DIR/v25_scores.json"
OUT_VLM_STYLE="$RESULT_DIR/qwen_resize_output_style_descrete.json"
OUT_VLM_STYLE_REASON="$RESULT_DIR/qwen_resize_output_style_reason_descrete.json"
OUT_VLM_CONTENT="$RESULT_DIR/qwen_resize_output_content_descrete.json"
OUT_VLM_CONTENT_REASON="$RESULT_DIR/qwen_resize_output_content_reason_descrete.json"
OUT_FOLLOW_SCORE="$RESULT_DIR/follow_scores.json"
OUT_FOLLOW_REASON="$RESULT_DIR/follow_reasons.json"
OUT_QWEN_REJECT_CREF="$RESULT_DIR/qwen_reject_cref.json"
OUT_QWEN_REJECT_SREF="$RESULT_DIR/qwen_reject_sref.json"
OUT_METRICS_CSV="$RESULT_DIR/metrics_mean.csv"

XINGPENG_IP="http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1"
XINGPENG_MODEL="qwen3vlw8a8"

FAILED_STEPS=()
SUCCEEDED_STEPS=()
run_step() {
  local name="$1"; shift
  echo
  echo "================================================================"
  echo "[$(date '+%F %T')] START: $name"
  echo "CMD: $*"
  echo "================================================================"
  "$@"
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] OK: $name"
    SUCCEEDED_STEPS+=("$name")
  else
    echo "[$(date '+%F %T')] FAILED($status): $name" >&2
    FAILED_STEPS+=("$name:$status")
  fi
  return 0
}

require_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "Missing directory: $p" >&2
    exit 2
  fi
}

require_dir "$SREF_ROOT"
require_dir "$RESULT_DIR"
require_dir "$CONTENT_DIR"
require_dir "$STYLE_DIR"
if [[ ! -f "$SREF_PROMPT" ]]; then
  echo "Missing prompt json: $SREF_PROMPT" >&2
  exit 2
fi

# Make target dir writable for metric JSON/CSV outputs. Target images may be root-owned from inference.
if ! ( touch "$RESULT_DIR/.metric_write_test" && rm -f "$RESULT_DIR/.metric_write_test" ) 2>/dev/null; then
  echo "[$(date '+%F %T')] RESULT_DIR not writable by $(id -un); trying sudo chmod for metric outputs..."
  sudo chmod a+rwx "$RESULT_DIR" || true
  sudo chmod a+rw "$RESULT_DIR"/*.json "$RESULT_DIR"/*.csv "$RESULT_DIR"/*.log 2>/dev/null || true
fi

if ! ( touch "$RESULT_DIR/.metric_write_test" && rm -f "$RESULT_DIR/.metric_write_test" ) 2>/dev/null; then
  echo "Still cannot write to RESULT_DIR: $RESULT_DIR" >&2
  exit 3
fi

# Snapshot current images so all metric files cover the same set.
SNAPSHOT_ROOT="/data/benchmark_metrics/.tmp/metric_snapshots/5030_sref_$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_IMG_DIR="$SNAPSHOT_ROOT/images"
mkdir -p "$SNAPSHOT_IMG_DIR"
export RESULT_DIR SNAPSHOT_IMG_DIR CONTENT_DIR STYLE_DIR SREF_PROMPT
$PYTHON_BIN - <<'PY'
import json, os
from pathlib import Path
exts={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
result=Path(os.environ['RESULT_DIR'])
snap=Path(os.environ['SNAPSHOT_IMG_DIR'])
snap.mkdir(parents=True, exist_ok=True)
imgs=sorted([p for p in result.iterdir() if p.is_file() and p.suffix.lower() in exts])
# Create symlinks with the same filenames. If symlink fails, fall back to copying only the link path text by hardlink then copy.
for p in imgs:
    dst=snap/p.name
    try:
        if dst.exists() or dst.is_symlink(): dst.unlink()
        dst.symlink_to(p)
    except Exception:
        import shutil
        shutil.copy2(p, dst)
keys=[p.stem for p in imgs]
(snap.parent/'keys.txt').write_text('\n'.join(keys)+'\n', encoding='utf-8')
summary={
    'result_dir': str(result),
    'snapshot_img_dir': str(snap),
    'num_images': len(imgs),
    'first10': [p.name for p in imgs[:10]],
}
(snap.parent/'snapshot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
EVAL_IMAGE_DIR="$SNAPSHOT_IMG_DIR"

cd /data/benchmark_metrics/benchmark_metrics

echo "[$(date '+%F %T')] Environment"
hostname || true
id || true
$PYTHON_BIN - <<'PY' || true
import sys
print('python', sys.executable)
try:
    import torch
    print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())
except Exception as e:
    print('torch check failed', e)
PY
nvidia-smi || true

# Encoder / image metrics
run_step "CSD style similarity" \
  $PYTHON_BIN "$RUNNER_PY" pair \
    --encoder csd \
    --dir_a "$STYLE_DIR" \
    --dir_b "$EVAL_IMAGE_DIR" \
    --out_json "$OUT_CSD_JSON" \
    --model dummy \
    --csd_arch vit_base \
    --csd_model_path "$CSD_MODEL_ONLY" \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite "$OVERWRITE_ENCODER"

run_step "OneIG style similarity" \
  $PYTHON_BIN "$RUNNER_PY" pair \
    --encoder oneig \
    --dir_a "$STYLE_DIR" \
    --dir_b "$EVAL_IMAGE_DIR" \
    --model dummy \
    --oneig_model_path "$CSD_MODEL" \
    --oneig_se_model_path "$ONEIG_MODEL" \
    --oneig_clip_model_path "$VIT_L" \
    --out_json "$OUT_ONEIG_JSON" \
    --gpus "$GPUS" \
    --overwrite "$OVERWRITE_ENCODER"

run_step "DINOv2 content similarity" \
  $PYTHON_BIN "$RUNNER_PY" pair \
    --encoder dinov2 \
    --dir_a "$CONTENT_DIR" \
    --dir_b "$EVAL_IMAGE_DIR" \
    --model "$DINOV2_MODEL" \
    --out_json "$OUT_DINOV2_JSON" \
    --gpus "$GPUS" \
    --overwrite "$OVERWRITE_ENCODER"

run_step "CAS content similarity" \
  $PYTHON_BIN "$RUNNER_PY" pair \
    --encoder cas \
    --dir_a "$CONTENT_DIR" \
    --dir_b "$EVAL_IMAGE_DIR" \
    --model "$CAS_MODEL" \
    --out_json "$OUT_CAS_JSON" \
    --gpus "$GPUS" \
    --overwrite "$OVERWRITE_ENCODER"

run_step "CLIP-T instruction/image-text" \
  $PYTHON_BIN "$RUNNER_PY" clip_t \
    --image_dir "$EVAL_IMAGE_DIR" \
    --prompt_json "$SREF_PROMPT" \
    --out_json "$OUT_CLIPCAP_JSON" \
    --model /mnt/jfs/model_zoo/openai/clip-vit-base-patch32 \
    --sim_metric cosine \
    --clipcap_text_mode first_sentence \
    --overwrite "$OVERWRITE_ENCODER"

run_step "LAION aesthetic" \
  $PYTHON_BIN "$RUNNER_PY" aesthetic \
    --backend laion \
    --image_dir "$EVAL_IMAGE_DIR" \
    --out_json "$OUT_LAION_JSON" \
    --laion_clip_model ViT-L-14 \
    --laion_clip_ckpt /mnt/jfs/model_zoo/open_clip/open_clip_model_ea4f182e96863ce2a27be5067cdb54d4.safetensors \
    --laion_linear_path ~/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite "$OVERWRITE_ENCODER"

run_step "Aesthetic v2.5" \
  $PYTHON_BIN "$RUNNER_PY" aesthetic \
    --backend v25 \
    --image_dir "$EVAL_IMAGE_DIR" \
    --out_json "$OUT_V25_AESTHETIC" \
    --v25_encoder_model_name /mnt/jfs/model_zoo/siglip-so400m-patch14-384/ \
    --dtype bfloat16 \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite "$OVERWRITE_ENCODER"

# VLM discrete metrics. Use --overwrite only when requested because these scripts use store_true.
VLM_OVERWRITE_ARGS=()
if [[ "$OVERWRITE_VLM" == "1" || "$OVERWRITE_VLM" == "true" || "$OVERWRITE_VLM" == "yes" ]]; then
  VLM_OVERWRITE_ARGS=(--overwrite)
fi

run_step "VLM style discrete score" \
  $PYTHON_BIN /data/benchmark_metrics/vlm_similarity/style_similarity_dir.py \
    --style_dir "$STYLE_DIR" \
    --output_dir "$EVAL_IMAGE_DIR" \
    --out_score_json "$OUT_VLM_STYLE" \
    --out_reason_json "$OUT_VLM_STYLE_REASON" \
    --base_url "$XINGPENG_IP" \
    --model "$XINGPENG_MODEL" \
    --num_procs "$NUM_PROCS_VLM" \
    "${VLM_OVERWRITE_ARGS[@]}"

run_step "VLM content discrete score" \
  $PYTHON_BIN /data/benchmark_metrics/vlm_similarity/content_similarity_dir.py \
    --content_dir "$CONTENT_DIR" \
    --output_dir "$EVAL_IMAGE_DIR" \
    --out_json "$OUT_VLM_CONTENT" \
    --out_reason_json "$OUT_VLM_CONTENT_REASON" \
    --base_url "$XINGPENG_IP" \
    --model "$XINGPENG_MODEL" \
    --num_procs "$NUM_PROCS_VLM" \
    "${VLM_OVERWRITE_ARGS[@]}"

run_step "VLM instruction-follow score" \
  $PYTHON_BIN /data/benchmark_metrics/vlm_similarity/edit_instruction_follow_dir.py \
    --image_dir "$EVAL_IMAGE_DIR" \
    --prompt_json "$SREF_PROMPT" \
    --out_score_json "$OUT_FOLLOW_SCORE" \
    --out_reason_json "$OUT_FOLLOW_REASON" \
    --base_url "$XINGPENG_IP" \
    --model "$XINGPENG_MODEL" \
    --instruction_text_mode first_sentence \
    --num_procs "$NUM_PROCS_VLM" \
    "${VLM_OVERWRITE_ARGS[@]}"

run_step "Triplet Qwen dual judge" \
  $PYTHON_BIN /data/benchmark_metrics/vlm_similarity/triplet_qwen_dual_judge.py \
    --content_dir "$CONTENT_DIR" \
    --style_dir "$STYLE_DIR" \
    --result_dir "$EVAL_IMAGE_DIR" \
    --output_content_json "$OUT_QWEN_REJECT_CREF" \
    --output_style_json "$OUT_QWEN_REJECT_SREF" \
    --endpoint "${XINGPENG_MODEL}@${XINGPENG_IP}" \
    --procs_per_endpoint "$NUM_PROCS_VLM" \
    "${VLM_OVERWRITE_ARGS[@]}"

# Aggregate means for existing metric JSONs.
jsons=()
for p in \
  "$OUT_DINOV2_JSON" \
  "$OUT_CAS_JSON" \
  "$OUT_ONEIG_JSON" \
  "$OUT_CLIPCAP_JSON" \
  "$OUT_CSD_JSON" \
  "$OUT_LAION_JSON" \
  "$OUT_V25_AESTHETIC" \
  "$OUT_VLM_STYLE" \
  "$OUT_VLM_CONTENT" \
  "$OUT_FOLLOW_SCORE" \
  "$OUT_QWEN_REJECT_CREF" \
  "$OUT_QWEN_REJECT_SREF"; do
  if [[ -f "$p" ]]; then jsons+=("$p"); fi
done
if [[ ${#jsons[@]} -gt 0 ]]; then
  run_step "Aggregate metrics_mean.csv" \
    $PYTHON_BIN /data/benchmark_metrics/batch_temp/json_means_to_csv.py \
      --jsons "${jsons[@]}" \
      --out_csv "$OUT_METRICS_CSV"
fi

# Final summary
$PYTHON_BIN - <<PY
import csv, json, os
from pathlib import Path
result=Path("$RESULT_DIR")
snap=Path("$SNAPSHOT_IMG_DIR")
metric_paths=[
"$OUT_DINOV2_JSON","$OUT_CAS_JSON","$OUT_ONEIG_JSON","$OUT_CLIPCAP_JSON","$OUT_ONEALIGN_JSON","$OUT_CSD_JSON","$OUT_LAION_JSON","$OUT_V25_AESTHETIC","$OUT_VLM_STYLE","$OUT_VLM_CONTENT","$OUT_FOLLOW_SCORE","$OUT_QWEN_REJECT_CREF","$OUT_QWEN_REJECT_SREF"]
print('\n================ FINAL FILE COUNTS ================')
print('result_dir', result)
print('snapshot_dir', snap)
print('snapshot_images', len([p for p in snap.iterdir() if p.is_file() or p.is_symlink()]))
for p in metric_paths:
    pp=Path(p)
    if pp.exists():
        try:
            data=json.load(open(pp, 'r', encoding='utf-8'))
            n=len(data) if hasattr(data, '__len__') else 'NA'
        except Exception as e:
            n=f'ERR:{e}'
        print(pp.name, n)
    else:
        print(pp.name, 'MISSING')
if Path("$OUT_METRICS_CSV").exists():
    print('\nmetrics_mean.csv:')
    print(Path("$OUT_METRICS_CSV").read_text(encoding='utf-8'))
PY

echo
printf 'Succeeded steps (%d):\n' "${#SUCCEEDED_STEPS[@]}"
printf '  %s\n' "${SUCCEEDED_STEPS[@]}"
printf 'Failed steps (%d):\n' "${#FAILED_STEPS[@]}"
printf '  %s\n' "${FAILED_STEPS[@]}"

echo "[$(date '+%F %T')] DONE script: $0"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  exit 1
fi
