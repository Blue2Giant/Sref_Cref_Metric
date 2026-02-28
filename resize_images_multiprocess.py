import os
import argparse
import shutil
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
"""
python /data/benchmark_metrics/resize_images_multiprocess.py --input_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/cref \
    --output_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new_resized_cref --num_workers 16
python /data/benchmark_metrics/resize_images_multiprocess.py --input_dir /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt_new \
    --output_dir /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt_resized_512_new --num_workers 16 --copy_json

python /data/benchmark_metrics/resize_images_multiprocess.py --input_dir /mnt/jfs/bench-bucket/sref_bench/bench_0228_content_prompt/ \
    --output_dir /mnt/jfs/bench-bucket/sref_bench/bench_0228_content_prompt_resize --num_workers 16 --copy_json
python /data/benchmark_metrics/resize_images_multiprocess.py --input_dir /mnt/jfs/bench-bucket/sref_bench/bench_0222_style/bench_1022_style/ \
    --output_dir /mnt/jfs/bench-bucket/sref_bench/bench_0222_style/bench_1022_style_resize --num_workers 16 --copy_json
"""

PREFERRED_KONTEXT_RESOLUTIONS: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

def resize_like_qwen(img: Image.Image) -> Image.Image:
    w, h = img.size
    aspect_ratio = w / float(h)
    _, target_w, target_h = min(
        (abs(aspect_ratio - (rw / float(rh))), rw, rh)
        for (rw, rh) in PREFERRED_KONTEXT_RESOLUTIONS
    )
    if (w, h) == (target_w, target_h):
        return img
    return img.resize((target_w, target_h), resample=_lanczos())
def resize_image_task(args):
    file_path, output_dir, copy_json = args
    try:
        filename = os.path.basename(file_path)
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, base + ".png")

        with Image.open(file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_resized = resize_like_qwen(img)
            img_resized.save(output_path, format="PNG")

        if copy_json:
            json_path = os.path.splitext(file_path)[0] + ".json"
            if os.path.exists(json_path):
                shutil.copy2(json_path, os.path.join(output_dir, base + ".json"))
                     
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch resize images using preferred aspect ratios and save as PNG.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of worker processes (default: CPU count).")
    parser.add_argument("--copy_json", action="store_true", help="Copy same-basename .json files if present.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    image_files = []
    for f in files:
        path = os.path.join(args.input_dir, f)
        try:
            with Image.open(path) as img:
                img.verify()
            image_files.append(path)
        except Exception:
            continue
    
    if not image_files:
        print("No image files found in the input directory.")
        return

    # Sort files to ensure deterministic order (though processing is parallel)
    image_files.sort()

    print(f"Found {len(image_files)} images. Starting resize with {args.num_workers} workers...")
    
    # Prepare arguments for workers
    tasks = [(f, args.output_dir, args.copy_json) for f in image_files]
    
    # Run with pool
    with Pool(processes=args.num_workers) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(pool.imap_unordered(resize_image_task, tasks), total=len(tasks), unit="img"))
        
    success_count = sum(results)
    print(f"Processing complete. Successfully resized {success_count}/{len(image_files)} images.")

if __name__ == "__main__":
    main()
