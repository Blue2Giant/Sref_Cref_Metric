import os
import argparse
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm
"""
python /data/benchmark_metrics/resize_images_multiprocess.py --input_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/cref \
    --output_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new_resized_cref --size 512 --num_workers 16
python /data/benchmark_metrics/resize_images_multiprocess.py --input_dir /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt_new \
    --output_dir /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt_resized_512_new --size 512 --num_workers 16
"""
def resize_image_task(args):
    """
    Worker function to resize a single image.
    args: (file_path, output_dir, target_size)
    """
    file_path, output_dir, target_size = args
    try:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        with Image.open(file_path) as img:
            width, height = img.size
            max_dim = max(width, height)
            
            # Calculate new size
            if max_dim != target_size:
                ratio = target_size / max_dim
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                # Use high-quality resampling
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                img_resized = img.copy()

            # Handle mode conversion for JPEG
            if ext in ['.jpg', '.jpeg'] and img_resized.mode in ('RGBA', 'P'):
                img_resized = img_resized.convert('RGB')
            
            # Save
            # quality=95 for JPEGs to maintain high quality
            if ext in ['.jpg', '.jpeg']:
                img_resized.save(output_path, quality=95)
            else:
                img_resized.save(output_path)
                     
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch resize images to a specific long side length using multiprocessing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--size", type=int, default=512, help="Target length for the long side (default: 512).")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of worker processes (default: CPU count).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    # Supported extensions
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Collect image files (top-level only)
    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    image_files = [os.path.join(args.input_dir, f) for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not image_files:
        print("No image files found in the input directory.")
        return

    # Sort files to ensure deterministic order (though processing is parallel)
    image_files.sort()

    print(f"Found {len(image_files)} images. Starting resize to long side {args.size} with {args.num_workers} workers...")
    
    # Prepare arguments for workers
    tasks = [(f, args.output_dir, args.size) for f in image_files]
    
    # Run with pool
    with Pool(processes=args.num_workers) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(pool.imap_unordered(resize_image_task, tasks), total=len(tasks), unit="img"))
        
    success_count = sum(results)
    print(f"Processing complete. Successfully resized {success_count}/{len(image_files)} images.")

if __name__ == "__main__":
    main()
