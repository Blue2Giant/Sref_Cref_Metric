import json
import argparse
import numpy as np
import sys
import os
"""
python /data/LoraPipeline/utils/sample_models_weighted.py --input_json /data/LoraPipeline/assets/flux_content_human.txt  --output_json  /data/LoraPipeline/assets/flux_content_human_sampled.txt 
"""
def main():
    parser = argparse.ArgumentParser(description="Weighted sampling of models based on scores.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the JSON file with model scores.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save the sampled model IDs. If not provided, prints to stdout.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of models to sample.")
    parser.add_argument("--power", type=float, default=4.0, help="Power for probability calculation: prob ~ score^power. Higher power means higher scores are much more likely to be picked. Default is 4.0 to emphasize score differences.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load JSON
    try:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
        
    if not data:
        print("Error: Empty JSON file.")
        sys.exit(1)

    # Extract IDs and scores
    model_ids = list(data.keys())
    scores = np.array(list(data.values()), dtype=float)
    
    # Handle potential negative scores if any (though sample showed positive)
    # If scores are close (e.g. 4.0-5.0), linear sampling (power=1) gives very flat distribution.
    # Using a higher power helps differentiate.
    # Alternatively, we could use softmax.
    
    min_score = np.min(scores)
    if min_score < 0:
        print(f"Warning: Found negative scores (min: {min_score}). Shifting to positive range for sampling.")
        scores = scores - min_score + 1e-6
    elif min_score == 0:
        scores = scores + 1e-6 # Avoid zero division or zero probability if desired

    # Calculate weights: score^power
    weights = np.power(scores, args.power)
    
    # Normalize to probabilities
    probs = weights / np.sum(weights)
    
    # Determine sample size
    num_samples = min(args.max_samples, len(model_ids))
    
    print(f"Total models: {len(model_ids)}")
    print(f"Sampling {num_samples} models...")
    print(f"Score range: {np.min(scores):.4f} - {np.max(scores):.4f}")
    print(f"Using power: {args.power}")
    
    # Sample without replacement
    sampled_indices = np.random.choice(
        len(model_ids), 
        size=num_samples, 
        replace=False, 
        p=probs
    )
    
    sampled_models = [model_ids[i] for i in sampled_indices]
    
    # Output
    if args.output_json:
        # Create directory if it doesn't exist
        out_dir = os.path.dirname(args.output_json)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        with open(args.output_json, 'w') as f:
            json.dump(sampled_models, f, indent=4)
        print(f"Saved sampled models to {args.output_json}")
    else:
        print(json.dumps(sampled_models, indent=4))

if __name__ == "__main__":
    main()
