import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Recursively delete all JSON files in a directory.")
    parser.add_argument("dir", help="Target directory path")
    
    args = parser.parse_args()
    
    root_dir = args.dir
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory.")
        return

    # Using glob for pattern matching - recursive by default as requested
    pattern = os.path.join(root_dir, "**", "*.json")
    
    # glob.glob returns a list of matching file paths
    # recursive=True is needed for ** to work properly
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"No JSON files found in '{root_dir}' or its subdirectories.")
        return

    print(f"Found {len(files)} JSON files.")
    
    # Direct deletion as requested for a script tool
    deleted_count = 0
    for file_path in files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    print(f"Successfully deleted {deleted_count} JSON files.")

if __name__ == "__main__":
    main()
