"""
upload_to_hf.py — PokéWorld.

Helper script to backup/upload datasets or model checkpoints to the Hugging Face Hub.

Pre-requisites:
    1. Activate conda environment:
       conda activate pokemon-rl
    2. Install Hugging Face Hub library:
       pip install -U huggingface_hub
    3. Login to Hugging Face:
       huggingface-cli login
       (generate a Write-Access token in your settings: https://huggingface.co/settings/tokens)

Usage:
    # Upload data folder to a Hugging Face dataset repository
    python scripts/upload_to_hf.py \
        --repo-id "username/poke-dreamer-dataset-v2" \
        --folder "data" \
        --repo-type "dataset"

    # Upload checkpoints folder to a Hugging Face model repository
    python scripts/upload_to_hf.py \
        --repo-id "username/poke-dreamer-models" \
        --folder "checkpoints" \
        --repo-type "model"
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("[ERROR] Hugging Face Hub library is not installed.")
    print("Please run: pip install -U huggingface_hub")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backup project files to Hugging Face Hub")
    p.add_argument("--repo-id", type=str, required=True,
                   help="Repository identifier on HF Hub (e.g. 'username/repo-name')")
    p.add_argument("--folder", type=Path, required=True,
                   help="Path to local folder to upload")
    p.add_argument("--repo-type", type=str, choices=["dataset", "model"], default="dataset",
                   help="Type of repository: 'dataset' or 'model' (default: 'dataset')")
    p.add_argument("--token", type=str, default=None,
                   help="HF Write Token (optional if already logged in via huggingface-cli)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.folder.exists():
        print(f"[ERROR] Local folder does not exist: {args.folder.resolve()}")
        sys.exit(1)

    api = HfApi()

    print(f"\n[hf_backup] Preparing to backup {args.folder.resolve()} to Hugging Face Hub...")
    print(f"[hf_backup] Target Repository: https://huggingface.co/{args.repo_type}s/{args.repo_id}")

    try:
        # 1. Create the repository if it doesn't already exist
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            exist_ok=True,
            token=args.token
        )
        print("[hf_backup] Repository verified/created successfully.")

        # 2. Upload the folder contents
        print(f"[hf_backup] Uploading all files in '{args.folder}' (this includes a progress bar)...")
        api.upload_folder(
            folder_path=str(args.folder),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=".",  # Uploads contents directly to the root of the repo
            token=args.token
        )
        print(f"\n[SUCCESS] Backup completed! View your files at:")
        print(f"https://huggingface.co/{args.repo_type}s/{args.repo_id}/tree/main")

    except Exception as e:
        print(f"\n[ERROR] Failed to upload: {e}")
        print("\nSuggestions:")
        print("  1. Make sure you generated a Write token (not Read token) on Hugging Face.")
        print("  2. Ensure you have run: huggingface-cli login")
        print("  3. Check that the repo name format is: username/repo-name")
        sys.exit(1)


if __name__ == "__main__":
    main()
