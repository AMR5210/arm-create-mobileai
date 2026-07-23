#!/usr/bin/env python3
"""Download the base model checkpoint used for all three benchmark variants."""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_LOCAL_DIR = Path("models") / "qwen3-0.6b-hf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo id")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)

    # Gated repos (e.g. Llama-3.2) require HF_TOKEN; Qwen3 is open and needs none.
    token = os.environ.get("HF_TOKEN")

    print(f"==> Downloading {args.repo_id} to {args.local_dir}")
    snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir, token=token)
    print("==> Download complete")


if __name__ == "__main__":
    main()
