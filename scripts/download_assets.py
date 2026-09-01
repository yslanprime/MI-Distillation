#!/usr/bin/env python3
"""Download the models and benchmarks used by MI-Distillation from the Hub.

Credentials come from the environment (HF_TOKEN), never from this file. Set
HF_ENDPOINT to use a mirror.

    python scripts/download_assets.py --list
    python scripts/download_assets.py --group teachers-32b --group students --group benchmarks
    python scripts/download_assets.py --repo Qwen/Qwen2.5-3B-Instruct --dest pretrained_model

The training problems are not covered here; see the README for how to prepare
them.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple


class Asset(NamedTuple):
    repo_id: str
    repo_type: str
    dest_root: str


# Grouped so users can fetch only the parts of the pipeline they intend to run.
ASSET_GROUPS: Dict[str, List[Asset]] = {
    "teachers-32b": [
        Asset("Qwen/QwQ-32B", "model", "pretrained_model"),
        Asset("Qwen/Qwen2.5-32B-Instruct", "model", "pretrained_model"),
    ],
    "teachers-14b": [
        Asset("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "model", "pretrained_model"),
        Asset("Qwen/Qwen2.5-14B-Instruct", "model", "pretrained_model"),
    ],
    "students": [
        Asset("Qwen/Qwen2.5-3B-Instruct", "model", "pretrained_model"),
        Asset("meta-llama/Llama-3.2-3B-Instruct", "model", "pretrained_model"),
    ],
    "benchmarks": [
        Asset("HuggingFaceH4/MATH-500", "dataset", "data/raw"),
        Asset("openai/gsm8k", "dataset", "data/raw"),
        Asset("Maxwell-Jia/AIME_2024", "dataset", "data/raw"),
        Asset("math-ai/amc23", "dataset", "data/raw"),
        Asset("Idavidrein/gpqa", "dataset", "data/raw"),
        Asset("math-ai/minervamath", "dataset", "data/raw"),
        Asset("Hothan/OlympiadBench", "dataset", "data/raw"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models and datasets from the HuggingFace Hub.")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        choices=sorted(ASSET_GROUPS),
        help="Asset group to download. Repeatable.",
    )
    parser.add_argument("--repo", action="append", default=[], help="Extra repo id to download. Repeatable.")
    parser.add_argument(
        "--repo_type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repo type for --repo entries.",
    )
    parser.add_argument("--dest", default="pretrained_model", help="Destination root for --repo entries.")
    parser.add_argument("--max_retries", type=int, default=5, help="Retries per repository. 0 means unlimited.")
    parser.add_argument("--list", action="store_true", help="List the known groups and exit.")
    return parser.parse_args()


def download(asset: Asset, max_retries: int) -> bool:
    from huggingface_hub import snapshot_download

    local_dir = Path(asset.dest_root) / asset.repo_id.split("/")[-1]
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    # HF_TOKEN is optional; huggingface_hub also honours a cached login.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[{asset.repo_id}] downloading to {local_dir} (attempt {attempt})")
            snapshot_download(
                repo_id=asset.repo_id,
                repo_type=asset.repo_type,
                local_dir=str(local_dir),
                token=token,
                resume_download=True,
                etag_timeout=100,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[{asset.repo_id}] failed: {exc}", file=sys.stderr)
            if max_retries and attempt >= max_retries:
                print(f"[{asset.repo_id}] giving up after {attempt} attempts.", file=sys.stderr)
                return False
            time.sleep(min(30, 3 * attempt))
        else:
            print(f"[{asset.repo_id}] done -> {local_dir}")
            return True


def main() -> None:
    args = parse_args()

    if args.list:
        for group, assets in sorted(ASSET_GROUPS.items()):
            print(f"{group}:")
            for asset in assets:
                print(f"  {asset.repo_type:8s} {asset.repo_id}  ->  {asset.dest_root}")
        return

    assets: List[Asset] = []
    for group in args.group:
        assets.extend(ASSET_GROUPS[group])
    for repo_id in args.repo:
        assets.append(Asset(repo_id, args.repo_type, args.dest))

    # De-duplicate while preserving order; groups intentionally overlap.
    seen = set()
    unique_assets = []
    for asset in assets:
        if asset.repo_id not in seen:
            seen.add(asset.repo_id)
            unique_assets.append(asset)

    if not unique_assets:
        raise SystemExit("Nothing to download. Pass --group and/or --repo, or --list to see the groups.")

    print(f"Downloading {len(unique_assets)} repositories.")
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("No HF_TOKEN set; gated repositories such as Llama-3.2 will fail without one.")

    failures = [asset.repo_id for asset in unique_assets if not download(asset, args.max_retries)]

    print()
    print(f"Succeeded: {len(unique_assets) - len(failures)}/{len(unique_assets)}")
    if failures:
        print("Failed:")
        for repo_id in failures:
            print(f"  - {repo_id}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
