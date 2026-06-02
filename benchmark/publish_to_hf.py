#!/usr/bin/env python3
"""Publish the RegProd-800 benchmark to the Hugging Face Hub as a dataset.

Prerequisites
-------------
    pip install huggingface_hub
    hf auth login            # or: huggingface-cli login
    # paste a *write* token from https://huggingface.co/settings/tokens

Usage
-----
    python publish_to_hf.py [repo_id]
    # repo_id defaults to "<your-hf-username>/regprod-800"

Uploads the text benchmark plus the ~496 MB fp16 activations bundle. The Hub
handles the large file over HTTP automatically (LFS-backed) — no local git-lfs
needed. Re-running is idempotent (exist_ok + per-file overwrite).
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
GITHUB = "https://github.com/adromero/langprod"

CARD_FRONTMATTER = """\
---
pretty_name: RegProd-800
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-classification
tags:
- interpretability
- mechanistic-interpretability
- representational-similarity-analysis
- probing
- transformers
size_categories:
- n<1K
configs:
- config_name: default
  data_files: stimuli.jsonl
---

"""

# Files shipped to the dataset repo (paths relative to this script's dir).
FILES = [
    "LICENSE",
    "stimuli.jsonl",
    "tasks.json",
    "baselines.json",
    "load.py",
    "reproduce.py",
    "build_benchmark.py",
    "make_activations_bundle.py",
    "activations/hidden_states_residual_fp16.h5",
]


def build_card() -> bytes:
    """HF dataset card = frontmatter + benchmark/README.md, with ../ links absolutised."""
    body = (HERE / "README.md").read_text()
    body = body.replace("](../", f"]({GITHUB}/blob/main/")  # repo-relative -> GitHub
    return (CARD_FRONTMATTER + body).encode("utf-8")


def main() -> None:
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{user}/regprod-800"
    print(f"authenticated as: {user}")
    print(f"target dataset:   https://huggingface.co/datasets/{repo_id}\n")

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=build_card(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", commit_message="Add dataset card",
    )
    for rel in FILES:
        p = HERE / rel
        if not p.exists():
            print(f"  skip (missing): {rel}")
            continue
        print(f"  uploading {rel} ({p.stat().st_size / 1e6:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(p), path_in_repo=rel,
            repo_id=repo_id, repo_type="dataset",
            commit_message=f"Add {rel}",
        )
    print(f"\nDone -> https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
