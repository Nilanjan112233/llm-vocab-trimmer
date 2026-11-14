"""Download everything needed to rebuild the trimmed model.

Files go into ./data (about 3.1 GB, mostly the source model):
  - gemma-4-E2B-it-Q4_0.gguf + mtp-gemma-4-E2B-it-Q4_0.gguf  (ggml-org/gemma-4-E2B-it-GGUF)
  - tokenizer.json                                            (google/gemma-4-E2B-it)
  - WikiText-103 train shard 0 + test split                   (Salesforce/wikitext)
  - UltraChat 200k test_sft split                             (HuggingFaceH4/ultrachat_200k)

Usage:  python scripts/download_data.py [--out data] [--skip-model]
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = [
    # (repo, repo_type, path in repo, local name, is_model)
    ("ggml-org/gemma-4-E2B-it-GGUF", "model", "gemma-4-E2B-it-Q4_0.gguf", "gemma-4-E2B-it-Q4_0.gguf", True),
    ("ggml-org/gemma-4-E2B-it-GGUF", "model", "mtp-gemma-4-E2B-it-Q4_0.gguf", "mtp-gemma-4-E2B-it-Q4_0.gguf", True),
    ("google/gemma-4-E2B-it", "model", "tokenizer.json", "tokenizer.json", False),
    ("Salesforce/wikitext", "dataset", "wikitext-103-raw-v1/train-00000-of-00002.parquet", "wikitext103_train0.parquet", False),
    ("Salesforce/wikitext", "dataset", "wikitext-103-raw-v1/test-00000-of-00001.parquet", "wikitext103_test.parquet", False),
    ("HuggingFaceH4/ultrachat_200k", "dataset", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", "ultrachat_test_sft.parquet", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip-model", action="store_true", help="only download tokenizer + corpus")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for repo, rtype, path, name, is_model in FILES:
        if is_model and a.skip_model:
            continue
        dst = out / name
        if dst.exists():
            print(f"have  {dst}")
            continue
        print(f"get   {repo}/{path}")
        src = hf_hub_download(repo, path, repo_type=rtype)
        shutil.copyfile(src, dst)

    # plain-text WikiText-103 test split for llama-perplexity
    wiki_txt = out / "wiki_test.txt"
    if not wiki_txt.exists():
        import pyarrow.parquet as pq
        lines = pq.read_table(out / "wikitext103_test.parquet").column("text").to_pylist()
        wiki_txt.write_text("".join(lines), encoding="utf-8")
        print(f"wrote {wiki_txt}")
    print("done")


if __name__ == "__main__":
    main()
