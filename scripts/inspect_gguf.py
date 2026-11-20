"""Show how much of a GGUF file is per-token tables (the part a vocab trim can shrink).

Groups the tensors into: per-token tensors (row dimension = vocab size, e.g. token_embd,
per_layer_token_embd, an untied output layer) and everything else (the transformer blocks,
norms, ...). Read-only.

Usage:  python scripts/inspect_gguf.py model.gguf [more.gguf ...] [--json]
"""
import argparse
import json
from pathlib import Path

from gguf import GGUFReader


def breakdown(path):
    r = GGUFReader(path)
    n_vocab = len(r.fields["tokenizer.ggml.tokens"].data)
    groups = {}
    for t in r.tensors:
        key = t.name if int(t.shape[-1]) == n_vocab else "other (transformer blocks, norms, ...)"
        g = groups.setdefault(key, {"params": 0, "bytes": 0, "per_token": key == t.name})
        g["params"] += int(t.n_elements)
        g["bytes"] += int(t.n_bytes)
    size = Path(path).stat().st_size
    return {"file": Path(path).name, "file_bytes": size, "n_vocab": n_vocab, "groups": groups}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf", nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args(argv)
    results = [breakdown(p) for p in a.gguf]
    if a.json:
        print(json.dumps(results, indent=2))
        return
    for res in results:
        size = res["file_bytes"]
        print(f"{res['file']}: {size / 1e9:.2f} GB, vocab {res['n_vocab']:,}")
        per_token = 0
        for name, g in sorted(res["groups"].items(), key=lambda kv: -kv[1]["bytes"]):
            per_token += g["bytes"] if g["per_token"] else 0
            print(f"  {name:<40} {g['params'] / 1e9:6.2f}B params  {g['bytes'] / 1e9:6.2f} GB  "
                  f"{100 * g['bytes'] / size:5.1f}%")
        print(f"  {'per-token tables in total':<40} {'':>14}  {per_token / 1e9:6.2f} GB  {100 * per_token / size:5.1f}%")


if __name__ == "__main__":
    main()
