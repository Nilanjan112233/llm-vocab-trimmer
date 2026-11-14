"""Step 1: decide which tokens to keep (English + code + chat).

How:
  1. Tokenize an English corpus (WikiText-103 Wikipedia text, UltraChat English conversations,
     the Python standard library source code, and the benchmark tasks) with the model's real
     tokenizer and count how often each token is used.
  2. Keep: all special/control/byte tokens, all whitespace-only tokens, all printable ASCII single
     characters, common Unicode punctuation/emoji, and every token used >= MIN_COUNT times.
  3. Spelling variants: for every kept word also keep its forms with/without the leading-space
     marker and in lower / Capital / UPPER case. Without this the model has to spell words like
     "whatsapp" from pieces inside JSON and makes typos (see README, "Lesson learned").
  4. Closure: for every kept token, replay the BPE merges that build it and keep every
     intermediate piece, so the trimmed tokenizer can still reach it.

Output: <out-dir>/keep_ids_<MIN_COUNT>[v].json  (sorted original token ids)

Usage:
  python scripts/build_keep_list.py --gguf data/gemma-4-E2B-it-Q4_0.gguf --data data 50
  python scripts/build_keep_list.py ... --no-variants 20 50
"""
import argparse
import collections
import json
import sysconfig
import time
from pathlib import Path

import pyarrow.parquet as pq
from gguf import GGUFReader
from tokenizers import Tokenizer

NORMAL = 1  # GGUF token type; everything else (control, user-defined, byte) is always kept
EXTRA_CHARS = set("’‘“”–—…•·°€£₹¥©®™×÷±≈≠≤≥→←↑↓✓✔✗✘★☆♥❤️😀😂🙂👍🙏🔥✅❌⚠️")


def read_vocab(gguf_path):
    f = GGUFReader(gguf_path).fields
    texts = [bytes(f["tokenizer.ggml.tokens"].parts[i]).decode("utf-8") for i in f["tokenizer.ggml.tokens"].data]
    types = [int(f["tokenizer.ggml.token_type"].parts[i][0]) for i in f["tokenizer.ggml.token_type"].data]
    merges = [bytes(f["tokenizer.ggml.merges"].parts[i]).decode("utf-8") for i in f["tokenizer.ggml.merges"].data]
    return texts, types, merges


def corpus_texts(data, tasks_path):
    def batched(name, items, size):
        batch = []
        for x in items:
            batch.append(x)
            if len(batch) >= size:
                yield name, batch
                batch = []
        if batch:
            yield name, batch

    wiki = pq.read_table(data / "wikitext103_train0.parquet").column("text").to_pylist()
    yield from batched("wiki", (line for line in wiki if line.strip()), 2000)

    chats = pq.read_table(data / "ultrachat_test_sft.parquet").column("messages").to_pylist()
    yield from batched("chat", (m["content"] for conv in chats for m in conv), 2000)

    def stdlib_files():
        for p in Path(sysconfig.get_paths()["stdlib"]).rglob("*.py"):
            if "site-packages" in p.parts:
                continue
            try:
                yield p.read_text(encoding="utf-8")
            except Exception:
                continue
    yield from batched("code", stdlib_files(), 200)

    tasks = json.load(open(tasks_path, encoding="utf-8"))
    yield "tasks", [t["input"] for t in tasks["tasks"]] + list(tasks["tools"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("min_counts", nargs="*", type=int, default=[50])
    ap.add_argument("--gguf", required=True, help="original (untrimmed) GGUF model")
    ap.add_argument("--data", default="data", help="folder with tokenizer.json and corpus parquet files")
    ap.add_argument("--tasks", default="bench/tasks.json")
    ap.add_argument("--out-dir", default="keep_lists")
    ap.add_argument("--no-variants", action="store_true", help="skip spelling variants (v1 behaviour)")
    a = ap.parse_args()
    data, out_dir = Path(a.data), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_texts, tok_types, merges = read_vocab(a.gguf)
    n = len(tok_texts)
    tk = Tokenizer.from_file(str(data / "tokenizer.json"))
    for i in (2, 105, 1000, 50000, 100000, 200000):  # tokenizer.json ids must match the GGUF ids
        assert tk.id_to_token(i) == tok_texts[i], (i, tk.id_to_token(i), tok_texts[i])

    counts, per_source = collections.Counter(), collections.Counter()
    t0 = time.time()
    for src, batch in corpus_texts(data, a.tasks):
        for enc in tk.encode_batch(batch, add_special_tokens=False):
            counts.update(enc.ids)
            per_source[src] += len(enc.ids)
    print("tokenized:", dict(per_source), f"in {time.time() - t0:.0f}s; distinct tokens used: {len(counts)}")

    rank = {}
    for r, m in enumerate(merges):
        x, y = m.split(" ", 1)
        rank.setdefault((x, y), r)
    tok_id = {t: i for i, t in enumerate(tok_texts)}

    def bpe_pieces(word):
        """Replay BPE on one token's text; return every intermediate piece created."""
        parts = list(word)
        made = set(parts)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
            made.add(parts[bi])
        return made

    for min_count in a.min_counts:
        keep = set()
        for i, (t, ty) in enumerate(zip(tok_texts, tok_types)):
            if ty != NORMAL:
                keep.add(i)                                  # control, user-defined, byte fallback
            elif set(t) <= set("▁\n\t\r "):
                keep.add(i)                                  # whitespace runs (code indentation)
            elif len(t) == 1 and (32 <= ord(t) < 127 or t in EXTRA_CHARS):
                keep.add(i)                                  # base characters
            elif t.startswith("<unused"):
                pass                                         # placeholder tokens, never used
            elif counts.get(i, 0) >= min_count:
                keep.add(i)

        n_var = 0
        if not a.no_variants:
            for i in list(keep):
                t = tok_texts[i]
                if tok_types[i] != NORMAL or len(t) < 2:
                    continue
                core = t[1:] if t.startswith("▁") else t
                if not core or not core[0].isalpha():
                    continue
                for c in {core, core.lower(), core.capitalize(), core.upper()}:
                    for v in (c, "▁" + c):
                        j = tok_id.get(v)
                        if j is not None and j not in keep:
                            keep.add(j)
                            n_var += 1

        added = 0
        for i in list(keep):
            if tok_types[i] != NORMAL:
                continue
            for piece in bpe_pieces(tok_texts[i]):
                j = tok_id.get(piece)
                if j is not None and j not in keep:
                    keep.add(j)
                    added += 1

        ids = sorted(keep)
        covered = sum(c for i, c in counts.items() if i in keep) / sum(counts.values())
        out = out_dir / f"keep_ids_{min_count}{'' if a.no_variants else 'v'}.json"
        json.dump(ids, open(out, "w"))
        print(f"min_count={min_count:>4}: keep {len(ids):>6} of {n} ({100 * len(ids) / n:4.1f}%), "
              f"variants +{n_var}, closure +{added}, corpus coverage {100 * covered:.3f}% -> {out}")


if __name__ == "__main__":
    main()
