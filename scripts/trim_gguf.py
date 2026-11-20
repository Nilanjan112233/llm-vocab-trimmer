"""Step 2: write a new GGUF that only has the kept tokens.

- Per-token tokenizer arrays (token texts, scores, types) are filtered; ids are renumbered
  in the same order.
- BPE merges are kept only if both parts and the result survive.
- Per-token tensors (any tensor whose row dimension equals the vocab size: token embeddings,
  Gemma's per-layer token embeddings, an untied output layer) are sliced row by row on the raw
  quantized bytes, so there is no re-quantization and no quality loss on the kept rows.
- Special token ids (bos/eos/unk/pad/mask/...) are remapped.

Works for the main model and for its MTP draft file (same keep list for both).

Usage:
  python scripts/trim_gguf.py <in.gguf> <out.gguf> <keep_ids.json> [--verify] [--force]

  --verify  re-open the written file and check that every kept row is bit-identical to the
            original row and that every other tensor is unchanged
  --force   overwrite <out.gguf> if it already exists
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter


def load_keep_ids(path, n_vocab):
    """Read a keep list (JSON array of original token ids) and check it is usable."""
    with open(path, encoding="utf-8") as fh:
        ids = json.load(fh)
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{path}: expected a non-empty JSON list of token ids")
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise ValueError(f"{path}: every entry must be an integer token id")
    keep = np.array(ids, dtype=np.int64)
    if keep.min() < 0 or keep.max() >= n_vocab:
        raise ValueError(f"{path}: ids must be in 0..{n_vocab - 1} (got {keep.min()}..{keep.max()}); "
                         "is this keep list for a different model?")
    if np.any(np.diff(keep) <= 0):
        raise ValueError(f"{path}: ids must be sorted and unique (new ids follow the old order)")
    return keep


def is_vocab_tensor(t, n_vocab):
    """GGUF lists shapes fastest dimension first, so the number of rows is the last entry."""
    return int(t.shape[-1]) == n_vocab


CHUNK_ROWS = 4096  # rows copied at a time, so the trim needs little RAM even for GB-sized tensors


class RowSlice:
    """The rows `keep` of a (memory-mapped) tensor, written to disk chunk by chunk.

    GGUFWriter.write_tensor_data only needs .nbytes, .tofile() and .byteswap(), so this avoids
    holding the sliced copy (hundreds of MB for Gemma's per-layer table) in RAM.
    """

    def __init__(self, data, keep, swap=False):
        self.data, self.keep, self.swap = data, keep, swap
        self.shape = (len(keep),) + tuple(data.shape[1:])
        self.dtype = data.dtype
        self.nbytes = len(keep) * (data.nbytes // data.shape[0])

    def chunks(self):
        for s in range(0, len(self.keep), CHUNK_ROWS):
            part = np.ascontiguousarray(self.data[self.keep[s:s + CHUNK_ROWS]])
            yield part.byteswap() if self.swap else part

    def byteswap(self, inplace=False):
        return RowSlice(self.data, self.keep, not self.swap)

    def tofile(self, fout):
        for part in self.chunks():
            part.tofile(fout)


def _same_bytes(x, y):
    """Compare two equally shaped arrays (or RowSlices) chunk by chunk."""
    xs = x.chunks() if isinstance(x, RowSlice) else (x[s:s + CHUNK_ROWS] for s in range(0, len(x), CHUNK_ROWS))
    ys = (y[s:s + CHUNK_ROWS] for s in range(0, len(y), CHUNK_ROWS))
    return all(np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)) for a, b in zip(xs, ys))


def _array_strings(field):
    return [bytes(field.parts[i]).decode("utf-8") for i in field.data]


def _array_scalars(field, cast):
    return [cast(field.parts[i][0]) for i in field.data]


def trim_gguf(src, dst, keep, log=print, reader=None):
    """Write `dst`: a copy of `src` with only the token ids in `keep` (sorted numpy int array).

    `reader` can be an already open GGUFReader for `src`. Returns a small dict of stats.
    """
    r = reader or GGUFReader(src)
    f = r.fields
    if "tokenizer.ggml.tokens" not in f:
        raise ValueError(f"{src}: no tokenizer.ggml.tokens - not a GGUF with an embedded vocab")
    n_vocab = len(f["tokenizer.ggml.tokens"].data)
    if keep.max() >= n_vocab:
        raise ValueError(f"keep list has id {keep.max()} but {src} has only {n_vocab} tokens")
    old2new = {int(o): i for i, o in enumerate(keep)}

    texts = _array_strings(f["tokenizer.ggml.tokens"])
    new_fields = {"tokenizer.ggml.tokens": ([texts[i] for i in keep], GGUFValueType.STRING)}
    if "tokenizer.ggml.scores" in f:
        scores = _array_scalars(f["tokenizer.ggml.scores"], float)
        new_fields["tokenizer.ggml.scores"] = ([scores[i] for i in keep], GGUFValueType.FLOAT32)
    if "tokenizer.ggml.token_type" in f:
        types = _array_scalars(f["tokenizer.ggml.token_type"], int)
        new_fields["tokenizer.ggml.token_type"] = ([types[i] for i in keep], GGUFValueType.INT32)

    merges, new_merges = [], []
    if "tokenizer.ggml.merges" in f:
        merges = _array_strings(f["tokenizer.ggml.merges"])
        kept_text = {texts[i] for i in old2new}
        for m in merges:
            x, y = m.split(" ", 1)
            if x in kept_text and y in kept_text and (x + y) in kept_text:
                new_merges.append(m)
        if merges and not new_merges:
            raise ValueError("no BPE merges survive with this keep list - the tokenizer would be unusable "
                             "(keep the merge pieces, see build_keep_list.py)")
        new_fields["tokenizer.ggml.merges"] = (new_merges, GGUFValueType.STRING)

    id_keys = [k for k in f if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")]
    dropped_ids = []

    arch = bytes(f["general.architecture"].parts[-1]).decode()
    w = GGUFWriter(dst, arch)
    for name, field in f.items():
        if name == "general.architecture" or name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        if name in new_fields:
            val, st = new_fields[name]
            w.add_key_value(name, val, GGUFValueType.ARRAY, sub_type=st)
        elif name in id_keys:
            old = int(field.contents())
            if old not in old2new:
                log(f"  WARNING {name}={old} not kept; dropping key")
                dropped_ids.append(name)
                continue
            w.add_key_value(name, old2new[old], vtype)
        elif vtype == GGUFValueType.ARRAY and name.startswith("tokenizer.") and len(field.data) == n_vocab:
            # any other per-token array (Gemma 4 has none) must be filtered too, or ids would not line up
            vals = field.contents()
            w.add_key_value(name, [vals[i] for i in keep], vtype, sub_type=sub)
        else:
            w.add_key_value(name, field.contents(), vtype, sub_type=sub)
    w.add_key_value("general.description", f"English-only vocab trim: {len(keep)}/{n_vocab} tokens",
                    GGUFValueType.STRING)

    datas, sliced = [], []
    for t in r.tensors:
        data = t.data
        if is_vocab_tensor(t, n_vocab):
            assert data.shape[0] == n_vocab, (t.name, data.shape)
            data = RowSlice(data, keep)
            sliced.append(t.name)
            log(f"  slice {t.name}: {n_vocab} -> {len(keep)} rows, {t.n_bytes / 1e6:.0f} MB -> {data.nbytes / 1e6:.0f} MB")
        w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
        datas.append(data)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for d in datas:
        w.write_tensor_data(d, tensor_endianess=r.endianess)
    w.close()
    log(f"done: {dst}  vocab {n_vocab} -> {len(keep)}, merges {len(merges)} -> {len(new_merges)}")
    return {"n_vocab": n_vocab, "n_kept": len(keep), "merges": len(merges), "merges_kept": len(new_merges),
            "sliced_tensors": sliced, "dropped_id_keys": dropped_ids}


def verify_trim(src, dst, keep, log=print):
    """Check `dst` against `src`: same vocab subset, kept rows bit-identical, other tensors unchanged."""
    a, b = GGUFReader(src), GGUFReader(dst)
    n_vocab = len(a.fields["tokenizer.ggml.tokens"].data)
    old_texts = _array_strings(a.fields["tokenizer.ggml.tokens"])
    new_texts = _array_strings(b.fields["tokenizer.ggml.tokens"])
    problems = []
    if new_texts != [old_texts[i] for i in keep]:
        problems.append("token list is not the kept subset of the original")
    tb = {t.name: t for t in b.tensors}
    for t in a.tensors:
        u = tb.pop(t.name, None)
        if u is None:
            problems.append(f"tensor {t.name} missing")
            continue
        if u.tensor_type != t.tensor_type:
            problems.append(f"tensor {t.name}: type changed")
            continue
        want = RowSlice(t.data, keep) if is_vocab_tensor(t, n_vocab) else t.data
        if tuple(want.shape) != tuple(u.data.shape) or not _same_bytes(want, u.data):
            problems.append(f"tensor {t.name}: data differs")
    problems += [f"unexpected tensor {n}" for n in tb]
    for p in problems:
        log(f"  VERIFY FAIL {p}")
    if not problems:
        log(f"verify ok: {len(a.tensors)} tensors, kept rows bit-identical, other tensors unchanged")
    return not problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trim the vocabulary of a GGUF model to a keep list.")
    ap.add_argument("src", help="original GGUF file")
    ap.add_argument("dst", help="output GGUF file")
    ap.add_argument("keep_ids", help="JSON list of original token ids to keep (from build_keep_list.py)")
    ap.add_argument("--verify", action="store_true", help="check the written file against the original")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    a = ap.parse_args(argv)

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_file():
        ap.error(f"input not found: {src}")
    if dst.exists() and dst.resolve() == src.resolve():
        ap.error("output must be a different file from the input")
    if dst.exists() and not a.force:
        ap.error(f"{dst} already exists (use --force to overwrite)")
    dst.parent.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(src)
    if "tokenizer.ggml.tokens" not in reader.fields:
        ap.error(f"{src}: no tokenizer.ggml.tokens - not a GGUF with an embedded vocab")
    try:
        keep = load_keep_ids(a.keep_ids, len(reader.fields["tokenizer.ggml.tokens"].data))
    except ValueError as e:
        ap.error(str(e))

    trim_gguf(src, dst, keep, reader=reader)
    del reader
    s0, s1 = src.stat().st_size, dst.stat().st_size
    print(f"size: {s0 / 1e9:.2f} GB -> {s1 / 1e9:.2f} GB ({100 * (1 - s1 / s0):.0f}% smaller)")
    if a.verify and not verify_trim(src, dst, keep):
        sys.exit(1)


if __name__ == "__main__":
    main()
