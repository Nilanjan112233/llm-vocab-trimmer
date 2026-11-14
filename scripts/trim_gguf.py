"""Step 2: write a new GGUF that only has the kept tokens.

- Token arrays (texts, scores, types) are filtered; ids are renumbered in the same order.
- BPE merges are kept only if both parts and the result survive.
- Per-token tensors (any tensor whose row dimension equals the vocab size: token embeddings,
  Gemma's per-layer token embeddings, an untied output layer) are sliced row by row on the raw
  quantized bytes, so there is no re-quantization and no quality loss on the kept rows.
- Special token ids (bos/eos/unk/pad/mask/...) are remapped.

Works for the main model and for its MTP draft file (same keep list for both).

Usage:  python scripts/trim_gguf.py <in.gguf> <out.gguf> <keep_ids.json>
"""
import argparse
import json

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("keep_ids")
    a = ap.parse_args()

    keep = np.array(json.load(open(a.keep_ids)), dtype=np.int64)
    r = GGUFReader(a.src)
    f = r.fields
    n_vocab = len(f["tokenizer.ggml.tokens"].data)
    old2new = {int(o): i for i, o in enumerate(keep)}

    texts = [bytes(f["tokenizer.ggml.tokens"].parts[i]).decode("utf-8") for i in f["tokenizer.ggml.tokens"].data]
    scores = [float(f["tokenizer.ggml.scores"].parts[i][0]) for i in f["tokenizer.ggml.scores"].data]
    types = [int(f["tokenizer.ggml.token_type"].parts[i][0]) for i in f["tokenizer.ggml.token_type"].data]
    merges = [bytes(f["tokenizer.ggml.merges"].parts[i]).decode("utf-8") for i in f["tokenizer.ggml.merges"].data]

    kept_text = {texts[i] for i in old2new}
    new_merges = []
    for m in merges:
        x, y = m.split(" ", 1)
        if x in kept_text and y in kept_text and (x + y) in kept_text:
            new_merges.append(m)

    new_fields = {
        "tokenizer.ggml.tokens": ([texts[i] for i in keep], GGUFValueType.STRING),
        "tokenizer.ggml.scores": ([scores[i] for i in keep], GGUFValueType.FLOAT32),
        "tokenizer.ggml.token_type": ([types[i] for i in keep], GGUFValueType.INT32),
        "tokenizer.ggml.merges": (new_merges, GGUFValueType.STRING),
    }
    id_keys = [k for k in f if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")]

    arch = bytes(f["general.architecture"].parts[-1]).decode()
    w = GGUFWriter(a.dst, arch)
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
                print(f"  WARNING {name}={old} not kept; dropping key")
                continue
            w.add_key_value(name, old2new[old], vtype)
        else:
            w.add_key_value(name, field.contents(), vtype, sub_type=sub)
    w.add_key_value("general.description", f"English-only vocab trim: {len(keep)}/{n_vocab} tokens",
                    GGUFValueType.STRING)

    datas = []
    for t in r.tensors:
        data = t.data
        if int(t.shape[-1]) == n_vocab:
            assert data.shape[0] == n_vocab, (t.name, data.shape)
            data = np.ascontiguousarray(data[keep])
            print(f"  slice {t.name}: {n_vocab} -> {len(keep)} rows, {t.n_bytes / 1e6:.0f} MB -> {data.nbytes / 1e6:.0f} MB")
        w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
        datas.append(data)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for d in datas:
        w.write_tensor_data(d, tensor_endianess=r.endianess)
    w.close()
    print(f"done: {a.dst}  vocab {n_vocab} -> {len(keep)}, merges {len(merges)} -> {len(new_merges)}")


if __name__ == "__main__":
    main()
