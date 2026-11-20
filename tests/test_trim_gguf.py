"""trim_gguf.py on a tiny synthetic GGUF (no model download needed)."""
import json

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import quantize

import trim_gguf

# id:   0        1        2    3    4    5     6      7
TOKENS = ["<pad>", "<bos>", "a", "b", "c", "ab", "abc", "zz"]
TYPES = [3, 3, 1, 1, 1, 1, 1, 1]          # 3 = control, 1 = normal
MERGES = ["a b", "ab c", "z z"]
N_VOCAB, N_EMBD = len(TOKENS), 32
KEEP = np.array([0, 1, 2, 3, 5])           # drop "c", "abc", "zz"


def quiet(*_):
    pass


def make_gguf(path, description=None):
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), "llama")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(TOKENS)
    w.add_token_scores([float(-i) for i in range(N_VOCAB)])
    w.add_token_types(TYPES)
    w.add_token_merges(MERGES)
    w.add_bos_token_id(1)
    w.add_pad_token_id(0)
    w.add_eos_token_id(7)                  # not in KEEP -> must be dropped
    if description:
        w.add_description(description)
    w.add_tensor("token_embd.weight", rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float32))
    q = quantize(rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float32), GGMLQuantizationType.Q8_0)
    w.add_tensor("per_layer_token_embd.weight", q, raw_dtype=GGMLQuantizationType.Q8_0)
    w.add_tensor("blk.0.attn_q.weight", rng.standard_normal((4, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def strings(field):
    return [bytes(field.parts[i]).decode() for i in field.data]


def scalars(field):
    return [field.parts[i][0].item() for i in field.data]


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "tiny.gguf"
    make_gguf(p)
    return p


def test_vocab_merges_and_special_ids(model, tmp_path):
    out = tmp_path / "out.gguf"
    stats = trim_gguf.trim_gguf(model, out, KEEP, log=quiet)
    f = GGUFReader(out).fields
    assert strings(f["tokenizer.ggml.tokens"]) == ["<pad>", "<bos>", "a", "b", "ab"]
    assert scalars(f["tokenizer.ggml.token_type"]) == [3, 3, 1, 1, 1]
    assert scalars(f["tokenizer.ggml.scores"]) == [0, -1, -2, -3, -5]
    assert strings(f["tokenizer.ggml.merges"]) == ["a b"]       # "ab c" and "z z" lose a part
    assert int(f["tokenizer.ggml.bos_token_id"].contents()) == 1
    assert int(f["tokenizer.ggml.padding_token_id"].contents()) == 0
    assert "tokenizer.ggml.eos_token_id" not in f                # dropped, not left pointing at a wrong token
    assert stats["dropped_id_keys"] == ["tokenizer.ggml.eos_token_id"]
    assert sorted(stats["sliced_tensors"]) == ["per_layer_token_embd.weight", "token_embd.weight"]


def test_kept_rows_are_bit_identical(model, tmp_path):
    out = tmp_path / "out.gguf"
    trim_gguf.trim_gguf(model, out, KEEP, log=quiet)
    a = {t.name: t for t in GGUFReader(model).tensors}
    b = {t.name: t for t in GGUFReader(out).tensors}
    for name in ("token_embd.weight", "per_layer_token_embd.weight"):
        assert b[name].tensor_type == a[name].tensor_type        # no re-quantization
        assert np.array_equal(b[name].data, a[name].data[KEEP])
    assert np.array_equal(b["blk.0.attn_q.weight"].data, a["blk.0.attn_q.weight"].data)
    assert trim_gguf.verify_trim(model, out, KEEP, log=quiet)


def test_chunked_writing_gives_the_same_file(model, tmp_path, monkeypatch):
    one = tmp_path / "one_chunk.gguf"
    trim_gguf.trim_gguf(model, one, KEEP, log=quiet)
    monkeypatch.setattr(trim_gguf, "CHUNK_ROWS", 2)                # 5 kept rows -> 3 chunks
    many = tmp_path / "many_chunks.gguf"
    trim_gguf.trim_gguf(model, many, KEEP, log=quiet)
    assert one.read_bytes() == many.read_bytes()
    assert trim_gguf.verify_trim(model, many, KEEP, log=quiet)


def test_verify_catches_a_wrong_file(model, tmp_path):
    out = tmp_path / "out.gguf"
    trim_gguf.trim_gguf(model, out, KEEP, log=quiet)
    assert not trim_gguf.verify_trim(model, out, np.array([0, 1, 2, 3, 4]), log=quiet)


def test_keep_list_without_any_merge_is_an_error(model, tmp_path):
    with pytest.raises(ValueError, match="no BPE merges"):
        trim_gguf.trim_gguf(model, tmp_path / "out.gguf", np.array([0, 1, 2, 3, 4]), log=quiet)


def test_existing_description_is_replaced(tmp_path):
    src = tmp_path / "desc.gguf"
    make_gguf(src, description="original text")
    out = tmp_path / "out.gguf"
    trim_gguf.trim_gguf(src, out, KEEP, log=quiet)
    assert GGUFReader(out).fields["general.description"].contents() == "English-only vocab trim: 5/8 tokens"


@pytest.mark.parametrize("ids, msg", [
    ([], "non-empty"),
    ([0, 1, 99], "0..7"),
    ([-1, 2], "0..7"),
    ([3, 1, 2], "sorted"),
    ([1, 1, 2], "sorted"),
    ([0, 1.5], "integer"),
])
def test_bad_keep_lists_are_rejected(tmp_path, ids, msg):
    p = tmp_path / "keep.json"
    p.write_text(json.dumps(ids))
    with pytest.raises(ValueError, match=msg):
        trim_gguf.load_keep_ids(p, N_VOCAB)


def test_cli_verify_and_overwrite_guard(model, tmp_path, capsys):
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps(KEEP.tolist()))
    out = tmp_path / "sub" / "out.gguf"                         # parent folder is created
    trim_gguf.main([str(model), str(out), str(keep), "--verify"])
    assert "verify ok" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        trim_gguf.main([str(model), str(out), str(keep)])        # refuses to overwrite
    trim_gguf.main([str(model), str(out), str(keep), "--force"])
