"""inspect_gguf.py groups per-token tensors apart from the rest."""
from inspect_gguf import breakdown, main
from test_trim_gguf import make_gguf


def test_breakdown(tmp_path, capsys):
    p = tmp_path / "tiny.gguf"
    make_gguf(p)
    res = breakdown(p)
    assert res["n_vocab"] == 8
    per_token = {n for n, g in res["groups"].items() if g["per_token"]}
    assert per_token == {"token_embd.weight", "per_layer_token_embd.weight"}
    assert sum(g["params"] for g in res["groups"].values()) == 8 * 32 * 2 + 4 * 32
    main([str(p)])
    assert "per-token tables in total" in capsys.readouterr().out
