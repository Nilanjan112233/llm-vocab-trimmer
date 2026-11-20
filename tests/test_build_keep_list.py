"""Selection rules of build_keep_list.py on a toy vocabulary."""
from build_keep_list import bpe_pieces, merge_ranks, select_keep_ids, spelling_variants

NORMAL, CONTROL, BYTE = 1, 3, 6
VOCAB = [
    ("<bos>", CONTROL), ("<0x41>", BYTE), ("<unused0>", NORMAL), ("▁▁", NORMAL),
    ("w", NORMAL), ("h", NORMAL), ("a", NORMAL), ("t", NORMAL), ("s", NORMAL),
    ("wh", NORMAL), ("at", NORMAL), ("what", NORMAL), ("whats", NORMAL),
    ("▁", NORMAL), ("▁whats", NORMAL), ("▁Whats", NORMAL), ("Whats", NORMAL),
    ("hola", NORMAL), ("é", NORMAL),
]
TEXTS = [t for t, _ in VOCAB]
TYPES = [ty for _, ty in VOCAB]
MERGES = ["w h", "a t", "wh at", "what s", "▁ whats"]
ID = {t: i for i, t in enumerate(TEXTS)}


def names(ids):
    return {TEXTS[i] for i in ids}


def test_bpe_pieces_replays_merges():
    rank = merge_ranks(MERGES)
    assert bpe_pieces("whats", rank) == {"w", "h", "a", "t", "s", "wh", "at", "what", "whats"}


def test_spelling_variants():
    v = spelling_variants("▁WhatsApp")
    assert {"whatsapp", "▁whatsapp", "WhatsApp", "▁WHATSAPP", "Whatsapp"} <= v
    assert spelling_variants("▁123") == set()


def test_always_kept_and_never_kept():
    ids, _, _ = select_keep_ids(TEXTS, TYPES, MERGES, counts={}, min_count=50)
    kept = names(ids)
    assert {"<bos>", "<0x41>", "▁▁", "w", "a"} <= kept       # special, byte, whitespace, ASCII
    assert "<unused0>" not in kept and "hola" not in kept and "é" not in kept


def test_frequency_variants_and_closure():
    counts = {ID["▁Whats"]: 100, ID["hola"]: 3}
    ids, n_var, n_closure = select_keep_ids(TEXTS, TYPES, MERGES, counts, min_count=50)
    kept = names(ids)
    assert "▁Whats" in kept and "hola" not in kept             # frequency threshold
    assert {"whats", "▁whats", "Whats"} <= kept                 # spelling variants of ▁Whats
    assert {"wh", "at", "what"} <= kept                         # BPE pieces needed to build "whats"
    assert (n_var, n_closure) == (3, 3)
    assert ids == sorted(ids)


def test_no_variants_mode():
    ids, n_var, _ = select_keep_ids(TEXTS, TYPES, MERGES, {ID["▁Whats"]: 100}, min_count=50, variants=False)
    assert n_var == 0 and "whats" not in names(ids)
