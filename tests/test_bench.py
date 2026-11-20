"""Scoring rules of bench.py and the task file."""
import json
from pathlib import Path

from bench import schema_for, score

TASK = {"tool": "send_message", "args": {"to": ["mom", "mother"], "text": ["late"]}}


def test_fully_right_is_case_insensitive():
    assert score(TASK, {"tool": "send_message", "args": {"to": "Mom", "text": "Running LATE"}}) == (True, True)


def test_right_tool_wrong_args():
    assert score(TASK, {"tool": "send_message", "args": {"to": "dad", "text": "late"}}) == (True, False)
    assert score(TASK, {"tool": "send_message", "args": {"to": "mom"}}) == (True, False)


def test_wrong_tool_or_unparsable_output():
    assert score(TASK, {"tool": "reply", "args": {"text": "ok"}}) == (False, False)
    assert score(TASK, {}) == (False, False)


def test_task_file_is_consistent():
    path = Path(__file__).resolve().parents[1] / "bench" / "tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["tasks"]) == 30
    assert len({t["id"] for t in data["tasks"]}) == 30
    tools = set(schema_for(data["tools"])["properties"]["tool"]["enum"])
    for t in data["tasks"]:
        assert t["tool"] in tools, t
        assert all(isinstance(w, list) and w for w in t["args"].values()), t


def test_summary_fields(tmp_path):
    from bench import summarize

    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1000)
    rows = [{"tool_ok": True, "args_ok": True, "seconds": 1.0},
            {"tool_ok": True, "args_ok": False, "seconds": 2.0}]
    s = summarize("m+mtp", model, 2.04, 3.02e9, rows, {"predicted_per_second": 21.46}, "x" * 500)
    assert s["full_accuracy"] == 0.5 and s["tool_accuracy"] == 1.0
    assert s["avg_task_seconds"] == 1.5 and s["gen_tokens_per_s"] == 21.5 and s["peak_ram_gb"] == 3.02
    assert len(s["speed_sample"]) == 300 and s["file"] == "m.gguf"
