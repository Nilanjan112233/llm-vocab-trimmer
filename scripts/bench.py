"""Benchmark one GGUF model on the assistant tool-calling tasks (bench/tasks.json).

Starts llama-server, sends every task with JSON-schema-constrained output
({"tool": ..., "args": {...}}), scores tool + args, records time per task, generation
speed and peak RAM of the server process, then stops the server.

Usage:
  python scripts/bench.py --server path/to/llama-server --model model.gguf [--draft mtp.gguf] [--name NAME]

Settings used for the published results: 4 threads, context 4096, temperature 0, reasoning off,
llama.cpp build b12360, Intel i5-11300H CPU. The --spec-type draft-mtp and --reasoning flags need a
recent llama.cpp build (b12360 works). Peak RAM is the server's resident memory, sampled after each
request. Timings are from a single pass, so expect a few percent of noise between runs.
"""
import argparse
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil

SPEED_PROMPT = "Explain in about 120 words why the sky is blue."


def build_system_prompt(tools):
    lines = [
        "You are a personal assistant on the user's phone. For every user message, choose exactly ONE tool and fill its args.",
        "Reply ONLY with JSON: {\"tool\": <tool name>, \"args\": {...}}.",
        "Tools:",
    ]
    lines += [f"- {name}: {desc}" for name, desc in tools.items()]
    return "\n".join(lines)


def schema_for(tools):
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": list(tools)},
            "args": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["tool", "args"],
    }


def score(task, out):
    """Returns (tool_ok, args_ok). Each expected arg needs any one of its keywords (case-insensitive)."""
    if out.get("tool") != task["tool"]:
        return False, False
    args = {k.lower(): str(v).lower() for k, v in (out.get("args") or {}).items()}
    for key, words in task["args"].items():
        if not any(w.lower() in args.get(key, "") for w in words):
            return True, False
    return True, True


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def summarize(name, model_path, load_s, peak_bytes, rows, timings, sample):
    n = len(rows)
    return {
        "model": name, "file": Path(model_path).name, "file_gb": round(Path(model_path).stat().st_size / 1e9, 2),
        "load_seconds": round(load_s, 1), "peak_ram_gb": round(peak_bytes / 1e9, 2),
        "tool_accuracy": round(sum(r["tool_ok"] for r in rows) / n, 3),
        "full_accuracy": round(sum(r["args_ok"] for r in rows) / n, 3),
        "avg_task_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "gen_tokens_per_s": round(timings.get("predicted_per_second", 0), 1),
        "prompt_tokens_per_s": round(timings.get("prompt_per_second", 0), 1),
        "speed_sample": sample[:300],
    }


class Server:
    def __init__(self, server, model, draft, port, threads, log_path, ctx=4096):
        cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-t", str(threads),
               "-c", str(ctx), "--reasoning", "off", "-np", "1"]
        if draft:
            cmd += ["--spec-type", "draft-mtp", "-md", draft]
        self.port = port
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.ps = psutil.Process(self.proc.pid)
        self.peak = 0

    def wait_ready(self, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early - see {self.log.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return time.time() - t0
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not ready after {timeout}s - see {self.log.name}")

    def chat(self, body, timeout=300):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        self.peak = max(self.peak, self.ps.memory_info().rss)
        return out

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except Exception:
            self.proc.kill()
        self.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="path to llama-server(.exe)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", help="MTP draft GGUF for speculative decoding (Gemma 4)")
    ap.add_argument("--name", help="result name (default: model file name)")
    ap.add_argument("--tasks", default="bench/tasks.json")
    ap.add_argument("--out", default="results/local")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096, help="context size passed to llama-server")
    a = ap.parse_args()

    for label, p in (("--server", a.server), ("--model", a.model), ("--draft", a.draft), ("--tasks", a.tasks)):
        if p and not Path(p).is_file():
            ap.error(f"{label}: file not found: {p}")
    if port_in_use(a.port):
        ap.error(f"port {a.port} is already in use (another llama-server?) - pick one with --port")

    model = Path(a.model)
    name = a.name or model.stem + ("+mtp" if a.draft else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(a.tasks, encoding="utf-8") as fh:
        data = json.load(fh)
    tasks, tools = data["tasks"], data["tools"]

    srv = Server(a.server, str(model), a.draft, a.port, a.threads, out / f"{name}.server.log", a.ctx)
    try:
        load_s = srv.wait_ready()
        sysmsg, schema = build_system_prompt(tools), schema_for(tools)
        rows = []
        for t in tasks:
            body = {"messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": t["input"]}],
                    "temperature": 0, "max_tokens": 120,
                    "response_format": {"type": "json_schema", "json_schema": {"name": "action", "schema": schema}}}
            t0 = time.time()
            resp = srv.chat(body)
            dt = time.time() - t0
            text = resp["choices"][0]["message"].get("content") or ""
            try:
                got = json.loads(text)
            except Exception:
                got = {}
            tool_ok, args_ok = score(t, got)
            rows.append({"id": t["id"], "input": t["input"], "expected": t["tool"], "got": got,
                         "raw": text if not got else None, "tool_ok": tool_ok, "args_ok": args_ok,
                         "seconds": round(dt, 2)})
            mark = "OK " if args_ok else ("~  " if tool_ok else "X  ")
            print(f"[{name}] {mark}{t['id']:>2} {dt:5.2f}s  {text[:90]}")

        sp = srv.chat({"messages": [{"role": "user", "content": SPEED_PROMPT}], "temperature": 0, "max_tokens": 200})
        sample = sp["choices"][0]["message"].get("content") or ""
        summary = summarize(name, model, load_s, srv.peak, rows, sp.get("timings", {}), sample)
        with open(out / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "tasks": rows}, fh, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
