"""
Local web UI for the 3-agent demo (Blue / Red / Red Advance).

    python scripts/demo_web.py            # http://127.0.0.1:8000
    python scripts/demo_web.py --port 8080

Serves scripts/demo_web.html and a /chat endpoint that drives the REAL pipeline
(same defense as the graded run) — Blue via OpenRouter, Red/Red Advance via
OpenAI. An HTML artifact on claude.ai cannot reach these local agents or your
.env keys, so the demo runs as a small local server (Python stdlib, no new deps).

Needs a filled .env (like `python src/main.py`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HTML = Path(__file__).with_suffix(".html")
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from core.utils import chat_with_agent  # noqa: E402


def _content(text: str):
    from google.genai import types

    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def _text_of(content) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


class BlueSession:
    def __init__(self):
        from assignment.pipeline import build_production_plugins
        from agents.agent import blue_verified_facts, create_blue_agent

        self.plugins = build_production_plugins()
        self.agent, self.runner = create_blue_agent(plugins=[])
        for p in self.plugins:
            if hasattr(p, "protect_prompt"):
                self.agent.instruction = p.protect_prompt(
                    self.agent.instruction, public_text=blue_verified_facts()
                )

    async def handle(self, text: str) -> dict:
        from guardrails.policy import labels_for_input

        ctx = SimpleNamespace(user_id="web-demo")
        labels = labels_for_input(text)
        for p in self.plugins:
            cb = getattr(p, "on_user_message_callback", None)
            if cb is None:
                continue
            blocked = await cb(invocation_context=ctx, user_message=_content(text))
            if blocked is not None:
                return {"agent": "blue", "reply": _text_of(blocked), "blocked": True,
                        "layer": p.name, "redacted": False, "labels": sorted(labels), "judge": None}
        try:
            reply, _ = await chat_with_agent(self.agent, self.runner, text)
        except Exception as exc:
            return {"agent": "blue", "reply": f"[LLM error] {type(exc).__name__}: {exc}",
                    "blocked": False, "layer": "error", "redacted": False,
                    "labels": sorted(labels), "judge": None}

        out = next((p for p in self.plugins if p.name == "output_guardrail"), None)
        blocked = redacted = False
        judge = None
        if out is not None:
            before = (out.blocked_count, out.redacted_count)
            resp = SimpleNamespace(content=_content(reply))
            resp = await out.after_model_callback(callback_context=None, llm_response=resp) or resp
            reply = _text_of(resp.content)
            labels |= getattr(out, "last_labels", set())
            blocked = out.blocked_count > before[0]
            redacted = out.redacted_count > before[1]
            j = getattr(out, "last_judge", None)
            if j:
                judge = {"reason": j["reason"], "verdict": j["verdict"][:80]}
        return {"agent": "blue", "reply": reply, "blocked": blocked,
                "layer": "output_guardrail" if (blocked or redacted) else None,
                "redacted": redacted, "labels": sorted(labels), "judge": judge}


class RedSession:
    def __init__(self, advance: bool):
        from agents.agent import create_red_agent_default
        from agents.guards_agent import create_red_agent_advance

        self.advance = advance
        self.agent, self.runner = (
            create_red_agent_advance() if advance else create_red_agent_default()
        )
        self.target = "red_advance" if advance else "red_default"

    async def handle(self, text: str) -> dict:
        from attacks.attacks import classify_attack_outcome

        name = "redadv" if self.advance else "red"
        try:
            reply, _ = await chat_with_agent(self.agent, self.runner, text)
        except Exception as exc:
            return {"agent": name, "reply": f"[LLM error] {type(exc).__name__}: {exc}",
                    "leaked": False, "layer": "error", "outcome": "error"}
        o = classify_attack_outcome(text, reply, target_name=self.target)
        return {"agent": name, "reply": reply, "leaked": o["leaked"],
                "layer": o["layer"], "outcome": o["blocked_at"].split(" —")[0]}


SESSIONS: dict = {}
LOCK = threading.Lock()


def get_session(agent: str):
    if agent not in SESSIONS:
        SESSIONS[agent] = BlueSession() if agent == "blue" else RedSession(agent == "redadv")
    return SESSIONS[agent]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            agent = req.get("agent", "blue")
            message = (req.get("message") or "").strip()
            if agent not in ("blue", "red", "redadv") or not message:
                self._send(400, json.dumps({"reply": "bad request"}))
                return
            with LOCK:  # agents + Blue rate-limit state are shared, serialize requests
                result = asyncio.run(get_session(agent).handle(message))
            self._send(200, json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            self._send(500, json.dumps({"reply": f"server error: {type(exc).__name__}: {exc}"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"VinBank guardrails demo → {url}   (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.shutdown()


if __name__ == "__main__":
    main()
