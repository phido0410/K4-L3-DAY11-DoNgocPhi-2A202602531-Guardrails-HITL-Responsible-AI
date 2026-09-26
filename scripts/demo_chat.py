"""
Interactive demo — chat with the three lab agents to test prompts by hand.

    python scripts/demo_chat.py            # start on Blue
    python scripts/demo_chat.py red        # start on Red

Agents (see README):
  blue    → create_blue_agent + full defense pipeline (rate limit → input →
            OpenRouter liquid/lfm-2.5-2.6b → output guardrail). Shows which
            layer acted, data-flow labels, and the LLM judge when it fires.
  red     → create_red_agent_default  (soft; leaks by design)   [gpt-4o-mini]
  redadv  → create_red_agent_advance  (hard target, B2)         [gpt-4o-mini]

For red / redadv each reply is classified with classify_attack_outcome and the
leak check from data/protected/vinbank_secrets.json (DEMO secrets).

Commands:  /blue  /red  /redadv  /reset  /help  /quit
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from core.utils import chat_with_agent  # noqa: E402

C_RESET, C_DIM, C_BLUE, C_RED, C_YEL, C_GRN = (
    "\033[0m", "\033[2m", "\033[36m", "\033[31m", "\033[33m", "\033[32m",
)


def _content(text: str):
    from google.genai import types

    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def _text_of(content) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


class BlueSession:
    """Drives the real defense pipeline for one user, keeping rate-limit state."""

    name = "blue"

    def __init__(self):
        from assignment.pipeline import build_production_plugins
        from agents.agent import blue_verified_facts, create_blue_agent

        self.plugins = build_production_plugins()
        self.agent, self.runner = create_blue_agent(plugins=[])
        for plugin in self.plugins:
            if hasattr(plugin, "protect_prompt"):
                self.agent.instruction = plugin.protect_prompt(
                    self.agent.instruction, public_text=blue_verified_facts()
                )
        self.user_id = "demo-user"

    async def send(self, text: str) -> None:
        from guardrails.policy import labels_for_input

        ctx = SimpleNamespace(user_id=self.user_id)
        labels = labels_for_input(text)

        # pre-LLM layers (rate limit, input guardrail)
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            blocked = await cb(invocation_context=ctx, user_message=_content(text))
            if blocked is not None:
                print(f"{C_YEL}  ⛔ blocked at [{plugin.name}]{C_RESET}")
                print(f"{C_BLUE}  Blue:{C_RESET} {_text_of(blocked)}")
                return

        # LLM
        try:
            reply, _ = await chat_with_agent(self.agent, self.runner, text)
        except Exception as exc:
            print(f"{C_RED}  LLM error: {type(exc).__name__}: {exc}{C_RESET}")
            return

        # post-LLM layer (output guardrail)
        out_plugin = next((p for p in self.plugins if p.name == "output_guardrail"), None)
        note = ""
        if out_plugin is not None:
            before = (out_plugin.blocked_count, out_plugin.redacted_count)
            llm_response = SimpleNamespace(content=_content(reply))
            llm_response = await out_plugin.after_model_callback(
                callback_context=None, llm_response=llm_response
            ) or llm_response
            reply = _text_of(llm_response.content)
            labels |= getattr(out_plugin, "last_labels", set())
            if out_plugin.blocked_count > before[0]:
                note = f"{C_YEL}  🛡  output guardrail BLOCKED the reply (leak/unsafe){C_RESET}"
            elif out_plugin.redacted_count > before[1]:
                note = f"{C_YEL}  ✂  output guardrail REDACTED PII{C_RESET}"
            judged = getattr(out_plugin, "last_judge", None)
            if judged:
                note += f"\n{C_DIM}  judge[{judged['reason']}] → {judged['verdict'][:60]}{C_RESET}"

        if note:
            print(note)
        if labels:
            print(f"{C_DIM}  data-flow labels: {sorted(labels)}{C_RESET}")
        print(f"{C_BLUE}  Blue:{C_RESET} {reply}")


class RedSession:
    """Soft (red) or hard (redadv) target — shows leak classification."""

    def __init__(self, advance: bool):
        from agents.agent import create_red_agent_default
        from agents.guards_agent import create_red_agent_advance

        self.name = "redadv" if advance else "red"
        self.color = C_RED
        self.agent, self.runner = (
            create_red_agent_advance() if advance else create_red_agent_default()
        )
        self.target_name = "red_advance" if advance else "red_default"

    async def send(self, text: str) -> None:
        from attacks.attacks import classify_attack_outcome

        try:
            reply, _ = await chat_with_agent(self.agent, self.runner, text)
        except Exception as exc:
            print(f"{C_RED}  LLM error: {type(exc).__name__}: {exc}{C_RESET}")
            return
        outcome = classify_attack_outcome(text, reply, target_name=self.target_name)
        tag = (
            f"{C_RED}LEAKED — secret disclosed{C_RESET}" if outcome["leaked"]
            else f"{C_GRN}{outcome['blocked_at']}{C_RESET}"
        )
        print(f"{C_DIM}  outcome: {tag}{C_DIM}  (layer={outcome['layer']}){C_RESET}")
        print(f"  {self.color}{self.name}:{C_RESET} {reply}")


def make_session(name: str):
    if name == "blue":
        return BlueSession()
    return RedSession(advance=(name == "redadv"))


BANNER = f"""{C_DIM}────────────────────────────────────────────────────────────
 Lab 11 demo — chat with Blue / Red / Red Advance
 switch: /blue  /red  /redadv    other: /reset  /help  /quit
────────────────────────────────────────────────────────────{C_RESET}"""

HELP = """Commands:
  /blue     talk to Blue (your defended assistant)
  /red      talk to Red (soft target — leaks by design)
  /redadv   talk to Red Advance (hard target, bonus B2)
  /reset    rebuild the current agent (clears Blue rate-limit window)
  /help     show this help
  /quit     exit
Anything else is sent to the current agent."""


async def main() -> None:
    start = sys.argv[1].lower().lstrip("/") if len(sys.argv) > 1 else "blue"
    if start not in {"blue", "red", "redadv"}:
        start = "blue"
    print(BANNER)
    session = make_session(start)
    print(f"{C_DIM}  active agent: {session.name}{C_RESET}")

    loop = asyncio.get_event_loop()
    while True:
        try:
            line = (await loop.run_in_executor(None, input, f"\n[{session.name}] you > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue
        low = line.lower()
        if low in {"/quit", "/exit", "/q"}:
            print("bye")
            return
        if low == "/help":
            print(HELP)
            continue
        if low in {"/blue", "/red", "/redadv"}:
            session = make_session(low.lstrip("/"))
            print(f"{C_DIM}  switched to {session.name}{C_RESET}")
            continue
        if low == "/reset":
            session = make_session(session.name)
            print(f"{C_DIM}  {session.name} rebuilt{C_RESET}")
            continue
        if low.startswith("/"):
            print(f"{C_DIM}  unknown command — /help{C_RESET}")
            continue
        await session.send(line)


if __name__ == "__main__":
    asyncio.run(main())
