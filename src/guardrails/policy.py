"""
Data-flow labels + sink policy — one table decides what may leave, and where.

Labels (what the data is / where it came from):
  confidential  credentials, internal hosts, system prompt
  customer_pii  phone, email, national id, card / account number
  untrusted     third-party content: email, RAG document, web page, tool output

Sinks (where the data goes):
  reply   answer shown to the customer
  egress  outbound call to another system
  action  side effect (transfer money, update profile, send statement …)

Everything derived from a request inherits the request's labels (taint), and
the strictest rule over all labels wins: block > hitl > redact > allow.

Why "confidential" is value-level here: the lab puts the secrets in the Blue
system prompt by design. Tainting the whole LLM context would taint every
reply, so confidential data is tracked by value (content_filter +
ProtectedContext) while "untrusted" is tracked per request.
"""
from __future__ import annotations

import re
from typing import Iterable

CONFIDENTIAL, CUSTOMER_PII, UNTRUSTED = "confidential", "customer_pii", "untrusted"
REPLY, EGRESS, ACTION = "reply", "egress", "action"
ALLOW, REDACT, HITL, BLOCK = "allow", "redact", "hitl", "block"

POLICY: dict[tuple[str, str], str] = {
    (CONFIDENTIAL, REPLY): BLOCK,
    (CONFIDENTIAL, EGRESS): BLOCK,
    (CONFIDENTIAL, ACTION): BLOCK,
    (CUSTOMER_PII, REPLY): REDACT,
    (CUSTOMER_PII, EGRESS): BLOCK,
    (CUSTOMER_PII, ACTION): HITL,
    (UNTRUSTED, REPLY): ALLOW,     # summarising an email is fine; injection is an input-layer job
    (UNTRUSTED, EGRESS): HITL,     # third-party text must not flow into internal systems unreviewed
    (UNTRUSTED, ACTION): HITL,     # an email can never authorise a side effect on its own
}
_STRICTNESS = {ALLOW: 0, REDACT: 1, HITL: 2, BLOCK: 3}

# content_filter issue types that make a value confidential (the rest is customer PII)
CONFIDENTIAL_ISSUE_TYPES = {
    "password", "api_key", "internal_host", "connection_string", "system_prompt", "encoded_credential",
}


def decide(labels: Iterable[str], sink: str) -> str:
    """Strictest decision for the given labels at a sink (no labels → allow)."""
    return max((POLICY[(label, sink)] for label in labels), key=_STRICTNESS.__getitem__, default=ALLOW)


def labels_from_issues(issue_types: Iterable[str]) -> set[str]:
    return {CONFIDENTIAL if t in CONFIDENTIAL_ISSUE_TYPES else CUSTOMER_PII for t in issue_types}


def labels_for_text(text: str, context=None) -> set[str]:
    """Value-level labels of a piece of text (same detectors as the output guardrail)."""
    from guardrails.output_guardrails import content_filter

    issues = content_filter(text or "", context)["issues"]
    return labels_from_issues(issue.split(":", 1)[0] for issue in issues)


# Third-party content pasted into the chat. Tool/RAG results should instead be
# labelled at the source (see agents.security_boundary.ExternalContent).
_EXTERNAL_CONTENT = re.compile(
    r"\b(?:summari[sz]e|translate|read|analy[sz]e|review|process|forward|tóm\s*tắt|dịch|đọc)\b[^.\n]{0,40}?"
    r"\b(?:e-?mails?|documents?|attachments?|files?|web\s*pages?|websites?|articles?|letters?|pdfs?|"
    r"tài\s*liệu|thư|văn\s*bản|tin\s*nhắn)\b"
    r"|^\s*(?:from|subject|to|cc)\s*:"
    r"|\b(?:external|third[\s-]party|forwarded)\s+(?:e-?mails?|documents?|content|messages?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def labels_for_input(text: str) -> set[str]:
    """Request-level taint: does the user message carry third-party content?"""
    return {UNTRUSTED} if _EXTERNAL_CONTENT.search(text or "") else set()
