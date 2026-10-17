"""Optional Claude transport for LLMPlanner. Off by default; nothing imports this unless you ask.

It is a plain HTTPS call with the standard library (no SDK) and has NOT been run against the live API
(there was no key when this was written). The planner it feeds is safe regardless: whatever comes back
is parsed strictly and must pass validate_plan before anything runs.
"""

from __future__ import annotations

import json
import os
import urllib.request

MODEL = "claude-sonnet-5-5"


def claude_transport(prompt: str, model: str = MODEL, max_tokens: int = 1024) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    body = json.dumps(
        {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https URL
        data = json.load(resp)
    return "".join(block.get("text", "") for block in data.get("content", []))
