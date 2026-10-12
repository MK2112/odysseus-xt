#!/usr/bin/env python3
"""Check that a served LoRA adapter is actually applied to inference.

A LoRA can be accepted, listed under ``/v1/models``, even logged as loaded —
yet produce byte-identical output to the base model (the adapter path is
inactive). Registration and HTTP 200s are NOT evidence that a LoRA is live.
This script sends the same deterministic prompts (temperature 0) to the base
id and the adapter id and compares the text::

    python3 scripts/verify_lora_active.py \\
        --base-url http://localhost:8000/v1 \\
        --base "LiquidAI/LFM2.5-1.2B-Instruct" \\
        --adapter dft

Works against any OpenAI-compatible endpoint (Transformers, vLLM, SGLang).
Exit codes: 0 = adapter ACTIVE (at least one prompt differs),
2 = adapter appears INACTIVE (all prompts identical), 1 = error.

Standard library only — runs anywhere Python 3 runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error

DEFAULT_PROMPTS = (
    "Reply with exactly one word: the capital of France.",
    "Count from 7 to 12, one number per line, nothing else.",
    "Rewrite this sentence in the past tense: She walks to the market every morning.",
    "What is 17 multiplied by 23? Reply with the number only.",
    "Write a haiku about a lighthouse. No title, no explanation.",
    "Explain why the sky is blue in exactly two sentences.",
)


def _chat_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _complete(chat_url: str, model: str, prompt: str, max_tokens: int, timeout: float, api_key: str = "") -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        chat_url,
        data=data,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise SystemExit(f"ERROR: HTTP {e.code} for model {model!r}: {detail}")
    except OSError as e:
        raise SystemExit(f"ERROR: cannot reach {chat_url}: {e}")
    try:
        return str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        raise SystemExit(f"ERROR: unexpected response shape for model {model!r}: {str(body)[:300]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a served LoRA adapter changes inference vs its base")
    parser.add_argument("--base-url", required=True, help="Endpoint root, e.g. http://localhost:8000/v1")
    parser.add_argument("--adapter-base-url", default="", help="Endpoint root serving the adapter (default: --base-url)")
    parser.add_argument("--base", required=True, help="Base model id as listed by /v1/models")
    parser.add_argument("--adapter", required=True, help="Adapter model id as listed by /v1/models")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds per request")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Extra prompt to include (repeatable; added to the built-in set).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_url = _chat_url(args.base_url)
    adapter_url = _chat_url(args.adapter_base_url or args.base_url)
    prompts = list(DEFAULT_PROMPTS) + [p for p in args.prompt if p and p.strip()]
    print(f"base:    {args.base} @ {base_url}")
    print(f"adapter: {args.adapter} @ {adapter_url}")
    diffs = 0
    for i, prompt in enumerate(prompts, 1):
        base_text = _complete(base_url, args.base, prompt, args.max_tokens, args.timeout, args.api_key)
        adapter_text = _complete(adapter_url, args.adapter, prompt, args.max_tokens, args.timeout, args.api_key)
        same = base_text.strip() == adapter_text.strip()
        diffs += not same
        print(f"[{i}/{len(prompts)}] {'SAME ' if same else 'DIFF '}: {prompt[:60]}")
        if not same and diffs == 1:
            print(f"  base:    {base_text.strip()[:220]!r}")
            print(f"  adapter: {adapter_text.strip()[:220]!r}")
    print()
    if diffs:
        print(f"ACTIVE: adapter output differs on {diffs}/{len(prompts)} prompt(s). The LoRA is live.")
        return 0
    print(
        f"INACTIVE: all {len(prompts)} prompt(s) byte-identical. The adapter path is almost "
        "certainly not applied — check the serve logs for base/adapter mismatch warnings, "
        "confirm the adapter repo + subfolder, and re-serve."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
