#!/usr/bin/env python3
"""OpenAI-compatible chat server for base LLM + PEFT LoRA adapter(s).

Covers the workflow from the Cookbook LoRA feature request::

    base = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    model = PeftModel.from_pretrained(base, "Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended")
    model_best = PeftModel.from_pretrained(base, "...", subfolder="best_model")

Usage::

    python3 scripts/transformers_server.py \\
        --model LiquidAI/LFM2.5-1.2B-Instruct \\
        --adapter dft=Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended@best_model \\
        --port 8000

Adapter spec grammar (repeatable ``--adapter``)::

    name=source[@subfolder|:subfolder] | source[@subfolder|:subfolder] | name=source | source

* ``name`` — the OpenAI model id exposed under ``/v1/models`` (``[A-Za-z0-9._-]+``).
* ``source`` — a HuggingFace repo id (``org/name``), a local directory, or a
  ``~``-relative path holding ``adapter_config.json``.
* ``subfolder`` — optional adapter subdirectory inside the repo (e.g.
  ``best_model``). ``@`` is preferred; ``:`` is accepted for non-Windows
  paths (a leading ``C:\\`` drive prefix is never treated as a separator).

The tokenizer defaults to the first adapter when that adapter ships tokenizer
files, else to the base model; ``--tokenizer`` overrides both. Each adapter
is addressable via the ``model`` field of ``/v1/chat/completions`` (like
vLLM ``--lora-modules``); an unknown ``model`` falls back to the active
adapter instead of 404ing so the picker stays usable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

# Sibling import: when launched as `python3 scripts/transformers_server.py`
# the script directory is already on sys.path; the insert below covers odd
# launchers (tests, REPLs) that import this file by path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peft_lora_common import (
    _target_device,
    _torch_dtype,
    adapter_base_id,
    adapter_matches_base,
    parse_adapter_spec,
    resolve_adapter_dir,
    resolve_tokenizer_source,
)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transformers_server")

_model = None
_tokenizer = None
_adapters: list[dict] = []
_base_id = ""
_active_adapter = ""
_lock = threading.Lock()
_args = None

_DEFAULT_ALLOWED_HOSTS = ["127.0.0.1", "localhost", "::1"]
_DEFAULT_CORS_ORIGINS: list = []


def _compute_allowed_hosts(bind_host: str, extras=None) -> list:
    seen = []
    for h in (bind_host, *_DEFAULT_ALLOWED_HOSTS, *(extras or [])):
        h = (h or "").strip()
        if h and h not in seen:
            seen.append(h)
    return seen


def _compute_cors_origins(extras=None) -> list:
    seen = []
    for o in (*_DEFAULT_CORS_ORIGINS, *(extras or [])):
        o = (o or "").strip()
        if o and o not in seen:
            seen.append(o)
    return seen


def _configure_security_middleware(application, allowed_hosts, allowed_origins):
    if application.middleware_stack is not None:
        raise RuntimeError("security middleware must be configured before the app starts serving")
    application.user_middleware.clear()
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )


@asynccontextmanager
async def lifespan(application):
    load_model()
    yield


app = FastAPI(title="Odysseus Transformers LoRA Server", lifespan=lifespan)
_configure_security_middleware(app, _DEFAULT_ALLOWED_HOSTS, _DEFAULT_CORS_ORIGINS)


def load_model() -> None:
    """Load the base model, tokenizer, and all configured adapters."""
    global _model, _tokenizer, _adapters, _base_id, _active_adapter
    if _model is not None:
        return
    if _args is None:
        raise RuntimeError("server args not initialised")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = _args.model
    _base_id = _args.served_model_name.strip() or base
    specs = [parse_adapter_spec(s) for s in (_args.adapter or [])]
    seen = set()
    for s in specs:
        if s["name"] in seen:
            raise ValueError(f"duplicate adapter name: {s['name']!r}")
        seen.add(s["name"])

    adapter_dirs = [resolve_adapter_dir(s["source"], s["subfolder"]) for s in specs]
    for spec, adir in zip(specs, adapter_dirs):
        # Fail fast on a base/adapter mismatch: applying an adapter trained
        # for a different checkpoint silently produces garbage. Warn-only
        # here (local snapshot paths legitimately differ from the recorded
        # repo id); the merge script enforces this strictly instead.
        try:
            recorded = adapter_base_id(adir)
            if recorded and not adapter_matches_base(adir, base):
                logger.warning(
                    "Adapter %s was trained for %r, but the base is %r. "
                    "If generations look wrong, serve the matching base — or run "
                    "scripts/verify_lora_active.py to check the adapter is live.",
                    spec["name"],
                    recorded,
                    base,
                )
        except (OSError, ValueError) as e:
            logger.warning("Could not read adapter config for %s: %s", spec["name"], e)
    tok_source = resolve_tokenizer_source(_args.tokenizer, adapter_dirs, base)
    logger.info("Loading tokenizer from %s", tok_source)
    _tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=_args.trust_remote_code)
    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype = _torch_dtype(_args.dtype)
    device = _target_device(_args.device)
    logger.info("Loading base model %s (dtype=%s device=%s)", base, _args.dtype, device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=_args.trust_remote_code,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )
    except TypeError:
        # transformers < 4.46 spells the kwarg torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=_args.trust_remote_code,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        )
    if device != "cuda":
        try:
            model = model.to(device)
        except Exception as e:
            logger.warning("Could not place model on %s (%s); staying on current device", device, e)
    model.eval()

    _adapters = []
    for spec, adir in zip(specs, adapter_dirs):
        if not _adapters:
            logger.info("Applying primary adapter %s from %s", spec["name"], adir)
            model = PeftModel.from_pretrained(model, adir)
        else:
            logger.info("Loading secondary adapter %s from %s", spec["name"], adir)
            model.load_adapter(adir, adapter_name=spec["name"])
        _adapters.append({**spec, "dir": adir})
    if _adapters:
        _active_adapter = _adapters[0]["name"]
        try:
            model.set_adapter(_active_adapter)
        except Exception as e:
            logger.warning("set_adapter(%s) failed: %s", _active_adapter, e)
    _model = model
    logger.info(
        "Ready: base=%s adapters=%s active=%s",
        _base_id,
        [a["name"] for a in _adapters],
        _active_adapter or "(base)",
    )


def _select_adapter(requested: str) -> str:
    """Resolve the requested model id to the effective id for this request.

    An exact adapter-name match activates that adapter (persistent via
    ``set_adapter``). The base id selects unadapted inference. Unknown ids
    fall back to the currently active adapter instead of failing, so stale
    picker entries keep working. Base-model selection is NOT applied here:
    PEFT's ``disable_adapter`` is a context manager, so ``_generate_for``
    wraps base generation in it per request.
    """
    name = (requested or "").strip()
    names = [a["name"] for a in _adapters]
    if name and name in names:
        if name != _active_adapter:
            with _lock:
                try:
                    _model.set_adapter(name)
                except Exception as e:
                    logger.warning("set_adapter(%s) failed: %s", name, e)
                    return _active_adapter or _base_id
                globals()["_active_adapter"] = name
        return name
    if name and (name == _base_id or (_args is not None and name == _args.model)):
        return _base_id
    return _active_adapter or _base_id


class ChatMessage(BaseModel):
    role: str = "user"
    content: object = ""


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = []
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: object = None


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and str(item.get("type", "")).lower() == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _prompt_from_messages(messages: list[ChatMessage]) -> str:
    chat = [{"role": m.role or "user", "content": _message_text(m.content)} for m in messages]
    chat = [m for m in chat if m["content"]]
    if not chat:
        raise HTTPException(400, "no message content")
    try:
        return _tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Models without a chat template: fall back to plain concatenation.
        return "\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\nassistant:"


def _stop_sequences(req: ChatRequest) -> list[str]:
    raw = req.stop
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(s) for s in raw if str(s or "").strip()][:8]


def _generate(prompt: str, req: ChatRequest) -> tuple[str, dict]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    max_new = req.max_tokens if req.max_tokens is not None else req.max_completion_tokens
    try:
        max_new = max(1, min(int(max_new or _args.max_tokens), 8192))
    except (TypeError, ValueError):
        max_new = _args.max_tokens
    temperature = 0.7 if req.temperature is None else float(req.temperature)
    top_p = 0.9 if req.top_p is None else float(req.top_p)
    do_sample = temperature > 0
    inputs = _tokenizer(prompt, return_tensors="pt")
    try:
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    except Exception:
        pass
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_kwargs: dict = {"max_new_tokens": max_new, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    stops = _stop_sequences(req)
    if stops:
        tokenizer = _tokenizer

        class _StopOnSubstrings(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                try:
                    text = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
                except Exception:
                    return False
                return any(s in text for s in stops)

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnSubstrings()])
    started = time.time()
    with _lock, torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][prompt_len:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Mirror vLLM: the stop sequence ends the output and is not part of it.
    for s in _stop_sequences(req):
        idx = text.find(s)
        if idx >= 0:
            text = text[:idx]
            break
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": int(new_tokens.shape[0]),
        "total_tokens": prompt_len + int(new_tokens.shape[0]),
    }
    logger.info("generate model=%s %dtok in %.2fs", _active_adapter or _base_id, usage["completion_tokens"], time.time() - started)
    return text, usage


def _generate_for(effective: str, prompt: str, req: ChatRequest) -> tuple[str, dict]:
    """Generate with the effective model id resolved by _select_adapter.

    Base-model requests run inside PEFT's ``disable_adapter`` context (it is
    a context manager, not a persistent switch); adapter requests generate
    with the already-activated adapter.
    """
    if _adapters and effective == _base_id:
        with _model.disable_adapter():
            return _generate(prompt, req)
    return _generate(prompt, req)


@app.get("/health")
def health():
    return {"status": "ok", "model": _base_id, "adapters": [a["name"] for a in _adapters]}


@app.get("/v1/models")
def list_models():
    data = [
        {
            "id": _base_id,
            "object": "model",
            "owned_by": "odysseus-transformers",
            "root": _args.model,
            "parent": None,
        }
    ]
    for a in _adapters:
        data.append(
            {
                "id": a["name"],
                "object": "model",
                "owned_by": "odysseus-transformers",
                "root": a["source"] + (f"@{a['subfolder']}" if a["subfolder"] else ""),
                "parent": _base_id,
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "model is still loading")
    effective = _select_adapter(req.model)
    prompt = _prompt_from_messages(req.messages)
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    if req.stream:
        text, usage = _generate_for(effective, prompt, req)

        def _framed():
            # Chunked SSE (single generation, streamed in slices) — keeps
            # the Odysseus chat UI's streaming path working without a
            # token-by-token streamer thread.
            step = 60
            for i in range(0, max(1, len(text)), step):
                delta = text[i : i + step]
                payload = (
                    '{"id":"%s","object":"chat.completion.chunk","created":%d,'
                    '"model":"%s","choices":[{"index":0,"delta":{"content":%s},'
                    '"finish_reason":null}]}' % (chat_id, created, effective, _json_dumps(delta))
                )
                yield f"data: {payload}\n\n"
            final = (
                '{"id":"%s","object":"chat.completion.chunk","created":%d,'
                '"model":"%s","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                '"usage":%s}' % (chat_id, created, effective, _json_dumps(usage))
            )
            yield f"data: {final}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_framed(), media_type="text/event-stream")
    text, usage = _generate_for(effective, prompt, req)
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": effective,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _json_dumps(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a base LLM + PEFT LoRA adapter(s) via OpenAI-compatible API")
    parser.add_argument("--model", required=True, help="Base model: HF repo id or local path/snapshot")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="LoRA adapter spec: name=source[@subfolder]. Repeatable. "
        "Example: dft=Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended@best_model",
    )
    parser.add_argument("--tokenizer", default="", help="Tokenizer source override (default: first adapter with tokenizer files, else base)")
    parser.add_argument("--served-model-name", default="", help="Model id reported for the base (default: --model)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--max-tokens", type=int, default=2048, help="Default max_new_tokens per request")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Additional Host header value to accept (DNS-rebinding allowlist). Can be repeated.",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Additional CORS origin to allow. Can be repeated. Defaults to no cross-origin access.",
    )
    return parser


def main() -> None:
    global _args
    _args = build_parser().parse_args()
    # Validate adapter specs before binding the port so typos fail fast
    # with a clear message instead of a half-started server.
    for spec in _args.adapter or []:
        try:
            parsed = parse_adapter_spec(spec)
        except ValueError as e:
            raise SystemExit(f"ERROR: invalid --adapter: {e}")
        logger.info("Configured adapter: %s -> %s%s", parsed["name"], parsed["source"], f"@{parsed['subfolder']}" if parsed["subfolder"] else "")
    if not _args.tokenizer:
        logger.info("Tokenizer: auto (first adapter with tokenizer files, else base model)")
    final_hosts = _compute_allowed_hosts(_args.host, _args.allowed_host)
    final_origins = _compute_cors_origins(_args.allowed_origin)
    _configure_security_middleware(app, final_hosts, final_origins)
    logger.info("security middleware: allowed_hosts=%s allowed_origins=%s", final_hosts, final_origins or "(none — default-deny)")
    uvicorn.run(app, host=_args.host, port=_args.port)


if __name__ == "__main__":
    main()
