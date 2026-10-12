# Transformers + PEFT LoRA Provider Shape

Last updated: dev@lora-transformers | 2026-10-12

## Scope

Canonical placeholder provider ID `transformers`; OpenAI Chat serving for a
base LLM plus PEFT LoRA adapter(s); generic identity-only inventory
normalization. There is no dedicated Transformers reader or model-card
detector. Served by `scripts/transformers_server.py` through the Cookbook
`Transformers` engine.

## Catalog Shape

`GET /v1/models` returns `object: list`, `data[]` model cards with `id`,
`object`, `owned_by: odysseus-transformers`, `root`, and `parent`. The base
card uses the served base id (`--served-model-name` or `--model`) with
`root` set to the base source and `parent: null`. Each adapter card uses the
adapter `name` from its `--adapter name=source[@subfolder]` spec, with `root`
set to the adapter source (+ `@subfolder` when present) and `parent` set to
the base id. The generic reader retains only identity/raw data.

LoRA cards use a different `id`, root path, and parent per adapter. Keep each
served ID endpoint scoped and do not merge it globally with the base
checkpoint (same rule as the vLLM provider shape).

## Runtime Capability

`POST /v1/chat/completions` accepts OpenAI chat messages, `temperature`,
`top_p`, `max_tokens`/`max_completion_tokens`, `stop` sequences (matched
against generated text, vLLM-style, and stripped from the output), and
`stream` (chunked SSE + `data: [DONE]`). The `model` field selects the active adapter by exact name;
an unknown `model` falls back to the currently active adapter instead of
404ing. There is no native tool-call parser on this backend: tool use works
through the model's text channel only. Prefer vLLM/SGLang when native
tool-calling, tensor parallelism, or high throughput is required.

The tokenizer defaults to the first adapter when that adapter ships
tokenizer files, else to the base model; `--tokenizer` overrides both.

## Fallback And Safety

Cookbook pins the base id plus every adapter name on the auto-registered
endpoint (`pinned_models`), so the picker lists the LoRA ids even before the
first `/v1/models` probe lands. The serve runner preflights
`import torch, transformers, peft, accelerate` in the launch Python and
refuses to start with an install hint when the deps are missing.

Serving refuses nothing about base/adapter fit at launch (local snapshot
paths legitimately differ from the recorded repo id), but it logs a warning
when the adapter's recorded base does not match `--model`.

## Merge And Verify Workflows

Two scripts cover the failure modes dynamic serving cannot:

- `scripts/merge_peft_lora.py` bakes an adapter (plus optional subfolder)
  into a standalone checkpoint servable anywhere the base architecture is
  supported. It aborts on a base/adapter mismatch unless
  `--allow-base-mismatch` is passed, and treats PEFT missing-key warnings
  as a hard failure (a warning-then-save merge is an unmodified base).
- `scripts/verify_lora_active.py` sends deterministic temperature-0 prompts
  to the base id and the adapter id on a running endpoint and verdicts
  ACTIVE (any prompt differs, exit 0) or INACTIVE (all identical, exit 2).
  Run it before any benchmark: registration and HTTP 200s do not prove the
  adapter path is live.

## Current Gaps

- Single active adapter per request; concurrent requests for different
  adapters serialize on a lock while switching.
- Streaming is chunked (one generation, sliced into SSE events), not
  token-by-token.
- No quantization flags; larger models should use vLLM/SGLang instead.
