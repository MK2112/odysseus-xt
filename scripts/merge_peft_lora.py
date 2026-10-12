#!/usr/bin/env python3
"""Bake a PEFT LoRA adapter into a standalone merged checkpoint.

Dynamic adapters are convenient, but they only work on backends with LoRA
support — and a silently inactive adapter (accepted, listed, yet producing
byte-identical output to the base) is the nastiest LoRA failure mode. A
merged checkpoint serves anywhere the base architecture is supported
(vLLM, SGLang, Transformers) with no adapter path to go stale::

    python3 scripts/merge_peft_lora.py \\
        --base LiquidAI/LFM2.5-1.2B-Instruct \\
        --adapter Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended@best_model \\
        --output /models/lfm2.5-1.2b-dft-merged

    # then serve the merged dir like any base model, e.g.
    python3 scripts/transformers_server.py --model /models/lfm2.5-1.2b-dft-merged
    # or: vllm serve /models/lfm2.5-1.2b-dft-merged

The adapter spec uses the same ``name=source[@subfolder]`` grammar as
``scripts/transformers_server.py`` (the name part is ignored here).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_peft_lora")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into a standalone checkpoint")
    parser.add_argument("--base", required=True, help="Base model: HF repo id or local path/snapshot")
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA adapter spec: [name=]source[@subfolder]. "
        "Example: Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended@best_model",
    )
    parser.add_argument("--output", required=True, help="Output directory for the merged checkpoint")
    parser.add_argument("--tokenizer", default="", help="Tokenizer source override (default: adapter when it ships tokenizer files, else base)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="Merge even when the adapter's recorded base differs from --base. "
        "By default a mismatch aborts: merging across checkpoints silently produces garbage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into a non-empty --output directory (default: abort).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        spec = parse_adapter_spec(args.adapter)
    except ValueError as e:
        raise SystemExit(f"ERROR: invalid --adapter: {e}")

    adapter_dir = resolve_adapter_dir(spec["source"], spec["subfolder"])
    logger.info("Adapter: %s%s", spec["source"], f"@{spec['subfolder']}" if spec["subfolder"] else "")

    # Strict base-match guard: a cross-checkpoint merge is silent garbage.
    # Warn-and-continue only when the adapter records no base at all
    # (nothing to compare against).
    try:
        recorded = adapter_base_id(adapter_dir)
    except (OSError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")
    if recorded:
        if not adapter_matches_base(adapter_dir, args.base):
            msg = (
                f"Adapter was trained for {recorded!r}, but --base is {args.base!r}. "
                "Merging across checkpoints silently produces a broken model. "
                "Pass the matching --base, or --allow-base-mismatch if you really mean it."
            )
            if args.allow_base_mismatch:
                logger.warning("%s Proceeding anyway (--allow-base-mismatch).", msg)
            else:
                raise SystemExit(f"ERROR: {msg}")
    else:
        logger.warning("Adapter records no base model; cannot verify --base %r matches.", args.base)

    out = os.path.abspath(os.path.expanduser(args.output))
    if os.path.exists(out) and os.listdir(out) and not args.overwrite:
        raise SystemExit(f"ERROR: output {out} exists and is not empty (pass --overwrite to reuse it).")
    os.makedirs(out, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok_source = resolve_tokenizer_source(args.tokenizer, [adapter_dir], args.base)
    logger.info("Loading tokenizer from %s", tok_source)
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=args.trust_remote_code)

    dtype = _torch_dtype(args.dtype)
    device = _target_device(args.device)
    logger.info("Loading base %s (dtype=%s device=%s)", args.base, args.dtype, device)
    load_kwargs: dict = {
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": args.trust_remote_code,
        "token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
    }
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, **load_kwargs)
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, **load_kwargs)
    if device != "cuda":
        try:
            base_model = base_model.to(device)
        except Exception as e:
            logger.warning("Could not place model on %s (%s); staying on current device", device, e)

    import warnings

    logger.info("Applying adapter from %s", adapter_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    # A successful save is NOT proof of a successful merge: PEFT warns about
    # missing adapter keys and still writes a model (effectively unmodified).
    # Treat those warnings as a hard failure.
    missing_key_warnings = [
        str(w.message)
        for w in caught
        if "missing" in str(w.message).lower() and "key" in str(w.message).lower()
        or "not used" in str(w.message).lower()
    ]
    if missing_key_warnings:
        raise SystemExit(
            "ERROR: adapter keys missing when applying the adapter — the merge would be "
            f"an unmodified base model. First warning: {missing_key_warnings[0][:300]} "
            "This usually means the adapter was trained for a different model "
            "architecture/loader than --base."
        )
    trainable, _total = model.get_nb_trainable_parameters()
    if not trainable:
        raise SystemExit("ERROR: adapter applied with zero trainable parameters — refusing to merge.")
    logger.info("Merging adapter into base weights…")
    merged = model.merge_and_unload()
    merged.eval()

    logger.info("Saving merged checkpoint to %s", out)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    logger.info("Done. Serve the merged dir like any base model, then run:")
    logger.info("  python3 scripts/verify_lora_active.py --base-url <endpoint> --base <id> --adapter <id>")


if __name__ == "__main__":
    main()
