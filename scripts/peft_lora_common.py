#!/usr/bin/env python3
"""Shared dependency-free helpers for PEFT LoRA workflows.

Used by ``scripts/transformers_server.py`` (serve base + adapter) and
``scripts/merge_peft_lora.py`` (bake an adapter into a standalone
checkpoint). Everything here is pure Python — no torch / transformers /
peft / fastapi imports — so it stays importable and unit-testable anywhere.

Adapter spec grammar (``name=source[@subfolder|:subfolder]``)::

    dft=Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended@best_model
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUBFOLDER_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
)


def _split_subfolder(location: str) -> tuple[str, str]:
    """Split ``source[@subfolder|:subfolder]`` into (source, subfolder).

    ``@`` is the preferred separator. ``:`` is accepted unless the location
    is a Windows drive path (``C:\\...`` / ``C:/...``), where the colon
    belongs to the drive prefix.
    """
    loc = (location or "").strip()
    if "@" in loc:
        src, _, sub = loc.rpartition("@")
        return src.strip(), sub.strip()
    if re.match(r"^[A-Za-z]:[\\/]", loc):
        return loc, ""
    if ":" in loc and not loc.startswith(":") and not loc.endswith(":"):
        src, _, sub = loc.rpartition(":")
        # A second colon means this is not a clean single separator
        # (e.g. a URI); leave the value untouched in that case.
        if ":" not in src:
            return src.strip(), sub.strip()
    return loc, ""


def parse_adapter_spec(spec: str) -> dict:
    """Parse one ``--adapter`` value into ``{name, source, subfolder}``.

    Raises :class:`ValueError` on malformed input.
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("empty adapter spec")
    if "=" in raw:
        name, _, location = raw.partition("=")
        name, location = name.strip(), location.strip()
        if not name:
            raise ValueError(f"adapter spec {raw!r} is missing a name before '='")
    else:
        name, location = "", raw
    if not location:
        raise ValueError(f"adapter spec {raw!r} is missing a source")
    source, subfolder = _split_subfolder(location)
    if not source:
        raise ValueError(f"adapter spec {raw!r} is missing a source")
    if subfolder:
        if not _SUBFOLDER_RE.match(subfolder) or ".." in subfolder.split("/"):
            raise ValueError(f"adapter spec {raw!r} has an invalid subfolder {subfolder!r}")
    if not name:
        leaf = source.rstrip("/").rsplit("/", 1)[-1].replace("--", "/").rsplit("/", 1)[-1]
        leaf = re.sub(r"[^A-Za-z0-9._-]+", "-", leaf).strip("-") or "lora"
        name = f"{leaf}-{subfolder}" if subfolder else leaf
    if not _NAME_RE.match(name):
        raise ValueError(f"adapter spec {raw!r} has an invalid name {name!r} (use [A-Za-z0-9._-])")
    return {"name": name, "source": source, "subfolder": subfolder}


def _looks_like_repo_id(source: str) -> bool:
    return bool(_REPO_RE.match(source)) and not os.path.exists(os.path.expanduser(source))


def _has_tokenizer_files(directory: str | Path) -> bool:
    try:
        names = {p.name for p in Path(directory).iterdir()}
    except OSError:
        return False
    return any(n in names for n in _TOKENIZER_FILES)


def resolve_adapter_dir(source: str, subfolder: str = "") -> str:
    """Resolve an adapter source (+ optional subfolder) to a local directory.

    Local paths are used directly; HuggingFace repo ids are fetched via
    ``snapshot_download`` (honouring ``HF_TOKEN``). The returned directory
    must contain ``adapter_config.json``.
    """
    candidate = os.path.expanduser(source)
    if os.path.isdir(candidate):
        path = os.path.join(candidate, subfolder) if subfolder else candidate
        if not os.path.isdir(path):
            raise FileNotFoundError(f"adapter subfolder not found: {path}")
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            raise FileNotFoundError(f"no adapter_config.json in {path}")
        return path
    if not _looks_like_repo_id(source):
        raise FileNotFoundError(
            f"adapter source is neither a local directory nor a <org>/<name> repo: {source!r}"
        )
    from huggingface_hub import snapshot_download

    allow = [f"{subfolder}/*"] if subfolder else None
    snap = snapshot_download(
        repo_id=source,
        allow_patterns=allow,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
    )
    path = os.path.join(snap, subfolder) if subfolder else snap
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(
            f"adapter_config.json not found for {source!r}"
            + (f" under subfolder {subfolder!r}" if subfolder else "")
        )
    return path


def resolve_tokenizer_source(explicit: str, adapter_dirs: list[str], base: str) -> str:
    """Pick the tokenizer source: explicit flag wins, else the first adapter
    that ships tokenizer files, else the base model."""
    if explicit and explicit.strip():
        return explicit.strip()
    for d in adapter_dirs:
        if d and _has_tokenizer_files(d):
            return d
    return base


def read_adapter_config(adapter_dir: str | Path) -> dict:
    """Read an adapter's ``adapter_config.json`` (raises with context)."""
    path = Path(adapter_dir) / "adapter_config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")
    except (OSError, ValueError) as e:
        raise ValueError(f"unreadable adapter_config.json in {adapter_dir}: {e}")


def adapter_base_id(adapter_dir: str | Path) -> str:
    """Return the adapter's recorded base model (``base_model_name_or_path``)."""
    return str(read_adapter_config(adapter_dir).get("base_model_name_or_path") or "").strip()


def normalize_model_id(value: str) -> str:
    """Normalize a model id for comparison: HF cache segments (``--``),
    local paths, and snapshot hash suffixes all reduce to a lowercase leaf
    name (``Org/Model`` and ``.../models--Org--Model/snapshots/<hash>``
    both become ``model``)."""
    text = (value or "").strip().rstrip("/")
    if not text:
        return ""
    parts = text.replace("\\", "/").split("/")
    leaf = ""
    for seg in parts:
        # HF cache layout: models--org--name (+ snapshots/<hash> below it).
        if seg.startswith("models--"):
            leaf = seg[len("models--") :].replace("--", "/")
            break
    else:
        leaf = parts[-1]
    if re.fullmatch(r"[0-9a-f]{30,}", leaf):
        return ""
    if "/" in leaf:
        leaf = leaf.rsplit("/", 1)[-1]
    return leaf.lower()


def adapter_matches_base(adapter_dir: str | Path, base: str) -> bool:
    """True when the adapter's recorded base matches the given base model.

    Compares normalized leaf names so ``Org/Model``, a local snapshot path,
    and an HF cache dir for the same checkpoint all agree. An empty recorded
    base (or an unresolvable snapshot hash) never matches — callers decide
    whether that warns or fails.
    """
    recorded = normalize_model_id(adapter_base_id(adapter_dir))
    if not recorded:
        return False
    wanted = normalize_model_id(base)
    return bool(wanted) and recorded == wanted


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get((name or "auto").lower(), torch.bfloat16)


def _target_device(explicit: str) -> str:
    want = (explicit or "auto").lower()
    if want in {"cuda", "cpu", "mps"}:
        return want
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
