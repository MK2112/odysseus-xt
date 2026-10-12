"""LoRA (Transformers + PEFT) serving support.

Covers the Cookbook `Transformers` backend added for base-LLM + LoRA-adapter
workflows such as::

    base = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    model = PeftModel.from_pretrained(base, "Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended")
    model_best = PeftModel.from_pretrained(base, "...", subfolder="best_model")

* Adapter-spec parsing in scripts/transformers_server.py (pure function, so
  the heavy torch/transformers/peft/uvicorn imports are stubbed here — the
  module must stay importable without them).
* Serve-command acceptance + diagnosis + LoRA-name extraction on the backend.
* Transformers recipe in static/js/cookbook-deps-recipes.js (node).
* Source-shape wiring in the browser-heavy cookbook modules.
"""

import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_HAS_NODE = shutil.which("node") is not None


def _load_common():
    """Import scripts/peft_lora_common.py (pure stdlib — no stubs needed)."""
    name = "odysseus_peft_lora_common_under_test"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "peft_lora_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tcommon():
    return _load_common()


def _load_transformers_server():
    """Import scripts/transformers_server.py with server-only deps stubbed.

    The spec parser under test is pure Python; torch/transformers/peft are
    only imported inside load_model(), while uvicorn/fastapi/pydantic live at
    module level for the ASGI app. Stub the module-level ones so this test
    runs without the serving stack installed.
    """
    name = "odysseus_transformers_server_under_test"
    if name in sys.modules:
        return sys.modules[name]
    sys.modules["uvicorn"] = types.ModuleType("uvicorn")
    fastapi = types.ModuleType("fastapi")

    class _App:
        def __init__(self, *a, **k):
            self.middleware_stack = None
            self.user_middleware = []

        def add_middleware(self, *a, **k):
            pass

        def get(self, *a, **k):
            def deco(f):
                return f

            return deco

        def post(self, *a, **k):
            def deco(f):
                return f

            return deco

    class _HTTPException(Exception):
        pass

    fastapi.FastAPI = _App
    fastapi.HTTPException = _HTTPException
    sys.modules["fastapi"] = fastapi
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = object
    sys.modules["fastapi.middleware.cors"] = cors
    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = object
    sys.modules["fastapi.responses"] = responses
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    sys.modules["pydantic"] = pydantic
    trusted = types.ModuleType("starlette.middleware.trustedhost")
    trusted.TrustedHostMiddleware = object
    sys.modules["starlette.middleware.trustedhost"] = trusted

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "transformers_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tserver():
    return _load_transformers_server()


def test_adapter_spec_with_subfolder(tcommon):
    parsed = tcommon.parse_adapter_spec(
        "dft=Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended@best_model"
    )
    assert parsed == {
        "name": "dft",
        "source": "Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended",
        "subfolder": "best_model",
    }


def test_adapter_spec_root_repo_gets_leaf_name(tcommon):
    parsed = tcommon.parse_adapter_spec(
        "Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended"
    )
    assert parsed["source"] == "Marcus2112/dft-lfm-1.2b-instruct-lora-r64-extended"
    assert parsed["subfolder"] == ""
    assert parsed["name"]  # auto-derived, non-empty, shell-safe
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parsed["name"])


def test_adapter_spec_colon_subfolder_and_windows_path(tcommon):
    parsed = tcommon.parse_adapter_spec("myadapter=Org/Repo:best_model")
    assert (parsed["name"], parsed["source"], parsed["subfolder"]) == (
        "myadapter",
        "Org/Repo",
        "best_model",
    )
    # A Windows drive colon is never treated as a subfolder separator.
    win = tcommon.parse_adapter_spec(r"win=C:\models\lora")
    assert win == {"name": "win", "source": r"C:\models\lora", "subfolder": ""}


@pytest.mark.parametrize(
    "bad",
    ["", "=Org/Repo", "bad name=Org/Repo", "n=", "n=Repo@../evil", "n=Repo@/abs"],
)
def test_adapter_spec_rejects_malformed(tcommon, bad):
    with pytest.raises(ValueError):
        tcommon.parse_adapter_spec(bad)


def test_tokenizer_prefers_explicit_then_adapter_then_base(tcommon):
    assert tcommon.resolve_tokenizer_source("tok-override", ["/a"], "Base/Model") == "tok-override"
    assert tcommon.resolve_tokenizer_source("", [], "Base/Model") == "Base/Model"


def test_tokenizer_prefers_adapter_with_tokenizer_files(tcommon, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "tokenizer.json").write_text("{}")
    assert tcommon.resolve_tokenizer_source("", [str(adapter)], "Base/Model") == str(adapter)


def test_server_reexports_shared_helpers(tserver):
    """The server must use the shared module, not a private copy."""
    for name in (
        "parse_adapter_spec",
        "resolve_adapter_dir",
        "resolve_tokenizer_source",
        "adapter_matches_base",
    ):
        assert getattr(tserver, name).__module__ == "peft_lora_common"


def test_normalize_model_id(tcommon):
    assert tcommon.normalize_model_id("Marcus2112/Some-Model") == "some-model"
    assert tcommon.normalize_model_id("/home/u/.cache/huggingface/hub/models--Org--Some-Model/snapshots/abc123") == "some-model"
    assert tcommon.normalize_model_id("/models/Some-Model") == "some-model"
    assert tcommon.normalize_model_id("") == ""


def test_adapter_matches_base(tcommon, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "LiquidAI/LFM2.5-1.2B-Instruct"}')
    assert tcommon.adapter_matches_base(str(adapter), "LiquidAI/LFM2.5-1.2B-Instruct")
    # Local snapshot paths of the same checkpoint agree.
    assert tcommon.adapter_matches_base(
        str(adapter), "/home/u/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/abc"
    )
    assert not tcommon.adapter_matches_base(str(adapter), "Other/Model")
    # An adapter recording no base never matches (caller decides warn vs fail).
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "adapter_config.json").write_text("{}")
    assert not tcommon.adapter_matches_base(str(empty), "LiquidAI/LFM2.5-1.2B-Instruct")


def test_read_adapter_config_errors(tcommon, tmp_path):
    with pytest.raises(FileNotFoundError):
        tcommon.read_adapter_config(str(tmp_path / "missing"))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "adapter_config.json").write_text("not json")
    with pytest.raises(ValueError):
        tcommon.read_adapter_config(str(bad))


def test_select_adapter_switches_and_falls_back(tserver):
    activated = []

    class _FakeModel:
        def set_adapter(self, name):
            activated.append(name)

    tserver._model = _FakeModel()
    tserver._adapters = [
        {"name": "dft", "source": "Org/A", "subfolder": "best_model", "dir": "/a"},
        {"name": "other", "source": "Org/B", "subfolder": "", "dir": "/b"},
    ]
    tserver._base_id = "Base/Model"
    tserver._active_adapter = "dft"

    class _Args:
        model = "Base/Model"

    tserver._args = _Args()
    try:
        assert tserver._select_adapter("other") == "other"
        assert activated == ["other"]
        assert tserver._select_adapter("Base/Model") == "Base/Model"
        # Unknown ids fall back to the active adapter, never 404.
        assert tserver._select_adapter("stale-entry") == "other"
        assert tserver._select_adapter("") == "other"
    finally:
        tserver._model = None
        tserver._adapters = []
        tserver._base_id = ""
        tserver._active_adapter = ""
        tserver._args = None


def test_serve_cmd_allows_transformers_server():
    """The serve-command validator must accept the generated backend command.

    _validate_serve_cmd lives in routes.cookbook_helpers (imports fastapi),
    so assert on the allowlist contract via source instead: the validator only
    permits python/python3-led commands, which is exactly what the
    Transformers backend emits.
    """
    helpers = (REPO / "routes" / "cookbook_helpers.py").read_text(encoding="utf-8")
    assert '"python"' in helpers and '"python3"' in helpers
    assert "transformers_server" in (REPO / "static" / "js" / "cookbook.js").read_text(
        encoding="utf-8"
    )


def test_backend_diagnoses_missing_peft_stack():
    helpers = (REPO / "routes" / "cookbook_helpers.py").read_text(encoding="utf-8")
    assert "transformers_lora" in helpers
    assert "peft" in helpers


def test_lora_name_extraction_contract():
    """_served_lora_names must expose adapter ids for endpoint pinning."""
    routes = (REPO / "routes" / "cookbook_routes.py").read_text(encoding="utf-8")
    assert "def _served_lora_names" in routes
    assert "--adapter" in routes and "--lora-modules" in routes
    # Pinned adapter ids join the endpoint's pinned models.
    assert "_served_lora_names(req.cmd)" in routes


def test_transformers_recipe_available_under_node():
    if not _HAS_NODE:
        pytest.skip("node binary not on PATH")
    script = (
        "import { pickRecipe, recipeCommands } from './static/js/cookbook-deps-recipes.js';\n"
        "const recipe = pickRecipe('transformers_lora', 'LiquidAI/LFM2.5-1.2B-Instruct');\n"
        "console.log(JSON.stringify({ label: recipe && recipe.label, cmds: recipeCommands(recipe, 'pip') }));\n"
    )
    res = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO,
        capture_output=True,
        timeout=20,
        text=True,
    )
    assert res.returncode == 0, f"node failed:\n{res.stderr}"
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["label"]
    joined = " ".join(payload["cmds"])
    assert "peft" in joined and "transformers" in joined


def test_stop_sequences_parsing(tserver):
    from types import SimpleNamespace

    assert tserver._stop_sequences(SimpleNamespace(stop=None)) == []
    assert tserver._stop_sequences(SimpleNamespace(stop="###")) == ["###"]
    assert tserver._stop_sequences(SimpleNamespace(stop=["a", "", "  ", "b"])) == ["a", "b"]
    assert tserver._stop_sequences(SimpleNamespace(stop=123)) == []
    many = tserver._stop_sequences(SimpleNamespace(stop=[f"s{i}" for i in range(20)]))
    assert len(many) == 8  # capped


def test_cookbook_js_builds_transformers_command():
    source = (REPO / "static" / "js" / "cookbook.js").read_text(encoding="utf-8")
    assert "backend === 'transformers'" in source
    assert "scripts/transformers_server.py --model" in source
    assert "--adapter" in source
    assert "--tokenizer" in source
    assert "--served-model-name" in source
    assert "--trust-remote-code" in source


def test_serve_panel_has_transformers_lora_ui():
    source = (REPO / "static" / "js" / "cookbookServe.js").read_text(encoding="utf-8")
    assert "['transformers','Transformers']" in source
    assert "hf_lora_adapters" in source
    assert "hf_tokenizer" in source
    assert "hf_served_model_name" in source
    assert "hf_max_tokens" in source
    assert "transformers_lora" in source
    assert "transformers_server" in source
    assert "hwfit-backend-transformers" in source


def test_shell_packages_probe_transformers_lora():
    source = (REPO / "routes" / "shell_routes.py").read_text(encoding="utf-8")
    assert '"transformers_lora"' in source
    assert "'transformers_lora':['transformers','peft']" in source


def test_transformers_provider_spec_exists():
    spec = (REPO / "specs" / "model-providers" / "transformers.md").read_text(encoding="utf-8")
    assert "endpoint scoped" in spec
    assert "--adapter" in spec
    assert "merge_peft_lora" in spec
    assert "verify_lora_active" in spec


def test_transformers_server_keeps_rebinding_defenses():
    """Pin the DNS-rebinding + wildcard-CORS defenses (mirrors
    test_diffusion_server_security.py, but via AST so no serving stack is
    needed to run it)."""
    import ast

    tree = ast.parse((REPO / "scripts" / "transformers_server.py").read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top.add(node.name)
        elif isinstance(node, ast.Assign):
            top.update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top.add(node.target.id)
    for name in (
        "_DEFAULT_ALLOWED_HOSTS",
        "_DEFAULT_CORS_ORIGINS",
        "_compute_allowed_hosts",
        "_compute_cors_origins",
        "_configure_security_middleware",
    ):
        assert name in top, f"transformers_server.py is missing {name}"
    source = (REPO / "scripts" / "transformers_server.py").read_text(encoding="utf-8")
    assert 'allow_origins=["*"]' not in source


def test_merge_script_aborts_on_base_mismatch(tmp_path):
    """The merge guard must fire before any torch import/weights load."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "Org/Other-Model"}')
    res = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "merge_peft_lora.py"),
            "--base",
            "Org/Base-Model",
            "--adapter",
            str(adapter),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "trained for" in combined and "allow-base-mismatch" in combined


def test_merge_script_rejects_bad_adapter_spec(tmp_path):
    res = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "merge_peft_lora.py"),
            "--base",
            "Org/Base-Model",
            "--adapter",
            "n=",
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode != 0
    assert "invalid --adapter" in (res.stdout + res.stderr).lower()


def test_verify_script_verdicts_against_stub_endpoint():
    """ACTIVE (exit 0) when any prompt differs, INACTIVE (exit 2) when all match."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    texts = {"base": "PARIS", "dft": "PARIS"}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            payload = {"choices": [{"message": {"content": texts.get(body.get("model"), "?")}}]}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"

        def _run():
            return subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "verify_lora_active.py"),
                    "--base-url",
                    url,
                    "--base",
                    "base",
                    "--adapter",
                    "dft",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

        res = _run()
        assert res.returncode == 2, res.stdout + res.stderr
        assert "INACTIVE" in res.stdout
        texts["dft"] = "Paris is the capital."
        res = _run()
        assert res.returncode == 0, res.stdout + res.stderr
        assert "ACTIVE" in res.stdout
    finally:
        server.shutdown()
