"""
Enigma AI Engine - Web API Server
==================================

FastAPI server that exposes the engine over HTTP.
Serves both the REST API and the web frontend.

Usage:
    python run.py --serve              # Start on port 8080
    python run.py --serve --port 9000  # Custom port

Endpoints:
    GET  /                  Web frontend
    GET  /api/health        Health check
    GET  /api/system        System/hardware info
    GET  /api/models        List available models
    GET  /api/models/status Current model status
    POST /api/models/load   Load a model
    POST /api/models/unload Unload current model
    POST /api/chat          Send a chat message
    GET  /api/profiles      List AI profiles
    GET  /api/profiles/{id} Get profile details
    POST /api/profiles/{id}/activate  Activate a profile
    GET  /api/config        Get generation config
    POST /api/config        Update generation config
    GET  /api/history       Get chat history
    DELETE /api/history     Clear chat history
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
API_DIR = Path(__file__).parent
STATIC_DIR = API_DIR / "static"
TEMPLATES_DIR = API_DIR / "templates"
PROJECT_ROOT = API_DIR.parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# App State (module-level singleton — one engine per process)
# ---------------------------------------------------------------------------

class AppState:
    """Holds the loaded engine, config overrides, and chat history."""

    def __init__(self):
        self.engine: Any = None
        self.model_path: str | None = None
        self.model_info: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        self.active_profile: str | None = None
        self.config_overrides: dict[str, Any] = {}
        self.start_time: float = time.time()

    # -- Engine management --------------------------------------------------

    def load_model(self, model_path: str) -> dict[str, Any]:
        """Load a model into the engine."""
        from enigma_engine.core import EnigmaEngine

        # Unload previous
        self.unload_model()

        self.engine = EnigmaEngine(model_path=model_path)
        self.model_path = model_path

        # Gather info
        param_count = 0
        if hasattr(self.engine, "model") and self.engine.model is not None:
            param_count = sum(p.numel() for p in self.engine.model.parameters())

        device = "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

        self.model_info = {
            "path": model_path,
            "parameters": param_count,
            "device": device,
            "loaded": True,
        }
        return self.model_info

    def unload_model(self):
        """Unload the current model and free memory."""
        if self.engine is not None:
            del self.engine
            self.engine = None
            self.model_path = None
            self.model_info = {}

            # Free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def chat(self, message: str, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        """Send a message to the engine and get a response."""
        if self.engine is None:
            raise RuntimeError("No model loaded")

        # Build kwargs from config overrides + per-request overrides
        kwargs: dict[str, Any] = {}
        if self.config_overrides:
            kwargs.update(self.config_overrides)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Try passing kwargs to engine.chat; fall back to no-kwargs if unsupported
        try:
            response = self.engine.chat(message, **kwargs)
        except TypeError:
            response = self.engine.chat(message)

        # Track history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})

        return response


state = AppState()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enigma AI Engine",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# Serve static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    temperature: float | None = None
    max_tokens: int | None = None

class ChatResponse(BaseModel):
    message: str
    tokens_used: int = 0

class ModelLoadRequest(BaseModel):
    path: str

class ConfigUpdate(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None

    def validated(self) -> dict[str, Any]:
        """Return only non-None fields, clamped to valid ranges."""
        limits = {
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "top_k": (1, 200),
            "max_tokens": (16, 4096),
            "repetition_penalty": (1.0, 2.0),
        }
        result: dict[str, Any] = {}
        for key, (lo, hi) in limits.items():
            val = getattr(self, key)
            if val is not None:
                clamped = max(lo, min(hi, val))
                result[key] = type(lo)(clamped)  # preserve int/float type
        return result


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web UI."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Enigma Engine</h1><p>Frontend not found.</p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Health & System
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "1.1.0",
        "uptime": round(time.time() - state.start_time, 1),
        "model_loaded": state.engine is not None,
    }


@app.get("/api/system")
async def system_info():
    """Return hardware and system information."""
    import platform

    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": "cpu",
        "torch_available": False,
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
    }

    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["vram_gb"] = round(vram, 1)
    except ImportError:
        pass

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass

    return info


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """List available model files."""
    models = []
    if MODELS_DIR.exists():
        for ext in ("*.pth", "*.pt", "*.gguf", "*.bin", "*.safetensors"):
            for p in MODELS_DIR.glob(ext):
                size_mb = p.stat().st_size / (1024 * 1024)
                models.append({
                    "name": p.stem,
                    "filename": p.name,
                    "path": str(p),
                    "size_mb": round(size_mb, 1),
                    "format": p.suffix.lstrip("."),
                })
        # Also check subdirectories one level deep
        for subdir in MODELS_DIR.iterdir():
            if subdir.is_dir():
                for ext in ("*.pth", "*.pt", "*.gguf", "*.bin", "*.safetensors"):
                    for p in subdir.glob(ext):
                        size_mb = p.stat().st_size / (1024 * 1024)
                        models.append({
                            "name": f"{subdir.name}/{p.stem}",
                            "filename": p.name,
                            "path": str(p),
                            "size_mb": round(size_mb, 1),
                            "format": p.suffix.lstrip("."),
                        })

    return {"models": models}


@app.get("/api/models/status")
async def model_status():
    """Get the status of the currently loaded model."""
    if state.engine is None:
        return {"loaded": False, "model": None}
    return {"loaded": True, "model": state.model_info}


@app.post("/api/models/load")
async def load_model(req: ModelLoadRequest):
    """Load a model by path."""
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(404, f"Model not found: {req.path}")
    try:
        info = state.load_model(req.path)
        return {"status": "ok", "model": info}
    except Exception as exc:
        logger.exception("Failed to load model")
        raise HTTPException(500, f"Failed to load model: {exc}")


@app.post("/api/models/unload")
async def unload_model():
    """Unload the current model and free memory."""
    state.unload_model()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send a chat message and get a response."""
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    try:
        response = state.chat(
            req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return {"message": response}
    except Exception as exc:
        logger.exception("Chat error")
        return JSONResponse(
            status_code=500,
            content={"error": f"Generation failed: {exc}"},
        )


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def get_history():
    """Get chat history."""
    return {"history": state.history}


@app.delete("/api/history")
async def clear_history():
    """Clear chat history."""
    state.history.clear()
    if state.engine is not None and hasattr(state.engine, "clear_history"):
        state.engine.clear_history()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@app.get("/api/profiles")
async def list_profiles():
    """List available AI profiles."""
    profiles = []
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                profiles.append({
                    "id": p.stem,
                    "name": data.get("name", p.stem),
                    "description": data.get("description", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue
    return {"profiles": profiles, "active": state.active_profile}


@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get full profile details."""
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Profile not found: {profile_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


@app.post("/api/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    """Activate an AI profile (applies its generation settings)."""
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Profile not found: {profile_id}")

    data = json.loads(path.read_text(encoding="utf-8"))
    state.active_profile = profile_id

    # Apply generation settings from profile
    gen = data.get("generation", {})
    if gen:
        state.config_overrides.update(gen)

    return {"status": "ok", "active": profile_id, "settings": gen}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    """Get current generation config."""
    from enigma_engine.config import CONFIG

    # Merge engine defaults with overrides
    config = {
        "temperature": CONFIG.get("temperature", 0.8),
        "top_p": CONFIG.get("top_p", 0.9),
        "top_k": CONFIG.get("top_k", 50),
        "max_tokens": CONFIG.get("max_gen", 100),
        "repetition_penalty": CONFIG.get("repetition_penalty", 1.1),
    }
    config.update(state.config_overrides)
    return config


@app.post("/api/config")
async def update_config(req: ConfigUpdate):
    """Update generation config (values clamped to valid ranges)."""
    updates = req.validated()
    state.config_overrides.update(updates)
    return {"status": "ok", "config": {**await get_config()}}


# ---------------------------------------------------------------------------
# Server runner (called from run.py --serve)
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8080, model_path: str | None = None):
    """Start the web server."""
    import uvicorn

    # Pre-load a model if specified
    if model_path:
        print(f"  Pre-loading model: {model_path}")
        try:
            state.load_model(model_path)
            print(f"  Model loaded: {state.model_info.get('parameters', 0):,} params")
        except Exception as exc:
            print(f"  [WARNING] Could not pre-load model: {exc}")
            print(f"  Server starting without a model. Load one via the web UI.")

    print(f"\n  Enigma Engine Web UI")
    print(f"  http://localhost:{port}")
    print(f"  API docs: http://localhost:{port}/api/docs")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
