"""
Enigma AI Engine - Local API Server
====================================

FastAPI server that exposes the engine over HTTP for local use.
Connect from any device on your local network (PC, phone, tablet).

Usage:
    python run.py --serve              # Start on port 8080
    python run.py --serve --port 9000  # Custom port

Endpoints:
    GET  /api/health        Health check
    GET  /api/system        System/hardware info
    GET  /api/models        List available models
    GET  /api/models/status Current model status
    POST /api/models/load   Load a model
    POST /api/models/unload Unload current model
    POST /api/chat          Send a chat message
    POST /api/chat/stream   Stream chat via SSE
    POST /api/batch         Batch inference (multiple prompts)
    GET  /api/profiles      List AI profiles
    GET  /api/profiles/{id} Get profile details
    POST /api/profiles/{id}/activate  Activate a profile
    GET  /api/config        Get generation config
    POST /api/config        Update generation config
    GET  /api/history       Get chat history
    DELETE /api/history     Clear chat history
    GET  /api/training/status  Training progress
    POST /api/train         Start training
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from pydantic import BaseModel, Field, field_validator

from enigma_engine import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
API_DIR = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# App State (module-level singleton — one engine per process)
# ---------------------------------------------------------------------------

class AppState:
    """Holds the loaded engine, config overrides, and chat history.

    All mutable state is guarded by ``_lock``.  Read-only accessors
    return *snapshots* (copy-on-write) so callers never hold a
    reference to the live internal list/dict.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.engine: Any = None
        self.model_path: str | None = None
        self._model_info: dict[str, Any] = {}
        self._history: list[dict[str, str]] = []
        self.active_profile: str | None = None
        self.config_overrides: dict[str, Any] = {}
        self.start_time: float = time.time()

    # -- Copy-on-write snapshots (Suggestion #8D) ----------------------

    @property
    def history(self) -> list[dict[str, str]]:
        """Direct access for internal mutations (under _lock)."""
        return self._history

    def history_snapshot(self) -> list[dict[str, str]]:
        """Return a shallow copy of chat history (safe for readers)."""
        with self._lock:
            return list(self._history)

    @property
    def model_info(self) -> dict[str, Any]:
        """Direct access for internal mutations."""
        return self._model_info

    @model_info.setter
    def model_info(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._model_info = value

    def model_info_snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of model info (safe for readers)."""
        with self._lock:
            return dict(self._model_info)

    # -- Engine management --------------------------------------------------

    def load_model(self, model_path: str) -> dict[str, Any]:
        """Load a model into the engine."""
        from enigma_engine.core import EnigmaEngine

        # Unload previous
        self.unload_model()

        engine = EnigmaEngine(model_path=model_path)

        # Gather info
        param_count = 0
        if hasattr(engine, "model") and engine.model is not None:
            param_count = sum(p.numel() for p in engine.model.parameters())

        device = "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

        info = {
            "path": model_path,
            "parameters": param_count,
            "device": device,
            "loaded": True,
        }

        with self._lock:
            self.engine = engine
            self.model_path = model_path
            self._model_info = info
        return dict(info)

    def unload_model(self):
        """Unload the current model and free memory."""
        with self._lock:
            if self.engine is not None:
                del self.engine
                self.engine = None
                self.model_path = None
                self._model_info = {}
                self._history.clear()

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
        with self._lock:
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": response})

        return response


state = AppState()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enigma AI Engine",
    version=__version__,
    docs_url="/api/docs",
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# CORS — opt-in only via --cors-origins  (Suggestion #6)
# No middleware = no attack surface.  Call enable_cors() at runtime to add it.
# ---------------------------------------------------------------------------

def enable_cors(origins: list[str]) -> None:
    """Add CORS middleware with the given origin list.

    Only call this *before* uvicorn starts (inside run_server).
    """
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", origins)




# ---------------------------------------------------------------------------
# Input Limits  (Suggestion #7)
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 32_768   # 32K chars (~8K tokens)
MAX_BATCH_PROMPTS = 50
MAX_PATH_LENGTH = 256

# ---------------------------------------------------------------------------
# Concurrency Lock  (Suggestion #7)
# ---------------------------------------------------------------------------
_inference_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    temperature: float | None = None
    max_tokens: int | None = None

class ChatResponse(BaseModel):
    message: str
    tokens_used: int = 0

class ModelLoadRequest(BaseModel):
    path: str = Field(..., max_length=MAX_PATH_LENGTH)

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
# Health & System
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": __version__,
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
    return {"loaded": True, "model": state.model_info_snapshot()}


@app.post("/api/models/load")
async def load_model(req: ModelLoadRequest):
    """Load a model by path."""
    path = Path(req.path).resolve()
    # Prevent path traversal — model must be inside MODELS_DIR
    models_root = MODELS_DIR.resolve()
    if not (path == models_root or str(path).startswith(str(models_root) + os.sep)):
        raise HTTPException(403, "Path must be inside the models directory")
    if not path.exists():
        raise HTTPException(404, f"Model not found: {req.path}")
    try:
        info = state.load_model(str(path))
        return {"status": "ok", "model": info}
    except Exception as exc:
        logger.exception("Failed to load model")
        raise HTTPException(500, f"Failed to load model: {exc}") from exc


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
    if not _inference_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"error": "Engine busy — another request is in progress."},
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
    finally:
        _inference_lock.release()


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat via Server-Sent Events.

    Returns an SSE stream where each event contains a token.
    Events:
        event: start   — generation begins
        event: token   — a single token (data.content)
        event: end     — generation complete (data.content = full text)
        event: error   — an error occurred
    """
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    if not _inference_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"error": "Engine busy — another request is in progress."},
        )

    # Build kwargs from config overrides + per-request overrides
    kwargs: dict[str, Any] = {}
    if state.config_overrides:
        kwargs.update(state.config_overrides)
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens

    def _sse_generator():
        """Yield SSE-formatted events from engine.stream_chat."""
        from enigma_engine.core.streaming import StreamChunk, StreamEvent

        try:
            # Start event
            start_chunk = StreamChunk(
                content="", event=StreamEvent.START,
                metadata={"message": req.message})
            yield start_chunk.to_sse()

            full_response = []
            try:
                gen = state.engine.stream_chat(
                    req.message, history=state.history_snapshot(), **kwargs)
                for token in gen:
                    full_response.append(token)
                    token_chunk = StreamChunk(
                        content=token, event=StreamEvent.TOKEN)
                    yield token_chunk.to_sse()

                # End event with full response
                combined = "".join(full_response)
                end_chunk = StreamChunk(
                    content=combined, event=StreamEvent.END)
                yield end_chunk.to_sse()

                # Track history
                with state._lock:
                    state._history.append(
                        {"role": "user", "content": req.message})
                    state._history.append(
                        {"role": "assistant", "content": combined})

            except Exception as exc:
                logger.exception("Stream chat error")
                err_msg = str(exc)
                err_chunk = StreamChunk(
                    content=err_msg, event=StreamEvent.ERROR)
                yield err_chunk.to_sse()
        finally:
            _inference_lock.release()

    return FastAPIStreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Batch Inference
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    prompts: list[str] = Field(..., max_length=MAX_BATCH_PROMPTS)
    max_tokens: int = 100
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    @field_validator("prompts")
    @classmethod
    def validate_prompt_lengths(cls, v: list[str]) -> list[str]:
        for i, p in enumerate(v):
            if len(p) > MAX_MESSAGE_LENGTH:
                raise ValueError(
                    f"prompts[{i}] exceeds {MAX_MESSAGE_LENGTH} characters")
        return v


@app.post("/api/batch")
async def batch_inference(req: BatchRequest):
    """Process multiple prompts in a single request.

    Uses batch_generate if available on the engine, otherwise
    falls back to sequential generation per prompt.
    """
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    if not req.prompts:
        return JSONResponse(
            status_code=400,
            content={"error": "Prompts list cannot be empty."},
        )
    if not _inference_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"error": "Engine busy \u2014 another request is in progress."},
        )

    kwargs: dict[str, Any] = {}
    if state.config_overrides:
        kwargs.update(state.config_overrides)
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p

    try:
        # Prefer batch_generate for efficiency
        if hasattr(state.engine, "batch_generate"):
            responses = state.engine.batch_generate(
                req.prompts, max_gen=req.max_tokens, **kwargs)
        else:
            # Sequential fallback
            responses = []
            for prompt in req.prompts:
                try:
                    resp = state.engine.chat(prompt, **kwargs)
                    responses.append(resp)
                except TypeError:
                    resp = state.engine.chat(prompt)
                    responses.append(resp)

        return {
            "responses": responses,
            "count": len(responses),
        }
    except Exception as exc:
        logger.exception("Batch inference error")
        return JSONResponse(
            status_code=500,
            content={"error": f"Batch generation failed: {exc}"},
        )
    finally:
        _inference_lock.release()


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def get_history():
    """Get chat history."""
    return {"history": state.history_snapshot()}


@app.delete("/api/history")
async def clear_history():
    """Clear chat history."""
    with state._lock:
        state._history.clear()
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


def _safe_profile_path(profile_id: str) -> Path:
    """Resolve profile path and reject traversal attempts."""
    # Strip path separators to prevent directory traversal
    safe_id = Path(profile_id).name
    path = (PROFILES_DIR / f"{safe_id}.json").resolve()
    if not str(path).startswith(str(PROFILES_DIR.resolve())):
        raise HTTPException(403, "Invalid profile ID")
    return path


@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get full profile details."""
    path = _safe_profile_path(profile_id)
    if not path.exists():
        raise HTTPException(404, f"Profile not found: {profile_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


@app.post("/api/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    """Activate an AI profile (applies its generation settings)."""
    path = _safe_profile_path(profile_id)
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
# Training
# ---------------------------------------------------------------------------

# Module-level training state for the API
_training_lock = threading.Lock()
_training_state: dict[str, Any] = {
    "active": False,
    "progress": 0,
    "message": "",
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
}


class TrainRequest(BaseModel):
    data_file: str = Field(..., max_length=MAX_PATH_LENGTH)
    epochs: int = Field(default=5, ge=1, le=1000)
    learning_rate: float = Field(default=0.00005, gt=0.0, le=1.0)
    batch_size: int = Field(default=4, ge=1, le=256)


@app.get("/api/training/status")
async def training_status():
    """Get the current training status."""
    with _training_lock:
        return dict(_training_state)


@app.post("/api/train")
async def start_training(req: TrainRequest):
    """Start a training run in the background.

    Requires a model to be loaded. Training data must exist in
    the data/ directory.
    """
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    with _training_lock:
        if _training_state["active"]:
            return JSONResponse(
                status_code=409,
                content={"error": "Training already in progress."},
            )

    # Resolve data file path safely
    data_dir = PROJECT_ROOT / "data"
    data_path = (data_dir / req.data_file).resolve()
    if not str(data_path).startswith(str(data_dir.resolve())):
        raise HTTPException(403, "Data file must be inside the data directory")
    if not data_path.exists():
        raise HTTPException(404, f"Data file not found: {req.data_file}")

    import threading

    def _run_training():
        """Background training thread."""
        try:
            with _training_lock:
                _training_state.update({
                    "active": True,
                    "progress": 0,
                    "message": "Initializing...",
                    "epoch": 0,
                    "total_epochs": req.epochs,
                    "loss": 0.0,
                })

            from enigma_engine.core.training import Trainer, TrainingConfig

            config = TrainingConfig(
                epochs=req.epochs,
                learning_rate=req.learning_rate,
                batch_size=req.batch_size,
            )

            model = state.engine.model
            tokenizer = state.engine.tokenizer
            trainer = Trainer(model, tokenizer, config)

            # Wire progress callback
            def on_progress(pct: int, msg: str):
                with _training_lock:
                    _training_state["progress"] = pct
                    _training_state["message"] = msg

            def on_epoch_complete(epoch: int, loss: float):
                with _training_lock:
                    _training_state["epoch"] = epoch + 1
                    _training_state["loss"] = loss

            trainer.on_progress = on_progress
            trainer.on_epoch_complete = on_epoch_complete

            data = data_path.read_text(encoding="utf-8")
            trainer.train(data)

            with _training_lock:
                _training_state.update({
                    "active": False,
                    "progress": 100,
                    "message": "Training complete",
                })
        except Exception as exc:
            logger.exception("Training error")
            with _training_lock:
                _training_state.update({
                    "active": False,
                    "progress": 0,
                    "message": f"Training failed: {exc}",
                })

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {
        "status": "started",
        "data_file": req.data_file,
        "epochs": req.epochs,
    }


# ---------------------------------------------------------------------------
# Server runner (called from run.py --serve)
# ---------------------------------------------------------------------------

def run_server(host: str = "127.0.0.1", port: int = 8080, model_path: str | None = None,
               api_key: str | None = None, cors_origins: list[str] | None = None):
    """Start the local API server.

    Args:
        host: Bind address. Defaults to 127.0.0.1 (localhost only).
              Use '0.0.0.0' to expose to the local network.
        port: Port number.
        model_path: Optional model to pre-load.
        api_key: Optional API key. When set, all /api/* requests must
                 include an ``Authorization: Bearer <key>`` header.
        cors_origins: Optional list of allowed CORS origins.
                      CORS middleware is only added when this is provided.
    """
    import uvicorn

    # Install CORS middleware only when explicitly requested
    if cors_origins:
        enable_cors(cors_origins)
    else:
        logger.info("CORS disabled (pass --cors-origins to enable)")

    # Install API-key middleware when a key is provided
    if api_key:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse as _JSONResponse

        _key = api_key  # capture in closure

        class _APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Skip key check for CORS preflight OPTIONS requests
                if request.method == "OPTIONS":
                    return await call_next(request)
                if request.url.path.startswith("/api/"):
                    auth = request.headers.get("authorization", "")
                    if auth != f"Bearer {_key}":
                        return _JSONResponse(
                            {"error": "Invalid or missing API key"},
                            status_code=401,
                        )
                return await call_next(request)

        app.add_middleware(_APIKeyMiddleware)

    # Pre-load a model if specified
    if model_path:
        logger.info("Pre-loading model: %s", model_path)
        try:
            state.load_model(model_path)
            params = f"{state.model_info_snapshot().get('parameters', 0):,}"
            logger.info("Model loaded: %s params", params)
        except Exception as exc:
            logger.warning("Could not pre-load model: %s", exc)
            logger.info(
                "Server starting without a model. "
                "Load one via POST /api/models/load")

    logger.info("Enigma Engine API Server")
    logger.info("http://localhost:%s", port)
    logger.info("API docs: http://localhost:%s/api/docs", port)
    logger.info("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port, log_level="info")
