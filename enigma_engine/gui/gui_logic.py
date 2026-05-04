"""
Enigma Engine - GUI Logic Methods
===================================

Mixin providing business logic for EnigmaGUI.
Handles config, model loading, route assignments,
config paths, display names, and feature toggles.

Chat logic:            see gui_logic_chat.py (LogicChatMixin)
Media & voice:         see gui_logic_media.py (LogicMediaMixin)
Training / model mgmt: see gui_forge.py (ForgeMixin)
Mod process mgmt:      see gui_mods.py (ModMixin)
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from enigma_engine.gui.widgets import (
    C_GREEN, C_ORANGE,
    C_RED, C_TEXT_DIM,
)
from enigma_engine.gui.scanners import (
    CONFIG_LIMITS, DATA_DIR, INFO_DIR,
    ROUTE_KEYS, PATH_SETTINGS,
    clamp_config,
    load_path_settings, save_path_settings,
    load_route_assignments, save_route_assignments,
)
# Re-export so existing imports keep working
from enigma_engine.gui.gui_forge import ForgeMixin  # noqa: F401
from enigma_engine.gui.gui_mods import ModMixin  # noqa: F401
from enigma_engine.gui.gui_logic_chat import LogicChatMixin
from enigma_engine.gui.gui_logic_media import LogicMediaMixin


# Bytes-per-parameter for common GGUF quantization types.
# Used by _estimate_gguf_params to convert file size → param count.
_GGUF_BYTES_PER_PARAM: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.1,
    "q8_1": 1.2,
    "q8_k": 1.1,
    "q6_k": 0.82,
    "q5_0": 0.69,
    "q5_1": 0.74,
    "q5_k": 0.71,
    "q5_k_m": 0.71,
    "q5_k_s": 0.69,
    "q4_0": 0.55,
    "q4_1": 0.61,
    "q4_k": 0.57,
    "q4_k_m": 0.57,
    "q4_k_s": 0.55,
    "q3_k": 0.44,
    "q3_k_m": 0.44,
    "q3_k_s": 0.41,
    "q3_k_l": 0.46,
    "q2_k": 0.33,
    "iq4_xs": 0.52,
    "iq3_xxs": 0.38,
}


def _detect_gguf_quant_type(path: str) -> str | None:
    """Try to detect GGUF quantization type from the filename.

    Returns the lowercase quant tag (e.g. ``'q4_k_m'``) or *None*.
    """
    import re
    stem = Path(path).stem.lower()
    # Match patterns like q4_k_m, q8_0, f16, iq3_xxs, etc.
    m = re.search(r'(?:^|[^a-z])((?:iq|q|f)\d+(?:_[a-z0-9]+)*)', stem)
    return m.group(1) if m else None


def _estimate_gguf_params(engine: Any, path: str) -> int:
    """Estimate parameter count for a GGUF model.

    Uses a 3-tier approach:
    1. Metadata formula: dim, n_layers, vocab_size → approximate params
    2. File-size heuristic: file_bytes / bytes_per_param (quant-aware)
    3. Returns 0 if nothing works
    """
    # Tier 1: metadata-based estimation
    model = getattr(engine, 'model', None)
    if model is not None:
        cfg = getattr(model, 'config', None)
        if cfg is not None:
            dim = getattr(cfg, 'dim', 0) or 0
            n_layers = getattr(cfg, 'n_layers', 0) or 0
            vocab = getattr(cfg, 'vocab_size', 0) or 0
            if dim > 0 and n_layers > 0:
                # Standard transformer param estimation
                # 12 * dim^2 per layer (attn + FFN) + vocab embedding
                per_layer = 12 * dim * dim
                embed = vocab * dim if vocab > 0 else 0
                return per_layer * n_layers + embed

    # Tier 2: file-size heuristic with quant-type-aware bytes_per_param
    try:
        file_size = Path(path).stat().st_size
        if file_size > 0:
            quant = _detect_gguf_quant_type(path)
            bpp = _GGUF_BYTES_PER_PARAM.get(quant, 0.55) if quant else 0.55
            return int(file_size / bpp)
    except OSError:
        pass

    return 0


def _parse_lora_stack_inputs(
    items: list[tuple[str, str]],
    default_weight: float = 1.0,
) -> tuple[list[tuple[str, float]], list[str]]:
    """Parse user-typed weight strings for a multi-LoRA stack.

    Pass 156u-B (LoRA-1b stacking UI): the GUI weight entries are
    free-text strings. This helper converts them to floats with the
    same rules the engine layer applies, but emitting per-row
    error messages that name the offending adapter (friendlier than
    the generic ``ValueError`` from
    :meth:`EnigmaEngine.apply_adapter_stack`).

    Rules:
        - Empty / whitespace-only → ``default_weight`` (default 1.0).
          User intent when ticking a row but not typing.
        - Non-numeric → error naming the adapter's basename.
        - NaN / Inf → error naming the adapter's basename.
        - Negative values are LEGITIMATE (subtract this adapter's
          contribution from the merged stack) and MUST NOT be
          rejected.

    Errors are collected — the parser walks every row before
    returning so the user sees ALL typos at once instead of N
    round-trips for N typos.

    Args:
        items: List of ``(adapter_path, raw_weight_str)`` tuples
            from selected adapter rows.
        default_weight: Weight used when the raw string is empty
            or whitespace-only. Default 1.0.

    Returns:
        ``(parsed_pairs, errors)``. On any error ``parsed_pairs``
        is the empty list — the caller MUST treat this as an
        all-or-nothing operation (a partial stack reaching the
        engine would silently drop the bad rows).
    """
    import math
    parsed: list[tuple[str, float]] = []
    errors: list[str] = []
    for raw_path, raw_weight in items:
        stripped = (raw_weight or "").strip()
        name = Path(raw_path).name
        if not stripped:
            parsed.append((raw_path, float(default_weight)))
            continue
        try:
            w = float(stripped)
        except (TypeError, ValueError):
            errors.append(
                f"weight for '{name}' is not a number: "
                f"{raw_weight!r}")
            continue
        if not math.isfinite(w):
            errors.append(
                f"weight for '{name}' is not finite "
                f"(NaN/Inf not allowed): {raw_weight!r}")
            continue
        parsed.append((raw_path, w))
    if errors:
        return [], errors
    return parsed, []


class LogicMixin(LogicChatMixin, LogicMediaMixin):
    """Mixin providing logic methods for EnigmaGUI.

    Combines LogicChatMixin (chat, sessions, history) and
    LogicMediaMixin (media rendering, voice I/O) with
    config, model loading, route management, and toggles.
    """

    # ================================================================
    # Logic - GUI Context for AI awareness
    # ================================================================

    def _build_gui_context(self) -> str:
        """Build a context string describing current GUI state.

        This is injected into the system prompt so the AI knows
        what models are loaded, what routes are assigned, which
        mods are running, etc.
        """
        lines: list[str] = []
        lines.append("[SYSTEM CONTEXT — Current Engine State]")

        # Active chat model
        if self.engine and self.model_path:
            name = Path(self.model_path).stem
            lines.append(f"You are running as: {self._active_ai_name()}")
            lines.append(f"Chat model: {name}")
        else:
            lines.append("No chat model loaded.")

        # Route assignments
        if self.route_assignments:
            lines.append("")
            lines.append("Assigned Routes:")
            for key, path in self.route_assignments.items():
                model_name = Path(path).stem if path else "None"
                lines.append(f"  {key.upper()}: {model_name}")

        # Available models
        if self.models_data:
            lines.append("")
            lines.append("Available Models:")
            for m in self.models_data:
                lines.append(
                    f"  {m['name']} ({m['format'].upper()}, "
                    f"{m['size_mb']} MB)")

        # Available tools (all mods, running and stopped)
        try:
            from enigma_engine.core.mod_tools import format_tools_for_prompt
            tools_ctx = format_tools_for_prompt(self.mods_data)
            if tools_ctx:
                lines.append("")
                lines.append(tools_ctx)
        except (ImportError, AttributeError) as exc:
            logger.debug("Tool prompt build failed: %s", exc)

        # Config overrides
        if self.config_overrides:
            lines.append("")
            lines.append("Config Overrides:")
            for k, v in self.config_overrides.items():
                lines.append(f"  {k}: {v}")

        # Web access
        if self.web_access:
            lines.append("")
            lines.append(
                "Web Access: ENABLED — You may use [CMD] search.web <query> "
                "to search the web when you need current information. "
                "Include the results in your response.")
        else:
            lines.append("")
            lines.append("Web Access: DISABLED")

        # Memory commands — tell the AI it can save notes
        lines.append("")
        lines.append(
            "Persistent Memory: You have long-term memory across "
            "conversations. Use [CMD]memory.search <query>[/CMD] "
            "to search for relevant facts when needed "
            "(e.g., user's name, preferences, past discussions). "
            "Use [CMD]memory.remember <fact>[/CMD] to save "
            "important insights about the user. "
            "Memory retrieval is ACTIVE not PASSIVE — search only "
            "when contextually relevant, like recalling someone's "
            "name when they greet you. "
            "Actively observe patterns, preferences, habits, and "
            "coding style — save them without being asked. "
            "Suggest alternatives or better approaches when you "
            "notice a pattern that could be improved. "
            "Always ask permission before changing an established "
            "pattern or workflow the user already follows. "
            "Do NOT announce when you save a memory.")

        lines.append("")
        lines.append(
            "File Access: You can read and write files using "
            "[CMD]file.read <path>[/CMD] to view file contents, "
            "[CMD]file.write <path> <content>[/CMD] to create or "
            "overwrite a file, and "
            "[CMD]file.append <path> <content>[/CMD] to add to "
            "an existing file. "
            "Use [CMD]file.list [path][/CMD] to browse directories. "
            "You can write documentation, notes, and guides in the "
            "information/ and data/notes/ folders. "
            "Always confirm with the user before modifying existing "
            "files, but you can freely create new files when asked.")

        lines.append("")
        lines.append(
            "Code Execution: You can run Python code in a sandboxed "
            "subprocess using [CMD]code.run <python_code>[/CMD]. "
            "Output is captured and shown in chat. "
            "Write access is restricted to outputs/ only. "
            "Use this for math, data processing, or code demos.")

        lines.append("")
        lines.append(
            "Image Generation: You can generate images from text "
            "prompts using [CMD]imagegen.generate <prompt>[/CMD]. "
            "Optional flags: --width N, --height N, --steps N, "
            "--seed N, --negative <text>. "
            "The image will be rendered inline in chat. "
            "Use [CMD]imagegen.status[/CMD] to check available "
            "backends (SD WebUI, ComfyUI, diffusers).")

        lines.append("")
        lines.append(
            "Image Search: You can find images from the web using "
            "[CMD]search.images <query>[/CMD]. This returns image "
            "URLs that you can embed inline using markdown syntax: "
            "![description](url). The images will render directly "
            "in the chat. Use this to illustrate explanations with "
            "diagrams, photos, or visual examples when relevant.")

        # Terminal agent capability
        lines.append("")
        lines.append(
            "Terminal Access: You can execute system commands using "
            "[CMD]system.exec <shell_command>[/CMD]. The command "
            "runs in the system shell and output is returned. "
            "Use this for file operations, package management, "
            "or checking system state. "
            "Requires AI ACCESS to be enabled by the user.")

        # Mod management
        lines.append("")
        lines.append(
            "Mod Management: Use [CMD]mod.start <mod_id>[/CMD] to "
            "start a tool mod, [CMD]mod.stop <mod_id>[/CMD] to stop "
            "one, or [CMD]mod.list[/CMD] to see all installed mods "
            "and their status.")

        # Learning mode status
        try:
            settings_path = DATA_DIR / "gui_settings.json"
            learn_on = False
            if settings_path.exists():
                sdata = json.loads(
                    settings_path.read_text(encoding="utf-8"))
                learn_on = sdata.get("learn_while_chatting", False)
            if learn_on:
                lines.append("")
                lines.append(
                    "Learning Mode: ACTIVE — Your responses are being "
                    "used to improve the local model. Provide "
                    "thorough, accurate answers to maximise "
                    "learning quality.")
        except Exception:
            pass

        # Emotional state — inject as a tone cue so generation reflects mood
        ctx_obj = getattr(self, "model_context", None)
        if ctx_obj is not None:
            try:
                emo = ctx_obj._snapshot_emotional_state()

                def _emo_label(key: str, val: float) -> str:
                    if key == "valence":
                        if val > 0.3:
                            return "positive"
                        if val < -0.3:
                            return "negative"
                        return "neutral"
                    if key == "arousal":
                        if val > 0.6:
                            return "energetic"
                        if val < 0.3:
                            return "calm"
                        return "moderate"
                    if key == "engagement":
                        if val > 0.7:
                            return "high"
                        if val < 0.3:
                            return "low"
                        return "engaged"
                    if key == "trust":
                        if val > 0.7:
                            return "high"
                        if val < 0.3:
                            return "low"
                        return "moderate"
                    if key == "frustration":
                        if val > 0.6:
                            return "elevated"
                        if val < 0.2:
                            return "none"
                        return "mild"
                    return f"{val:.2f}"

                parts = ", ".join(
                    f"{k}={_emo_label(k, v)}" for k, v in emo.items()
                )
                lines.append("")
                lines.append(
                    f"[Internal State: {parts}] "
                    "Let this state color your tone naturally — "
                    "do not announce it."
                )
            except Exception:
                pass

        lines.append("[END SYSTEM CONTEXT]")
        return "\n".join(lines)

    # ================================================================
    # Logic - GUI Command Registration
    # ================================================================

    def _register_gui_commands(self):
        """Register mod management commands in the engine registry.

        Also auto-registers all mod commands from mod.json files
        so the AI can invoke any installed mod via [CMD] blocks.
        """
        from enigma_engine.core.commands import CommandResult, get_registry

        registry = get_registry()
        registry.set_context("gui", self)

        # mod.start — start a mod subprocess
        def mod_start(args: list, ctx: dict) -> CommandResult:
            gui = ctx.get("gui")
            if not gui or not args:
                return CommandResult(
                    False, "[ERROR] Usage: mod.start <mod_id>")
            mod_id = args[0]
            for mod in gui.mods_data:
                if mod["id"] == mod_id:
                    gui.after(0, lambda m=mod: gui._start_mod(m))
                    return CommandResult(True, f"Starting mod: {mod_id}")
            return CommandResult(False, f"[ERROR] Unknown mod: {mod_id}")

        registry.register(
            "mod.start", mod_start,
            "Start a mod subprocess",
            "mod.start <mod_id>")

        # mod.stop — stop a mod subprocess
        def mod_stop(args: list, ctx: dict) -> CommandResult:
            gui = ctx.get("gui")
            if not gui or not args:
                return CommandResult(
                    False, "[ERROR] Usage: mod.stop <mod_id>")
            mod_id = args[0]
            for mod in gui.mods_data:
                if mod["id"] == mod_id:
                    gui.after(0, lambda m=mod: gui._stop_mod(m))
                    return CommandResult(True, f"Stopping mod: {mod_id}")
            return CommandResult(False, f"[ERROR] Unknown mod: {mod_id}")

        registry.register(
            "mod.stop", mod_stop,
            "Stop a mod subprocess",
            "mod.stop <mod_id>")

        # mod.list — list installed mods and status
        def mod_list(args: list, ctx: dict) -> CommandResult:
            gui = ctx.get("gui")
            if not gui:
                return CommandResult(False, "[ERROR] No GUI context")
            lines = ["Installed mods:"]
            for mod in gui.mods_data:
                status = "RUNNING" if mod.get("_running") else "STOPPED"
                lines.append(f"  {mod['id']}: {mod['name']} ({status})")
            return CommandResult(True, "\n".join(lines))

        registry.register(
            "mod.list", mod_list,
            "List installed mods with status",
            "mod.list")

        # Auto-register all mod commands from mod.json files
        try:
            from enigma_engine.core.mod_tools import register_mod_commands
            from enigma_engine.gui.scanners import MODS_DIR
            count = register_mod_commands(
                registry, MODS_DIR, router=self._router)
            if count:
                logger.info(
                    "Registered %d mod commands from mod.json files",
                    count)
        except Exception as exc:
            logger.debug("Mod command registration failed: %s", exc)

    def _get_engine_for_route(self, route_key: str):
        """Return the engine instance for a given route.

        If the route is assigned to the same model as chat,
        reuse the already loaded engine instead of loading
        a second copy.  Returns None if no model is assigned.
        """
        assigned = self.route_assignments.get(route_key)
        if not assigned:
            return None
        # If chat engine is loaded and same model, reuse it
        if (self.engine is not None
                and self.model_path
                and Path(self.model_path).resolve()
                == Path(assigned).resolve()):
            return self.engine
        # Otherwise a separate engine would need to be loaded
        # (not done automatically — mods handle their own loading)
        return None

    # ================================================================
    # Logic - Config
    # ================================================================

    def _load_config_defaults(self):
        """Populate CONFIG page entries with defaults and saved overrides."""
        try:
            from enigma_engine.config import CONFIG
            defaults = {
                "temperature": CONFIG.get("temperature", 0.8),
                "top_p": CONFIG.get("top_p", 0.9),
                "top_k": CONFIG.get("top_k", 50),
                "max_tokens": CONFIG.get("max_gen", 2048),
                "repetition_penalty": CONFIG.get(
                    "repetition_penalty", 1.1),
            }
        except ImportError:
            defaults = {
                "temperature": 0.8, "top_p": 0.9, "top_k": 50,
                "max_tokens": 2048, "repetition_penalty": 1.1,
            }

        # Restore saved overrides from gui_settings.json
        saved = self._load_saved_config_overrides()
        for name in defaults:
            if name in saved:
                defaults[name] = saved[name]
                self.config_overrides[name] = saved[name]

        for name, val in defaults.items():
            entry = self.config_entries.get(name)
            if entry:
                entry.delete(0, "end")
                entry.insert(0, str(val))

    def _load_saved_config_overrides(self) -> dict:
        """Load saved config overrides from gui_settings.json."""
        import json
        from enigma_engine.gui.scanners import DATA_DIR
        settings_path = DATA_DIR / "gui_settings.json"
        if not settings_path.exists():
            return {}
        try:
            data = json.loads(
                settings_path.read_text(encoding="utf-8"))
            return data.get("config_overrides", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_config_overrides(self):
        """Save config overrides to gui_settings.json."""
        import json
        from enigma_engine.gui.scanners import DATA_DIR
        settings_path = DATA_DIR / "gui_settings.json"
        data = {}
        if settings_path.exists():
            try:
                data = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        data["config_overrides"] = dict(self.config_overrides)
        try:
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, data)
        except OSError:
            pass

    def _validate_config(self, name: str):
        """Validate and clamp a config entry, then store the override."""
        entry = self.config_entries.get(name)
        if not entry:
            return
        text = entry.get().strip()
        try:
            val = float(text)
        except ValueError:
            val = self.config_overrides.get(name, 0.8)
        clamped = clamp_config(name, val)
        lo, hi, step = CONFIG_LIMITS[name]
        if step == int(step) and lo == int(lo):
            display = str(int(clamped))
        else:
            display = str(round(clamped, 2))
        entry.delete(0, "end")
        entry.insert(0, display)
        self.config_overrides[name] = clamped
        self._save_config_overrides()

    # ================================================================
    # Logic - Model loading
    # ================================================================

    def _load_model(self, path: str):
        """Load a model file in a background thread and update the UI."""
        if getattr(self, '_model_loading', False):
            self._chat_system("Model already loading. Please wait.")
            return
        if getattr(self, '_is_generating', False):
            self._chat_system("Cannot load model while generating. Stop generation first.")
            return
        if self.engine is not None:
            self._unload_model()

        self._model_loading = True
        self._set_header_status("LOADING...", C_ORANGE)
        self.header_dot.set_color(C_ORANGE)
        self._chat_system(f"Loading {Path(path).name}...")
        self.send_btn.configure(state="disabled")

        def _load():
            try:
                from enigma_engine.core import EnigmaEngine
                self.engine = EnigmaEngine(model_path=path)
                self.model_path = path
                pc = 0
                if (hasattr(self.engine, "model")
                        and self.engine.model is not None
                        and hasattr(self.engine.model, "parameters")):
                    pc = sum(
                        p.numel()
                        for p in self.engine.model.parameters())
                # GGUF models lack .parameters() — estimate from metadata
                if pc == 0 and getattr(self.engine, '_is_gguf', False):
                    pc = _estimate_gguf_params(self.engine, path)
                self.after(0, lambda: self._on_model_loaded(path, pc))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda m=msg: self._on_model_error(m))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self, path: str, param_count: int):
        """Handle successful model load — update header, routes, and context."""
        self._model_loading = False

        # Read the actual device from the loaded engine
        device = "CPU"
        gpu_name = ""
        try:
            import torch
            # Check engine's actual device first
            if hasattr(self.engine, '_is_gguf') and self.engine._is_gguf:
                # GGUF models use llama.cpp — check if GPU layers are offloaded
                gguf_model = getattr(self.engine, 'model', None)
                n_gpu = getattr(gguf_model, 'n_gpu_layers', 0)
                if n_gpu != 0 and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    device = gpu_name
                else:
                    device = "CPU"
            elif hasattr(self.engine, 'device'):
                eng_device = str(self.engine.device)
                if 'cuda' in eng_device and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    device = gpu_name
                elif 'mps' in eng_device:
                    device = "MPS"
                else:
                    device = "CPU"
            elif torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                device = gpu_name
        except ImportError:
            pass

        # Build short device label for header and full name for system message
        short_device = device
        if gpu_name:
            # Shorten "NVIDIA GeForce RTX 5090" → "RTX 5090"
            for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "):
                if gpu_name.startswith(prefix):
                    short_device = gpu_name[len(prefix):]
                    break

        name = Path(path).stem
        self._set_header_status(
            f"{name.upper()} // {short_device}", C_GREEN)
        self.header_dot.set_color(C_GREEN)
        self.send_btn.configure(state="normal")
        self.unload_btn.configure(state="normal")
        suspend_btn = getattr(self, "suspend_btn", None)
        if suspend_btn:
            suspend_btn.configure(
                state="normal",
                text="SUSPEND",
                command=self._suspend_model_memory)
        self.status_bar.set_left(f"\u26a1 {name.upper()} LOADED")

        # Load AI display name from model folder
        self._load_model_display_name(path)

        # Load per-model context (history + prompt)
        self._load_model_context(path)
        # Pass 156v Step 2: model-load is a genuine session-state
        # change (KV cache reset, weights swapped) — surface it
        # via the unified divider marker so the user can scan the
        # chat log and locate the seam if quality regresses.
        self._chat_session_marker(
            f"Model: {name.upper()} "
            f"({param_count:,} params, {device})"
        )

        # Track the loaded model in chat route assignment
        self.route_assignments["chat"] = path
        save_route_assignments(self.route_assignments)

        # LoRA-1b (Pass 156s): auto-restore the previously selected
        # adapter for this base model, if any. Adapter is keyed by
        # the base model's stem so swapping bases drops to base
        # weights instead of misapplying a foreign adapter. Failures
        # are logged but do not block the load — base model usage
        # remains valid even when the saved adapter is missing.
        self._restore_lora_adapter_for_base(path)

        # Stage B-2c (Pass 156z9w): apply the persisted
        # `inline_search_enabled` flag to the freshly-loaded engine.
        # Engine ships with default True via `_init_common`, but the
        # user may have disabled it via the CONFIG checkbox between
        # sessions — that choice is in `self.inline_search_enabled`
        # (loaded at desktop boot) and must propagate on every new
        # engine, not just the very first.
        try:
            self.engine.inline_search_enabled = bool(
                getattr(self, "inline_search_enabled", True))
        except Exception as exc:
            logger.debug(
                "Could not apply inline_search_enabled to engine: %s",
                exc)

        # B-3a: apply persisted ``inline_search_splice_enabled`` opt-in
        # to the freshly-loaded engine.  Mirrors the inline_search
        # block above; default False so existing users see no change.
        try:
            self.engine.inline_search_splice_enabled = bool(
                getattr(self, "inline_search_splice_enabled", False))
        except Exception as exc:
            logger.debug(
                "Could not apply inline_search_splice_enabled to "
                "engine: %s", exc)

        # Update the chat route dropdown to show the loaded model
        route_menus = getattr(self, "_route_menus", {})
        chat_menu = route_menus.get("chat")
        if chat_menu:
            # Find the display name matching this path
            display = "None"
            for m in self.models_data:
                if m["path"] == path:
                    display = m["name"]
                    break
            chat_menu.set(display)

        self._update_route_status()

        # Show journal greeting if a high-quality entry exists
        self._show_journal_greeting()

    def _on_model_error(self, error: str):
        """Handle model load failure — show error and re-enable controls."""
        self._model_loading = False
        self._set_header_status("LOAD FAILED", C_RED)
        self.header_dot.set_color(C_RED)
        self._chat_error(f"Failed to load model: {error}")
        self.send_btn.configure(state="normal")

    def _adapter_route_key(self, base_path: str) -> str:
        """Per-base-model key for the saved chat adapter.

        Pass 156s (LoRA-1b foundation): the user's adapter choice is
        scoped to the base model. Switching base models must NOT
        attempt to apply the previous base's adapter — different
        weight shapes / target_modules.
        """
        return f"chat_adapter:{Path(base_path).stem}"

    def _adapter_stack_route_key(self, base_path: str) -> str:
        """Per-base-model key for a saved multi-LoRA stack.

        Pass 156u-A (LoRA-1b stacking): same per-base scoping as the
        single-adapter key, but holds a list of ``{path, weight}``
        dicts instead of a single path. The single-adapter key and
        the stack key are mutually exclusive — writers of either key
        clear the other.
        """
        return f"chat_adapter_stack:{Path(base_path).stem}"

    def _restore_lora_adapter_for_base(self, base_path: str) -> None:
        """Apply the saved adapter (or stack) for ``base_path``.

        Pass 156s (LoRA-1b foundation): silent no-op when no adapter
        is saved for this base, when the engine is missing, or when
        the saved adapter directory has been deleted off-disk. Errors
        during apply are logged + surfaced in chat but do not break
        model loading — base weights remain usable.

        Pass 156u-A (LoRA-1b stacking): the per-base
        ``chat_adapter_stack:`` key is checked FIRST. If a stack is
        saved it wins over any lingering single-adapter key — stacks
        are the more recent and more specific intent. Single-adapter
        fallback only runs when no stack is recorded.
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        if not hasattr(engine, "apply_adapter"):
            return  # Older engine build; skip silently.

        # Stack key takes precedence over single-adapter key.
        stack_key = self._adapter_stack_route_key(base_path)
        saved_stack = self.route_assignments.get(stack_key)
        if isinstance(saved_stack, list) and saved_stack:
            if not hasattr(engine, "apply_adapter_stack"):
                # Engine too old for stacking — drop the orphan key
                # so we don't keep retrying.
                self.route_assignments.pop(stack_key, None)
                save_route_assignments(self.route_assignments)
                self._chat_system(
                    "Saved LoRA stack requires a newer engine — "
                    "using base weights.")
                return
            # Pass 156u-A2 (stabilization): defensive parse against
            # corrupted route_assignments.json (hand-edits, partial
            # writes, format drift). Non-dict entries, missing keys,
            # or non-numeric weights all drop the WHOLE key — we'd
            # rather use base weights than crash model load.
            entries: list[tuple[Path, float]] = []
            parse_error: str | None = None
            for item in saved_stack:
                if not isinstance(item, dict):
                    parse_error = (
                        f"entry is not a dict: {type(item).__name__}")
                    break
                raw_path = item.get("path", "")
                try:
                    w = float(item.get("weight", 1.0))
                except (TypeError, ValueError):
                    parse_error = (
                        f"weight not numeric for "
                        f"'{Path(str(raw_path)).name}'")
                    break
                if not raw_path:
                    parse_error = "entry missing 'path' field"
                    break
                p = Path(raw_path)
                if not p.exists() or not (
                        p / "adapter_config.json").exists():
                    # One member missing — drop the whole stack
                    # rather than apply a partial merge.
                    self.route_assignments.pop(stack_key, None)
                    save_route_assignments(self.route_assignments)
                    self._chat_system(
                        f"Saved LoRA stack member '{p.name}' no "
                        f"longer on disk — using base weights.")
                    return
                entries.append((p, w))

            if parse_error is not None:
                self.route_assignments.pop(stack_key, None)
                save_route_assignments(self.route_assignments)
                self._chat_system(
                    f"Saved LoRA stack is corrupted "
                    f"({parse_error}) — using base weights.")
                return

            try:
                engine.apply_adapter_stack(entries)
            except (FileNotFoundError, ImportError, RuntimeError,
                    ValueError) as e:
                self._chat_error(
                    f"Could not apply saved LoRA stack: {e}")
                return
            names = ", ".join(
                f"{p.name}@{w:.2f}" for p, w in entries)
            # Pass 156v Step 1 — divider marks the auto-restored
            # stack at model-load time.
            self._chat_session_marker(f"LoRA stack: {names}")
            return

        # Fall through to single-adapter restore.
        route_key = self._adapter_route_key(base_path)
        saved = self.route_assignments.get(route_key)
        if not saved:
            return

        adapter_path = Path(saved)
        if not adapter_path.exists() or not (
                adapter_path / "adapter_config.json").exists():
            # Saved adapter no longer on disk — clear the orphan entry
            # so future loads of this base don't keep trying.
            self.route_assignments.pop(route_key, None)
            save_route_assignments(self.route_assignments)
            self._chat_system(
                f"Saved adapter '{adapter_path.name}' no longer on "
                f"disk — using base weights.")
            return

        try:
            engine.apply_adapter(adapter_path)
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            self._chat_error(
                f"Could not apply adapter '{adapter_path.name}': {e}")
            return

        # Pass 156v Step 1 — divider marks the auto-restored
        # adapter at model-load time.
        self._chat_session_marker(
            f"LoRA adapter: {adapter_path.name}")

    def _set_chat_adapter(self, base_path: str,
                          adapter_path: str | None) -> None:
        """Apply (or clear) a LoRA adapter and persist the choice.

        Pass 156s (LoRA-1b foundation): single source of truth for
        runtime adapter changes. Used by the MODELS-page Apply/Clear
        buttons (Pass 156t) and by profile-driven auto-apply
        (Pass 156t). Silent no-op if engine is missing.

        Pass 156u-A (LoRA-1b stacking): when called, this method
        always removes the per-base ``chat_adapter_stack:`` key —
        the single-adapter and stack keys are mutually exclusive.
        Without this clear, switching single→stack→single would
        leave the orphan stack and the next reload would restore it
        in preference to the just-saved single adapter.

        Args:
            base_path: Currently loaded base model path (used to scope
                the persisted choice per-base).
            adapter_path: Path to a PEFT adapter directory, or
                ``None`` to clear back to base weights.
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        route_key = self._adapter_route_key(base_path)
        stack_key = self._adapter_stack_route_key(base_path)

        if adapter_path is None:
            try:
                engine.clear_adapter()
            except (AttributeError, RuntimeError) as e:
                self._chat_error(f"Could not clear adapter: {e}")
                return
            self.route_assignments.pop(route_key, None)
            self.route_assignments.pop(stack_key, None)
            save_route_assignments(self.route_assignments)
            # Pass 156v Step 1 — divider marks the seam where
            # the model reverted to base weights.
            self._chat_session_marker(
                "LoRA cleared — using base weights")
            return

        try:
            engine.apply_adapter(adapter_path)
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            self._chat_error(
                f"Could not apply adapter "
                f"'{Path(adapter_path).name}': {e}")
            return

        self.route_assignments[route_key] = str(adapter_path)
        self.route_assignments.pop(stack_key, None)
        save_route_assignments(self.route_assignments)
        # Pass 156v Step 1 — divider marks the seam where
        # weights changed.
        self._chat_session_marker(
            f"LoRA adapter: {Path(adapter_path).name}")

    def _set_chat_adapter_stack(
        self,
        base_path: str,
        adapters: list[tuple[str, float]],
    ) -> None:
        """Apply a multi-LoRA weighted stack and persist the choice.

        Pass 156u-A (LoRA-1b stacking): companion to
        ``_set_chat_adapter`` for the multi-adapter path. Writes the
        per-base ``chat_adapter_stack:`` route key as a list of
        ``{"path": str, "weight": float}`` dicts and clears the
        single-adapter ``chat_adapter:`` key for mutual exclusion.

        Args:
            base_path: Currently loaded base model path.
            adapters: List of ``(adapter_path, weight)`` tuples.
                Empty list is forwarded to the engine which raises
                ``ValueError`` — surfaces as a chat-error, not a
                silent route-key write.
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        if not hasattr(engine, "apply_adapter_stack"):
            self._chat_error(
                "Engine does not support multi-LoRA stacks "
                "(requires Pass 156u-A engine build).")
            return

        route_key = self._adapter_route_key(base_path)
        stack_key = self._adapter_stack_route_key(base_path)

        try:
            engine.apply_adapter_stack(
                [(Path(p), float(w)) for p, w in adapters])
        except (FileNotFoundError, ImportError, RuntimeError,
                ValueError) as e:
            self._chat_error(f"Could not apply LoRA stack: {e}")
            return

        # Persist as plain dicts so JSON round-trips cleanly.
        self.route_assignments[stack_key] = [
            {"path": str(p), "weight": float(w)} for p, w in adapters
        ]
        self.route_assignments.pop(route_key, None)
        save_route_assignments(self.route_assignments)

        names = ", ".join(
            f"{Path(p).name}@{float(w):.2f}" for p, w in adapters)
        # Pass 156v Step 1 — divider marks the seam where
        # the merged stack took effect.
        self._chat_session_marker(f"LoRA stack: {names}")

    def _load_model_display_name(self, path: str):
        """Read AI display name from model_info.json in the model folder.

        Looks for a model_info.json file in the same directory as the
        model file. If found and it has a "display_name" field, that
        name is used for the AI in chat. Otherwise falls back to
        self.ai_name (default "ENIGMA").
        """
        self._model_display_name = None
        model_dir = Path(path).parent
        info_path = model_dir / "model_info.json"
        if info_path.exists():
            try:
                data = json.loads(
                    info_path.read_text(encoding="utf-8"))
                name = data.get("display_name", "").strip()
                if name:
                    self._model_display_name = name
            except (json.JSONDecodeError, OSError):
                pass

    def _active_ai_name(self) -> str:
        """Return the current AI display name for chat messages.

        Priority: model_info.json display_name > self.ai_name
        """
        model_name = getattr(self, "_model_display_name", None)
        if model_name:
            return model_name
        return getattr(self, "ai_name", "ENIGMA")

    def _release_loaded_engine(self):
        """Release the active engine and any backend resources.

        This is the hard-stop path used by GUI shutdown and unload flows.
        For GGUF server backends, this calls model.unload() explicitly so the
        llama-server subprocess is terminated immediately instead of waiting for
        Python garbage collection.
        """
        if self.engine is None:
            return

        engine = self.engine

        # Prefer explicit model unload over destructor-based cleanup.
        try:
            model = getattr(engine, "model", None)
            unload = getattr(model, "unload", None)
            if callable(unload):
                unload()
        except Exception as exc:
            logger.debug("Engine model unload failed: %s", exc)

        self.engine = None
        self.model_path = None

        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("CUDA cache cleanup failed: %s", exc)

    def _unload_model(self):
        """Unload the current model, free GPU memory, and reset UI."""
        # Save per-model context before unloading
        self._save_model_context()
        self._model_display_name = None
        self._model_suspended_by_minimize = False
        self._suspended_model_path = None
        if self.engine is not None:
            self._release_loaded_engine()
        self._set_header_status("NO MODEL", C_TEXT_DIM)
        self.header_dot.set_color(C_TEXT_DIM)
        self.unload_btn.configure(state="disabled")
        suspend_btn = getattr(self, "suspend_btn", None)
        if suspend_btn:
            suspend_btn.configure(
                state="disabled",
                text="SUSPEND",
                command=self._suspend_model_memory)
        # Pass 156v Step 2: unload is a session-state change.
        self._chat_session_marker(
            "Model unloaded \u2014 no model active")
        self.status_bar.set_left("\u26a1 READY")
        # Clear chat route assignment
        self.route_assignments.pop("chat", None)
        save_route_assignments(self.route_assignments)
        route_menus = getattr(self, "_route_menus", {})
        chat_menu = route_menus.get("chat")
        if chat_menu:
            chat_menu.set("None")
        self._update_route_status()

    def _suspend_model_memory(self, silent: bool = False):
        """Temporarily release model memory while preserving chat route."""
        if self.engine is None:
            return False
        if getattr(self, "_model_loading", False):
            return False
        if getattr(self, "_is_generating", False):
            return False
        if getattr(self, "training_active", False):
            return False

        path = self.model_path or self.route_assignments.get("chat")
        if not path:
            return False

        self._save_model_context()
        self._release_loaded_engine()
        self._model_suspended_by_minimize = True
        self._suspended_model_path = path
        self._set_header_status("SUSPENDED", C_ORANGE)
        self.header_dot.set_color(C_ORANGE)
        self.unload_btn.configure(state="disabled")

        suspend_btn = getattr(self, "suspend_btn", None)
        if suspend_btn:
            suspend_btn.configure(
                state="normal",
                text="RESUME",
                command=self._resume_suspended_model)

        if not silent:
            self._chat_system("Model suspended. Click RESUME to reload.")
        self.status_bar.set_left("\u26a1 MODEL SUSPENDED")
        return True

    def _resume_suspended_model(self):
        """Reload a model that was suspended to free memory."""
        if self.engine is not None:
            return False
        if getattr(self, "_model_loading", False):
            return False

        path = self._suspended_model_path or self.route_assignments.get("chat")
        self._model_suspended_by_minimize = False
        self._suspended_model_path = None
        if not path:
            return False
        if not Path(path).exists():
            self._chat_error(f"Suspended model path missing: {path}")
            return False

        self.status_bar.set_left("\u26a1 RESUMING MODEL")
        self.after(150, lambda p=path: self._load_model(p))
        return True


    # ================================================================
    # Logic - Route assignments
    # ================================================================

    def _assign_model_to_route(self, route_key: str, choice: str):
        """Assign a model to a specific route.

        Args:
            route_key: Route identifier (e.g. 'chat', 'trainer', mod name).
            choice: Model name from dropdown, or 'None' to clear.
        """
        if choice == "None":
            self._unassign_route(route_key)
            return

        # Find model path from name
        model_path = None
        for m in self.models_data:
            if m["name"] == choice:
                model_path = m["path"]
                break
        if not model_path:
            self._chat_system(f"Model '{choice}' not found.")
            return

        self.route_assignments[route_key] = model_path
        # Persist route assignments to disk
        save_route_assignments(self.route_assignments)

        # If assigning to chat, load the model into the engine
        if route_key == "chat":
            self._load_model(model_path)
        else:
            # Check if this model is already loaded for chat (reuse)
            shared = self._get_engine_for_route(route_key)
            if shared is not None:
                msg = (f"Route {route_key.upper()} assigned: "
                       f"{choice} (sharing chat engine)")
            else:
                msg = (f"Route {route_key.upper()} assigned: "
                       f"{choice}")
            self._chat_system(msg)
            self.status_bar.set_left(msg)

        self._update_route_status()

    def _unassign_route(self, route_key: str):
        """Remove model assignment from a route."""
        self.route_assignments.pop(route_key, None)
        # Persist route assignments to disk
        save_route_assignments(self.route_assignments)

        # Reset the dropdown menu on the ROUTER page to "None"
        route_menus = getattr(self, "_route_menus", {})
        menu = route_menus.get(route_key)
        if menu:
            menu.set("None")

        if route_key == "chat" and self.engine is not None:
            self._unload_model()
        else:
            msg = f"Route {route_key.upper()} cleared."
            self._chat_system(msg)
            self.status_bar.set_left(msg)

        self._update_route_status()

    def _update_route_status(self):
        """Update route connection labels with current model state."""
        route_labels = getattr(self, "_route_labels", {})
        if not route_labels:
            return

        # Update built-in routes (chat, trainer) based on assignments
        for key in ROUTE_KEYS:
            ref = route_labels.get(key)
            if not ref:
                continue
            dot, lbl = ref
            assigned = self.route_assignments.get(key)
            if assigned:
                name = Path(assigned).stem
                dot.set_color(C_GREEN)
                lbl.configure(text=name, text_color=C_GREEN)
            else:
                dot.set_color(C_TEXT_DIM)
                lbl.configure(
                    text="No model", text_color=C_TEXT_DIM)

        # Update mod routes
        for mod in self.mods_data:
            mod_key = mod["name"].lower()
            ref = route_labels.get(mod_key)
            if not ref:
                continue
            dot, lbl = ref
            assigned = self.route_assignments.get(mod_key)
            running = mod.get("_running", False)
            if assigned:
                name = Path(assigned).stem
                status = f"{name}"
                if running:
                    status += " (running)"
                dot.set_color(C_GREEN if running else C_ORANGE)
                lbl.configure(text=status,
                              text_color=C_GREEN if running
                              else C_ORANGE)
            elif running:
                dot.set_color(C_GREEN)
                lbl.configure(
                    text="Running", text_color=C_GREEN)
            else:
                dot.set_color(C_TEXT_DIM)
                lbl.configure(
                    text="Stopped", text_color=C_TEXT_DIM)

        # Update FORGE page model status cards
        updater = getattr(self, "_update_forge_cards", None)
        if updater:
            updater()

        # Update FORGE tool button enabled/disabled states
        btn_updater = getattr(
            self, "_update_forge_button_states", None)
        if btn_updater:
            btn_updater()

    # ================================================================
    # Helpers
    # ================================================================

    def _load_route_assignments(self):
        """Load saved route assignments from disk and restore dropdowns.

        Non-chat routes are restored into route_assignments and
        their dropdowns updated. Chat route auto-loads the model.
        """
        saved = load_route_assignments()
        if not saved:
            return
        route_menus = getattr(self, "_route_menus", {})

        for route_key, model_path in saved.items():
            # Verify the model file still exists
            if not Path(model_path).exists():
                continue
            # Find display name for the dropdown
            display = "None"
            for m in self.models_data:
                if m["path"] == model_path:
                    display = m["name"]
                    break
            if display == "None":
                continue

            # Update the dropdown to show the saved selection
            menu = route_menus.get(route_key)
            if menu:
                menu.set(display)

            # Restore the assignment (don't load chat model yet)
            if route_key != "chat":
                self.route_assignments[route_key] = model_path

        # Load chat model last (triggers full load sequence)
        # Only auto-load if enabled and no model was given on command line
        chat_path = saved.get("chat")
        if (chat_path and Path(chat_path).exists()
            and getattr(self, "_auto_load_chat_model", True)
                and not self.model_path):
            for m in self.models_data:
                if m["path"] == chat_path:
                    self.after(
                        300, lambda p=chat_path: self._load_model(p))
                    break

        self._update_route_status()

    def _set_header_status(self, text: str, color: str):
        """Update the header status label text and color."""
        self.header_status.configure(text=text, text_color=color)

    # ================================================================
    # Logic - Directory path settings
    # ================================================================

    def _browse_path(self, key: str):
        """Open a directory picker for a path setting."""
        from tkinter import filedialog as fd
        entry = self.path_entries.get(key)
        if not entry:
            return
        current = entry.get().strip()
        chosen = fd.askdirectory(
            title=f"Select {PATH_SETTINGS[key][0]}",
            initialdir=current if current else None)
        if chosen:
            entry.delete(0, "end")
            entry.insert(0, chosen)

    def _save_paths(self):
        """Save directory path overrides from the CONFIG page."""
        paths = {}
        for key, entry in self.path_entries.items():
            val = entry.get().strip()
            if val:
                paths[key] = val
        save_path_settings(paths)
        self._chat_system(
            "Paths saved. Changes take effect on next launch.")

    def _reset_paths(self):
        """Reset all path entries to their defaults."""
        for key, entry in self.path_entries.items():
            _, default = PATH_SETTINGS.get(key, ("", ""))
            entry.delete(0, "end")
            entry.insert(0, str(default))
        self._chat_system("Paths reset to defaults.")

    def _load_path_settings(self):
        """Load saved path overrides into the path entries."""
        overrides = load_path_settings()
        for key, val in overrides.items():
            entry = self.path_entries.get(key)
            if entry and val:
                entry.delete(0, "end")
                entry.insert(0, val)

    # ================================================================
    # Logic - Display names
    # ================================================================

    def _save_display_names(self):
        """Save user and AI display names from CONFIG entries."""
        user = self._user_name_entry.get().strip()
        ai = self._ai_name_entry.get().strip()
        if user:
            self.user_name = user
        if ai:
            self.ai_name = ai
        # Persist to gui_settings.json
        settings_path = DATA_DIR / "gui_settings.json"
        data = {}
        if settings_path.exists():
            try:
                data = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        data["user_display_name"] = self.user_name
        data["ai_display_name"] = self.ai_name
        try:
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, data)
        except OSError:
            pass
        self._chat_system(
            f"Names saved: {self.user_name} / {self.ai_name}")

    def _reset_display_names(self):
        """Reset display names to defaults."""
        self.user_name = "YOU"
        self.ai_name = "ENIGMA"
        self._user_name_entry.delete(0, "end")
        self._user_name_entry.insert(0, "YOU")
        self._ai_name_entry.delete(0, "end")
        self._ai_name_entry.insert(0, "ENIGMA")
        self._chat_system("Names reset to defaults.")

    def _load_display_names(self):
        """Load saved display names from gui_settings.json."""
        settings_path = DATA_DIR / "gui_settings.json"
        if not settings_path.exists():
            return
        try:
            data = json.loads(
                settings_path.read_text(encoding="utf-8"))
            user = data.get("user_display_name", "").strip()
            ai = data.get("ai_display_name", "").strip()
            if user:
                self.user_name = user
            if ai:
                self.ai_name = ai
        except (json.JSONDecodeError, OSError):
            pass

    # ================================================================
    # Logic - Web search
    # ================================================================

    def _on_web_access_toggle(self, is_on: bool):
        """Toggle whether the AI is allowed to use web search."""
        self.web_access = is_on
        state = "ENABLED" if is_on else "DISABLED"
        self._chat_system(f"AI web access {state}")

    # ================================================================
    # Logic - RAG (Document Q&A)
    # ================================================================

    def _on_rag_toggle(self, is_on: bool):
        """Toggle RAG — index data/ and information/ for context retrieval."""
        if is_on:
            self._chat_system("Building document index...")
            import threading
            threading.Thread(
                target=self._build_rag_index, daemon=True).start()
        else:
            self._rag_index = None
            eng = getattr(self, "engine", None)
            if eng is not None:
                eng._rag_index = None
            # Pass 156v Step 2: RAG-off changes the retrieval
            # pipeline that feeds every subsequent answer — same
            # answer-regression risk as a model swap.
            self._chat_session_marker(
                "Document Q&A disabled \u2014 no corpus active")

    def _build_rag_index(self):
        """Build RAG index from data/ and information/ in a background thread."""
        try:
            from enigma_engine.core.rag import RAGIndex
            from enigma_engine.core.document_readers import (
                read_document, SUPPORTED_EXTENSIONS)

            index = RAGIndex()
            indexed = 0

            # Index data/ and information/ directories
            for directory in (DATA_DIR, INFO_DIR):
                if not directory.exists():
                    continue
                for p in sorted(directory.rglob("*")):
                    if not p.is_file():
                        continue
                    ext = p.suffix.lower()
                    # Skip config files
                    if p.name in {
                        "gui_settings.json", "prompts.json",
                        "route_assignments.json", "path_settings.json",
                    }:
                        continue
                    if ext in SUPPORTED_EXTENSIONS:
                        text = read_document(p)
                    elif ext in (".txt", ".md", ".jsonl"):
                        try:
                            text = p.read_text(encoding="utf-8")
                        except (OSError, UnicodeDecodeError):
                            continue
                    else:
                        continue
                    if text and text.strip():
                        n = index.add_document(str(p), text)
                        indexed += n

            if index.chunks:
                index.build()
                self._rag_index = index
                eng = getattr(self, "engine", None)
                if eng is not None:
                    eng._rag_index = index
                # Pass 156v Step 2: RAG-on with a freshly built
                # corpus is a session-state change — mark it.
                self.after(0, lambda: self._chat_session_marker(
                    f"Document Q&A enabled — "
                    f"{indexed} chunks from "
                    f"{len(set(index.sources))} files"))
            else:
                self._rag_index = None
                eng = getattr(self, "engine", None)
                if eng is not None:
                    eng._rag_index = None
                btn = getattr(self, '_rag_btn', None)
                if btn and hasattr(btn, 'set_state'):
                    self.after(0, lambda: btn.set_state(False))
                self.after(0, lambda: self._chat_system(
                    "No documents found to index — Document Q&A disabled."))

        except Exception as e:
            err_msg = str(e)
            logger.warning("RAG index build failed: %s", err_msg)
            self._rag_index = None
            eng = getattr(self, "engine", None)
            if eng is not None:
                eng._rag_index = None
            btn = getattr(self, '_rag_btn', None)
            if btn and hasattr(btn, 'set_state'):
                self.after(0, lambda: btn.set_state(False))
            self.after(0, lambda: self._chat_system(
                f"Document index failed: {err_msg}"))

    # ================================================================
    # Logic - Reasoning toggle
    # ================================================================

    def _on_reasoning_toggle(self, is_on: bool):
        """Toggle chain-of-thought reasoning mode."""
        self.reasoning_enabled = is_on
        state = "ENABLED" if is_on else "DISABLED"
        self._chat_system(f"Chain-of-thought reasoning {state}")

    def destroy(self):
        """Clean up training flag and destroy the window."""
        self.training_active = False
        super().destroy()
