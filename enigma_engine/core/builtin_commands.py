"""
Built-in command implementations for the Enigma Engine command system.

All default commands are registered here via ``register_builtin_commands()``.
The function is called once by :func:`commands.get_registry` when the global
registry is first created.
"""
from __future__ import annotations

import json

from .commands import CommandResult, CommandRegistry


# ``Dict`` is referenced in annotation strings throughout this module.
# With ``from __future__ import annotations`` the annotations are never
# evaluated at runtime, so we only need the name available in the module
# namespace for tools that *do* resolve annotations (e.g. get_type_hints).
from typing import Dict


# ── Constants (extracted from inline magic numbers) ──────────────
# HTTP timeouts (seconds)
HTTP_TIMEOUT_HEALTH = 2         # Quick health/connectivity check
HTTP_TIMEOUT_SHORT = 5          # TTS, subprocess communicate
HTTP_TIMEOUT_DEFAULT = 10       # Standard web requests
HTTP_TIMEOUT_FETCH = 15         # Fetching page content
HTTP_TIMEOUT_LONG = 30          # Large downloads, subprocess
HTTP_TIMEOUT_GENERATE = 120     # Image/video generation endpoints

# Output truncation limits (characters)
PREVIEW_LIMIT = 50              # Short previews of text content
SNIPPET_LIMIT = 100             # Code/search snippet length
CONTENT_PREVIEW = 500           # File content preview
OUTPUT_LIMIT = 1000             # Command output cap
FETCH_MAX_CHARS = 2000          # Web page text extraction cap
ERROR_TAIL = 1000               # Stderr tail capture

# List display limits (items)
LIST_DISPLAY_LIMIT = 20         # Max items shown in list output
SEARCH_MAX_RESULTS = 5          # Default DDG search count
HISTORY_DEFAULT_COUNT = 20      # Default history items to show
NOTES_DISPLAY_LIMIT = 10        # Max notes shown in list
CONFIG_LIST_LIMIT = 6           # Config value preview lines

# Polling
POLL_ITERATIONS = 60            # Max iterations for generation poll
POLL_INTERVAL = 2               # Seconds between poll checks


def register_builtin_commands(registry: CommandRegistry) -> None:
    """Register every built-in command on *registry*."""

    # ========== Config Commands ==========

    def config_get(args: list[str], ctx: Dict) -> CommandResult:
        """Get a config value."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: config.get <key>")

        config = ctx.get("config", {})
        key = args[0]

        if key in config:
            return CommandResult(True, f"[OK] {key} = {config[key]}", config[key])
        return CommandResult(False, f"[ERROR] Unknown config: {key}")

    def config_set(args: list[str], ctx: Dict) -> CommandResult:
        """Set a config value."""
        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: config.set <key> <value>")

        config = ctx.get("config")
        if config is None:
            config = {}
            ctx["config"] = config

        key = args[0]
        value = args[1]

        # Type conversion
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string

        config[key] = value
        return CommandResult(True, f"[OK] {key} = {value}")

    def config_list(args: list[str], ctx: Dict) -> CommandResult:
        """List all config values."""
        config = ctx.get("config", {})
        if not config:
            return CommandResult(True, "[OK] No config values set")

        lines = ["[OK] Config:"]
        for k, v in sorted(config.items()):
            lines.append(f"  {k} = {v}")
        return CommandResult(True, "\n".join(lines), config)

    registry.register("config.get", config_get, "Get config value", "config.get <key>")
    registry.register("config.set", config_set, "Set config value", "config.set <key> <value>")
    registry.register("config.list", config_list, "List all config", "config.list")

    # ========== Model Commands ==========

    def model_info(args: list[str], ctx: Dict) -> CommandResult:
        """Show current model info."""
        engine = ctx.get("engine")
        if not engine:
            return CommandResult(False, "[ERROR] No model loaded")

        try:
            model = engine.model
            if hasattr(model, 'config'):
                cfg = model.config
                info = f"Model: {cfg.dim}d, {cfg.n_layers} layers, {cfg.vocab_size} vocab"
            else:
                info = f"Model: {type(model).__name__}"
            return CommandResult(True, f"[OK] {info}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def model_list(args: list[str], ctx: Dict) -> CommandResult:
        """List available models."""
        from pathlib import Path
        models_dir = Path("models")

        if not models_dir.exists():
            return CommandResult(False, "[ERROR] No models directory")

        models = []
        # .pth files
        for f in models_dir.glob("*.pth"):
            models.append(f.stem)
        # Subdirs with model.pt
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "model.pt").exists():
                models.append(d.name)
        # GGUF files
        for f in models_dir.rglob("*.gguf"):
            models.append(f.relative_to(models_dir).as_posix())

        if not models:
            return CommandResult(True, "[OK] No models found")

        return CommandResult(True, f"[OK] Models: {', '.join(models)}", models)

    def model_load(args: list[str], ctx: Dict) -> CommandResult:
        """Load a model by path."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: model.load <path>")

        from pathlib import Path
        model_path = Path(args[0])

        # Check if path exists (try models/ folder too)
        if not model_path.exists():
            model_path = Path("models") / args[0]
        if not model_path.exists():
            # Try adding .gguf extension
            model_path = Path("models") / (args[0] + ".gguf")
        if not model_path.exists():
            return CommandResult(False, f"[ERROR] Model not found: {args[0]}")

        engine = ctx.get("engine")
        if engine:
            try:
                from enigma_engine.core.inference import EnigmaEngine
                new_engine = EnigmaEngine(model_path=str(model_path))
                # Clean up old engine to free GPU memory
                old_engine = ctx.get("engine")
                ctx["engine"] = new_engine
                if old_engine is not None:
                    del old_engine
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                return CommandResult(True, f"[OK] Loaded model: {model_path.name}")
            except Exception as e:
                return CommandResult(False, f"[ERROR] Failed to load: {e}")

        return CommandResult(False, "[ERROR] No engine in context")

    def model_switch(args: list[str], ctx: Dict) -> CommandResult:
        """Switch to a different model by name."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: model.switch <name>")

        name = args[0]

        # Try to load the named model
        from pathlib import Path
        model_path = Path("models") / name

        # Check various locations
        candidates = [
            model_path / "model.pt",
            Path("models") / f"{name}.pth",
            Path("models") / f"{name}.gguf",
        ]

        for candidate in candidates:
            if candidate.exists():
                return model_load([str(candidate)], ctx)

        return CommandResult(False, f"[ERROR] Model not found: {name}")

    def model_download(args: list[str], ctx: Dict) -> CommandResult:
        """Download a model from HuggingFace."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: model.download <repo_id> [filename]")

        from pathlib import Path

        repo_id = args[0]
        filename = args[1] if len(args) > 1 else None

        try:
            from enigma_engine.core.download_progress import DownloadTracker

            tracker = DownloadTracker(show_cli=True, cache_dir=Path("models"))

            if filename:
                # Download specific file (e.g., a GGUF)
                path = tracker.download_file(repo_id, filename)
            else:
                # Download entire model
                path = tracker.download_model(repo_id)

            if path:
                return CommandResult(True, f"[OK] Downloaded to: {path}", str(path))
            return CommandResult(False, "[ERROR] Download failed")

        except ImportError:
            return CommandResult(False, "[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Download failed: {e}")

    registry.register("model.info", model_info, "Show current model info", "model.info")
    registry.register("model.list", model_list, "List available models", "model.list")
    registry.register("model.load", model_load, "Load model by path", "model.load <path>")
    registry.register("model.switch", model_switch, "Switch to model by name", "model.switch <name>")
    registry.register("model.download", model_download, "Download from HuggingFace", "model.download <repo_id> [filename]")

    # ========== System Commands ==========

    def system_info(args: list[str], ctx: Dict) -> CommandResult:
        """Show system hardware info."""
        try:
            from enigma_engine.core.hardware_detection import detect_hardware
            hw = detect_hardware()
            lines = [
                "[OK] System Info:",
                f"  Device: {hw.device}",
                f"  GPU: {hw.gpu_name or 'None'}",
                f"  VRAM: {hw.gpu_vram_gb:.1f} GB" if hw.gpu_vram_gb else "  VRAM: N/A",
                f"  RAM: {hw.ram_gb:.1f} GB",
            ]
            return CommandResult(True, "\n".join(lines), hw)
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def system_clear(args: list[str], ctx: Dict) -> CommandResult:
        """Clear terminal output."""
        terminal = ctx.get("terminal")
        if terminal and hasattr(terminal, 'terminal'):
            terminal.terminal.clear()
        return CommandResult(True, "[OK] Cleared")

    registry.register("system.info", system_info, "Show hardware info", "system.info")
    registry.register("system.clear", system_clear, "Clear terminal", "system.clear")

    # ========== Clipboard Commands ==========

    def clipboard_copy(args: list[str], ctx: Dict) -> CommandResult:
        """Copy text to clipboard."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: clipboard.copy <text>")

        text = " ".join(args)

        try:
            import subprocess
            import sys
            # Use platform-native clipboard
            process = subprocess.Popen(
                ["clip"] if sys.platform == "win32" else ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.communicate(text.encode("utf-8"), timeout=HTTP_TIMEOUT_SHORT)
            preview = text[:PREVIEW_LIMIT] + "..." if len(text) > PREVIEW_LIMIT else text
            return CommandResult(True, f"[OK] Copied to clipboard: {preview}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to copy: {e}")

    registry.register("clipboard.copy", clipboard_copy, "Copy text to clipboard", "clipboard.copy <text>")

    # ========== Stop Command ==========

    def stop_cmd(args: list[str], ctx: Dict) -> CommandResult:
        """Stop current AI generation."""
        engine = ctx.get("engine")

        if engine:
            # Set cancel flag that the generation loop checks
            engine._cancel_generation = True
            return CommandResult(True, "[OK] Stop signal sent")

        return CommandResult(True, "[OK] No active generation to stop")

    registry.register("stop", stop_cmd, "Stop current generation", "stop")

    # ========== File Commands ==========

    def file_list(args: list[str], ctx: Dict) -> CommandResult:
        """List directory contents."""
        from pathlib import Path

        path = Path(args[0]) if args else Path(".")

        blocked = _check_blocked_path(str(path), ctx)
        if blocked:
            return CommandResult(False, blocked)

        if not path.exists():
            return CommandResult(False, f"[ERROR] Path not found: {path}")

        if path.is_file():
            return CommandResult(True, f"[OK] {path.name} ({path.stat().st_size} bytes)")

        items = []
        for item in sorted(path.iterdir()):
            if item.name.startswith("."):
                continue
            suffix = "/" if item.is_dir() else ""
            items.append(f"{item.name}{suffix}")

        if not items:
            return CommandResult(True, "[OK] (empty directory)")

        return CommandResult(True, f"[OK] {', '.join(items[:LIST_DISPLAY_LIMIT])}" +
                           (" ..." if len(items) > LIST_DISPLAY_LIMIT else ""), items)

    def _check_blocked_path(path_str: str, ctx: Dict) -> str | None:
        """Check if a path is blocked by config patterns. Returns error msg or None."""
        from pathlib import Path
        from fnmatch import fnmatch
        config = ctx.get("config", {})
        blocked_paths = config.get("blocked_paths", [])
        blocked_patterns = config.get("blocked_patterns", [])
        resolved = str(Path(path_str).resolve())
        name = Path(path_str).name
        for bp in blocked_paths:
            if resolved == str(Path(bp).resolve()):
                return f"[ERROR] Access denied: {path_str} is a blocked path"
        for pat in blocked_patterns:
            if fnmatch(name, pat) or fnmatch(name.lower(), pat.lower()):
                return f"[ERROR] Access denied: {path_str} matches blocked pattern '{pat}'"
        return None

    def file_read(args: list[str], ctx: Dict) -> CommandResult:
        """Read file contents."""
        from pathlib import Path

        if not args:
            return CommandResult(False, "[ERROR] Usage: file.read <path>")

        blocked = _check_blocked_path(args[0], ctx)
        if blocked:
            return CommandResult(False, blocked)

        path = Path(args[0])
        if not path.exists():
            return CommandResult(False, f"[ERROR] File not found: {path}")

        if path.is_dir():
            return CommandResult(False, f"[ERROR] {path} is a directory")

        try:
            content = path.read_text(encoding="utf-8")
            # Truncate for display
            if len(content) > CONTENT_PREVIEW:
                display = content[:CONTENT_PREVIEW] + f"\n... ({len(content)} chars total)"
            else:
                display = content
            return CommandResult(True, f"[OK] {path.name}:\n{display}", content)
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    registry.register("file.list", file_list, "List directory", "file.list [path]")
    registry.register("file.read", file_read, "Read file", "file.read <path>")

    def file_write(args: list[str], ctx: Dict) -> CommandResult:
        """Write content to a file (overwrites existing)."""
        from pathlib import Path

        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: file.write <path> <content>")

        blocked = _check_blocked_path(args[0], ctx)
        if blocked:
            return CommandResult(False, blocked)

        path = Path(args[0]).resolve()  # Resolve to absolute path
        # Join remaining args as content (handles spaces)
        content = " ".join(args[1:])

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            from enigma_engine.core.safe_save import atomic_write_text
            atomic_write_text(path, content)
            msg = f"[OK] Wrote {len(content)} chars to {path}"
            # Include path in data so GUI can provide file access options
            return CommandResult(True, msg, data={"path": str(path), "exists": True})
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to write: {e}")

    def file_append(args: list[str], ctx: Dict) -> CommandResult:
        """Append content to a file."""
        from pathlib import Path

        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: file.append <path> <content>")

        blocked = _check_blocked_path(args[0], ctx)
        if blocked:
            return CommandResult(False, blocked)

        path = Path(args[0]).resolve()  # Resolve to absolute path
        # Join remaining args as content (handles spaces)
        content = " ".join(args[1:])

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Append with newline if file exists and doesn't end with newline
            if path.exists():
                existing = path.read_text(encoding="utf-8")
                if existing and not existing.endswith("\n"):
                    content = "\n" + content

            with open(path, "a", encoding="utf-8") as f:
                f.write(content)

            msg = f"[OK] Appended {len(content)} chars to {path}"
            # Include path in data so GUI can provide file access options
            return CommandResult(True, msg, data={"path": str(path), "exists": True})
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to append: {e}")

    registry.register("file.write", file_write, "Write to file", "file.write <path> <content>")
    registry.register("file.append", file_append, "Append to file", "file.append <path> <content>")

    # ========== Memory Commands ==========

    def memory_save(args: list[str], ctx: Dict) -> CommandResult:
        """Save current conversation to memory."""
        from pathlib import Path
        if not args:
            return CommandResult(False, "[ERROR] Usage: memory.save <name>")

        # Sanitise name — prevent path traversal
        name = Path(args[0]).name
        if not name or name in (".", ".."):
            return CommandResult(False, "[ERROR] Invalid memory name")

        # Get memory directory from context or use default
        memory_dir = ctx.get("memory_dir")
        if memory_dir is None:
            memory_dir = Path("memory")

        memory_dir.mkdir(parents=True, exist_ok=True)

        # Get current conversation from context
        messages = ctx.get("chat_messages", [])
        if not messages:
            return CommandResult(False, "[ERROR] No conversation to save (empty)")

        import time

        # Save the conversation
        memory_file = memory_dir / f"{name}.json"
        data = {
            "name": name,
            "saved_at": time.time(),
            "message_count": len(messages),
            "messages": messages
        }

        try:
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(memory_file, data)
            return CommandResult(True, f"[OK] Saved conversation as '{name}' ({len(messages)} messages)")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to save: {e}")

    def memory_load(args: list[str], ctx: Dict) -> CommandResult:
        """Load a conversation from memory."""
        from pathlib import Path
        if not args:
            return CommandResult(False, "[ERROR] Usage: memory.load <name>")

        # Sanitise name — prevent path traversal
        name = Path(args[0]).name
        if not name or name in (".", ".."):
            return CommandResult(False, "[ERROR] Invalid memory name")

        # Get memory directory from context or use default
        memory_dir = ctx.get("memory_dir")
        if memory_dir is None:
            memory_dir = Path("memory")

        memory_file = memory_dir / f"{name}.json"

        if not memory_file.exists():
            return CommandResult(False, f"[ERROR] Memory not found: {name}")

        try:
            data = json.loads(memory_file.read_text(encoding="utf-8"))
            messages = data.get("messages", [])

            # Store loaded messages in context for use by chat system
            ctx["chat_messages"] = messages

            return CommandResult(True, f"[OK] Loaded '{name}' ({len(messages)} messages)", messages)
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to load: {e}")

    def memory_list(args: list[str], ctx: Dict) -> CommandResult:
        """List all saved memories."""
        # Get memory directory from context or use default
        memory_dir = ctx.get("memory_dir")
        if memory_dir is None:
            from pathlib import Path
            memory_dir = Path("memory")

        if not memory_dir.exists():
            return CommandResult(True, "[OK] No saved memories")

        memories = []
        for f in sorted(memory_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                name = f.stem
                count = data.get("message_count", len(data.get("messages", [])))
                memories.append(f"{name} ({count} messages)")
            except Exception:
                memories.append(f"{f.stem} (invalid)")

        if not memories:
            return CommandResult(True, "[OK] No saved memories")

        return CommandResult(True, f"[OK] Memories: {', '.join(memories)}", memories)

    registry.register("memory.save", memory_save, "Save conversation to memory", "memory.save <name>")
    registry.register("memory.load", memory_load, "Load conversation from memory", "memory.load <name>")
    registry.register("memory.list", memory_list, "List saved memories", "memory.list")

    # ========== Persistent Memory (AI Notes) ==========

    def memory_remember(args: list[str], ctx: Dict) -> CommandResult:
        """Save a fact to persistent memory.

        The AI can call this to remember something important.
        Facts persist across conversations and restarts.
        """
        if not args:
            return CommandResult(
                False,
                "[ERROR] Usage: memory.remember <fact to remember>")
        from .memory import get_memory
        fact = " ".join(args)
        mem = get_memory()
        if mem.add(fact):
            return CommandResult(
                True, f"[OK] Remembered: {fact}")
        return CommandResult(
            True, f"[OK] Already known: {fact}")

    def memory_forget(args: list[str], ctx: Dict) -> CommandResult:
        """Remove a fact from persistent memory."""
        if not args:
            return CommandResult(
                False,
                "[ERROR] Usage: memory.forget <fact or keyword>")
        from .memory import get_memory
        query = " ".join(args)
        mem = get_memory()
        if mem.remove(query):
            return CommandResult(True, f"[OK] Forgot: {query}")
        return CommandResult(
            False, f"[ERROR] No matching memory: {query}")

    def memory_notes(args: list[str], ctx: Dict) -> CommandResult:
        """Show all persistent memory notes."""
        from .memory import get_memory
        mem = get_memory()
        if not mem.facts:
            return CommandResult(
                True, "[OK] No persistent memories yet.")
        lines = [f"[OK] {mem.count} memories:"]
        for i, fact in enumerate(mem.facts):
            lines.append(f"  {i + 1}. {fact}")
        return CommandResult(True, "\n".join(lines), mem.facts)

    def memory_clear_notes(args: list[str], ctx: Dict) -> CommandResult:
        """Clear all persistent memory notes."""
        from .memory import get_memory
        mem = get_memory()
        mem.clear()
        return CommandResult(True, "[OK] All persistent memories cleared.")

    def memory_search(args: list[str], ctx: Dict) -> CommandResult:
        """Search for specific facts in persistent memory.

        This makes memory retrieval active instead of passive.
        The AI can search its memory when something is relevant,
        instead of seeing all facts upfront.
        """
        if not args:
            return CommandResult(
                False,
                "[ERROR] Usage: memory.search <query>")
        from .memory import get_memory
        query = " ".join(args).lower()
        mem = get_memory()
        if mem.disabled:
            return CommandResult(
                True, "[OK] Memory is disabled.")
        if not mem.facts:
            return CommandResult(
                True, "[OK] No memories to search.")

        # Find facts containing the query (case-insensitive substring match)
        matches = [f for f in mem.facts if query in f.lower()]

        if not matches:
            return CommandResult(
                True, f"[OK] No memories found matching '{query}'.")

        lines = [f"[OK] Found {len(matches)} matching memor{'y' if len(matches) == 1 else 'ies'}:"]
        for i, fact in enumerate(matches):
            lines.append(f"  {i + 1}. {fact}")
        return CommandResult(True, "\n".join(lines), matches)

    registry.register(
        "memory.remember", memory_remember,
        "Remember a fact about the user",
        "memory.remember <fact>")
    registry.register(
        "memory.forget", memory_forget,
        "Forget a fact from memory",
        "memory.forget <keyword>")
    registry.register(
        "memory.notes", memory_notes,
        "Show all remembered facts",
        "memory.notes")
    registry.register(
        "memory.clear_notes", memory_clear_notes,
        "Clear all persistent memories",
        "memory.clear_notes")
    registry.register(
        "memory.search", memory_search,
        "Search for specific facts in memory",
        "memory.search <query>")

    # ========== Emotional State Commands ==========

    def emotions_show(args: list[str], ctx: Dict) -> CommandResult:
        """Show the AI's current emotional state."""
        model_context = ctx.get("model_context")
        if model_context is None:
            return CommandResult(False, "[ERROR] No model context loaded")
        state = getattr(model_context, "emotional_state", None)
        if state is None:
            return CommandResult(False, "[ERROR] Emotional state not available")
        lines = ["[OK] Current emotional state:"]
        labels = {
            "valence": ("Valence", "-1.0 neg .. +1.0 pos"),
            "arousal": ("Arousal", "0.0 calm .. 1.0 energized"),
            "engagement": ("Engagement", "0.0 bored .. 1.0 interested"),
            "trust": ("Trust", "0.0 guarded .. 1.0 open"),
            "frustration": ("Frustration", "0.0 patient .. 1.0 frustrated"),
        }
        for key, (label, desc) in labels.items():
            val = state.get(key, 0.0)
            bar_len = 20
            if key == "valence":
                # Valence is -1 to 1, map to 0-20
                fill = int((val + 1.0) / 2.0 * bar_len)
            else:
                fill = int(val * bar_len)
            fill = max(0, min(bar_len, fill))
            bar = "#" * fill + "-" * (bar_len - fill)
            lines.append(f"  {label:12s} [{bar}] {val:+.2f}  ({desc})")
        return CommandResult(True, "\n".join(lines))

    def emotions_reset(args: list[str], ctx: Dict) -> CommandResult:
        """Reset the AI's emotional state to neutral baseline."""
        model_context = ctx.get("model_context")
        if model_context is None:
            return CommandResult(False, "[ERROR] No model context loaded")
        if hasattr(model_context, "reset_emotional_state"):
            model_context.reset_emotional_state()
            return CommandResult(True, "[OK] Emotional state reset to baseline")
        return CommandResult(False, "[ERROR] Emotional state not available")

    registry.register(
        "emotions.show", emotions_show,
        "Show the AI's current emotional state",
        "emotions.show")
    registry.register(
        "emotions.reset", emotions_reset,
        "Reset emotional state to neutral baseline",
        "emotions.reset")

    # ========== Training Commands ==========

    def train_status(args: list[str], ctx: Dict) -> CommandResult:
        """Show training status."""
        engine = ctx.get("engine")
        if engine and hasattr(engine, '_training_active') and engine._training_active:
            return CommandResult(True, "[OK] Training in progress...")
        return CommandResult(True, "[OK] No training active")

    def train_start(args: list[str], ctx: Dict) -> CommandResult:
        """Start training on data."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: train.start <data_file> [--epochs N]")

        data_file = args[0]
        epochs = 10  # default

        # Parse --epochs
        for i, arg in enumerate(args):
            if arg == "--epochs" and i + 1 < len(args):
                try:
                    epochs = int(args[i + 1])
                except ValueError:
                    pass

        # This would need async handling
        return CommandResult(True, f"[OK] Would train on {data_file} for {epochs} epochs (use Training tab)")

    def train_stop(args: list[str], ctx: Dict) -> CommandResult:
        """Stop training."""
        return CommandResult(True, "[OK] Training stopped (if any)")

    registry.register("train.status", train_status, "Show training status", "train.status")
    registry.register("train.start", train_start, "Start training", "train.start <data> [--epochs N]")
    registry.register("train.stop", train_stop, "Stop training", "train.stop")

    # ========== Mod Commands ==========

    def mod_list(args: list[str], ctx: Dict) -> CommandResult:
        """List installed mods."""
        from pathlib import Path

        mods_dir = Path("mods")
        if not mods_dir.exists():
            return CommandResult(True, "[OK] No mods directory")

        mods = []
        for mod_folder in sorted(mods_dir.iterdir()):
            if mod_folder.is_dir() and mod_folder.name != "_template":
                mod_json = mod_folder / "mod.json"
                if mod_json.exists():
                    try:
                        with open(mod_json, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        name = config.get('name', mod_folder.name)
                        mod_type = config.get('type', 'unknown')
                        mods.append(f"{name} ({mod_type})")
                    except Exception:
                        mods.append(f"{mod_folder.name} (invalid)")

        if not mods:
            return CommandResult(True, "[OK] No mods installed")

        return CommandResult(True, f"[OK] Mods: {', '.join(mods)}", mods)

    def mod_status(args: list[str], ctx: Dict) -> CommandResult:
        """Show mod router status."""
        try:
            from enigma_engine.router import get_router
            router = get_router()

            if not router.running:
                return CommandResult(True, "[OK] Router not running. Start it from Mods tab.")

            status = router.get_status()
            connected = status.get('connected_mods', 0)
            training = status.get('training', {})

            lines = [
                "[OK] Router Status:",
                f"  Running on port {status['port']}",
                f"  Connected mods: {connected}",
            ]

            if training.get('running'):
                lines.append(f"  Training: {training.get('examples_processed', 0)} examples processed")

            if connected > 0:
                for mod in status.get('mods', []):
                    lines.append(f"  - {mod['name']} ({mod['mod_id']})")

            return CommandResult(True, "\n".join(lines), status)
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def mod_start(args: list[str], ctx: Dict) -> CommandResult:
        """Start the mod router."""
        try:
            from enigma_engine.router import get_router
            router = get_router()

            if router.running:
                return CommandResult(True, "[OK] Router already running")

            if router.start():
                return CommandResult(True, f"[OK] Router started on port {router.port}")
            else:
                return CommandResult(False, "[ERROR] Failed to start router")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def mod_stop(args: list[str], ctx: Dict) -> CommandResult:
        """Stop the mod router."""
        try:
            from enigma_engine.router import get_router
            router = get_router()

            if not router.running:
                return CommandResult(True, "[OK] Router not running")

            router.stop()
            return CommandResult(True, "[OK] Router stopped")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def mod_send(args: list[str], ctx: Dict) -> CommandResult:
        """Send command to a mod."""
        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: mod.send <mod_id> <command> [args...]")

        mod_id = args[0]
        command = args[1]
        cmd_args = args[2:] if len(args) > 2 else []

        try:
            from enigma_engine.router import get_router
            router = get_router()

            if not router.running:
                return CommandResult(False, "[ERROR] Router not running")

            message = {
                'type': 'command',
                'command': command,
                'args': cmd_args
            }

            if router.send_to_mod(mod_id, message):
                return CommandResult(True, f"[OK] Sent '{command}' to {mod_id}")
            else:
                return CommandResult(False, f"[ERROR] Mod not connected: {mod_id}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    registry.register("mod.list", mod_list, "List installed mods", "mod.list")
    registry.register("mod.status", mod_status, "Show router status", "mod.status")
    registry.register("mod.start", mod_start, "Start mod router", "mod.start")
    registry.register("mod.stop", mod_stop, "Stop mod router", "mod.stop")
    registry.register("mod.send", mod_send, "Send command to mod", "mod.send <mod_id> <cmd> [args]")

    # ========== Search Commands ==========

    def search_files(args: list[str], ctx: Dict) -> CommandResult:
        """Search for files by glob pattern."""
        from pathlib import Path
        import glob

        if not args:
            return CommandResult(False, "[ERROR] Usage: search.files <pattern>")

        pattern = args[0]

        try:
            matches = glob.glob(pattern, recursive=True)

            if not matches:
                return CommandResult(True, f"[OK] No files matching: {pattern}")

            # Limit results for display
            display_matches = matches[:LIST_DISPLAY_LIMIT]
            names = [Path(m).name for m in display_matches]
            extra = f" ... (+{len(matches) - LIST_DISPLAY_LIMIT} more)" if len(matches) > LIST_DISPLAY_LIMIT else ""

            return CommandResult(True, f"[OK] Found {len(matches)} files: {', '.join(names)}{extra}", matches)
        except Exception as e:
            return CommandResult(False, f"[ERROR] Search failed: {e}")

    def search_content(args: list[str], ctx: Dict) -> CommandResult:
        """Search for text inside files."""
        from pathlib import Path

        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: search.content <directory> <query>")

        directory = Path(args[0])
        query = " ".join(args[1:])  # Allow queries with spaces

        if not directory.exists():
            return CommandResult(False, f"[ERROR] Directory not found: {directory}")

        try:
            matches = []
            # Search in common text files
            extensions = ["*.txt", "*.py", "*.md", "*.json", "*.yaml", "*.yml", "*.toml"]

            for ext in extensions:
                for filepath in directory.rglob(ext):
                    try:
                        content = filepath.read_text(encoding="utf-8", errors="ignore")
                        if query.lower() in content.lower():
                            matches.append(filepath.name)
                    except Exception:
                        continue

            if not matches:
                return CommandResult(True, f"[OK] No files contain: {query}")

            # Deduplicate and limit
            unique_matches = list(set(matches))[:LIST_DISPLAY_LIMIT]
            return CommandResult(True, f"[OK] Found in {len(matches)} files: {', '.join(unique_matches)}", unique_matches)
        except Exception as e:
            return CommandResult(False, f"[ERROR] Search failed: {e}")

    registry.register("search.files", search_files, "Find files by pattern", "search.files <pattern>")
    registry.register("search.content", search_content, "Search text in files", "search.content <dir> <query>")

    # ========== Web Commands ==========

    def search_web(args: list[str], ctx: Dict) -> CommandResult:
        """Search the web and return results."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: search.web <query>")

        query = " ".join(args)

        try:
            from enigma_engine.core.web_utils import ddg_search

            results = ddg_search(query, max_results=SEARCH_MAX_RESULTS)
            if not results:
                return CommandResult(True, f"[OK] No results for: {query}")

            lines = [f"[OK] Web search: {query}"]
            for i, r in enumerate(results, 1):
                lines.append(f"\n{i}. {r['title']}")
                snippet = r.get("snippet", "")
                lines.append(f"   {snippet[:SNIPPET_LIMIT]}...")

            return CommandResult(True, "\n".join(lines), results)

        except ImportError:
            return CommandResult(False, "[ERROR] requests library not installed. Run: pip install requests")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def web_fetch(args: list[str], ctx: Dict) -> CommandResult:
        """Fetch content from a URL."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: web.fetch <url>")

        url = args[0]

        try:
            import requests
            from enigma_engine.core.web_utils import (
                fetch_page_text, _validate_url)

            content_type = ""
            try:
                _validate_url(url)
                resp = requests.head(
                    url, headers={"User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; "
                        "x64) AppleWebKit/537.36")},
                    timeout=HTTP_TIMEOUT_DEFAULT, allow_redirects=True)
                content_type = resp.headers.get(
                    "Content-Type", "")
            except Exception:
                pass

            if not content_type or "text/html" in content_type:
                content = fetch_page_text(url, max_chars=FETCH_MAX_CHARS)
                if not content:
                    return CommandResult(
                        False, f"[ERROR] No content at: {url}")
                return CommandResult(
                    True,
                    f"[OK] Fetched {url}:\n\n{content}",
                    content)
            else:
                # Non-HTML — fetch raw text (validate to prevent SSRF)
                _validate_url(url)
                resp = requests.get(url, headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; "
                        "Win64; x64) AppleWebKit/537.36")},
                    timeout=HTTP_TIMEOUT_FETCH)
                resp.raise_for_status()
                content = resp.text[:FETCH_MAX_CHARS]
                return CommandResult(
                    True,
                    f"[OK] Fetched {url}:\n\n{content}",
                    content)

        except ImportError:
            return CommandResult(False, "[ERROR] requests library not installed")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to fetch: {e}")

    registry.register("search.web", search_web, "Search the web", "search.web <query>")

    def search_images(args: list[str], ctx: Dict) -> CommandResult:
        """Search the web for images and return URLs."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: search.images <query>")

        query = " ".join(args)

        try:
            from enigma_engine.core.web_utils import ddg_image_search

            results = ddg_image_search(query, max_results=SEARCH_MAX_RESULTS)
            if not results:
                return CommandResult(True, f"[OK] No image results for: {query}")

            lines = [f"[OK] Image search: {query}"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "")[:SNIPPET_LIMIT]
                url = r.get("url", "")
                lines.append(f"\n{i}. {title}")
                lines.append(f"   ![{title}]({url})")

            return CommandResult(True, "\n".join(lines), results)

        except ImportError:
            return CommandResult(False, "[ERROR] requests library not installed. Run: pip install requests")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    registry.register("search.images", search_images, "Search for images", "search.images <query>")
    registry.register("web.fetch", web_fetch, "Fetch URL content", "web.fetch <url>")

    # ========== Note Commands ==========

    def note_add(args: list[str], ctx: Dict) -> CommandResult:
        """Add a quick note."""
        from pathlib import Path
        from datetime import datetime

        if not args:
            return CommandResult(False, "[ERROR] Usage: note.add <text>")

        text = " ".join(args)

        # Get notes directory from context or use default
        notes_dir = ctx.get("notes_dir")
        if notes_dir is None:
            notes_dir = Path("data/notes")

        notes_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped note file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_file = notes_dir / f"note_{timestamp}.txt"

        try:
            from enigma_engine.core.safe_save import atomic_write_text
            atomic_write_text(note_file, text)
            return CommandResult(True, f"[OK] Note saved: {note_file.name}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to save note: {e}")

    def note_list(args: list[str], ctx: Dict) -> CommandResult:
        """List recent notes."""
        from pathlib import Path

        # Get notes directory from context or use default
        notes_dir = ctx.get("notes_dir")
        if notes_dir is None:
            notes_dir = Path("data/notes")

        if not notes_dir.exists():
            return CommandResult(True, "[OK] No notes yet")

        try:
            notes = sorted(notes_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)

            if not notes:
                return CommandResult(True, "[OK] No notes yet")

            # Show last 10 notes with previews
            lines = ["[OK] Recent notes:"]
            for note in notes[:NOTES_DISPLAY_LIMIT]:
                try:
                    content = note.read_text(encoding="utf-8")
                    preview = content[:PREVIEW_LIMIT].replace("\n", " ")
                    if len(content) > PREVIEW_LIMIT:
                        preview += "..."
                    lines.append(f"  {note.name}: {preview}")
                except Exception:
                    lines.append(f"  {note.name}: (unreadable)")

            return CommandResult(True, "\n".join(lines), [n.name for n in notes[:NOTES_DISPLAY_LIMIT]])
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to list notes: {e}")

    registry.register("note.add", note_add, "Add a quick note", "note.add <text>")
    registry.register("note.list", note_list, "List recent notes", "note.list")

    # ========== History Command ==========

    def history_cmd(args: list[str], ctx: Dict) -> CommandResult:
        """Show command history."""
        registry = ctx.get("registry")
        if not registry:
            return CommandResult(True, "[OK] No history available")

        count = HISTORY_DEFAULT_COUNT
        if args:
            try:
                count = int(args[0])
            except ValueError:
                pass

        history = registry.get_history(count)

        if not history:
            return CommandResult(True, "[OK] No commands in history")

        lines = [f"[OK] Last {len(history)} commands:"]
        for i, cmd in enumerate(history, 1):
            lines.append(f"  {i}. {cmd}")

        return CommandResult(True, "\n".join(lines), history)

    registry.register("history", history_cmd, "Show command history", "history [count]")

    # ========== Training Data Commands ==========

    def train_data_add(args: list[str], ctx: Dict) -> CommandResult:
        """Add a training example to training.txt."""
        from pathlib import Path

        if not args:
            return CommandResult(False, "[ERROR] Usage: train.data.add <Q: question A: answer>")

        text = " ".join(args)

        # Get data directory from context or use default
        data_dir = ctx.get("data_dir")
        if data_dir is None:
            data_dir = Path("data")

        data_dir.mkdir(parents=True, exist_ok=True)
        training_file = data_dir / "training.txt"

        try:
            # Append to training file
            with open(training_file, "a", encoding="utf-8") as f:
                f.write(f"\n{text}\n")

            return CommandResult(True, f"[OK] Added training example to {training_file.name}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to add training data: {e}")

    def train_data_list(args: list[str], ctx: Dict) -> CommandResult:
        """List training data entries."""
        from pathlib import Path

        # Get data directory from context or use default
        data_dir = ctx.get("data_dir")
        if data_dir is None:
            data_dir = Path("data")

        training_file = data_dir / "training.txt"

        if not training_file.exists():
            return CommandResult(True, "[OK] No training data file yet")

        try:
            content = training_file.read_text(encoding="utf-8")
            lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]

            if not lines:
                return CommandResult(True, "[OK] Training file is empty")

            # Count Q/A pairs roughly
            q_count = sum(1 for l in lines if l.startswith("Q:"))

            # Show preview
            preview = "\n".join(lines[:CONFIG_LIST_LIMIT])
            if len(lines) > CONFIG_LIST_LIMIT:
                preview += f"\n... ({len(lines)} total lines, ~{q_count} Q&A pairs)"

            return CommandResult(True, f"[OK] Training data:\n{preview}", lines)
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to read training data: {e}")

    registry.register("train.data.add", train_data_add, "Add training example", "train.data.add <text>")
    registry.register("train.data.list", train_data_list, "List training data", "train.data.list")

    # ========== Shell Command ==========

    def shell_cmd(args: list[str], ctx: Dict) -> CommandResult:
        """Run a terminal command (restricted to safe commands)."""
        import shlex
        import subprocess

        if not args:
            return CommandResult(False, "[ERROR] Usage: shell <command>")

        # Join all args as the command (handles spaces)
        command = " ".join(args)

        # Safety: Only allow whitelisted commands
        # Extract the base command (first word)
        base_cmd = args[0].lower().strip()

        ALLOWED_COMMANDS = {
            "python", "pip", "dir", "ls", "cat", "type", "echo",
            "git", "cd", "pwd", "whoami", "hostname", "date",
            "find", "grep", "head", "tail", "wc", "sort",
            "pytest", "where", "which", "env", "set",
        }

        BLOCKED_PATTERNS = [
            "rm -rf", "del /f", "format", "mkfs", "dd if=",
            "shutdown", "reboot", "taskkill", "kill -9",
            ":(){", "fork", ">(", "curl | sh", "wget | sh",
        ]

        if base_cmd not in ALLOWED_COMMANDS:
            return CommandResult(
                False,
                f"[ERROR] Command '{base_cmd}' not in allowed list. "
                f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        # Check for dangerous patterns within allowed commands
        command_lower = command.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in command_lower:
                return CommandResult(False, f"[ERROR] Blocked dangerous pattern: {pattern}")

        try:
            # shell=False — args list prevents shell injection
            cmd_list = shlex.split(command)
            result = subprocess.run(
                cmd_list,
                shell=False,
                capture_output=True,
                text=True,
                timeout=HTTP_TIMEOUT_LONG,
                cwd=ctx.get("cwd", None)
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    # Truncate long output
                    if len(output) > OUTPUT_LIMIT:
                        output = output[:OUTPUT_LIMIT] + f"\n... ({len(output)} chars total)"
                    return CommandResult(True, f"[OK] {output}", {"stdout": output, "returncode": 0})
                return CommandResult(True, "[OK] Command completed (no output)")
            else:
                msg = error if error else f"Exit code: {result.returncode}"
                return CommandResult(False, f"[ERROR] {msg}", {"stderr": error, "returncode": result.returncode})

        except subprocess.TimeoutExpired:
            return CommandResult(False, "[ERROR] Command timed out (30s limit)")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to run command: {e}")

    registry.register("shell", shell_cmd, "Run terminal command", "shell <command>")

    # ========== Code Sandbox ==========

    def code_run(args: list[str], ctx: Dict) -> CommandResult:
        """Execute Python code in a sandboxed subprocess.

        The code runs in a fresh Python process with a 30-second
        timeout and restricted filesystem access (outputs/ only).
        """
        import subprocess
        import sys
        import tempfile
        import textwrap
        from pathlib import Path

        if not args:
            return CommandResult(
                False,
                "[ERROR] Usage: code.run <python_code>")

        code = " ".join(args)
        # Unwrap code blocks that the AI may wrap in triple backticks
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first and last lines (``` markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        # Safety: disallow dangerous operations
        forbidden = [
            "shutil.rmtree", "os.remove", "os.rmdir", "os.unlink",
            "os.system", "os.popen",
            "os.exec", "os.spawn",
            "os.open(", "os.fdopen(", "os.rename(", "os.replace(",
            "__import__(", "subprocess.call",
            "subprocess.Popen", "subprocess.run",
            "subprocess.check_call", "subprocess.check_output",
            "subprocess.getoutput", "subprocess.getstatusoutput",
            "exec(", "eval(",
            "importlib", "compile(",
            "ctypes", "sys.modules",
        ]
        for pattern in forbidden:
            if pattern in code:
                return CommandResult(
                    False,
                    f"[ERROR] Forbidden operation: {pattern}")

        # Write code to a temp file and execute
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
                encoding="utf-8"
            ) as f:
                # Prepend safety wrapper
                wrapper = textwrap.dedent("""\
                    import sys
                    import os
                    # Restrict write access to outputs/ directory
                    _original_open = open
                    def _safe_open(path, mode='r', *a, **kw):
                        if any(m in str(mode) for m in ('w', 'a', 'x')):
                            from pathlib import Path
                            p = Path(path).resolve()
                            outputs = Path('outputs').resolve()
                            try:
                                p.relative_to(outputs)
                            except ValueError:
                                raise PermissionError(
                                    f"Write access restricted to outputs/ only")
                        return _original_open(path, mode, *a, **kw)
                    import builtins
                    builtins.open = _safe_open
                """)
                f.write(wrapper + "\n" + code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=HTTP_TIMEOUT_LONG,
                cwd=str(Path(".")),
            )

            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    if len(output) > FETCH_MAX_CHARS:
                        output = (
                            output[:FETCH_MAX_CHARS]
                            + f"\n... ({len(output)} chars total)")
                    return CommandResult(
                        True, f"[OK]\n{output}",
                        {"stdout": output, "returncode": 0})
                return CommandResult(
                    True, "[OK] Code executed (no output)")
            else:
                # Show last ERROR_TAIL chars of error
                if len(error) > ERROR_TAIL:
                    error = error[-ERROR_TAIL:]
                return CommandResult(
                    False, f"[ERROR]\n{error}",
                    {"stderr": error, "returncode": result.returncode})

        except subprocess.TimeoutExpired:
            try:
                Path(tmp_path).unlink()
            except (OSError, NameError):
                pass
            return CommandResult(
                False, "[ERROR] Code execution timed out (30s limit)")
        except Exception as e:
            try:
                Path(tmp_path).unlink()
            except (OSError, NameError):
                pass
            return CommandResult(
                False, f"[ERROR] Failed to execute code: {e}")

    registry.register(
        "code.run", code_run,
        "Execute Python code in a sandboxed subprocess",
        "code.run <python_code>")

    # ========== Image Generation Commands ==========

    def imagegen_generate(args: list[str], ctx: Dict) -> CommandResult:
        """Generate an image from a text prompt.

        Detects available backends automatically:
        1. Stable Diffusion WebUI (AUTOMATIC1111) on port 7860
        2. ComfyUI on port 8188
        3. Local diffusers library (requires GPU)

        If no backend is running, falls back to the imagegen mod
        via the router (if connected).
        """
        if not args:
            return CommandResult(
                False,
                "[ERROR] Usage: imagegen.generate <prompt> "
                "[--width N] [--height N] [--steps N] [--seed N] "
                "[--negative <text>]")

        from pathlib import Path as _Path
        from datetime import datetime as _dt

        # Parse args: everything before -- flags is the prompt
        prompt_parts: list[str] = []
        width = 512
        height = 512
        steps = 20
        seed = -1
        negative = ""
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--width" and i + 1 < len(args):
                try:
                    width = int(args[i + 1])
                except ValueError:
                    return CommandResult(False, f"[ERROR] --width must be a number, got '{args[i + 1]}'")
                i += 2; continue
            elif a == "--height" and i + 1 < len(args):
                try:
                    height = int(args[i + 1])
                except ValueError:
                    return CommandResult(False, f"[ERROR] --height must be a number, got '{args[i + 1]}'")
                i += 2; continue
            elif a == "--steps" and i + 1 < len(args):
                try:
                    steps = int(args[i + 1])
                except ValueError:
                    return CommandResult(False, f"[ERROR] --steps must be a number, got '{args[i + 1]}'")
                i += 2; continue
            elif a == "--seed" and i + 1 < len(args):
                try:
                    seed = int(args[i + 1])
                except ValueError:
                    return CommandResult(False, f"[ERROR] --seed must be a number, got '{args[i + 1]}'")
                i += 2; continue
            elif a == "--negative" and i + 1 < len(args):
                neg_parts: list[str] = []
                i += 1
                while i < len(args) and not args[i].startswith("--"):
                    neg_parts.append(args[i]); i += 1
                negative = " ".join(neg_parts); continue
            else:
                prompt_parts.append(a); i += 1

        prompt = " ".join(prompt_parts)
        if not prompt:
            return CommandResult(
                False, "[ERROR] Prompt text is required")

        output_dir = _Path("outputs/images")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try backends in order
        generated_path = None
        backend_used = None

        # --- Backend 1: Stable Diffusion WebUI ---
        try:
            import requests
            url = "http://127.0.0.1:7860"
            r = requests.get(
                f"{url}/sdapi/v1/sd-models", timeout=HTTP_TIMEOUT_HEALTH)
            if r.status_code == 200:
                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg_scale": 7.5,
                    "seed": seed,
                }
                resp = requests.post(
                    f"{url}/sdapi/v1/txt2img",
                    json=payload, timeout=HTTP_TIMEOUT_GENERATE)
                if resp.status_code == 200:
                    import base64
                    data = resp.json()
                    images = data.get("images", [])
                    if images:
                        img_bytes = base64.b64decode(images[0])
                        timestamp = _dt.now().strftime(
                            "%Y%m%d_%H%M%S")
                        filename = f"img_{timestamp}.png"
                        saved = output_dir / filename
                        saved.write_bytes(img_bytes)
                        generated_path = str(saved)
                        backend_used = "SD WebUI"
        except Exception:
            pass

        # --- Backend 2: ComfyUI ---
        if generated_path is None:
            try:
                import requests
                url = "http://127.0.0.1:8188"
                r = requests.get(
                    f"{url}/system_stats", timeout=HTTP_TIMEOUT_HEALTH)
                if r.status_code == 200:
                    # ComfyUI needs a workflow — use simple
                    # txt2img with polling
                    import time as _time
                    workflow = {
                        "3": {
                            "class_type": "KSampler",
                            "inputs": {
                                "cfg": 7.5,
                                "denoise": 1,
                                "latent_image": ["5", 0],
                                "model": ["4", 0],
                                "negative": ["7", 0],
                                "positive": ["6", 0],
                                "sampler_name": "euler",
                                "scheduler": "normal",
                                "seed": seed if seed > 0 else 42,
                                "steps": steps,
                            },
                        },
                        "4": {
                            "class_type":
                                "CheckpointLoaderSimple",
                            "inputs": {
                                "ckpt_name":
                                    "v1-5-pruned-emaonly"
                                    ".safetensors",
                            },
                        },
                        "5": {
                            "class_type": "EmptyLatentImage",
                            "inputs": {
                                "batch_size": 1,
                                "height": height,
                                "width": width,
                            },
                        },
                        "6": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {
                                "clip": ["4", 1],
                                "text": prompt,
                            },
                        },
                        "7": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {
                                "clip": ["4", 1],
                                "text": negative,
                            },
                        },
                        "8": {
                            "class_type": "VAEDecode",
                            "inputs": {
                                "samples": ["3", 0],
                                "vae": ["4", 2],
                            },
                        },
                        "9": {
                            "class_type": "SaveImage",
                            "inputs": {
                                "filename_prefix": "enigma",
                                "images": ["8", 0],
                            },
                        },
                    }
                    resp = requests.post(
                        f"{url}/prompt",
                        json={"prompt": workflow},
                        timeout=HTTP_TIMEOUT_GENERATE)
                    if resp.status_code == 200:
                        prompt_id = resp.json().get(
                            "prompt_id")
                        # Poll for completion
                        for _ in range(POLL_ITERATIONS):
                            _time.sleep(POLL_INTERVAL)
                            hist = requests.get(
                                f"{url}/history/{prompt_id}",
                                timeout=HTTP_TIMEOUT_SHORT)
                            if hist.status_code == 200:
                                hd = hist.json()
                                if prompt_id in hd:
                                    outs = hd[prompt_id].get(
                                        "outputs", {})
                                    if ("9" in outs
                                            and outs["9"].get(
                                                "images")):
                                        img_info = (
                                            outs["9"]["images"][0])
                                        fname = img_info.get(
                                            "filename", "")
                                        # Fetch image data
                                        img_resp = requests.get(
                                            f"{url}/view?"
                                            f"filename={fname}",
                                            timeout=HTTP_TIMEOUT_DEFAULT)
                                        if (img_resp.status_code
                                                == 200):
                                            timestamp = (
                                                _dt.now().strftime(
                                                    "%Y%m%d_%H%M%S"))
                                            saved = (
                                                output_dir
                                                / f"img_{timestamp}"
                                                  ".png")
                                            saved.write_bytes(
                                                img_resp.content)
                                            generated_path = str(
                                                saved)
                                            backend_used = "ComfyUI"
                                        break
            except Exception:
                pass

        # --- Backend 3: Local diffusers ---
        if generated_path is None:
            try:
                import torch
                from diffusers import StableDiffusionPipeline

                device = ("cuda" if torch.cuda.is_available()
                          else "cpu")
                if device == "cuda" and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                elif device == "cuda":
                    dtype = torch.float16
                else:
                    dtype = torch.float32
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=dtype)
                pipe = pipe.to(device)

                generator = None
                if seed > 0:
                    generator = torch.Generator(
                        device).manual_seed(seed)

                image = pipe(
                    prompt,
                    negative_prompt=negative or None,
                    width=width, height=height,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]

                timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img_{timestamp}.png"
                saved = output_dir / filename
                image.save(str(saved))
                generated_path = str(saved)
                backend_used = "diffusers (local)"

                # Free VRAM
                del pipe
                torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception:
                pass

        # --- Fallback: mod router ---
        if generated_path is None:
            try:
                from enigma_engine.router import get_router
                router = get_router()
                if router.running:
                    message = {
                        "type": "command",
                        "command": "generate",
                        "args": {
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "steps": steps,
                            "seed": seed,
                            "negative_prompt": negative,
                        },
                    }
                    if router.send_to_mod("imagegen", message):
                        # Mod saves to outputs/images/ — poll
                        import time as _time
                        import glob
                        before = set(glob.glob(
                            str(output_dir / "img_*.png")))
                        for _ in range(POLL_ITERATIONS // 2):
                            _time.sleep(POLL_INTERVAL)
                            after = set(glob.glob(
                                str(output_dir / "img_*.png")))
                            new = after - before
                            if new:
                                generated_path = max(
                                    new, key=lambda p: (
                                        _Path(p).stat().st_mtime))
                                backend_used = "imagegen mod"
                                break
            except Exception:
                pass

        if generated_path is None:
            return CommandResult(
                False,
                "[ERROR] No image generation backend available. "
                "Start Stable Diffusion WebUI (port 7860), "
                "ComfyUI (port 8188), install diffusers, "
                "or start the imagegen mod.")

        return CommandResult(
            True,
            f"[OK] Image generated via {backend_used}: "
            f"{generated_path}",
            {"path": generated_path, "backend": backend_used})

    def imagegen_status(args: list[str], ctx: Dict) -> CommandResult:
        """Check which image generation backends are available."""
        backends: list[str] = []

        try:
            import requests
            try:
                r = requests.get(
                    "http://127.0.0.1:7860/sdapi/v1/sd-models",
                    timeout=HTTP_TIMEOUT_HEALTH)
                if r.status_code == 200:
                    count = len(r.json())
                    backends.append(
                        f"SD WebUI (port 7860) — "
                        f"{count} model(s)")
            except Exception:
                pass

            try:
                r = requests.get(
                    "http://127.0.0.1:8188/system_stats",
                    timeout=HTTP_TIMEOUT_HEALTH)
                if r.status_code == 200:
                    backends.append("ComfyUI (port 8188)")
            except Exception:
                pass
        except ImportError:
            pass

        try:
            from diffusers import StableDiffusionPipeline  # noqa: F401
            backends.append("diffusers (local)")
        except ImportError:
            pass

        # Check imagegen mod
        try:
            from enigma_engine.router import get_router
            router = get_router()
            if router.running:
                mods = router.get_connected_mods()
                for m in mods:
                    if m["mod_id"] == "imagegen":
                        backends.append("imagegen mod (connected)")
                        break
        except Exception:
            pass

        if not backends:
            return CommandResult(
                True,
                "[OK] No image generation backends detected.\n"
                "Options:\n"
                "  • Start Stable Diffusion WebUI on port 7860\n"
                "  • Start ComfyUI on port 8188\n"
                "  • Install diffusers: pip install diffusers\n"
                "  • Start the imagegen mod")

        lines = ["[OK] Available backends:"]
        for b in backends:
            lines.append(f"  • {b}")
        return CommandResult(True, "\n".join(lines))

    registry.register(
        "imagegen.generate", imagegen_generate,
        "Generate image from text prompt",
        "imagegen.generate <prompt> [--width N] [--height N] "
        "[--steps N] [--seed N] [--negative <text>]")
    registry.register(
        "imagegen.status", imagegen_status,
        "Check available image generation backends",
        "imagegen.status")

    # ========== Help Command ==========

    def help_cmd(args: list[str], ctx: Dict) -> CommandResult:
        """Show help."""
        registry = ctx.get("registry")
        if registry:
            if args:
                return CommandResult(True, registry.get_help(args[0]))
            return CommandResult(True, registry.get_help())
        return CommandResult(True, "[OK] Type 'help' for commands")

    registry.register("help", help_cmd, "Show help", "help [command]")
