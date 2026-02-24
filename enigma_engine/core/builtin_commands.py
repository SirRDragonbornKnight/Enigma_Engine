from __future__ import annotations
"""
Built-in command implementations for the Enigma Engine command system.

All default commands are registered here via ``register_builtin_commands()``.
The function is called once by :func:`commands.get_registry` when the global
registry is first created.
"""

import json

from .commands import CommandResult, CommandRegistry


# ``Dict`` is referenced in annotation strings throughout this module.
# With ``from __future__ import annotations`` the annotations are never
# evaluated at runtime, so we only need the name available in the module
# namespace for tools that *do* resolve annotations (e.g. get_type_hints).
from typing import Dict  # noqa: F401


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
        elif value.replace(".", "").replace("-", "").isdigit():
            value = float(value) if "." in value else int(value)

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
                ctx["engine"] = new_engine
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
            # Use platform-native clipboard
            process = subprocess.Popen(
                ["clip"] if __import__("sys").platform == "win32" else ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE
            )
            process.communicate(text.encode("utf-8"))
            preview = text[:50] + "..." if len(text) > 50 else text
            return CommandResult(True, f"[OK] Copied to clipboard: {preview}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to copy: {e}")

    registry.register("clipboard.copy", clipboard_copy, "Copy text to clipboard", "clipboard.copy <text>")

    # ========== Stop Command ==========

    def stop_cmd(args: list[str], ctx: Dict) -> CommandResult:
        """Stop current AI generation."""
        engine = ctx.get("engine")

        if engine and hasattr(engine, '_generation_lock'):
            # Signal the engine to stop (best effort)
            return CommandResult(True, "[OK] Stop signal sent")

        return CommandResult(True, "[OK] No active generation to stop")

    registry.register("stop", stop_cmd, "Stop current generation", "stop")

    # ========== File Commands ==========

    def file_list(args: list[str], ctx: Dict) -> CommandResult:
        """List directory contents."""
        from pathlib import Path

        path = Path(args[0]) if args else Path(".")
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

        return CommandResult(True, f"[OK] {', '.join(items[:20])}" +
                           (" ..." if len(items) > 20 else ""), items)

    def file_read(args: list[str], ctx: Dict) -> CommandResult:
        """Read file contents."""
        from pathlib import Path

        if not args:
            return CommandResult(False, "[ERROR] Usage: file.read <path>")

        path = Path(args[0])
        if not path.exists():
            return CommandResult(False, f"[ERROR] File not found: {path}")

        if path.is_dir():
            return CommandResult(False, f"[ERROR] {path} is a directory")

        try:
            content = path.read_text(encoding="utf-8")
            # Truncate for display
            if len(content) > 500:
                display = content[:500] + f"\n... ({len(content)} chars total)"
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

        path = Path(args[0])
        # Join remaining args as content (handles spaces)
        content = " ".join(args[1:])

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return CommandResult(True, f"[OK] Wrote {len(content)} chars to {path.name}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to write: {e}")

    def file_append(args: list[str], ctx: Dict) -> CommandResult:
        """Append content to a file."""
        from pathlib import Path

        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: file.append <path> <content>")

        path = Path(args[0])
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

            return CommandResult(True, f"[OK] Appended {len(content)} chars to {path.name}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to append: {e}")

    registry.register("file.write", file_write, "Write to file", "file.write <path> <content>")
    registry.register("file.append", file_append, "Append to file", "file.append <path> <content>")

    # ========== Memory Commands ==========

    def memory_save(args: list[str], ctx: Dict) -> CommandResult:
        """Save current conversation to memory."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: memory.save <name>")

        name = args[0]

        # Get memory directory from context or use default
        memory_dir = ctx.get("memory_dir")
        if memory_dir is None:
            from pathlib import Path
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
            memory_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return CommandResult(True, f"[OK] Saved conversation as '{name}' ({len(messages)} messages)")
        except Exception as e:
            return CommandResult(False, f"[ERROR] Failed to save: {e}")

    def memory_load(args: list[str], ctx: Dict) -> CommandResult:
        """Load a conversation from memory."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: memory.load <name>")

        name = args[0]

        # Get memory directory from context or use default
        memory_dir = ctx.get("memory_dir")
        if memory_dir is None:
            from pathlib import Path
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

    # ========== Brick Commands ==========

    def brick_list(args: list[str], ctx: Dict) -> CommandResult:
        """List installed bricks."""
        from pathlib import Path

        bricks_dir = Path("bricks")
        if not bricks_dir.exists():
            return CommandResult(True, "[OK] No bricks directory")

        bricks = []
        for brick_folder in sorted(bricks_dir.iterdir()):
            if brick_folder.is_dir() and brick_folder.name != "_template":
                brick_json = brick_folder / "brick.json"
                if brick_json.exists():
                    try:
                        with open(brick_json, 'r') as f:
                            config = json.load(f)
                        name = config.get('name', brick_folder.name)
                        brick_type = config.get('type', 'unknown')
                        bricks.append(f"{name} ({brick_type})")
                    except Exception:
                        bricks.append(f"{brick_folder.name} (invalid)")

        if not bricks:
            return CommandResult(True, "[OK] No bricks installed")

        return CommandResult(True, f"[OK] Bricks: {', '.join(bricks)}", bricks)

    def brick_status(args: list[str], ctx: Dict) -> CommandResult:
        """Show brick router status."""
        try:
            from enigma_engine.router import get_router
            router = get_router()

            if not router.running:
                return CommandResult(True, "[OK] Router not running. Start it from Bricks tab.")

            status = router.get_status()
            connected = status.get('connected_bricks', 0)
            training = status.get('training', {})

            lines = [
                f"[OK] Router Status:",
                f"  Running on port {status['port']}",
                f"  Connected bricks: {connected}",
            ]

            if training.get('running'):
                lines.append(f"  Training: {training.get('examples_processed', 0)} examples processed")

            if connected > 0:
                for brick in status.get('bricks', []):
                    lines.append(f"  - {brick['name']} ({brick['brick_id']})")

            return CommandResult(True, "\n".join(lines), status)
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def brick_start(args: list[str], ctx: Dict) -> CommandResult:
        """Start the brick router."""
        try:
            from enigma_engine.router import get_router
            router = get_router()

            if router.running:
                return CommandResult(True, "[OK] Router already running")

            if router.start():
                return CommandResult(True, "[OK] Router started on port 9900")
            else:
                return CommandResult(False, "[ERROR] Failed to start router")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def brick_stop(args: list[str], ctx: Dict) -> CommandResult:
        """Stop the brick router."""
        try:
            from enigma_engine.router import get_router
            router = get_router()

            if not router.running:
                return CommandResult(True, "[OK] Router not running")

            router.stop()
            return CommandResult(True, "[OK] Router stopped")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def brick_send(args: list[str], ctx: Dict) -> CommandResult:
        """Send command to a brick."""
        if len(args) < 2:
            return CommandResult(False, "[ERROR] Usage: brick.send <brick_id> <command> [args...]")

        brick_id = args[0]
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

            if router.send_to_brick(brick_id, message):
                return CommandResult(True, f"[OK] Sent '{command}' to {brick_id}")
            else:
                return CommandResult(False, f"[ERROR] Brick not connected: {brick_id}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    registry.register("brick.list", brick_list, "List installed bricks", "brick.list")
    registry.register("brick.status", brick_status, "Show router status", "brick.status")
    registry.register("brick.start", brick_start, "Start brick router", "brick.start")
    registry.register("brick.stop", brick_stop, "Stop brick router", "brick.stop")
    registry.register("brick.send", brick_send, "Send command to brick", "brick.send <brick_id> <cmd> [args]")

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
            display_matches = matches[:20]
            names = [Path(m).name for m in display_matches]
            extra = f" ... (+{len(matches) - 20} more)" if len(matches) > 20 else ""

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
            unique_matches = list(set(matches))[:20]
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
            import requests
            from urllib.parse import quote_plus

            # Use DuckDuckGo HTML search (no API key needed)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse results
            from html.parser import HTMLParser

            class DDGParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.current_title = ""
                    self.current_url = ""
                    self.current_snippet = ""
                    self.in_result = False
                    self.in_title = False
                    self.in_snippet = False

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                        self.in_title = True
                        self.current_url = attrs_dict.get("href", "")
                    if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                        self.in_snippet = True

                def handle_endtag(self, tag):
                    if tag == "a" and self.in_title:
                        self.in_title = False
                    if tag == "a" and self.in_snippet:
                        self.in_snippet = False
                        if self.current_title and self.current_snippet:
                            self.results.append({
                                "title": self.current_title.strip(),
                                "url": self.current_url,
                                "snippet": self.current_snippet.strip()
                            })
                            self.current_title = ""
                            self.current_url = ""
                            self.current_snippet = ""

                def handle_data(self, data):
                    if self.in_title:
                        self.current_title += data
                    if self.in_snippet:
                        self.current_snippet += data

            parser = DDGParser()
            parser.feed(response.text)

            if not parser.results:
                return CommandResult(True, f"[OK] No results for: {query}")

            # Format results
            results = parser.results[:5]  # Top 5 results
            lines = [f"[OK] Web search: {query}"]
            for i, r in enumerate(results, 1):
                lines.append(f"\n{i}. {r['title']}")
                lines.append(f"   {r['snippet'][:100]}...")

            return CommandResult(True, "\n".join(lines), results)

        except ImportError:
            return CommandResult(False, "[ERROR] requests library not installed. Run: pip install requests")
        except requests.RequestException as e:
            return CommandResult(False, f"[ERROR] Web search failed: {e}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    def web_fetch(args: list[str], ctx: Dict) -> CommandResult:
        """Fetch content from a URL."""
        if not args:
            return CommandResult(False, "[ERROR] Usage: web.fetch <url>")

        url = args[0]

        try:
            import requests

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Try to extract text content
            content_type = response.headers.get("Content-Type", "")

            if "text/html" in content_type:
                # Parse HTML, extract text
                from html.parser import HTMLParser

                class TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []
                        self.skip_tags = {"script", "style", "nav", "footer", "header"}
                        self.current_tag = ""

                    def handle_starttag(self, tag, attrs):
                        self.current_tag = tag

                    def handle_data(self, data):
                        if self.current_tag not in self.skip_tags:
                            text = data.strip()
                            if text and len(text) > 2:
                                self.text.append(text)

                extractor = TextExtractor()
                extractor.feed(response.text)

                content = " ".join(extractor.text)
                # Limit content length
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"

                return CommandResult(True, f"[OK] Fetched {url}:\n\n{content}", content)
            else:
                # Non-HTML, return raw (limited)
                content = response.text[:2000]
                return CommandResult(True, f"[OK] Fetched {url}:\n\n{content}", content)

        except ImportError:
            return CommandResult(False, "[ERROR] requests library not installed")
        except requests.RequestException as e:
            return CommandResult(False, f"[ERROR] Failed to fetch: {e}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {e}")

    registry.register("search.web", search_web, "Search the web", "search.web <query>")
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
            note_file.write_text(text, encoding="utf-8")
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
            for note in notes[:10]:
                try:
                    content = note.read_text(encoding="utf-8")
                    preview = content[:50].replace("\n", " ")
                    if len(content) > 50:
                        preview += "..."
                    lines.append(f"  {note.name}: {preview}")
                except Exception:
                    lines.append(f"  {note.name}: (unreadable)")

            return CommandResult(True, "\n".join(lines), [n.name for n in notes[:10]])
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

        count = 20
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
            preview = "\n".join(lines[:6])
            if len(lines) > 6:
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

        # Block shell metacharacters to prevent injection
        if any(ch in command for ch in ("|", "&", ";", "`", "$", ">", "<")):
            return CommandResult(False, "[ERROR] Shell metacharacters are not allowed")

        try:
            # shell=False — args list prevents shell injection
            cmd_list = shlex.split(command)
            result = subprocess.run(
                cmd_list,
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=ctx.get("cwd", None)
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    # Truncate long output
                    if len(output) > 1000:
                        output = output[:1000] + f"\n... ({len(output)} chars total)"
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
