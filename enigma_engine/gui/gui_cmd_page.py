"""
Enigma Engine - CMD Page Builder
===================================

Mixin providing a dual-mode terminal page for EnigmaGUI.

SYSTEM mode: Runs real PowerShell commands on the host machine.
ENGINE mode: Runs AI engine commands (config.*, file.*, model.*, etc.)
             The AI can also be asked questions and will auto-execute
             any commands it generates.

AI ACCESS toggle: When enabled, the AI can send real system commands
                  through the SYSTEM shell. When disabled, the AI is
                  restricted to engine commands only.
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import time

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT_DIM, C_BORDER, C_BORDER_ACCENT,
    C_CYAN, C_GREEN, C_GREEN_DIM, C_INPUT, C_ORANGE,
    C_PANEL, C_RED, C_SURFACE, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_CMD, FONT_MONO, FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableLabel, SelectableTextbox,
    Tooltip,
)

# Mode constants
MODE_SYSTEM = "SYSTEM"
MODE_ENGINE = "ENGINE"


class CMDPageMixin:
    """Mixin providing a dual-mode terminal page for EnigmaGUI.

    SYSTEM mode runs real PowerShell commands.
    ENGINE mode runs AI engine commands and can ask the AI questions.
    AI ACCESS toggle lets the AI execute real system commands.
    """

    # ================================================================
    # PAGE: CMD - Dual-Mode Terminal
    # ================================================================

    def _build_page_cmd(self):
        """Build the CMD terminal page with mode toggle."""
        page = self._make_page("CMD")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # State
        self._cmd_mode: str = MODE_SYSTEM
        self._cmd_ai_access: bool = False
        self._cmd_history: list[str] = []
        self._cmd_history_idx: int = -1
        self._cmd_proc: subprocess.Popen | None = None
        self._cmd_cwd: str = os.getcwd()
        self._cmd_busy = False
        self._cmd_start_time = time.time()

        # ------- Top bar -------
        top = ctk.CTkFrame(page, fg_color="transparent", height=44)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))

        SectionLabel(top, "Terminal", color=C_CYAN).pack(
            side="left", fill="x", expand=True)

        # AI ACCESS toggle (right side)
        self._cmd_ai_label = SelectableLabel(
            top, text="AI ACCESS", font=FONT_TINY,
            text_color=C_TEXT_DIM)
        self._cmd_ai_label.pack(side="right", padx=(4, 0))

        self._cmd_ai_btn = ctk.CTkButton(
            top, text="OFF", width=50, height=28,
            font=FONT_TINY, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM,
            command=self._cmd_toggle_ai_access)
        self._cmd_ai_btn.pack(side="right", padx=(0, 4))

        # Mode toggle
        self._cmd_mode_btn = ctk.CTkSegmentedButton(
            top, values=[MODE_SYSTEM, MODE_ENGINE],
            font=FONT_SMALL, height=28,
            fg_color=C_SURFACE,
            selected_color=C_ACCENT_DIM,
            selected_hover_color=C_ACCENT_DIM,
            unselected_color=C_SURFACE,
            unselected_hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM,
            text_color_disabled=C_TEXT_DIM,
            command=self._cmd_switch_mode)
        self._cmd_mode_btn.set(MODE_SYSTEM)
        self._cmd_mode_btn.pack(side="right", padx=(0, 12))

        # Clear button
        clear_btn = ctk.CTkButton(
            top, text="CLEAR", width=70, height=28,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM, command=self._cmd_clear)
        clear_btn.pack(side="right", padx=(0, 6))

        # ------- Terminal output -------
        terminal_frame = ctk.CTkFrame(page, fg_color="transparent")
        terminal_frame.grid(row=1, column=0, sticky="nsew",
                            padx=10, pady=(2, 2))
        terminal_frame.grid_columnconfigure(0, weight=1)
        terminal_frame.grid_rowconfigure(0, weight=1)

        output_frame = HUDFrame(terminal_frame, glow_color=C_BORDER)
        output_frame.grid(row=0, column=0, sticky="nsew")
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        self.cmd_output = SelectableTextbox(
            output_frame, wrap="word",
            font=FONT_CMD, fg_color=C_PANEL, text_color=C_GREEN,
            border_width=0, corner_radius=2)
        self.cmd_output.grid(
            row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Color tags
        tb = self.cmd_output._textbox
        tb.tag_configure("prompt", foreground=C_CYAN,
                         font=("Consolas", 15, "bold"))
        tb.tag_configure("command", foreground=C_TEXT_BRIGHT)
        tb.tag_configure("output", foreground=C_GREEN)
        tb.tag_configure("error", foreground=C_RED)
        tb.tag_configure("info", foreground=C_TEXT_DIM)
        tb.tag_configure("ai_output", foreground=C_ORANGE)
        tb.tag_configure("divider", foreground=C_BORDER_ACCENT)

        # ------- Status strip -------
        self._cmd_status_strip = ctk.CTkFrame(
            page, fg_color=C_SURFACE, height=24,
            corner_radius=0)
        self._cmd_status_strip.grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 0))

        self._cmd_stat_labels: dict[str, SelectableLabel] = {}
        for key, default in [
            ("device", "CPU"),
            ("ram", "RAM: --"),
            ("gpu", ""),
            ("model", "No model"),
            ("uptime", "0m"),
        ]:
            lbl = SelectableLabel(
                self._cmd_status_strip, text=default,
                font=FONT_TINY, text_color=C_TEXT_DIM)
            lbl.pack(side="left", padx=(8, 12))
            self._cmd_stat_labels[key] = lbl

        # Start status strip updater
        self._cmd_update_status_strip()

        # ------- Input line -------
        input_row = ctk.CTkFrame(page, fg_color="transparent")
        input_row.grid(row=3, column=0, sticky="ew",
                       padx=10, pady=(4, 8))
        input_row.grid_columnconfigure(1, weight=1)

        # Prompt label (changes per mode)
        self._cmd_prompt_label = SelectableLabel(
            input_row, text="PS>",
            font=("Consolas", 15, "bold"),
            text_color=C_CYAN, width=40)
        self._cmd_prompt_label.grid(row=0, column=0, padx=(0, 4))

        self.cmd_input = ctk.CTkEntry(
            input_row, height=38, font=FONT_MONO,
            fg_color=C_INPUT, border_color=C_BORDER_ACCENT,
            border_width=1, text_color=C_TEXT_BRIGHT,
            corner_radius=2,
            placeholder_text="Type a system command...",
            placeholder_text_color=C_TEXT_DIM)
        self.cmd_input.grid(row=0, column=1, sticky="ew")
        self.cmd_input.bind("<Return>", self._cmd_on_enter)
        self.cmd_input.bind("<Up>", self._cmd_history_up)
        self.cmd_input.bind("<Down>", self._cmd_history_down)

        run_btn = ctk.CTkButton(
            input_row, text="RUN", width=70, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a5a2a",
            text_color=C_GREEN, command=self._cmd_execute)
        run_btn.grid(row=0, column=2, padx=(4, 0))

        # Tooltips for CMD page interactive elements
        Tooltip(self._cmd_ai_btn,
                "Allow AI to run system commands")
        # Note: CTkSegmentedButton does not support .bind() — skip tooltip
        Tooltip(clear_btn, "Clear terminal output")
        Tooltip(run_btn, "Execute command (Enter)")

        # Right-click context menu on CMD input
        self.cmd_input.bind(
            "<Button-3>", self._cmd_input_context_menu)

        # Welcome
        self._cmd_welcome()

    # ================================================================
    # CMD - Context menu
    # ================================================================

    def _cmd_input_context_menu(self, event):
        """Show right-click context menu for CMD input entry."""
        import tkinter as tk
        menu = tk.Menu(self, tearoff=0)
        entry = self.cmd_input
        menu.add_command(
            label="Cut",
            command=lambda: (
                entry.clipboard_clear(),
                entry.clipboard_append(entry.get()),
                entry.delete(0, "end"),
            ) if entry.get() else None)
        menu.add_command(
            label="Copy",
            command=lambda: (
                entry.clipboard_clear(),
                entry.clipboard_append(entry.get()),
            ) if entry.get() else None)
        menu.add_command(
            label="Paste",
            command=lambda: entry.insert(
                "end", self.clipboard_get()) if self._safe_clipboard()
            else None)
        menu.add_separator()
        menu.add_command(
            label="Select All",
            command=lambda: entry.select_range(0, "end"))
        menu.tk_popup(event.x_root, event.y_root)

    def _safe_clipboard(self) -> bool:
        """Return True if clipboard has text content."""
        try:
            self.clipboard_get()
            return True
        except Exception:
            return False

    # ================================================================
    # CMD - Mode switching
    # ================================================================

    def _cmd_switch_mode(self, mode: str):
        """Switch between SYSTEM and ENGINE mode."""
        self._cmd_mode = mode
        if mode == MODE_SYSTEM:
            short = os.path.basename(self._cmd_cwd)
            self._cmd_prompt_label.configure(text=f"PS {short}>")
            self.cmd_input.configure(
                placeholder_text="Type a system command...")
        else:
            self._cmd_prompt_label.configure(text="ENG>")
            self.cmd_input.configure(
                placeholder_text=(
                    "Engine command or 'ask <question>'..."))
        self._cmd_write("info",
                        f"[Switched to {mode} mode]\n")
        self._cmd_write("divider", "\n")

    def _cmd_toggle_ai_access(self):
        """Toggle AI access to real system commands."""
        self._cmd_ai_access = not self._cmd_ai_access
        if self._cmd_ai_access:
            self._cmd_ai_btn.configure(
                text="ON", fg_color=C_GREEN_DIM,
                text_color=C_GREEN)
            self._cmd_ai_label.configure(text_color=C_GREEN)
            self._cmd_write(
                "info",
                "[AI ACCESS ON] AI can now execute "
                "system commands.\n")
        else:
            self._cmd_ai_btn.configure(
                text="OFF", fg_color=C_SURFACE,
                text_color=C_TEXT_DIM)
            self._cmd_ai_label.configure(text_color=C_TEXT_DIM)
            self._cmd_write(
                "info",
                "[AI ACCESS OFF] AI restricted to "
                "engine commands only.\n")
        self._cmd_write("divider", "\n")

    # ================================================================
    # CMD - Command execution (router)
    # ================================================================

    def _cmd_on_enter(self, event):
        """Handle Enter key in command input."""
        self._cmd_execute()
        return "break"

    def _cmd_execute(self):
        """Route command to the active mode handler."""
        text = self.cmd_input.get().strip()
        if not text:
            return

        self.cmd_input.delete(0, "end")

        # Add to history
        if not self._cmd_history or self._cmd_history[-1] != text:
            self._cmd_history.append(text)
        self._cmd_history_idx = -1

        # Handle universal clear
        if text.lower() in ("cls", "clear"):
            self._cmd_clear()
            return

        # Handle cancel / Ctrl+C for running system commands
        if text.lower() in ("cancel", "ctrl+c", "kill") and self._cmd_busy:
            proc = getattr(self, "_cmd_proc", None)
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
                self._cmd_proc = None
            self._cmd_busy = False
            self._cmd_write("error", "[cancelled]\n")
            self._cmd_write("divider", "\n")
            return

        if self._cmd_mode == MODE_SYSTEM:
            self._cmd_run_system(text)
        else:
            self._cmd_run_engine(text)

    # ================================================================
    # CMD - SYSTEM mode (real PowerShell)
    # ================================================================

    def _cmd_run_system(self, text: str):
        """Execute a real system command via PowerShell."""
        if self._cmd_busy:
            self._cmd_write("error",
                            "[busy] Wait for current command.\n")
            return

        self._cmd_write("prompt", f"PS {self._cmd_cwd}> ")
        self._cmd_write("command", text + "\n")

        self._cmd_busy = True

        def _run():
            try:
                marker = "---ENIGMA_CWD_MARKER---"
                full_cmd = (
                    f"{text}; "
                    f"Write-Host '{marker}'; "
                    f"(Get-Location).Path"
                )

                proc = subprocess.Popen(
                    ["powershell", "-NoProfile",
                     "-NonInteractive", "-Command", full_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self._cmd_cwd,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self._cmd_proc = proc

                output_lines: list[str] = []
                found_marker = False

                for line in proc.stdout:
                    if marker in line:
                        found_marker = True
                        continue
                    if found_marker:
                        new_cwd = line.strip()
                        if new_cwd and os.path.isdir(new_cwd):
                            self._cmd_cwd = new_cwd
                        continue
                    output_lines.append(line)

                proc.wait()
                self._cmd_proc = None
                text_out = "".join(output_lines)

                def _show():
                    if text_out.strip():
                        self._cmd_write("output", text_out)
                        if not text_out.endswith("\n"):
                            self._cmd_write("output", "\n")
                    if proc.returncode and proc.returncode != 0:
                        self._cmd_write(
                            "error",
                            f"[exit code {proc.returncode}]\n")
                    self._cmd_write("divider", "\n")
                    self._cmd_busy = False
                    short = os.path.basename(self._cmd_cwd)
                    self._cmd_prompt_label.configure(
                        text=f"PS {short}>")

                self.after(0, _show)

            except Exception as exc:
                err_msg = str(exc)
                def _err():
                    self._cmd_write("error", f"[!] {err_msg}\n")
                    self._cmd_write("divider", "\n")
                    self._cmd_busy = False
                self.after(0, _err)
                self._cmd_proc = None

        threading.Thread(target=_run, daemon=True).start()

    # ================================================================
    # CMD - ENGINE mode (AI command registry)
    # ================================================================

    def _cmd_run_engine(self, text: str):
        """Execute an engine command or ask the AI."""
        # Show the command
        self._cmd_write("prompt", "ENG> ")
        self._cmd_write("command", text + "\n")

        # Built-in engine terminal commands
        cmd_lower = text.lower()
        if cmd_lower == "help":
            self._cmd_engine_help()
            return
        if cmd_lower == "history":
            self._cmd_show_history()
            return
        if cmd_lower.startswith("ask "):
            self._cmd_ask_ai(text[4:].strip())
            return
        # Info commands
        _info_cmds = {
            "status": self._cmd_show_status,
            "sysinfo": self._cmd_show_sysinfo,
            "gpu": self._cmd_show_gpu,
            "memory": self._cmd_show_memory,
            "models": self._cmd_show_models,
            "routes": self._cmd_show_routes,
            "sessions": self._cmd_show_sessions,
            "mods": self._cmd_show_mods,
            "data": self._cmd_show_data,
            "uptime": self._cmd_show_uptime,
        }
        if cmd_lower in _info_cmds:
            _info_cmds[cmd_lower]()
            return

        # Execute via command registry
        try:
            from enigma_engine.core.commands import get_registry
            registry = get_registry()
            if self.engine is not None:
                registry.set_context("engine", self.engine)
                if hasattr(self, "config_overrides"):
                    registry.set_context(
                        "config", self.config_overrides)
            registry.set_context("registry", registry)

            result = registry.execute(text)
            tag = "output" if result.success else "error"
            self._cmd_write(tag, result.message + "\n")
        except ImportError:
            self._cmd_write(
                "error", "[!] Command system not available.\n")
        except Exception as exc:
            self._cmd_write("error", f"[!] {exc}\n")

        self._cmd_write("divider", "\n")

    def _cmd_ask_ai(self, question: str):
        """Send a question to the AI and handle its response.

        If AI ACCESS is on, any system commands the AI generates
        in [CMD] blocks will be executed in the real shell.
        Otherwise, only engine commands are executed.
        """
        if not question:
            self._cmd_write("error", "[!] Usage: ask <question>\n")
            return

        if self.engine is None:
            self._cmd_write(
                "error",
                "[!] No model loaded. Load one from ROUTER.\n")
            return

        self._cmd_write("info", "Thinking...\n")

        def _generate():
            try:
                kwargs = {}
                if hasattr(self, "config_overrides"):
                    kwargs.update(self.config_overrides)
                try:
                    resp = self.engine.chat(question, **kwargs)
                except TypeError:
                    resp = self.engine.chat(question)

                from enigma_engine.core.commands import (
                    parse_commands, get_registry)
                clean_text, commands = parse_commands(resp)

                def _show():
                    if clean_text:
                        self._cmd_write(
                            "ai_output", clean_text + "\n")

                    if commands:
                        self._cmd_write(
                            "info",
                            f"\n[{len(commands)} command(s) "
                            f"from AI]\n")

                        registry = get_registry()
                        if self.engine is not None:
                            registry.set_context(
                                "engine", self.engine)

                        for cmd_str in commands:
                            self._cmd_execute_ai_command(
                                cmd_str, registry)

                    self._cmd_write("divider", "\n")

                self.after(0, _show)
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda: self._cmd_write(
                    "error", f"[!] {err_msg}\n"))

        threading.Thread(target=_generate, daemon=True).start()

    def _cmd_execute_ai_command(self, cmd_str: str, registry):
        """Execute a single AI-generated command.

        If the command is a known engine command, run it through
        the registry. If AI ACCESS is on and the command is not
        recognized by the registry, run it as a system command.
        """
        self._cmd_write("prompt", "AI>> ")
        self._cmd_write("command", cmd_str + "\n")

        # Try engine command first
        try:
            result = registry.execute(cmd_str)
            if result.success:
                self._cmd_write("output", result.message + "\n")
                return
            # If the registry says unknown and AI ACCESS is on,
            # try as a system command
            is_unknown = ("unknown" in result.message.lower()
                          or "not found" in result.message.lower())
            if is_unknown and self._cmd_ai_access:
                self._cmd_write(
                    "info",
                    "[AI ACCESS] Running as system command...\n")
                self._cmd_run_system(cmd_str)
                return
            # Otherwise show the engine error
            self._cmd_write("error", result.message + "\n")
        except Exception:
            if self._cmd_ai_access:
                self._cmd_write(
                    "info",
                    "[AI ACCESS] Running as system command...\n")
                self._cmd_run_system(cmd_str)
            else:
                self._cmd_write(
                    "error",
                    f"[!] Unknown command: {cmd_str}\n")
                self._cmd_write(
                    "info",
                    "[Turn on AI ACCESS to allow "
                    "system commands]\n")

    # ================================================================
    # CMD - Engine help
    # ================================================================

    def _cmd_engine_help(self):
        """Show engine commands and terminal info."""
        try:
            from enigma_engine.core.commands import get_registry
            registry = get_registry()
            help_text = registry.get_help()
            self._cmd_write("output", help_text + "\n")
        except ImportError:
            self._cmd_write(
                "error", "[!] Command system not available.\n")

        self._cmd_write("info", "\nTERMINAL COMMANDS\n")
        terminal_cmds = [
            ("help", "Show this help"),
            ("clear", "Clear the terminal"),
            ("history", "Show command history"),
            ("ask <question>", "Ask the AI a question"),
            ("status", "Show loaded model, routes, and engine state"),
            ("sysinfo", "Show hardware and system information"),
            ("gpu", "Show GPU details and memory usage"),
            ("memory", "Show RAM and VRAM usage"),
            ("models", "List all available models"),
            ("routes", "Show current route assignments"),
            ("sessions", "List saved chat sessions"),
            ("mods", "List installed mods and their status"),
            ("profiles", "List AI profiles"),
            ("data", "List training data files"),
            ("uptime", "Show session uptime"),
        ]
        for name, desc in terminal_cmds:
            self._cmd_write(
                "info", f"  {name:<30} - {desc}\n")
        self._cmd_write("divider", "\n")

    # ================================================================
    # CMD - History navigation
    # ================================================================

    def _cmd_history_up(self, event):
        """Navigate command history backwards."""
        if not self._cmd_history:
            return "break"
        if self._cmd_history_idx == -1:
            self._cmd_history_idx = len(self._cmd_history) - 1
        elif self._cmd_history_idx > 0:
            self._cmd_history_idx -= 1
        else:
            return "break"
        self.cmd_input.delete(0, "end")
        self.cmd_input.insert(
            0, self._cmd_history[self._cmd_history_idx])
        return "break"

    def _cmd_history_down(self, event):
        """Navigate command history forwards."""
        if not self._cmd_history or self._cmd_history_idx == -1:
            return "break"
        if self._cmd_history_idx < len(self._cmd_history) - 1:
            self._cmd_history_idx += 1
            self.cmd_input.delete(0, "end")
            self.cmd_input.insert(
                0, self._cmd_history[self._cmd_history_idx])
        else:
            self._cmd_history_idx = -1
            self.cmd_input.delete(0, "end")
        return "break"

    def _cmd_show_history(self):
        """Display command history."""
        if not self._cmd_history:
            self._cmd_write("info", "No commands in history.\n")
            return
        self._cmd_write("info", "Command history:\n")
        for i, cmd in enumerate(self._cmd_history[-20:], 1):
            self._cmd_write("output", f"  {i:>3}  {cmd}\n")
        self._cmd_write("divider", "\n")

    # ================================================================
    # CMD - Output helpers
    # ================================================================

    def _cmd_write(self, tag: str, text: str):
        """Write tagged text to the terminal output."""
        self.cmd_output._textbox.insert("end", text, tag)
        self.cmd_output.see("end")

    def _cmd_clear(self):
        """Clear the terminal output."""
        self.cmd_output.clear()
        self._cmd_write("prompt", "ENIGMA ENGINE TERMINAL\n")
        self._cmd_write("divider", "" + "\n")
        self._cmd_write("info", "  Terminal cleared.\n\n")

    def _cmd_welcome(self):
        """Show welcome message with system information."""
        self._cmd_write("prompt", "ENIGMA ENGINE TERMINAL\n")
        self._cmd_write("divider",
                        "" + "\n")

        # System info
        try:
            from enigma_engine.core.hardware_detection import (
                detect_hardware)
            hw = detect_hardware()
            self._cmd_write("info",
                            f"  System   : {platform.system()} "
                            f"{platform.release()} "
                            f"({platform.machine()})\n")
            self._cmd_write("info",
                            f"  CPU      : {hw.cpu_cores} cores\n")
            ram_used = hw.ram_gb - hw.available_ram_gb
            self._cmd_write("info",
                            f"  RAM      : {ram_used:.1f} / "
                            f"{hw.ram_gb:.1f} GB\n")
            if hw.gpu_available:
                self._cmd_write("output",
                                f"  GPU      : {hw.gpu_name}\n")
                self._cmd_write("output",
                                f"  VRAM     : "
                                f"{hw.gpu_vram_gb:.1f} GB\n")
                if hw.cuda_version:
                    self._cmd_write("info",
                                    f"  CUDA     : "
                                    f"{hw.cuda_version}\n")
            else:
                self._cmd_write("info",
                                "  GPU      : None detected\n")
        except Exception:
            self._cmd_write("info",
                            f"  System   : {platform.system()} "
                            f"{platform.release()}\n")

        self._cmd_write("divider", "\n")

        # Model / route status
        engine = getattr(self, "engine", None)
        model_path = getattr(self, "model_path", None)
        routes = getattr(self, "route_assignments", {})

        if engine is not None and model_path:
            from pathlib import Path
            name = Path(model_path).stem
            self._cmd_write("output",
                            f"  Model    : {name}\n")
            if (hasattr(engine, "model") and engine.model
                    and hasattr(engine.model, "parameters")):
                pc = sum(p.numel()
                         for p in engine.model.parameters())
                self._cmd_write("info",
                                f"  Params   : {pc:,}\n")
        else:
            self._cmd_write("info",
                            "  Model    : None loaded\n")

        if routes:
            assigned = [f"{k}={Path(v).stem}"
                        for k, v in routes.items() if v]
            if assigned:
                self._cmd_write("info",
                                f"  Routes   : "
                                f"{', '.join(assigned)}\n")

        self._cmd_write("divider", "\n")

        # Counts
        try:
            from enigma_engine.gui.scanners import (
                scan_models, scan_mods, scan_sessions,
                scan_training_data)
            n_models = len(scan_models())
            n_mods = len(scan_mods())
            n_sessions = len(scan_sessions())
            n_data = len(scan_training_data())
            counts = (f"{n_models} models  |  "
                      f"{n_mods} mods  |  "
                      f"{n_sessions} sessions  |  "
                      f"{n_data} data files")
            self._cmd_write("info", f"  {counts}\n")
        except Exception:
            pass

        self._cmd_write("divider", "\n")
        self._cmd_write("info",
                        "  Type 'help' for commands  |  "
                        "'ask <question>' to talk to the AI\n")
        self._cmd_write("info",
                        f"  CWD: {self._cmd_cwd}\n")
        self._cmd_write("divider", "\n")

    # ================================================================
    # CMD - Activity log (callable from any page/thread)
    # ================================================================

    def _cmd_activity(self, tag: str, text: str):
        """Thread-safe activity log to the CMD terminal.

        Call this from any thread to pipe AI activity into
        the CMD output so the user can see what the AI is doing.
        Tags: info, output, error, ai_output, prompt, command, divider
        """
        self.after(0, lambda: self._cmd_write(tag, text))

    def _cmd_execute_ai_commands(self, commands: list[str]) -> str:
        """Execute [CMD] blocks from an AI response.

        Runs each command through the engine registry.
        Returns any output text to append to the chat response.
        Also stores generated image paths in self._cmd_image_paths
        for inline rendering by the chat display.
        """
        if not commands:
            return ""

        self._cmd_activity(
            "info",
            f"\n[AI executing {len(commands)} command(s)]\n")

        results: list[str] = []
        image_paths: list[str] = []
        try:
            from enigma_engine.core.commands import get_registry
            registry = get_registry()
            if self.engine is not None:
                registry.set_context("engine", self.engine)
            registry.set_context("registry", registry)
        except ImportError:
            self._cmd_activity(
                "error", "[!] Command system not available.\n")
            return ""

        for cmd_str in commands:
            self._cmd_activity("prompt", "AI>> ")
            self._cmd_activity("command", cmd_str + "\n")

            try:
                result = registry.execute(cmd_str)
                if result.success:
                    self._cmd_activity(
                        "output", result.message + "\n")
                    results.append(result.message)
                    # Collect image paths from command data
                    if (result.data
                            and isinstance(result.data, dict)
                            and "path" in result.data):
                        p = result.data["path"]
                        ext = str(p).lower().rsplit(".", 1)[-1]
                        if ext in (
                            "png", "jpg", "jpeg",
                            "bmp", "webp", "gif",
                        ):
                            image_paths.append(str(p))
                else:
                    self._cmd_activity(
                        "error", result.message + "\n")
            except Exception as exc:
                self._cmd_activity(
                    "error", f"[!] {exc}\n")

        # Store image paths for the caller to render inline
        self._cmd_image_paths = image_paths

        self._cmd_activity("divider", "\n")
        return "\n".join(results)

    # ================================================================
    # CMD - Info commands
    # ================================================================

    def _cmd_show_status(self):
        """Show loaded model, routes, and engine state."""
        from pathlib import Path

        self._cmd_write("prompt", "STATUS\n")
        self._cmd_write("divider", "\n")

        # Engine / model
        engine = getattr(self, "engine", None)
        model_path = getattr(self, "model_path", None)
        if engine is not None and model_path:
            name = Path(model_path).stem
            self._cmd_write("output", f"  Model    : {name}\n")
            self._cmd_write("info",
                            f"  Path     : {model_path}\n")
            if (hasattr(engine, "model") and engine.model
                    and hasattr(engine.model, "parameters")):
                pc = sum(p.numel()
                         for p in engine.model.parameters())
                self._cmd_write("info",
                                f"  Params   : {pc:,}\n")
            if hasattr(engine, "model") and engine.model:
                cfg = getattr(engine.model, "config", None)
                if cfg:
                    self._cmd_write(
                        "info",
                        f"  Arch     : {cfg.dim}d, "
                        f"{cfg.n_layers}L, "
                        f"{cfg.n_heads}H, "
                        f"ctx {cfg.max_seq_len}\n")
            fmt = Path(model_path).suffix.lstrip(".")
            self._cmd_write("info", f"  Format   : {fmt}\n")
        else:
            self._cmd_write("info",
                            "  Model    : None loaded\n")

        # Routes
        routes = getattr(self, "route_assignments", {})
        if routes:
            self._cmd_write("divider", "\n")
            self._cmd_write("prompt", "  ROUTES\n")
            for key, path in routes.items():
                if path:
                    self._cmd_write(
                        "info",
                        f"    {key:<12} → {Path(path).stem}\n")
        else:
            self._cmd_write("info",
                            "  Routes   : None assigned\n")

        # Generation state
        is_gen = getattr(self, "_is_generating", False)
        self._cmd_write("info",
                        f"  Generating: "
                        f"{'YES' if is_gen else 'No'}\n")

        # Uptime
        elapsed = int(time.time() - self._cmd_start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        if hrs:
            up = f"{hrs}h {mins}m {secs}s"
        else:
            up = f"{mins}m {secs}s"
        self._cmd_write("info", f"  Uptime   : {up}\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_sysinfo(self):
        """Show hardware and system information."""
        self._cmd_write("prompt", "SYSTEM INFO\n")
        self._cmd_write("divider", "\n")

        self._cmd_write("info",
                        f"  OS       : {platform.system()} "
                        f"{platform.release()}\n")
        self._cmd_write("info",
                        f"  Version  : {platform.version()}\n")
        self._cmd_write("info",
                        f"  Arch     : {platform.machine()}\n")
        self._cmd_write("info",
                        f"  Python   : {platform.python_version()}"
                        "\n")
        self._cmd_write("info",
                        f"  CWD      : {self._cmd_cwd}\n")

        try:
            from enigma_engine.core.hardware_detection import (
                detect_hardware)
            hw = detect_hardware()
            self._cmd_write("info",
                            f"  CPU      : {hw.cpu_cores} cores, "
                            f"{hw.cpu_threads} threads\n")
            self._cmd_write("info",
                            f"  RAM      : {hw.ram_gb:.1f} GB "
                            f"total, "
                            f"{hw.available_ram_gb:.1f} GB free\n")
            self._cmd_write("info",
                            f"  Type     : {hw.hardware_type}\n")
            if hw.is_arm:
                self._cmd_write("info", "  ARM      : Yes\n")
            if hw.is_raspberry_pi:
                self._cmd_write("info",
                                f"  Pi       : {hw.pi_model}\n")
            if hw.is_apple_silicon:
                self._cmd_write("info",
                                "  Apple Si : Yes\n")
        except Exception:
            self._cmd_write("info",
                            "  [hardware detection unavailable]\n")

        # PyTorch info
        try:
            import torch
            self._cmd_write("info",
                            f"  PyTorch  : {torch.__version__}\n")
            if torch.cuda.is_available():
                self._cmd_write("output",
                                f"  CUDA     : "
                                f"{torch.version.cuda}\n")
        except ImportError:
            self._cmd_write("info",
                            "  PyTorch  : Not installed\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_gpu(self):
        """Show GPU details and memory usage."""
        self._cmd_write("prompt", "GPU INFO\n")
        self._cmd_write("divider", "\n")

        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    name = torch.cuda.get_device_name(i)
                    total = props.total_memory / (1024 ** 3)
                    alloc = (torch.cuda.memory_allocated(i)
                             / (1024 ** 3))
                    cached = (torch.cuda.memory_reserved(i)
                              / (1024 ** 3))
                    free = total - alloc

                    self._cmd_write("output",
                                    f"  GPU {i}   : {name}\n")
                    self._cmd_write("info",
                                    f"  Total    : "
                                    f"{total:.2f} GB\n")
                    self._cmd_write("info",
                                    f"  Used     : "
                                    f"{alloc:.2f} GB\n")
                    self._cmd_write("info",
                                    f"  Cached   : "
                                    f"{cached:.2f} GB\n")
                    self._cmd_write("info",
                                    f"  Free     : "
                                    f"{free:.2f} GB\n")

                    # Usage bar
                    pct = alloc / total if total > 0 else 0
                    bar_len = 30
                    filled = int(bar_len * pct)
                    bar = ("█" * filled
                           + "░" * (bar_len - filled))
                    color = ("output" if pct < 0.7
                             else "error" if pct > 0.9
                             else "ai_output")
                    self._cmd_write(
                        color,
                        f"  VRAM     : [{bar}] "
                        f"{pct * 100:.0f}%\n")

                    if i < torch.cuda.device_count() - 1:
                        self._cmd_write("divider", "\n")
            elif (hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available()):
                self._cmd_write("output",
                                "  GPU      : Apple Silicon "
                                "(MPS)\n")
                self._cmd_write("info",
                                "  VRAM     : Unified memory\n")
            else:
                self._cmd_write("info",
                                "  No GPU detected.\n")
        except ImportError:
            self._cmd_write("info",
                            "  PyTorch not installed — "
                            "cannot detect GPU.\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_memory(self):
        """Show RAM and VRAM usage."""
        self._cmd_write("prompt", "MEMORY\n")
        self._cmd_write("divider", "\n")

        try:
            import psutil
            mem = psutil.virtual_memory()
            total = mem.total / (1024 ** 3)
            used = mem.used / (1024 ** 3)
            avail = mem.available / (1024 ** 3)
            pct = mem.percent / 100

            self._cmd_write("info",
                            f"  RAM Total : {total:.1f} GB\n")
            self._cmd_write("info",
                            f"  RAM Used  : {used:.1f} GB\n")
            self._cmd_write("info",
                            f"  RAM Free  : {avail:.1f} GB\n")

            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            color = ("output" if pct < 0.7
                     else "error" if pct > 0.9
                     else "ai_output")
            self._cmd_write(
                color,
                f"  RAM       : [{bar}] "
                f"{pct * 100:.0f}%\n")
        except ImportError:
            self._cmd_write("info",
                            "  [psutil not installed]\n")

        # VRAM if available
        try:
            import torch
            if torch.cuda.is_available():
                self._cmd_write("divider", "\n")
                for i in range(torch.cuda.device_count()):
                    total = (torch.cuda.get_device_properties(i)
                             .total_memory / (1024 ** 3))
                    alloc = (torch.cuda.memory_allocated(i)
                             / (1024 ** 3))
                    pct = alloc / total if total > 0 else 0
                    bar_len = 30
                    filled = int(bar_len * pct)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    color = ("output" if pct < 0.7
                             else "error" if pct > 0.9
                             else "ai_output")
                    self._cmd_write("info",
                                    f"  VRAM {i}    : "
                                    f"{alloc:.1f} / "
                                    f"{total:.1f} GB\n")
                    self._cmd_write(
                        color,
                        f"  VRAM      : [{bar}] "
                        f"{pct * 100:.0f}%\n")
        except ImportError:
            pass

        self._cmd_write("divider", "\n")

    def _cmd_show_models(self):
        """List all available models."""
        from enigma_engine.gui.scanners import scan_models

        models = scan_models()
        self._cmd_write("prompt",
                        f"MODELS ({len(models)})\n")
        self._cmd_write("divider", "\n")

        if not models:
            self._cmd_write("info", "  No models found.\n")
        else:
            # Group by format
            native = [m for m in models
                      if m["format"] in ("pth", "pt")]
            external = [m for m in models
                        if m["format"] not in ("pth", "pt")]

            if native:
                self._cmd_write("output", "  NATIVE\n")
                for m in native:
                    size = (f"{m['size_mb']:.0f} MB"
                            if m["size_mb"] < 1024
                            else f"{m['size_mb'] / 1024:.1f} GB")
                    self._cmd_write(
                        "info",
                        f"    {m['name']:<30} "
                        f"{size:>10}  .{m['format']}\n")

            if external:
                self._cmd_write("output", "\n  EXTERNAL\n")
                for m in external:
                    size = (f"{m['size_mb']:.0f} MB"
                            if m["size_mb"] < 1024
                            else f"{m['size_mb'] / 1024:.1f} GB")
                    self._cmd_write(
                        "info",
                        f"    {m['name']:<30} "
                        f"{size:>10}  .{m['format']}\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_routes(self):
        """Show current route assignments."""
        from pathlib import Path

        routes = getattr(self, "route_assignments", {})
        self._cmd_write("prompt", "ROUTE ASSIGNMENTS\n")
        self._cmd_write("divider", "\n")

        if not routes or not any(routes.values()):
            self._cmd_write("info",
                            "  No routes assigned.\n")
        else:
            for key, path in routes.items():
                if path:
                    name = Path(path).stem
                    fmt = Path(path).suffix.lstrip(".")
                    exists = Path(path).exists()
                    tag = "output" if exists else "error"
                    self._cmd_write(
                        tag,
                        f"  {key:<12} → {name} "
                        f"(.{fmt})"
                        f"{'  [MISSING]' if not exists else ''}"
                        "\n")

        # Show unassigned routes
        from enigma_engine.gui.scanners import ROUTE_KEYS
        mods = []
        try:
            from enigma_engine.gui.scanners import scan_mods
            mods = scan_mods()
        except Exception:
            pass
        all_keys = list(ROUTE_KEYS)
        for b in mods:
            all_keys.append(b["id"])
        unassigned = [k for k in all_keys
                      if not routes.get(k)]
        if unassigned:
            self._cmd_write("info",
                            f"\n  Unassigned: "
                            f"{', '.join(unassigned)}\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_sessions(self):
        """List saved chat sessions."""
        from enigma_engine.gui.scanners import scan_sessions

        sessions = scan_sessions()
        self._cmd_write("prompt",
                        f"SESSIONS ({len(sessions)})\n")
        self._cmd_write("divider", "\n")

        if not sessions:
            self._cmd_write("info",
                            "  No saved sessions.\n")
        else:
            for s in sessions[-20:]:  # Show last 20
                self._cmd_write(
                    "info",
                    f"  {s['name']:<40} "
                    f"{s['messages']} msgs\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_mods(self):
        """List installed mods and their status."""
        from enigma_engine.gui.scanners import scan_mods

        mods = scan_mods()
        self._cmd_write("prompt",
                        f"MODS ({len(mods)})\n")
        self._cmd_write("divider", "\n")

        if not mods:
            self._cmd_write("info",
                            "  No mods installed.\n")
        else:
            for b in mods:
                # Check if mod process is running
                procs = getattr(self, "mod_processes", {})
                running = (b["id"] in procs
                           and procs[b["id"]].poll() is None)
                tag = "output" if running else "info"
                dot = "●" if running else "○"
                self._cmd_write(
                    tag,
                    f"  {dot} {b['name']:<20} "
                    f"v{b['version']:<6} "
                    f"port {b['port']}\n")
                if b["description"]:
                    self._cmd_write(
                        "info",
                        f"    {b['description']}\n")
                if b["commands"]:
                    cmds = ", ".join(b["commands"])
                    self._cmd_write(
                        "info",
                        f"    Commands: {cmds}\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_data(self):
        """List training data files."""
        from enigma_engine.gui.scanners import scan_training_data

        files = scan_training_data()
        self._cmd_write("prompt",
                        f"TRAINING DATA ({len(files)})\n")
        self._cmd_write("divider", "\n")

        if not files:
            self._cmd_write("info",
                            "  No training data found.\n")
        else:
            for f in files:
                size = (f"{f['size_kb']:.0f} KB"
                        if f["size_kb"] < 1024
                        else f"{f['size_kb'] / 1024:.1f} MB")
                self._cmd_write(
                    "info",
                    f"  {f['name']:<30} {size:>10}\n")

        self._cmd_write("divider", "\n")

    def _cmd_show_uptime(self):
        """Show session uptime."""
        elapsed = int(time.time() - self._cmd_start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        if hrs:
            up = f"{hrs}h {mins}m {secs}s"
        else:
            up = f"{mins}m {secs}s"
        self._cmd_write("info", f"Session uptime: {up}\n")
        self._cmd_write("divider", "\n")

    # ================================================================
    # CMD - Live status strip
    # ================================================================

    def _cmd_update_status_strip(self):
        """Periodically update the status strip with live info."""
        labels = getattr(self, "_cmd_stat_labels", None)
        if not labels:
            return

        # Device
        device = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                device = "CUDA"
            elif (hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available()):
                device = "MPS"
        except ImportError:
            pass
        labels["device"].configure(text=device)

        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            used = mem.used / (1024 ** 3)
            total = mem.total / (1024 ** 3)
            labels["ram"].configure(
                text=f"RAM: {used:.1f}/{total:.0f}G")
        except ImportError:
            labels["ram"].configure(text="RAM: --")

        # GPU VRAM
        try:
            import torch
            if torch.cuda.is_available():
                alloc = (torch.cuda.memory_allocated(0)
                         / (1024 ** 3))
                total = (torch.cuda.get_device_properties(0)
                         .total_memory / (1024 ** 3))
                labels["gpu"].configure(
                    text=f"VRAM: {alloc:.1f}/{total:.0f}G")
            else:
                labels["gpu"].configure(text="")
        except ImportError:
            labels["gpu"].configure(text="")

        # Model
        engine = getattr(self, "engine", None)
        model_path = getattr(self, "model_path", None)
        if engine is not None and model_path:
            from pathlib import Path
            name = Path(model_path).stem
            if len(name) > 20:
                name = name[:18] + ".."
            labels["model"].configure(
                text=f"Model: {name}",
                text_color=C_GREEN)
        else:
            labels["model"].configure(
                text="No model",
                text_color=C_TEXT_DIM)

        # Uptime
        elapsed = int(time.time() - self._cmd_start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, _ = divmod(rem, 60)
        if hrs:
            labels["uptime"].configure(text=f"Up: {hrs}h{mins}m")
        else:
            labels["uptime"].configure(text=f"Up: {mins}m")

        # Refresh every 5 seconds ONLY if CMD page is visible
        if self._current_page == "CMD":
            self.after(5000, self._cmd_update_status_strip)
