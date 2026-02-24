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
import subprocess
import threading

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_BG, C_BORDER, C_BORDER_ACCENT,
    C_CYAN, C_GREEN, C_GREEN_DIM, C_INPUT, C_ORANGE,
    C_PANEL, C_RED, C_SURFACE, C_TEXT, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_CMD, FONT_MONO, FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableTextbox,
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

        # ------- Top bar -------
        top = ctk.CTkFrame(page, fg_color="transparent", height=44)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))

        SectionLabel(top, "Terminal", color=C_CYAN).pack(
            side="left", fill="x", expand=True)

        # AI ACCESS toggle (right side)
        self._cmd_ai_label = ctk.CTkLabel(
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
        ctk.CTkButton(
            top, text="CLEAR", width=70, height=28,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM, command=self._cmd_clear
        ).pack(side="right", padx=(0, 6))

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

        # ------- Input line -------
        input_row = ctk.CTkFrame(page, fg_color="transparent")
        input_row.grid(row=2, column=0, sticky="ew",
                       padx=10, pady=(4, 8))
        input_row.grid_columnconfigure(1, weight=1)

        # Prompt label (changes per mode)
        self._cmd_prompt_label = ctk.CTkLabel(
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

        ctk.CTkButton(
            input_row, text="RUN", width=70, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a5a2a",
            text_color=C_GREEN, command=self._cmd_execute
        ).grid(row=0, column=2, padx=(4, 0))

        # Welcome
        self._cmd_welcome()

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
                def _err():
                    self._cmd_write("error", f"[!] {exc}\n")
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
        if text.lower() == "help":
            self._cmd_engine_help()
            return
        if text.lower() == "history":
            self._cmd_show_history()
            return
        if text.lower().startswith("ask "):
            self._cmd_ask_ai(text[4:].strip())
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
                self.after(0, lambda: self._cmd_write(
                    "error", f"[!] {exc}\n"))

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
        for name, desc in [
                ("help", "Show this help"),
                ("clear", "Clear the terminal"),
                ("history", "Show command history"),
                ("ask <question>", "Ask the AI a question")]:
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
        self.cmd_output.configure(state="normal")
        self.cmd_output._textbox.insert("end", text, tag)
        self.cmd_output.configure(state="disabled")
        self.cmd_output.see("end")

    def _cmd_clear(self):
        """Clear the terminal output."""
        self.cmd_output.configure(state="normal")
        self.cmd_output.delete("1.0", "end")
        self.cmd_output.configure(state="disabled")
        self._cmd_welcome()

    def _cmd_welcome(self):
        """Show welcome message in terminal."""
        self._cmd_write("info", "Enigma Engine Terminal\n")
        self._cmd_write(
            "info",
            f"  SYSTEM: PowerShell  //  {self._cmd_cwd}\n")
        self._cmd_write(
            "info",
            "  ENGINE: AI commands  //  "
            "'ask <question>' to talk to the AI\n")
        self._cmd_write(
            "info",
            "  Toggle mode with the SYSTEM/ENGINE buttons. "
            "Use 'cls' to clear.\n")
        self._cmd_write("divider", "\n")
