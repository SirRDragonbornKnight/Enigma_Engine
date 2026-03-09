"""
Enigma Engine - GUI Mod Logic
==================================

Mixin for mod subprocess management, status updates,
command sending, and auto-start.
Extracted from gui_logic.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import customtkinter as ctk

from enigma_engine.gui.widgets import C_GREEN, C_TEXT_DIM

logger = logging.getLogger(__name__)


class ModMixin:
    """Mixin providing mod process management for EnigmaGUI.

    Expects the host class to have:
    - mod_processes: dict[str, Popen]
    - mods_data: list[dict]
    - _chat_system, _chat_error
    """

    def _launch_mod(self, mod: dict):
        """Launch a mod subprocess. Returns the Popen or None."""
        mod_main = Path(mod["path"]) / "main.py"
        if not mod_main.exists():
            return None
        try:
            proc = subprocess.Popen(
                [sys.executable, str(mod_main)],
                cwd=str(Path(mod["path"])),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
            # Brief delay to catch immediate crashes (missing imports, etc.)
            try:
                proc.wait(timeout=1.0)
                # Process exited within 1s — it crashed
                stderr_out = ""
                if proc.stderr:
                    raw = proc.stderr.read()
                    stderr_out = raw.decode("utf-8", errors="replace")[-500:]
                    proc.stderr.close()
                logger.warning(
                    "Mod %s exited immediately (code %s): %s",
                    mod.get("id", "?"), proc.returncode,
                    stderr_out.strip())
                return None
            except subprocess.TimeoutExpired:
                # Still running after 1s — that's good, close stderr
                if proc.stderr:
                    proc.stderr.close()
            self.mod_processes[mod["id"]] = proc
            mod["_running"] = True
            self._update_mod_page_status(mod)
            return proc
        except Exception as exc:
            logger.warning("Failed to launch mod %s: %s",
                           mod.get("id", "?"), exc)
            return None

    def _start_mod(self, mod: dict):
        """Start a mod subprocess."""
        if mod.get("_running", False):
            return
        proc = self._launch_mod(mod)
        if proc is None:
            self._mod_log(mod, f"No main.py found for {mod['name']}")
            return
        self._mod_log(
            mod,
            f"Started: {mod['name']} (port {mod.get('port', '?')})")
        self._chat_system(
            f"Mod online: {mod['name']} "
            f"(port {mod.get('port', '?')})")

    def _stop_mod(self, mod: dict):
        """Stop a mod subprocess."""
        mod_id = mod["id"]
        if not mod.get("_running", False):
            return
        proc = self.mod_processes.get(mod_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass  # process already exited between terminate timeout and kill
        self.mod_processes.pop(mod_id, None)
        mod["_running"] = False
        self._update_mod_page_status(mod)
        self._mod_log(mod, f"Stopped: {mod['name']}")
        self._chat_system(f"Mod stopped: {mod['name']}")

    def _toggle_mod(self, mod: dict):
        """Toggle a mod on or off."""
        if mod.get("_running", False):
            self._stop_mod(mod)
        else:
            self._start_mod(mod)

    def _update_mod_page_status(self, mod: dict):
        """Update the status widgets on a mod's page."""
        running = mod.get("_running", False)
        dot = mod.get("_page_dot")
        lbl = mod.get("_page_status")
        start_btn = mod.get("_start_btn")
        stop_btn = mod.get("_stop_btn")
        if dot:
            dot.set_color(C_GREEN if running else C_TEXT_DIM)
        if lbl:
            lbl.configure(
                text="RUNNING" if running else "STOPPED",
                text_color=C_GREEN if running else C_TEXT_DIM)
        if start_btn:
            start_btn.configure(
                state="disabled" if running else "normal")
        if stop_btn:
            stop_btn.configure(
                state="normal" if running else "disabled")

    def _mod_log(self, mod: dict, text: str):
        """Append text to a mod's output log."""
        log_widget = mod.get("_log")
        if not log_widget:
            return
        log_widget.write(f"{text}\n")

    def _send_mod_command(self, mod: dict, command: str):
        """Gather UI widget values and log the command."""
        if not mod.get("_running", False):
            self._mod_log(
                mod, f"Cannot send '{command}': mod not running")
            return

        # Collect args from UI widgets
        ui_widgets = mod.get("_ui_widgets", {})
        args: dict[str, str] = {}
        for w_id, widget in ui_widgets.items():
            if isinstance(widget, ctk.CTkEntry):
                val = widget.get().strip()
                if val:
                    args[w_id] = val
            elif isinstance(widget, ctk.CTkTextbox):
                val = widget.get("1.0", "end").strip()
                if val:
                    args[w_id] = val
            elif isinstance(widget, ctk.CTkOptionMenu):
                val = widget.get()
                if val:
                    args[w_id] = val
            elif isinstance(widget, ctk.CTkCheckBox):
                var = getattr(widget, "_mod_var", None)
                if var is not None:
                    args[w_id] = str(var.get())

        self._mod_log(
            mod, f">> {command} {args if args else ''}")

        # Send command via router if available
        try:
            from enigma_engine.router import ModRouter  # noqa: F401
            if hasattr(self, "_router") and self._router:
                msg = {
                    "type": "command",
                    "data": {"command": command, "args": args}}
                sent = self._router.send_to_mod(mod["id"], msg)
                if sent:
                    self._mod_log(mod, "   Command sent")
                else:
                    self._mod_log(
                        mod, "   Mod not connected to router")
            else:
                self._mod_log(
                    mod,
                    "   Command queued (router not active)")
        except ImportError:
            self._mod_log(
                mod, "   Command queued (router not available)")

    def _auto_start_mod(self, mod: dict):
        """Auto-start a mod on GUI launch."""
        proc = self._launch_mod(mod)
        if proc is None:
            logger.info("Mod '%s' did not auto-start",
                        mod.get("id", "?"))
