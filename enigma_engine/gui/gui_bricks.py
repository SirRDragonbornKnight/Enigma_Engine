"""
Enigma Engine - GUI Brick Logic
==================================

Mixin for brick subprocess management, status updates,
command sending, and auto-start.
Extracted from gui_logic.py to keep files under 800 lines.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import customtkinter as ctk

from enigma_engine.gui.widgets import C_GREEN, C_TEXT_DIM


class BrickMixin:
    """Mixin providing brick process management for EnigmaGUI.

    Expects the host class to have:
    - brick_processes: dict[str, Popen]
    - bricks_data: list[dict]
    - _chat_system, _chat_error
    """

    def _launch_brick(self, brick: dict):
        """Launch a brick subprocess. Returns the Popen or None."""
        brick_main = Path(brick["path"]) / "main.py"
        if not brick_main.exists():
            return None
        try:
            proc = subprocess.Popen(
                [sys.executable, str(brick_main)],
                cwd=str(Path(brick["path"])),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            self.brick_processes[brick["id"]] = proc
            brick["_running"] = True
            self._update_brick_page_status(brick)
            return proc
        except Exception:
            return None

    def _start_brick(self, brick: dict):
        """Start a brick subprocess."""
        if brick.get("_running", False):
            return
        proc = self._launch_brick(brick)
        if proc is None:
            self._brick_log(brick, f"No main.py found for {brick['name']}")
            return
        self._brick_log(
            brick,
            f"Started: {brick['name']} (port {brick.get('port', '?')})")
        self._chat_system(
            f"Brick online: {brick['name']} "
            f"(port {brick.get('port', '?')})")

    def _stop_brick(self, brick: dict):
        """Stop a brick subprocess."""
        brick_id = brick["id"]
        if not brick.get("_running", False):
            return
        proc = self.brick_processes.get(brick_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.brick_processes.pop(brick_id, None)
        brick["_running"] = False
        self._update_brick_page_status(brick)
        self._brick_log(brick, f"Stopped: {brick['name']}")
        self._chat_system(f"Brick stopped: {brick['name']}")

    def _toggle_brick(self, brick: dict):
        """Toggle a brick on or off."""
        if brick.get("_running", False):
            self._stop_brick(brick)
        else:
            self._start_brick(brick)

    def _update_brick_page_status(self, brick: dict):
        """Update the status widgets on a brick's page."""
        running = brick.get("_running", False)
        dot = brick.get("_page_dot")
        lbl = brick.get("_page_status")
        start_btn = brick.get("_start_btn")
        stop_btn = brick.get("_stop_btn")
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

    def _brick_log(self, brick: dict, text: str):
        """Append text to a brick's output log."""
        log_widget = brick.get("_log")
        if not log_widget:
            return
        log_widget.configure(state="normal")
        log_widget.insert("end", f"{text}\n")
        log_widget.configure(state="disabled")
        log_widget.see("end")

    def _send_brick_command(self, brick: dict, command: str):
        """Gather UI widget values and log the command."""
        if not brick.get("_running", False):
            self._brick_log(
                brick, f"Cannot send '{command}': brick not running")
            return

        # Collect args from UI widgets
        ui_widgets = brick.get("_ui_widgets", {})
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
                var = getattr(widget, "_brick_var", None)
                if var is not None:
                    args[w_id] = str(var.get())

        self._brick_log(
            brick, f">> {command} {args if args else ''}")

        # Send command via router if available
        try:
            from enigma_engine.router import BrickRouter
            if hasattr(self, "_router") and self._router:
                msg = {
                    "type": "command",
                    "data": {"command": command, "args": args}}
                sent = self._router.send_to_brick(brick["id"], msg)
                if sent:
                    self._brick_log(brick, "   Command sent")
                else:
                    self._brick_log(
                        brick, "   Brick not connected to router")
            else:
                self._brick_log(
                    brick,
                    f"   Command queued (router not active)")
        except ImportError:
            self._brick_log(
                brick, f"   Command queued (router not available)")

    def _auto_start_brick(self, brick: dict):
        """Auto-start a brick on GUI launch."""
        self._launch_brick(brick)
