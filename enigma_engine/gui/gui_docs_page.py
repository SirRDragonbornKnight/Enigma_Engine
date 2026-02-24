"""
Enigma Engine - GUI Docs Page
================================

Mixin providing the DOCS page for EnigmaGUI.
Handles documentation files, profiles, and brick docs.

Features:
    - File browser with category sections
    - Full text editor with save/delete
    - Create new files (guides, profiles)
    - Brick docs auto-discovered from bricks/<id>/docs/
    - Profile management (create, edit, delete)
"""
from __future__ import annotations

import json
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_BORDER, C_CYAN, C_GREEN,
    C_GREEN_DIM, C_INPUT, C_PANEL, C_PURPLE, C_RED,
    C_SURFACE, C_TEXT, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_BODY, FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, Tooltip,
)
from enigma_engine.gui.scanners import (
    BRICKS_DIR, INFO_DIR, PROFILES_DIR, scan_docs,
)


class DocsPageMixin:
    """Mixin that builds and manages the DOCS page."""

    # ================================================================
    # Page builder
    # ================================================================

    def _build_page_docs(self):
        page = self._make_page("DOCS")
        page.grid_columnconfigure(0, weight=1, minsize=220)
        page.grid_columnconfigure(1, weight=3)
        page.grid_rowconfigure(1, weight=1)

        # Top bar
        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=10, pady=(8, 2))

        SectionLabel(top, "Documentation").pack(
            side="left", fill="x", expand=True)

        # New file button
        new_btn = ctk.CTkButton(
            top, text="+ NEW", width=80, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_ACCENT, command=self._docs_new_file)
        new_btn.pack(side="right", padx=(4, 0))
        Tooltip(new_btn, "Create new file")

        # New profile button
        new_prof_btn = ctk.CTkButton(
            top, text="+ PROFILE", width=100, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_PURPLE, command=self._docs_new_profile)
        new_prof_btn.pack(side="right", padx=(4, 0))
        Tooltip(new_prof_btn, "Create new AI profile")

        # Left column: file browser
        browser = ctk.CTkFrame(page, fg_color="transparent")
        browser.grid(row=1, column=0, sticky="nsew",
                     padx=(10, 4), pady=(2, 8))
        browser.grid_columnconfigure(0, weight=1)
        browser.grid_rowconfigure(0, weight=1)

        browser_frame = HUDFrame(browser, glow_color=C_BORDER)
        browser_frame.pack(fill="both", expand=True)
        browser_frame.grid_columnconfigure(0, weight=1)
        browser_frame.grid_rowconfigure(0, weight=1)

        self._docs_browser = ctk.CTkScrollableFrame(
            browser_frame, fg_color=C_PANEL,
            corner_radius=0)
        self._docs_browser.grid(
            row=0, column=0, sticky="nsew", padx=2, pady=2)
        self._docs_browser.grid_columnconfigure(0, weight=1)

        # Right column: editor
        editor_col = ctk.CTkFrame(page, fg_color="transparent")
        editor_col.grid(row=1, column=1, sticky="nsew",
                        padx=(4, 10), pady=(2, 8))
        editor_col.grid_columnconfigure(0, weight=1)
        editor_col.grid_rowconfigure(1, weight=1)

        # Editor header (filename + buttons)
        editor_top = ctk.CTkFrame(editor_col, fg_color="transparent")
        editor_top.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        editor_top.grid_columnconfigure(0, weight=1)

        self._docs_filename = ctk.CTkLabel(
            editor_top, text="Select a file", font=FONT_SMALL,
            text_color=C_TEXT_DIM, anchor="w")
        self._docs_filename.grid(row=0, column=0, sticky="w")

        # Button row
        btn_frame = ctk.CTkFrame(editor_top, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        save_btn = ctk.CTkButton(
            btn_frame, text="SAVE", width=70, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a5a2a",
            text_color=C_GREEN, command=self._docs_save)
        save_btn.pack(side="left", padx=(0, 4))
        Tooltip(save_btn, "Save changes")

        del_btn = ctk.CTkButton(
            btn_frame, text="DELETE", width=70, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color="#5a1a1a",
            text_color=C_RED, command=self._docs_delete)
        del_btn.pack(side="left", padx=(0, 4))
        Tooltip(del_btn, "Delete this file")

        reload_btn = ctk.CTkButton(
            btn_frame, text="RELOAD", width=70, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM, command=self._docs_refresh)
        reload_btn.pack(side="left")
        Tooltip(reload_btn, "Refresh file list")

        # Editor text area
        editor_frame = HUDFrame(editor_col, glow_color=C_BORDER)
        editor_frame.grid(row=1, column=0, sticky="nsew")
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(0, weight=1)

        self._docs_editor = ctk.CTkTextbox(
            editor_frame, wrap="word", font=FONT_BODY,
            fg_color=C_PANEL, text_color=C_TEXT,
            border_width=0, corner_radius=0)
        self._docs_editor.grid(
            row=0, column=0, sticky="nsew", padx=4, pady=4)

        # State
        self._docs_current_path: str | None = None
        self._docs_items: list[dict] = []

        # Populate browser
        self._docs_refresh()

    # ================================================================
    # File browser
    # ================================================================

    def _docs_refresh(self):
        """Rescan docs and rebuild the file browser."""
        self._docs_items = scan_docs()

        # Clear browser
        for w in self._docs_browser.winfo_children():
            w.destroy()

        # Group by category
        categories: dict[str, list[dict]] = {}
        for item in self._docs_items:
            cat = item["category"]
            categories.setdefault(cat, []).append(item)

        # Render categories in order: guides first, profiles, bricks
        order = ["guides", "profiles"]
        for cat in sorted(categories.keys()):
            if cat not in order:
                order.append(cat)

        for cat in order:
            items = categories.get(cat, [])
            if not items:
                continue

            # Category header
            if cat == "guides":
                label = "GUIDES"
                color = C_CYAN
            elif cat == "profiles":
                label = "PROFILES"
                color = C_PURPLE
            elif cat.startswith("brick:"):
                brick_name = cat.split(":", 1)[1].upper()
                label = f"BRICK: {brick_name}"
                color = C_ACCENT
            else:
                label = cat.upper()
                color = C_TEXT_DIM

            ctk.CTkLabel(
                self._docs_browser, text=f"  {label}",
                font=FONT_TINY, text_color=color,
                anchor="w"
            ).pack(fill="x", pady=(8, 2))

            # File entries
            for item in items:
                self._docs_add_browser_item(item)

    def _docs_add_browser_item(self, item: dict):
        """Add a clickable file entry to the browser."""
        btn = ctk.CTkButton(
            self._docs_browser,
            text=f"  {item['name']}",
            font=FONT_SMALL, anchor="w",
            height=30, corner_radius=2,
            fg_color="transparent",
            hover_color=C_SURFACE,
            text_color=C_TEXT,
            command=lambda p=item["path"]: self._docs_open(p))
        btn.pack(fill="x", pady=1)

    # ================================================================
    # Editor operations
    # ================================================================

    def _docs_open(self, path: str):
        """Load a file into the editor."""
        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            self._docs_editor.delete("1.0", "end")
            self._docs_editor.insert("1.0", f"Error reading file:\n{e}")
            return

        self._docs_current_path = path
        p = Path(path)
        self._docs_filename.configure(
            text=p.name, text_color=C_TEXT_BRIGHT)
        self._docs_editor.delete("1.0", "end")
        self._docs_editor.insert("1.0", content)

        # Highlight current selection in browser
        for w in self._docs_browser.winfo_children():
            if isinstance(w, ctk.CTkButton):
                w.configure(fg_color="transparent")
        # Find and highlight the matching button
        for w in self._docs_browser.winfo_children():
            if isinstance(w, ctk.CTkButton):
                try:
                    if w.cget("command"):
                        # Check by matching text to item name
                        btn_text = w.cget("text").strip()
                        item_name = p.stem.replace("_", " ").title()
                        # Also check profile names
                        for item in self._docs_items:
                            if item["path"] == path:
                                item_name = item["name"]
                                break
                        if btn_text == item_name:
                            w.configure(fg_color=C_SURFACE)
                except Exception:
                    pass

    def _docs_save(self):
        """Save the editor content back to the current file."""
        if not self._docs_current_path:
            return

        content = self._docs_editor.get("1.0", "end").strip()
        path = Path(self._docs_current_path)

        try:
            path.write_text(content + "\n", encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Save Error", str(e))
            return

        # Update status
        self._docs_filename.configure(
            text=f"{path.name} (saved)", text_color=C_GREEN)
        self.after(2000, lambda: self._docs_filename.configure(
            text=path.name, text_color=C_TEXT_BRIGHT))

        # If it was a profile, re-scan profiles
        if path.parent == PROFILES_DIR and path.suffix == ".json":
            self.profiles_data = scan_docs()
            # Refresh to pick up name changes
            self._docs_refresh()

    def _docs_delete(self):
        """Delete the current file after confirmation."""
        if not self._docs_current_path:
            return

        path = Path(self._docs_current_path)
        confirm = messagebox.askyesno(
            "Delete File",
            f"Permanently delete {path.name}?")
        if not confirm:
            return

        try:
            path.unlink()
        except OSError as e:
            messagebox.showerror("Delete Error", str(e))
            return

        self._docs_current_path = None
        self._docs_filename.configure(
            text="File deleted", text_color=C_RED)
        self._docs_editor.delete("1.0", "end")
        self._docs_refresh()

        # Re-scan profiles if a profile was deleted
        if path.parent == PROFILES_DIR:
            from enigma_engine.gui.scanners import scan_profiles
            self.profiles_data = scan_profiles()

    # ================================================================
    # New file creation
    # ================================================================

    def _docs_new_file(self):
        """Create a new documentation file in information/."""
        dialog = ctk.CTkInputDialog(
            text="File name (without extension):",
            title="New Documentation File")
        name = dialog.get_input()
        if not name or not name.strip():
            return

        name = name.strip().replace(" ", "_").lower()
        if not name.endswith((".md", ".txt")):
            name += ".md"

        path = INFO_DIR / name
        if path.exists():
            messagebox.showwarning(
                "Exists", f"{name} already exists.")
            return

        # Create with template
        INFO_DIR.mkdir(parents=True, exist_ok=True)
        title = path.stem.replace("_", " ").title()
        path.write_text(
            f"# {title}\n\nWrite your content here.\n",
            encoding="utf-8")

        self._docs_refresh()
        self._docs_open(str(path))

    def _docs_new_profile(self):
        """Create a new AI profile in profiles/."""
        dialog = ctk.CTkInputDialog(
            text="Profile name:",
            title="New AI Profile")
        name = dialog.get_input()
        if not name or not name.strip():
            return

        profile_id = name.strip().replace(" ", "_").lower()
        path = PROFILES_DIR / f"{profile_id}.json"
        if path.exists():
            messagebox.showwarning(
                "Exists", f"Profile '{profile_id}' already exists.")
            return

        # Create profile with template
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profile = {
            "name": name.strip(),
            "id": profile_id,
            "version": "1.0",
            "description": "",
            "system_prompt": "You are a helpful AI assistant.",
            "personality": {
                "tone": "helpful",
                "verbosity": "balanced",
                "formality": "casual",
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 2048,
                "repetition_penalty": 1.1,
            },
        }
        path.write_text(
            json.dumps(profile, indent=2) + "\n",
            encoding="utf-8")

        # Re-scan profiles for the rest of the app
        from enigma_engine.gui.scanners import scan_profiles
        self.profiles_data = scan_profiles()

        self._docs_refresh()
        self._docs_open(str(path))
