"""
Enigma Engine - GUI Docs Page
================================

Mixin providing the DOCS page for EnigmaGUI.
Handles documentation files and mod docs.

Features:
    - File browser with category sections and search filter
    - Full text editor with save/delete and Ctrl+S shortcut
    - Create new documentation files
    - Mod docs auto-discovered from mods/<id>/docs/
    - Notes files from data/notes/
    - Unsaved changes detection with visual indicator
    - Live line/word count in editor footer
"""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_BORDER, C_CYAN, C_GREEN,
    C_INPUT, C_ORANGE, C_PANEL, C_RED,
    C_SURFACE, C_TEXT, C_TEXT_BRIGHT,
    C_TEXT_DIM,
    FONT_BODY, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableLabel, Tooltip,
    themed_button, wire_hotkeys,
)
from enigma_engine.gui.scanners import (
    INFO_DIR, scan_docs,
)

# Color for notes category
C_YELLOW = "#eab308"


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
        new_btn = themed_button(
            top, "+ NEW", style="action",
            width=80, height=34,
            font=FONT_SMALL,
            command=self._docs_new_file)
        new_btn.pack(side="right", padx=(4, 0))
        Tooltip(new_btn, "Create new file")

        # Left column: file browser with search
        browser = ctk.CTkFrame(page, fg_color="transparent")
        browser.grid(row=1, column=0, sticky="nsew",
                     padx=(10, 4), pady=(2, 8))
        browser.grid_columnconfigure(0, weight=1)
        browser.grid_rowconfigure(1, weight=1)

        # Search bar at top of browser
        self._docs_search_entry = ctk.CTkEntry(
            browser, font=FONT_TINY, height=30,
            fg_color=C_INPUT, text_color=C_TEXT_BRIGHT,
            border_color=C_BORDER, border_width=1,
            placeholder_text="Search files...",
            placeholder_text_color=C_TEXT_DIM,
            corner_radius=2)
        self._docs_search_entry.grid(
            row=0, column=0, sticky="ew", pady=(0, 4))
        wire_hotkeys(self._docs_search_entry)
        self._docs_search_entry.bind(
            "<KeyRelease>", lambda e: self._docs_filter_browser())

        browser_frame = HUDFrame(browser, glow_color=C_BORDER)
        browser_frame.grid(row=1, column=0, sticky="nsew")
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

        self._docs_filename = SelectableLabel(
            editor_top, text="Select a file", font=FONT_SMALL,
            text_color=C_TEXT_DIM, anchor="w")
        self._docs_filename.grid(row=0, column=0, sticky="w")
        self._docs_filename.bind(
            "<Button-1>", lambda e: self._docs_start_rename())
        Tooltip(self._docs_filename, "Click to rename")

        # Hidden rename entry (shown when user clicks filename)
        self._docs_rename_entry = ctk.CTkEntry(
            editor_top, font=FONT_SMALL, height=28,
            fg_color=C_INPUT, text_color=C_TEXT_BRIGHT,
            border_color=C_ACCENT, border_width=1)
        wire_hotkeys(self._docs_rename_entry)
        self._docs_rename_entry.bind(
            "<Return>", lambda e: self._docs_finish_rename())
        self._docs_rename_entry.bind(
            "<Escape>", lambda e: self._docs_cancel_rename())

        # File path label (shows full path when a file is open)
        self._docs_path_label = SelectableLabel(
            editor_top, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._docs_path_label.grid(row=1, column=0, sticky="w")

        # Button row
        btn_frame = ctk.CTkFrame(editor_top, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        save_btn = themed_button(
            btn_frame, "SAVE", style="primary",
            width=70, height=30,
            font=FONT_SMALL,
            command=self._docs_save)
        save_btn.pack(side="left", padx=(0, 4))
        Tooltip(save_btn, "Save changes (Ctrl+S)")

        del_btn = themed_button(
            btn_frame, "DELETE", style="danger",
            width=70, height=30,
            font=FONT_SMALL,
            command=self._docs_delete)
        del_btn.pack(side="left", padx=(0, 4))
        Tooltip(del_btn, "Delete this file")

        reload_btn = themed_button(
            btn_frame, "RELOAD", style="secondary",
            width=70, height=30,
            font=FONT_SMALL,
            command=self._docs_refresh)
        reload_btn.pack(side="left")
        Tooltip(reload_btn, "Refresh file list")

        # Inline unsaved changes bar (hidden until needed)
        self._docs_unsaved_bar = ctk.CTkFrame(
            editor_col, fg_color="transparent")
        # Not gridded until _docs_check_unsaved shows it
        SelectableLabel(
            self._docs_unsaved_bar, text="Unsaved changes",
            font=FONT_TINY, text_color=C_ORANGE, anchor="w"
        ).pack(side="left", padx=(4, 6))
        themed_button(
            self._docs_unsaved_bar, "SAVE", style="primary",
            width=50, height=24,
            font=FONT_TINY,
            command=self._docs_unsaved_save
        ).pack(side="left", padx=(0, 4))
        themed_button(
            self._docs_unsaved_bar, "DISCARD", style="danger",
            width=60, height=24,
            font=FONT_TINY,
            command=self._docs_unsaved_discard
        ).pack(side="left", padx=(0, 4))
        themed_button(
            self._docs_unsaved_bar, "CANCEL", style="secondary",
            width=55, height=24,
            font=FONT_TINY,
            command=self._docs_unsaved_cancel
        ).pack(side="left")

        # Inline delete confirmation bar (hidden until DELETE is clicked)
        self._docs_delete_bar = ctk.CTkFrame(
            editor_col, fg_color="transparent")
        self._docs_delete_label = SelectableLabel(
            self._docs_delete_bar, text="Delete file?",
            font=FONT_TINY, text_color=C_RED, anchor="w")
        self._docs_delete_label.pack(side="left", padx=(4, 6))
        themed_button(
            self._docs_delete_bar, "YES", style="danger",
            width=40, height=24,
            font=FONT_TINY,
            command=self._docs_confirm_delete
        ).pack(side="left", padx=(0, 4))
        themed_button(
            self._docs_delete_bar, "NO", style="secondary",
            width=40, height=24,
            font=FONT_TINY,
            command=self._docs_delete_bar.grid_forget
        ).pack(side="left")

        # Pending action for unsaved changes flow
        self._docs_pending_action = None

        # Editor text area
        editor_frame = HUDFrame(editor_col, glow_color=C_BORDER)
        editor_frame.grid(row=1, column=0, sticky="nsew")
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(0, weight=1)

        self._docs_editor = ctk.CTkTextbox(
            editor_frame, wrap="word", font=FONT_BODY,
            fg_color=C_PANEL, text_color=C_TEXT,
            border_width=0, corner_radius=0, undo=True)
        self._docs_editor.grid(
            row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Enable undo/redo on the underlying tk.Text widget
        inner = self._docs_editor._textbox
        inner.configure(undo=True, maxundo=-1, autoseparators=True)

        # Ctrl+Z undo / Ctrl+Y redo
        self._docs_editor.bind(
            "<Control-z>", lambda e: self._docs_undo())
        self._docs_editor.bind(
            "<Control-y>", lambda e: self._docs_redo())

        # Track edits for unsaved detection
        self._docs_editor.bind(
            "<KeyRelease>", lambda e: self._docs_on_edit())

        # Ctrl+S keyboard shortcut
        self._docs_editor.bind(
            "<Control-s>", lambda e: self._docs_keyboard_save())

        # Ctrl+F keyboard shortcut for find bar
        self._docs_editor.bind(
            "<Control-f>", lambda e: self._docs_toggle_find())

        # Right-click context menu
        self._docs_editor.bind(
            "<Button-3>", self._docs_editor_context_menu)

        # Editor footer: stats strip
        footer = ctk.CTkFrame(editor_col, fg_color="transparent",
                              height=20)
        footer.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        footer.grid_columnconfigure(0, weight=1)

        self._docs_stats_label = SelectableLabel(
            footer, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="e")
        self._docs_stats_label.grid(row=0, column=0, sticky="e")

        # State
        self._docs_current_path: str | None = None
        self._docs_items: list[dict] = []
        self._docs_saved_content: str = ""
        self._docs_modified: bool = False

        # Populate browser
        self._docs_refresh()

        # Cancel any existing auto-save timer before starting a new one
        old_id = getattr(self, "_docs_auto_save_id", None)
        if old_id is not None:
            self.after_cancel(old_id)
        # Start auto-save timer (30s interval)
        self._docs_auto_save_id: str | None = None
        self._docs_auto_save_id = self.after(30_000, self._docs_auto_save)

    # ================================================================
    # File browser
    # ================================================================

    def _docs_refresh(self):
        """Rescan docs and rebuild the file browser."""
        self._docs_items = scan_docs()
        self._docs_rebuild_browser(self._docs_items)

    def _docs_rebuild_browser(self, items: list[dict]):
        """Rebuild the file browser with the given items list."""
        # Clear browser
        for w in self._docs_browser.winfo_children():
            w.destroy()

        # Group by category
        categories: dict[str, list[dict]] = {}
        for item in items:
            cat = item["category"]
            categories.setdefault(cat, []).append(item)

        # Render categories in order
        order = ["guides", "trainer", "data", "prompts", "notes"]
        for cat in sorted(categories.keys()):
            if cat not in order:
                order.append(cat)

        for cat in order:
            cat_items = categories.get(cat, [])
            if not cat_items:
                continue

            # Category header
            if cat == "guides":
                label = "GUIDES"
                color = C_CYAN
            elif cat == "trainer":
                label = "TRAINER"
                color = C_GREEN
            elif cat == "data":
                label = "TRAINING DATA"
                color = C_GREEN
            elif cat == "prompts":
                label = "PROMPTS"
                color = C_ORANGE
            elif cat == "notes":
                label = "NOTES"
                color = C_YELLOW
            elif cat.startswith("mod:"):
                mod_name = cat.split(":", 1)[1].upper()
                label = f"MOD: {mod_name}"
                color = C_ACCENT
            else:
                label = cat.upper()
                color = C_TEXT_DIM

            SelectableLabel(
                self._docs_browser, text=f"  {label}",
                font=FONT_TINY, text_color=color,
                anchor="w"
            ).pack(fill="x", pady=(8, 2))

            # File entries
            for item in cat_items:
                self._docs_add_browser_item(item)

    def _docs_filter_browser(self):
        """Filter the file browser by search text."""
        query = self._docs_search_entry.get().strip().lower()
        if not query:
            self._docs_rebuild_browser(self._docs_items)
            return

        filtered = [
            item for item in self._docs_items
            if query in item["name"].lower()
            or query in item.get("filename", "").lower()
            or query in item.get("category", "").lower()
        ]
        self._docs_rebuild_browser(filtered)

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
        # Show file path on hover
        Tooltip(btn, str(item["path"]))

    # ================================================================
    # Editor operations
    # ================================================================

    def _docs_open(self, path: str):
        """Load a file into the editor."""
        # Check for unsaved changes before switching
        if self._docs_modified and self._docs_current_path:
            self._docs_pending_action = lambda p=path: self._docs_open(p)
            if not self._docs_check_unsaved():
                return  # Inline bar shown, will callback

        p = Path(path)
        ext = p.suffix.lower()

        # PDF / DOCX — extract text (read-only display)
        if ext in (".pdf", ".docx"):
            try:
                from enigma_engine.core.document_readers import read_document
                content = read_document(path)
                if content is None:
                    lib = "pymupdf" if ext == ".pdf" else "python-docx"
                    content = (
                        f"Cannot read {ext} files.\n\n"
                        f"Install the required library:\n"
                        f"  pip install {lib}"
                    )
                elif len(content) > 500_000:
                    content = (
                        content[:500_000]
                        + f"\n\n--- Truncated (showing first 500,000 of "
                        f"{len(content):,} chars) ---"
                    )
            except Exception as e:
                content = f"Error reading {p.name}:\n{e}"
        else:
            try:
                content = p.read_text(encoding="utf-8")
                if len(content) > 500_000:
                    content = (
                        content[:500_000]
                        + f"\n\n--- Truncated (showing first 500,000 of "
                        f"{len(content):,} chars) ---"
                    )
            except OSError as e:
                self._docs_editor.delete("1.0", "end")
                self._docs_editor.insert("1.0", f"Error reading file:\n{e}")
                return

        # Clean up stale inline widgets from the previous file
        self._docs_cancel_rename()
        if hasattr(self, "_docs_delete_bar") \
                and self._docs_delete_bar.winfo_ismapped():
            self._docs_delete_bar.grid_forget()
        if hasattr(self, "_docs_find_bar") \
                and self._docs_find_bar.winfo_ismapped():
            self._docs_toggle_find()

        self._docs_current_path = path
        self._docs_saved_content = content
        self._docs_modified = False
        p = Path(path)
        self._docs_filename.configure(
            text=p.name, text_color=C_TEXT_BRIGHT)
        self._docs_path_label.configure(text=str(p.parent))
        self._docs_editor.delete("1.0", "end")
        self._docs_editor.insert("1.0", content)
        self._docs_update_stats()

        # Highlight current selection in browser
        for w in self._docs_browser.winfo_children():
            if isinstance(w, ctk.CTkButton):
                w.configure(fg_color="transparent")
        # Find and highlight the matching button
        for w in self._docs_browser.winfo_children():
            if isinstance(w, ctk.CTkButton):
                try:
                    if w.cget("command"):
                        btn_text = w.cget("text").strip()
                        item_name = p.stem.replace("_", " ").title()
                        for item in self._docs_items:
                            if item["path"] == path:
                                item_name = item["name"]
                                break
                        if btn_text == item_name:
                            w.configure(fg_color=C_SURFACE)
                except Exception:
                    pass

    def _docs_on_edit(self):
        """Called on each key release in the editor to track changes."""
        self._docs_mark_modified()
        self._docs_update_stats()

    def _docs_mark_modified(self):
        """Mark the current file as modified if content differs."""
        if not self._docs_current_path:
            return
        current = self._docs_editor.get("1.0", "end").strip()
        saved = self._docs_saved_content.strip()
        was_modified = self._docs_modified
        self._docs_modified = (current != saved)
        # Update filename indicator
        if self._docs_modified and not was_modified:
            p = Path(self._docs_current_path)
            self._docs_filename.configure(
                text=f"● {p.name}", text_color=C_ORANGE)
        elif not self._docs_modified and was_modified:
            p = Path(self._docs_current_path)
            self._docs_filename.configure(
                text=p.name, text_color=C_TEXT_BRIGHT)

    def _docs_check_unsaved(self) -> bool:
        """Show inline bar for unsaved changes before switching files.

        Returns:
            True if no unsaved changes (caller should proceed).
            False if unsaved changes exist (inline bar shown,
            action deferred via _docs_pending_action).
        """
        if not self._docs_modified or not self._docs_current_path:
            return True
        # Show inline unsaved bar — caller should abort
        self._docs_unsaved_bar.grid(
            row=2, column=0, sticky="ew", pady=(2, 0))
        return False

    def _docs_unsaved_save(self):
        """Save then run the pending action."""
        self._docs_unsaved_bar.grid_forget()
        self._docs_save()
        action = self._docs_pending_action
        self._docs_pending_action = None
        if action:
            action()

    def _docs_unsaved_discard(self):
        """Discard changes and run the pending action."""
        self._docs_unsaved_bar.grid_forget()
        self._docs_modified = False
        action = self._docs_pending_action
        self._docs_pending_action = None
        if action:
            action()

    def _docs_unsaved_cancel(self):
        """Cancel — abort the pending action."""
        self._docs_unsaved_bar.grid_forget()
        self._docs_pending_action = None

    def _docs_keyboard_save(self):
        """Handle Ctrl+S keyboard shortcut."""
        self._docs_save()
        return "break"

    def _docs_undo(self):
        """Handle Ctrl+Z undo in the docs editor."""
        try:
            self._docs_editor._textbox.edit_undo()
        except Exception:
            pass  # Nothing to undo
        self._docs_on_edit()
        return "break"

    def _docs_redo(self):
        """Handle Ctrl+Y redo in the docs editor."""
        try:
            self._docs_editor._textbox.edit_redo()
        except Exception:
            pass  # Nothing to redo
        self._docs_on_edit()
        return "break"

    def _docs_update_stats(self):
        """Update the line/word count in the editor footer."""
        content = self._docs_editor.get("1.0", "end").rstrip("\n")
        lines = content.count("\n") + 1 if content else 0
        words = len(content.split()) if content else 0
        chars = len(content)
        self._docs_stats_label.configure(
            text=f"{lines} lines  ·  {words} words  ·  {chars} chars")

    def _docs_save(self):
        """Save the editor content back to the current file."""
        if not self._docs_current_path:
            return

        content = self._docs_editor.get("1.0", "end").strip()
        path = Path(self._docs_current_path)

        try:
            from enigma_engine.core.safe_save import atomic_write_text
            atomic_write_text(path, content + "\n")
        except OSError as e:
            self.status_bar.set_left(f"Save failed: {e}")
            return

        # Track saved state
        self._docs_saved_content = content
        self._docs_modified = False

        # Update status
        saved_name = path.name
        self._docs_filename.configure(
            text=f"{saved_name} (saved)", text_color=C_GREEN)
        self.after(2000, lambda n=saved_name:
                   self._docs_filename.configure(
                       text=n, text_color=C_TEXT_BRIGHT))

    def _docs_delete(self):
        """Show inline delete confirmation for the current file."""
        if not self._docs_current_path:
            return

        path = Path(self._docs_current_path)
        self._docs_delete_label.configure(
            text=f"Delete {path.name}?")
        self._docs_delete_bar.grid(
            row=4, column=0, sticky="ew", pady=(2, 0))

    def _docs_confirm_delete(self):
        """Actually delete the file after inline confirmation."""
        self._docs_delete_bar.grid_forget()
        if not self._docs_current_path:
            return

        path = Path(self._docs_current_path)

        try:
            path.unlink()
        except OSError as e:
            self.status_bar.set_left(f"Delete failed: {e}")
            return

        self._docs_current_path = None
        self._docs_saved_content = ""
        self._docs_modified = False
        self._docs_filename.configure(
            text="File deleted", text_color=C_RED)
        self._docs_path_label.configure(text="")
        self._docs_stats_label.configure(text="")
        self._docs_editor.delete("1.0", "end")
        self._docs_refresh()

    # ================================================================
    # Inline rename
    # ================================================================

    def _docs_start_rename(self):
        """Show an inline entry over the filename label to rename."""
        if not self._docs_current_path:
            return

        path = Path(self._docs_current_path)
        # Hide label, show entry in same grid slot
        self._docs_filename.grid_forget()
        self._docs_rename_entry.grid(row=0, column=0, sticky="ew")
        self._docs_rename_entry.delete(0, "end")
        self._docs_rename_entry.insert(0, path.stem)
        self._docs_rename_entry.focus_set()
        self._docs_rename_entry.select_range(0, "end")
        # Cancel on focus loss
        self._docs_rename_entry.bind(
            "<FocusOut>", lambda e: self._docs_finish_rename())

    def _docs_cancel_rename(self):
        """Cancel rename — restore the filename label."""
        self._docs_rename_entry.grid_forget()
        self._docs_filename.grid(row=0, column=0, sticky="w")

    def _docs_finish_rename(self):
        """Apply the rename from the entry field."""
        if not self._docs_current_path:
            self._docs_cancel_rename()
            return

        new_stem = self._docs_rename_entry.get().strip()
        old_path = Path(self._docs_current_path)

        # Validate
        _RESERVED_NAMES = frozenset({
            'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)),
            *(f'LPT{i}' for i in range(1, 10)),
        })
        if (not new_stem
                or new_stem == old_path.stem
                or any(c in new_stem for c in r'\/:*?"<>|')
                or '..' in new_stem
                or new_stem.upper() in _RESERVED_NAMES):
            self._docs_cancel_rename()
            return

        new_name = new_stem + old_path.suffix
        new_path = old_path.parent / new_name

        if new_path.exists():
            self.status_bar.set_left(f"{new_name} already exists.")
            self._docs_cancel_rename()
            return

        try:
            old_path.rename(new_path)
        except OSError as e:
            self.status_bar.set_left(f"Rename failed: {e}")
            self._docs_cancel_rename()
            return

        # Restore label with new name
        self._docs_current_path = str(new_path)
        self._docs_cancel_rename()
        self._docs_filename.configure(
            text=new_path.name, text_color=C_TEXT_BRIGHT)
        self._docs_path_label.configure(text=str(new_path))
        self._docs_refresh()

    # ================================================================
    # New file creation
    # ================================================================

    def _docs_new_file(self):
        """Create a new blank documentation file and open it."""
        INFO_DIR.mkdir(parents=True, exist_ok=True)

        # Find next available untitled name
        name = "untitled.md"
        counter = 2
        while (INFO_DIR / name).exists():
            name = f"untitled_{counter}.md"
            counter += 1

        path = INFO_DIR / name
        path.write_text("", encoding="utf-8")

        self._docs_refresh()
        self._docs_open(str(path))

    # ================================================================
    # Editor context menu
    # ================================================================

    def _docs_editor_context_menu(self, event):
        """Show right-click context menu for the DOCS editor."""
        import tkinter as tk
        menu = tk.Menu(self, tearoff=0)
        tb = self._docs_editor._textbox
        has_sel = bool(tb.tag_ranges("sel"))
        menu.add_command(
            label="Cut",
            state="normal" if has_sel else "disabled",
            command=lambda: (
                self.clipboard_clear(),
                self.clipboard_append(tb.get("sel.first", "sel.last")),
                tb.delete("sel.first", "sel.last"),
            ))
        menu.add_command(
            label="Copy",
            state="normal" if has_sel else "disabled",
            command=lambda: (
                self.clipboard_clear(),
                self.clipboard_append(tb.get("sel.first", "sel.last")),
            ))
        menu.add_command(
            label="Paste",
            command=self._docs_paste)
        menu.add_separator()
        menu.add_command(
            label="Select All",
            command=lambda: (
                tb.tag_add("sel", "1.0", "end-1c"),
                tb.mark_set("insert", "end-1c"),
            ))
        menu.add_separator()
        menu.add_command(
            label="Find (Ctrl+F)",
            command=self._docs_toggle_find)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.destroy()

    def _docs_paste(self):
        """Paste clipboard text into DOCS editor, replacing selection."""
        try:
            text = self.clipboard_get()
        except Exception:
            return
        tb = self._docs_editor._textbox
        if tb.tag_ranges("sel"):
            tb.delete("sel.first", "sel.last")
        tb.insert("insert", text)

    # ================================================================
    # Find bar (Ctrl+F)
    # ================================================================

    def _docs_toggle_find(self):
        """Toggle the find bar visibility in the DOCS editor."""
        if hasattr(self, "_docs_find_bar") and \
                self._docs_find_bar.winfo_ismapped():
            self._docs_find_bar.grid_forget()
            # Clear highlights
            self._docs_editor._textbox.tag_remove(
                "find_hl", "1.0", "end")
            return
        self._docs_show_find_bar()

    def _docs_show_find_bar(self):
        """Create and show the find bar below the editor toolbar."""
        if not hasattr(self, "_docs_find_bar"):
            # Build the find bar once, in the editor column
            parent = self._docs_editor.master.master  # editor_col
            self._docs_find_bar = ctk.CTkFrame(
                parent, fg_color=C_SURFACE, height=36,
                corner_radius=0)
            self._docs_find_entry = ctk.CTkEntry(
                self._docs_find_bar, width=240, height=30,
                font=FONT_SMALL, fg_color=C_INPUT,
                border_color=C_BORDER, border_width=1,
                text_color=C_TEXT_BRIGHT,
                placeholder_text="Find...",
                placeholder_text_color=C_TEXT_DIM)
            self._docs_find_entry.pack(
                side="left", padx=(6, 4), pady=3)
            wire_hotkeys(self._docs_find_entry)
            self._docs_find_entry.bind(
                "<Return>", lambda e: self._docs_find_next())
            self._docs_find_entry.bind(
                "<Escape>", lambda e: self._docs_toggle_find())
            themed_button(
                self._docs_find_bar, "\u25b2",
                style="secondary", width=30, height=28,
                font=FONT_TINY,
                command=self._docs_find_prev
            ).pack(side="left", padx=1, pady=3)
            themed_button(
                self._docs_find_bar, "\u25bc",
                style="secondary", width=30, height=28,
                font=FONT_TINY,
                command=self._docs_find_next
            ).pack(side="left", padx=1, pady=3)
            self._docs_find_count = SelectableLabel(
                self._docs_find_bar, text="",
                font=FONT_TINY, text_color=C_TEXT_DIM)
            self._docs_find_count.pack(
                side="left", padx=(8, 0), pady=3)
            themed_button(
                self._docs_find_bar, "\u2716",
                style="icon", width=28, height=28,
                font=FONT_TINY,
                command=self._docs_toggle_find
            ).pack(side="right", padx=4, pady=3)
            # Configure highlight tag
            self._docs_editor._textbox.tag_configure(
                "find_hl", background="#3d3d00", foreground=C_TEXT_BRIGHT)
            self._docs_editor._textbox.tag_configure(
                "find_current", background="#665500",
                foreground=C_TEXT_BRIGHT)
        # Show find bar (grid row between toolbar and editor)
        self._docs_find_bar.grid(
            row=2, column=0, sticky="ew", pady=(0, 0))
        # Shift editor and footer rows down
        self._docs_find_entry.focus_set()
        self._docs_find_idx = "1.0"

    def _docs_find_next(self):
        """Find and highlight the next occurrence."""
        query = self._docs_find_entry.get()
        if not query:
            return
        tb = self._docs_editor._textbox
        tb.tag_remove("find_hl", "1.0", "end")
        tb.tag_remove("find_current", "1.0", "end")
        # Count all matches and highlight them
        count = 0
        start = "1.0"
        while True:
            pos = tb.search(query, start, stopindex="end",
                            nocase=True)
            if not pos:
                break
            end_pos = f"{pos}+{len(query)}c"
            tb.tag_add("find_hl", pos, end_pos)
            count += 1
            start = end_pos
        # Find next from current position
        search_from = getattr(self, "_docs_find_idx", "1.0")
        pos = tb.search(query, search_from, stopindex="end",
                        nocase=True)
        if not pos:
            # Wrap around
            pos = tb.search(query, "1.0", stopindex="end",
                            nocase=True)
        if pos:
            end_pos = f"{pos}+{len(query)}c"
            tb.tag_add("find_current", pos, end_pos)
            tb.see(pos)
            tb.mark_set("insert", end_pos)
            self._docs_find_idx = end_pos
        self._docs_find_count.configure(
            text=f"{count} match{'es' if count != 1 else ''}")

    def _docs_find_prev(self):
        """Find and highlight the previous occurrence."""
        query = self._docs_find_entry.get()
        if not query:
            return
        tb = self._docs_editor._textbox
        tb.tag_remove("find_hl", "1.0", "end")
        tb.tag_remove("find_current", "1.0", "end")
        # Count and highlight all
        count = 0
        start = "1.0"
        while True:
            pos = tb.search(query, start, stopindex="end",
                            nocase=True)
            if not pos:
                break
            end_pos = f"{pos}+{len(query)}c"
            tb.tag_add("find_hl", pos, end_pos)
            count += 1
            start = end_pos
        # Find previous from current position
        search_from = getattr(self, "_docs_find_idx", "end")
        pos = tb.search(query, search_from, stopindex="1.0",
                        backwards=True, nocase=True)
        if not pos:
            pos = tb.search(query, "end", stopindex="1.0",
                            backwards=True, nocase=True)
        if pos:
            end_pos = f"{pos}+{len(query)}c"
            tb.tag_add("find_current", pos, end_pos)
            tb.see(pos)
            tb.mark_set("insert", pos)
            self._docs_find_idx = pos
        self._docs_find_count.configure(
            text=f"{count} match{'es' if count != 1 else ''}")

    # ================================================================
    # Auto-save
    # ================================================================

    def _docs_auto_save(self):
        """Auto-save the current document if modified."""
        if getattr(self, "_docs_modified", False) and \
                self._docs_current_path:
            try:
                content = self._docs_editor.get("1.0", "end").strip()
                from enigma_engine.core.safe_save import atomic_write_text
                atomic_write_text(Path(self._docs_current_path),
                                  content + "\n")
                self._docs_saved_content = content
                self._docs_modified = False
                # Update filename indicator
                if hasattr(self, "_docs_filename"):
                    name = Path(self._docs_current_path).name
                    self._docs_filename.configure(
                        text=name, text_color=C_TEXT_BRIGHT)
            except Exception:
                pass  # Silently skip — user can manual-save
        # Re-schedule (30 seconds)
        if hasattr(self, "after"):
            self._docs_auto_save_id = self.after(30_000, self._docs_auto_save)
