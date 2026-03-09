"""
Enigma Engine - Forge Model Operations
=========================================

Model management: import, create, copy, rename, delete,
quantize, export GGUF.
Split from gui_forge.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import scan_models, MODELS_DIR
from enigma_engine.gui.widgets import C_TEXT, C_TEXT_DIM

logger = logging.getLogger(__name__)


class ForgeModelsMixin:
    """Model management operations for the Forge.

    Expects the host class to have ForgeMixin setup attributes.
    """

    # ================================================================
    # Model management (refresh / delete / create / import)
    # ================================================================

    def _model_op_busy(self) -> bool:
        """Check if a model operation is already running.

        Returns True (and shows a warning) if an operation is in
        progress.  All heavy model operations (copy, import, create,
        delete) must call this before starting work.
        """
        if getattr(self, "_model_op_in_progress", False):
            self._models_msg(
                "Another model operation is in progress...",
                "#f97316")
            return True
        return False

    def _import_model(self):
        """Import an external model file into the models/ directory.

        Opens a file dialog so the user can pick a model file from
        anywhere on their system.  The file is copied into models/.
        """
        if self._model_op_busy():
            return

        import shutil
        from tkinter import filedialog

        filetypes = [
            ("All model files",
             "*.gguf *.pth *.pt *.bin *.safetensors"),
            ("GGUF models", "*.gguf"),
            ("PyTorch models", "*.pth *.pt *.bin"),
            ("Safetensors", "*.safetensors"),
            ("All files", "*.*"),
        ]
        src = filedialog.askopenfilename(
            title="Import model",
            filetypes=filetypes)
        if not src:
            return  # User cancelled

        src_path = Path(src)
        dest_dir = MODELS_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src_path.name

        if dest.exists():
            self._models_msg(
                f"'{src_path.name}' already exists in models/.",
                "#e8e8e8")
            return

        size_mb = round(src_path.stat().st_size / (1024 * 1024), 1)
        self._model_op_in_progress = True
        self._models_msg(
            f"Importing {src_path.name} ({size_mb} MB)...",
            "#e8e8e8")

        def _do_import():
            try:
                shutil.copy2(str(src_path), str(dest))
                name = dest.stem
                self.after(0, lambda: self._models_msg(
                    f"Imported: {name} ({size_mb} MB)", "#22c55e"))
                self.after(0, self._refresh_models)
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda m=msg: self._models_msg(
                    f"Import failed: {m}", "#ef4444"))
            finally:
                self._model_op_in_progress = False

        threading.Thread(target=_do_import, daemon=True).start()

    def _download_huggingface(self):
        """Download a model from HuggingFace Hub into models/.

        Reads the repo ID from the inline _hf_repo_entry field on
        the MODELS page, then downloads in a background thread
        using DownloadTracker from download_progress.py.
        """
        if self._model_op_busy():
            return

        hf_entry = getattr(self, "_hf_repo_entry", None)
        if hf_entry is None:
            return
        repo_id = hf_entry.get().strip()
        if not repo_id:
            self._models_msg(
                "Enter a HuggingFace model ID.", "#e8e8e8")
            hf_entry.focus_set()
            return
        # Clear the entry after reading
        hf_entry.delete(0, "end")

        self._model_op_in_progress = True
        self._models_msg(
            f"Downloading {repo_id} from HuggingFace...",
            "#e8e8e8")

        def _on_progress(progress):
            """Update status label with download progress."""
            pct = progress.percentage
            if pct > 0:
                msg = f"Downloading {repo_id}: {pct:.0f}%"
            else:
                msg = f"Downloading {repo_id}..."
            self.after(0, lambda m=msg: self._models_msg(
                m, "#e8e8e8"))

        def _do_download():
            try:
                from enigma_engine.core.download_progress import (
                    DownloadTracker)
                dest_dir = MODELS_DIR
                dest_dir.mkdir(parents=True, exist_ok=True)
                tracker = DownloadTracker(
                    callback=_on_progress,
                    show_cli=False,
                    cache_dir=dest_dir)
                result = tracker.download_model(repo_id)
                if result is not None:
                    name = result.name
                    self.after(0, lambda n=name: self._models_msg(
                        f"Downloaded: {n}", "#22c55e"))
                    self.after(0, self._refresh_models)
                else:
                    self.after(0, lambda: self._models_msg(
                        "Download failed — check logs for details.",
                        "#ef4444"))
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda m=err_msg: self._models_msg(
                    f"Download failed: {m}", "#ef4444"))
            finally:
                self._model_op_in_progress = False

        threading.Thread(target=_do_download, daemon=True).start()

    def _refresh_models(self):
        """Refresh model cards and route dropdowns.

        Runs ``scan_models()`` in a background thread because it loads
        every native .pth/.pt file to count parameters, which blocks
        the GUI on large model directories.
        """
        def _scan_and_update():
            models = scan_models()

            def _update():
                self.models_data = models
                for w in self.model_cards_frame.winfo_children():
                    w.destroy()
                if self.models_data:
                    self._populate_model_cards(
                        self.model_cards_frame)
                model_names = (
                    ["None"]
                    + [m["name"] for m in self.models_data])
                route_menus = getattr(self, "_route_menus", {})
                for menu in route_menus.values():
                    menu.configure(values=model_names)

            self.after(0, _update)

        threading.Thread(
            target=_scan_and_update, daemon=True).start()

    def _delete_model(self, model: dict):
        """Show inline delete confirmation on the model card.

        The actual deletion happens in _confirm_delete_model after
        the user clicks YES on the inline bar.
        """
        if self._model_op_busy():
            return

        cards_frame = getattr(self, "model_cards_frame", None)
        if cards_frame is None:
            return
        # Hide all delete rows first, then show the matching one
        for card in cards_frame.winfo_children():
            dr = getattr(card, "_delete_row", None)
            if dr is not None:
                dr.grid_forget()
                dr_model = getattr(dr, "_model", None)
                if dr_model and dr_model.get("path") == model.get("path"):
                    dr.grid(
                        row=4, column=0, columnspan=2,
                        sticky="w", pady=(4, 0))

    def _confirm_delete_model(self, model: dict, delete_row=None):
        """Delete a model file after inline confirmation.

        Runs file deletion and model refresh in a background thread
        so the GUI doesn't freeze (scan_models loads every .pth to
        count params, which is slow for large models).
        """
        if delete_row:
            delete_row.grid_forget()

        if self._model_op_busy():
            return

        path = Path(model["path"])
        name = model["name"]

        # Unassign from any routes using this model
        for route_key, assigned in list(
                self.route_assignments.items()):
            if assigned == str(path):
                self._unassign_route(route_key)

        # Unload if currently loaded (quick flag reset on main thread)
        if self.model_path and Path(self.model_path) == path:
            # Save context, clear engine reference, update UI
            self._save_model_context()
            self._model_display_name = None
            self._release_loaded_engine()
            self._set_header_status("NO MODEL", C_TEXT_DIM)
            self.header_dot.set_color(C_TEXT_DIM)
            self.unload_btn.configure(state="disabled")
            self.status_bar.set_left("\u26a1 READY")
            self.route_assignments.pop("chat", None)
            route_menus = getattr(self, "_route_menus", {})
            chat_menu = route_menus.get("chat")
            if chat_menu:
                chat_menu.set("None")
            self._update_route_status()

        self._model_op_in_progress = True
        self._models_msg(f"Deleting {name}...", "#e8e8e8")

        def _do_delete():
            # Heavy work: free GPU memory, delete file, rescan
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass
            try:
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:
                    path.unlink()
                self.after(0, lambda: self._models_msg(
                    f"Deleted: {name}", "#22c55e"))
            except OSError as exc:
                err_msg = str(exc)
                self.after(0, lambda: self._models_msg(
                    f"Failed to delete: {err_msg}", "#ef4444"))
                self._model_op_in_progress = False
                return
            self.after(0, self._refresh_models)
            self._model_op_in_progress = False

        threading.Thread(target=_do_delete, daemon=True).start()

    # ----------------------------------------------------------------
    # Models page status helpers
    # ----------------------------------------------------------------

    def _models_msg(self, text: str, color: str | None = None):
        """Show a message on the MODELS page status label and status bar."""
        label = getattr(self, "_models_status", None)
        if label:
            label.configure(text=text,
                            text_color=color or C_TEXT)
        self.status_bar.set_left(text)

    def _create_new_model(self):
        """Create a new blank model with the default small preset."""
        if self._model_op_busy():
            return

        name = self.new_model_name.get().strip()
        if not name:
            self._models_msg("Enter a model name.", C_TEXT)
            return

        # Sanitize: only alphanumeric, underscore, hyphen
        safe = "".join(
            c for c in name if c.isalnum() or c in "_-")
        if not safe:
            self._models_msg("Invalid model name.", C_TEXT)
            return

        out_path = MODELS_DIR / f"{safe}.pth"
        if out_path.exists():
            self._models_msg(f"Model '{safe}' already exists.", C_TEXT)
            return

        self._model_op_in_progress = True
        self._models_msg(f"Creating {safe}...", C_TEXT)

        def _create():
            try:
                from enigma_engine.core.model import (
                    MODEL_PRESETS as MP, Enigma)
                from enigma_engine.core.tokenizer import get_tokenizer

                tokenizer = get_tokenizer("auto")

                # Always use the small preset for blank models
                preset = MP["small"]
                preset.vocab_size = tokenizer.vocab_size

                model = Enigma(config=preset)
                pc = sum(p.numel() for p in model.parameters())

                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                }, out_path)

                msg = f"Created: {safe} ({pc:,} params)"
                self.after(0, lambda: self._models_msg(
                    msg, "#22c55e"))
                self.after(0, self._refresh_models)
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda: self._models_msg(
                    f"Failed: {err_msg}", "#ef4444"))
            finally:
                self._model_op_in_progress = False

        threading.Thread(target=_create, daemon=True).start()

    # ================================================================
    # Copy model
    # ================================================================

    def _copy_model(self, model: dict):
        """Copy an existing model file with a new name.

        Preserves all trained weights so the copy can be used
        independently (e.g. for fine-tuning experiments).
        Guards against concurrent copies to prevent disk flooding.
        """
        import shutil

        if self._model_op_busy():
            return
        self._model_op_in_progress = True

        src = Path(model["path"])

        # For directory-based models (HF, sharded), copy the whole dir
        if src.is_dir():
            base = src.name
            idx = 1
            dest = src.parent / f"{base}_copy"
            while dest.exists():
                idx += 1
                dest = src.parent / f"{base}_copy{idx}"

            self._models_msg(
                f"Copying {model['name']} → {dest.name}...", "#e8e8e8")

            def _do_copy_dir():
                try:
                    shutil.copytree(str(src), str(dest))
                    size_mb = round(sum(
                        f.stat().st_size for f in dest.rglob("*")
                        if f.is_file()) / (1024 * 1024), 1)
                    self.after(0, lambda: self._models_msg(
                        f"Copied: {model['name']} → {dest.name} "
                        f"({size_mb} MB)", "#22c55e"))
                    self.after(0, self._refresh_models)
                except Exception as exc:
                    msg = str(exc)
                    self.after(0, lambda m=msg: self._models_msg(
                        f"Copy failed: {m}", "#ef4444"))
                finally:
                    self._model_op_in_progress = False

            threading.Thread(
                target=_do_copy_dir, daemon=True).start()
            return

        # Single-file model (.pth, .gguf, etc.)
        base = src.stem
        suffix = src.suffix

        # Find a unique name: base_copy, base_copy2, …
        idx = 1
        dest = src.parent / f"{base}_copy{suffix}"
        while dest.exists():
            idx += 1
            dest = src.parent / f"{base}_copy{idx}{suffix}"

        self._models_msg(
            f"Copying {model['name']} → {dest.stem}...", "#e8e8e8")

        def _do_copy():
            try:
                shutil.copy2(str(src), str(dest))
                size_mb = round(dest.stat().st_size / (1024 * 1024), 1)
                self.after(0, lambda: self._models_msg(
                    f"Copied: {model['name']} → {dest.stem} "
                    f"({size_mb} MB)", "#22c55e"))
                self.after(0, self._refresh_models)
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda m=msg: self._models_msg(
                    f"Copy failed: {m}", "#ef4444"))
            finally:
                self._model_op_in_progress = False

        threading.Thread(target=_do_copy, daemon=True).start()

    # ================================================================
    # Rename model
    # ================================================================

    def _rename_model(self, model: dict, new_name: str | None = None):
        """Rename a model file.

        Called from right-click context menu on model cards.
        If *new_name* is provided it is used directly (inline
        rename flow); otherwise does nothing.

        Updates route assignments if the model is assigned to
        any routes, and unloads the model if it is currently loaded.
        """
        if not new_name:
            return

        # ---- Rename ----
        safe = "".join(
            c for c in new_name if c.isalnum() or c in "_-.")
        if not safe:
            self._models_msg("Invalid name.", "#ef4444")
            return

        src = Path(model["path"])
        old_name = src.stem

        dest = src.parent / f"{safe}{src.suffix}"
        # Compare as strings — Path.__eq__ is case-insensitive on
        # Windows so "base.pth" == "Base.pth" would silently bail.
        case_only_change = (
            str(dest).lower() == str(src).lower()
            and str(dest) != str(src)
        )
        if str(dest) == str(src):
            return  # Truly no change

        if not case_only_change and dest.exists():
            self._models_msg(
                f"'{safe}{src.suffix}' already exists.",
                "#ef4444")
            return

        # Update route assignments that point to this model
        old_path_str = str(src)
        new_path_str = str(dest)
        for route_key, assigned in list(
                self.route_assignments.items()):
            if assigned == old_path_str:
                self.route_assignments[route_key] = new_path_str

        # Unload if currently loaded
        if self.model_path and Path(self.model_path) == src:
            self._unload_model()

        try:
            # Case-only rename on Windows needs two-step via tmp
            if case_only_change:
                tmp = src.parent / f"{safe}_tmp_rename{src.suffix}"
                src.rename(tmp)
                tmp.rename(dest)
            else:
                src.rename(dest)
            from enigma_engine.gui.scanners import (
                save_route_assignments)
            save_route_assignments(self.route_assignments)

            # Rename model context directory so chat history follows
            try:
                from enigma_engine.core.model_context import (
                    model_key_from_path, _CONTEXTS_DIR)
                old_key = model_key_from_path(str(src))
                new_key = model_key_from_path(str(dest))
                if old_key != new_key:
                    old_ctx_dir = _CONTEXTS_DIR / old_key
                    new_ctx_dir = _CONTEXTS_DIR / new_key
                    if old_ctx_dir.exists() and not new_ctx_dir.exists():
                        old_ctx_dir.rename(new_ctx_dir)
            except Exception:
                pass  # Context rename is best-effort

            self._models_msg(
                f"Renamed: {old_name} → {safe}", "#22c55e")
            self._refresh_models()
        except OSError as exc:
            self._models_msg(
                f"Rename failed: {exc}", "#ef4444")

    # ================================================================
    # Quantize student model
    # ================================================================

    def _quantize_student(self):
        """Quantize the STUDENT model in a background thread."""
        from enigma_engine.gui.scanners import ROUTE_KEYS
        student_path = self.route_assignments.get(ROUTE_KEYS[1])
        if not student_path or not Path(student_path).exists():
            self._log("[!] No student model assigned.\n")
            return

        mode = getattr(self, "quantize_mode_var", None)
        mode = mode.get() if mode else "int8"

        def _run():
            try:
                self._log(f"\n--- QUANTIZE ({mode}) ---\n")
                self._log(f"Loading: {Path(student_path).name}")

                from enigma_engine.core.model import (
                    Enigma, ModelConfig as ForgeConfig)
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location="cpu")
                cfg_dict = (
                    checkpoint.get("model_config")
                    or checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
                config = ForgeConfig(**cfg_dict)
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                model.load_state_dict(state_dict)

                self._log(f"Quantizing with mode={mode}...")
                model.quantize(mode=mode)

                # Save quantized model alongside original
                out = Path(student_path)
                stem = out.stem
                out_path = out.parent / f"{stem}_q{mode}{out.suffix}"
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                }, out_path)

                size_mb = out_path.stat().st_size / (1024 * 1024)
                self._log(f"Saved: {out_path.name} ({size_mb:.1f} MB)")
                self._log("--- QUANTIZE COMPLETE ---\n")
                self.after(0, self._refresh_models)
            except Exception as exc:
                err_msg = str(exc)
                self._log(f"[ERROR] Quantize failed: {err_msg}\n")

        import threading
        threading.Thread(
            target=_run, daemon=True,
            name="quantize-student").start()

    # ================================================================
    # Export student model to GGUF
    # ================================================================

    def _export_student_gguf(self):
        """Export the STUDENT model to GGUF in a background thread."""
        from enigma_engine.gui.scanners import ROUTE_KEYS, MODELS_DIR
        student_path = self.route_assignments.get(ROUTE_KEYS[1])
        if not student_path or not Path(student_path).exists():
            self._log("[!] No student model assigned.\n")
            return

        qtype = getattr(self, "export_gguf_mode_var", None)
        qtype = qtype.get() if qtype else "Q8_0"

        def _run():
            try:
                self._log(f"\n--- EXPORT GGUF ({qtype}) ---\n")
                self._log(f"Loading: {Path(student_path).name}")

                from enigma_engine.core.model import (
                    Enigma, ModelConfig as ForgeConfig)
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location="cpu")
                cfg_dict = (
                    checkpoint.get("model_config")
                    or checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
                config = ForgeConfig(**cfg_dict)
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                model.load_state_dict(state_dict)

                from enigma_engine.core.tokenizer import (
                    get_tokenizer)
                tokenizer = get_tokenizer("auto")

                stem = Path(student_path).stem
                out_path = str(
                    MODELS_DIR / f"{stem}-{qtype}.gguf")

                self._log(f"Exporting as {qtype}...")
                from enigma_engine.core.gguf import export_to_gguf
                export_to_gguf(
                    model, out_path,
                    quant_type=qtype,
                    tokenizer=tokenizer)

                size_mb = Path(out_path).stat().st_size / (1024 * 1024)
                self._log(
                    f"Saved: {Path(out_path).name} "
                    f"({size_mb:.1f} MB)")
                self._log("--- EXPORT COMPLETE ---\n")
                self.after(0, self._refresh_models)
            except Exception as exc:
                err_msg = str(exc)
                self._log(
                    f"[ERROR] GGUF export failed: {err_msg}\n")

        import threading
        threading.Thread(
            target=_run, daemon=True,
            name="export-gguf").start()

    # ================================================================
    # Resize model
    # ================================================================

    @staticmethod
    def _transfer_weights(src_sd: dict, dst_sd: dict) -> dict:
        """Transfer weights from *src_sd* into *dst_sd*.

        For each key present in both dicts the overlapping slice is
        copied.  Keys only in *dst_sd* are left untouched (random init).
        Returns the updated *dst_sd*.
        """

        for key in dst_sd:
            if key not in src_sd:
                continue
            s = src_sd[key]
            d = dst_sd[key]

            if s.shape == d.shape:
                dst_sd[key] = s.clone()
                continue

            # Copy the overlapping region dimension-by-dimension
            slices = tuple(
                slice(0, min(sd, dd))
                for sd, dd in zip(s.shape, d.shape)
            )
            d[slices] = s[slices].clone()

        return dst_sd

