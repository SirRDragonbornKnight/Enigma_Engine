"""
Enigma Engine - GUI Forge / Training Logic
=============================================

Mixin for training, tokenizer training, and model
management (create / delete / refresh).
Extracted from gui_logic.py to keep files under 800 lines.
"""
from __future__ import annotations

import threading
from pathlib import Path

from enigma_engine.gui.scanners import scan_models, DATA_DIR
from enigma_engine.gui.widgets import C_TEXT


class ForgeMixin:
    """Mixin providing training, model management, and data editing
    for EnigmaGUI.

    Expects the host class to have:
    - training_active, models_data, training_files
    - train_data_var, epochs_entry, batch_entry, lr_entry
    - model_size_var, vocab_entry
    - train_model_btn, stop_train_btn, train_tok_btn, train_log
    - data_editor, data_file_label, _editing_data_path
    - model_cards_frame, new_model_name, new_model_size
    - status_bar, route_assignments
    - _populate_model_cards, _chat_system, _chat_error
    - _unassign_route, _unload_model
    """

    @staticmethod
    def _model_config_dict(model) -> dict:
        """Extract model config as a serializable dict."""
        c = model.config
        return {
            "vocab_size": c.vocab_size, "dim": c.dim,
            "n_layers": c.n_layers, "n_heads": c.n_heads,
            "n_kv_heads": c.n_kv_heads,
            "hidden_dim": c.hidden_dim,
            "max_seq_len": c.max_seq_len, "dropout": c.dropout,
        }

    # ================================================================
    # Training data selection
    # ================================================================

    def _on_data_selected(self, choice: str):
        for f in self.training_files:
            if choice.startswith(f["name"]):
                self.train_data_var.set(f["path"])
                self._load_data_into_editor(f["path"])
                break

    def _log(self, text: str):
        def _do():
            self.train_log.configure(state="normal")
            self.train_log.insert("end", text + "\n")
            self.train_log.configure(state="disabled")
            self.train_log.see("end")
        self.after(0, _do)

    # ================================================================
    # Data file editing
    # ================================================================

    def _load_data_into_editor(self, path: str):
        """Load a data file into the FORGE data editor."""
        editor = getattr(self, "data_editor", None)
        if editor is None:
            return
        try:
            content = Path(path).read_text(encoding="utf-8")
            editor.delete("1.0", "end")
            editor.insert("1.0", content)
            self.data_file_label.configure(
                text=Path(path).name, text_color=C_TEXT)
            self._editing_data_path = path
        except OSError as exc:
            self._log(f"[!] Failed to load: {exc}")

    def _save_data_file(self):
        """Save the editor content back to the data file."""
        path = getattr(self, "_editing_data_path", None)
        if not path:
            self._log("[!] No data file loaded.")
            return
        try:
            content = self.data_editor.get("1.0", "end").strip()
            Path(path).write_text(content + "\n", encoding="utf-8")
            self._log(f"Saved: {Path(path).name}")
            self._refresh_data_files()
        except OSError as exc:
            self._log(f"[!] Failed to save: {exc}")

    def _new_data_file(self):
        """Create a new training data file in data/."""
        from tkinter import simpledialog
        name = simpledialog.askstring(
            "New Data File",
            "Enter filename (without extension):",
            parent=self)
        if not name:
            return
        safe = "".join(
            c for c in name.strip()
            if c.isalnum() or c in "_- ")
        if not safe:
            self._log("[!] Invalid filename.")
            return
        path = DATA_DIR / f"{safe}.txt"
        if path.exists():
            self._log(f"[!] '{safe}.txt' already exists.")
            return
        try:
            path.write_text("# Training Data\n\n", encoding="utf-8")
            self._log(f"Created: {safe}.txt")
            self._refresh_data_files()
            self._load_data_into_editor(str(path))
            self.train_data_var.set(str(path))
        except OSError as exc:
            self._log(f"[!] Failed to create: {exc}")

    def _refresh_data_files(self):
        """Re-scan data files and update the dropdown."""
        from enigma_engine.gui.scanners import scan_training_data
        self.training_files = scan_training_data()
        data_opts = [
            f"{f['name']} ({f['size_kb']} KB)"
            for f in self.training_files]
        menu = getattr(self, "train_data_menu", None)
        if menu and data_opts:
            menu.configure(values=data_opts)

    # ================================================================
    # Model training
    # ================================================================

    def _start_model_training(self):
        if self.training_active:
            return
        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log("[!] No training data selected or file missing.")
            return

        try:
            epochs = int(self.epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Epochs must be 1-1000.")
            return

        try:
            batch_size = int(self.batch_entry.get())
            if batch_size < 1 or batch_size > 256:
                raise ValueError
        except ValueError:
            self._log("[!] Batch size must be 1-256.")
            return

        try:
            lr = float(self.lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return

        model_size = self.model_size_var.get()
        self.training_active = True
        self.train_model_btn.configure(state="disabled")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 TRAINING...")

        self._log("--- TRAINING INITIATED ---")
        self._log(f"Data    : {Path(data_path).name}")
        self._log(f"Model   : {model_size}")
        self._log(
            f"Epochs  : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")

        def _train():
            try:
                import torch
                from enigma_engine.core.model import (
                    MODEL_PRESETS, Enigma)
                from enigma_engine.core.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.tokenizer import get_tokenizer

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                tokenizer = get_tokenizer("auto")
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                self._log(f"Building {model_size} model...")
                preset = MODEL_PRESETS.get(
                    model_size, MODEL_PRESETS["small"])
                preset.vocab_size = tokenizer.vocab_size
                model = Enigma(config=preset)
                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")
                model = model.to(device)

                text = Path(data_path).read_text(encoding="utf-8")
                self._log(f"Data    : {len(text):,} chars loaded")

                config = TrainingConfig(
                    epochs=epochs, batch_size=batch_size,
                    learning_rate=lr,
                    save_every=max(1, epochs // 5),
                    checkpoint_dir="models/checkpoints",
                    use_amp=torch.cuda.is_available())

                trainer = Trainer(
                    model, tokenizer, config, device=device)

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    self._log(
                        f"  Epoch {epoch:>3d}  |  "
                        f"loss {loss:.4f}")
                trainer.on_epoch_complete = on_epoch

                self._log("Training...\n")
                state = trainer.train(text)

                out = Path("models") / f"enigma_{model_size}.pth"
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }, out)

                tok_path = (
                    out.parent
                    / f"enigma_{model_size}_tokenizer.json")
                if hasattr(tokenizer, "save"):
                    tokenizer.save(tok_path)

                self._log("\n--- TRAINING COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Saved to  : {out}")
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- TRAINING STOPPED ---")
            except Exception as exc:
                self._log(f"\n[!] Training failed: {exc}")
            finally:
                self.training_active = False
                self.after(0, lambda: self.train_model_btn.configure(
                    state="normal"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_train, daemon=True).start()

    # ================================================================
    # Tokenizer training
    # ================================================================

    def _start_tokenizer_training(self):
        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log("[!] No training data selected.")
            return
        try:
            vocab_size = int(self.vocab_entry.get())
            if vocab_size < 100 or vocab_size > 100000:
                raise ValueError
        except ValueError:
            self._log("[!] Vocab size must be 100-100000.")
            return

        self.train_tok_btn.configure(state="disabled")
        self._log("--- TOKENIZER TRAINING ---")
        self._log(
            f"Data  : {Path(data_path).name}  "
            f"|  Vocab: {vocab_size}")

        def _train_tok():
            try:
                from enigma_engine.core.bpe_tokenizer import (
                    BPETokenizer)
                text = Path(data_path).read_text(encoding="utf-8")
                self._log(f"Loaded {len(text):,} chars")

                tokenizer = BPETokenizer()
                tokenizer.train(
                    [text], vocab_size=vocab_size, verbose=False)

                out = Path("models") / "tokenizer.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                tokenizer.save(out)

                test = "Hello, how are you?"
                enc = tokenizer.encode(test)
                dec = tokenizer.decode(enc)

                self._log(f"Saved   : {out}")
                self._log(f"Vocab   : {tokenizer.vocab_size}")
                self._log(
                    f"Test    : '{test}' -> "
                    f"{len(enc)} tokens -> '{dec}'")
                self._log("--- TOKENIZER READY ---")
            except Exception as exc:
                self._log(f"[!] Failed: {exc}")
            finally:
                self.after(0, lambda: self.train_tok_btn.configure(
                    state="normal"))

        threading.Thread(target=_train_tok, daemon=True).start()

    def _stop_training(self):
        self.training_active = False
        self._log("Stopping after current epoch...")

    # ================================================================
    # Model management (refresh / delete / create)
    # ================================================================

    def _refresh_models(self):
        """Refresh model cards and route dropdowns."""
        self.models_data = scan_models()
        for w in self.model_cards_frame.winfo_children():
            w.destroy()
        if self.models_data:
            self._populate_model_cards(self.model_cards_frame)
        # Update route dropdown options
        model_names = ["None"] + [m["name"] for m in self.models_data]
        route_menus = getattr(self, "_route_menus", {})
        for menu in route_menus.values():
            menu.configure(values=model_names)

    def _delete_model(self, model: dict):
        """Delete a model file after confirmation."""
        from tkinter import messagebox
        path = Path(model["path"])
        name = model["name"]

        if not messagebox.askyesno(
                "Delete Model",
                f"Delete {name}?\n\nThis cannot be undone."):
            return

        # Unassign from any routes using this model
        for route_key, assigned in list(
                self.route_assignments.items()):
            if assigned == str(path):
                self._unassign_route(route_key)

        # Unload if currently loaded
        if self.model_path and Path(self.model_path) == path:
            self._unload_model()

        try:
            path.unlink()
            self._chat_system(f"Deleted: {name}")
        except OSError as exc:
            self._chat_error(f"Failed to delete: {exc}")
            return

        self._refresh_models()

    def _create_new_model(self):
        """Create a new untrained model from a preset."""
        name = self.new_model_name.get().strip()
        if not name:
            self._chat_system("Enter a model name.")
            return

        # Sanitize: only alphanumeric, underscore, hyphen
        safe = "".join(
            c for c in name if c.isalnum() or c in "_-")
        if not safe:
            self._chat_system("Invalid model name.")
            return

        out_path = Path("models") / f"{safe}.pth"
        if out_path.exists():
            self._chat_system(f"Model '{safe}' already exists.")
            return

        size = self.new_model_size.get()
        self._chat_system(f"Creating {safe} ({size})...")

        def _create():
            try:
                import torch
                from enigma_engine.core.model import (
                    MODEL_PRESETS, Enigma)
                from enigma_engine.core.tokenizer import get_tokenizer

                tokenizer = get_tokenizer("auto")
                preset = MODEL_PRESETS.get(
                    size, MODEL_PRESETS["small"])
                preset.vocab_size = tokenizer.vocab_size
                model = Enigma(config=preset)
                pc = sum(p.numel() for p in model.parameters())

                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                }, out_path)

                self.after(0, lambda: self._chat_system(
                    f"Created: {safe} ({size}, {pc:,} params)"))
                self.after(0, self._refresh_models)
            except Exception as exc:
                self.after(0, lambda: self._chat_error(
                    f"Failed to create model: {exc}"))

        threading.Thread(target=_create, daemon=True).start()
