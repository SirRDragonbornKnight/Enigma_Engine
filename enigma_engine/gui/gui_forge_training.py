"""
Enigma Engine - Forge Training Modes (Basic)
================================================

Training mode implementations: Solo, DPO, Vision, LoRA.
Split from gui_forge.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ForgeTrainingMixin:
    """Training mode implementations: Solo, DPO, Vision, LoRA.

    Expects the host class to have ForgeMixin setup attributes.
    """

    # ================================================================
    # Fine-tune existing model (TRAINER route)
    # ================================================================

    def _start_solo_training(self):
        """Train the model assigned to the STUDENT route on data.

        Solo training is standard supervised fine-tuning — the STUDENT
        learns directly from the selected training data without any
        guidance from the TRAINER model.
        """
        if self.training_active:
            return

        # Check STUDENT route assignment (model to be trained)
        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            # Try to find any data file as fallback
            fallback = DATA_DIR / "instructions.txt"
            if fallback.exists():
                data_path = str(fallback)
                self._log(
                    f"No data selected — using {fallback.name}")
            else:
                txt_files = list(DATA_DIR.glob("*.txt"))
                if txt_files:
                    data_path = str(txt_files[0])
                    self._log(
                        f"No data selected — using "
                        f"{txt_files[0].name}")
                else:
                    self._log(
                        "[!] No training data available.\n"
                        "    Add a .txt file to data/ or use "
                        "Guided mode instead.")
                    return

        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Fine-tune epochs must be 1-1000.")
            return

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return

        model_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 SOLO TRAINING...")

        self._log("--- SOLO TRAINING INITIATED ---")
        self._log(f"Student : {model_name}")
        self._log(f"Data    : {Path(data_path).name}")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _finetune():
            losses = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.tokenizer import get_tokenizer

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Load existing model
                self._log(f"Loading {model_name}...")
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location=device)
                # Prefer model_config, fall back to config, skip TrainingConfig
                cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
                config = ForgeConfig(**cfg_dict)
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                model.load_state_dict(state_dict)
                model = model.to(device)

                tokenizer = get_tokenizer("auto")
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                text = Path(data_path).read_text(encoding="utf-8")
                self._log(f"Data    : {len(text):,} chars loaded")

                forge_params = self._read_forge_train_params()
                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=forge_params["batch_size"],
                    learning_rate=lr,
                    max_grad_accumulation=forge_params["max_grad_accumulation"],
                    use_gradient_checkpointing=forge_params["use_gradient_checkpointing"],
                    save_every=max(1, epochs // 5),
                    checkpoint_dir=str(MODELS_DIR / "checkpoints"),
                    use_amp=torch.cuda.is_available(),
                    run_evaluation=True)  # Enable before/after evaluation

                trainer = Trainer(
                    model, tokenizer, train_config)

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    losses.append(loss)
                    pct = int(epoch / epochs * 100)
                    self._update_forge_progress(
                        pct, f"Epoch {epoch}/{epochs}")
                    self._log(
                        f"  Epoch {epoch:>3d}  |  "
                        f"loss {loss:.4f}")
                trainer.on_epoch_complete = on_epoch

                self._log("Training...\n")
                state = trainer.train(text)

                # Log evaluation results if available
                ppl_before = None
                ppl_after = None
                if hasattr(state, "before_eval") and hasattr(state, "after_eval"):
                    before = state.before_eval
                    after = state.after_eval
                    ppl_before = before["perplexity"]
                    ppl_after = after["perplexity"]
                    improvement = ppl_before - ppl_after
                    improvement_pct = (improvement / ppl_before * 100) if ppl_before > 0 else 0
                    self._log("\n--- EVALUATION RESULTS ---")
                    self._log(f"Before: perplexity = {ppl_before:.2f}")
                    self._log(f"After:  perplexity = {ppl_after:.2f}")
                    self._log(f"Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")

                # Save back to the student model file
                out = Path(student_path)
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }, out)

                self._log("\n--- SOLO TRAINING COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Saved to  : {out}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Solo", model_name, epochs,
                    state.best_loss,
                    before_perplexity=ppl_before,
                    after_perplexity=ppl_after)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- SOLO TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(f"\n[!] Solo training failed: {exc}")
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_finetune, daemon=True).start()

    # ================================================================
    # DPO training (preference optimization)
    # ================================================================

    def _start_dpo_training(self):
        """Train STUDENT with Direct Preference Optimization.

        Requires a JSONL data file where each line has:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        if self.training_active:
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        # DPO requires a JSONL file with preference pairs
        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log(
                "[!] No data file selected.\n"
                "    DPO requires a JSONL file with:\n"
                '    {"prompt": "...", "chosen": "...", '
                '"rejected": "..."}')
            return

        if not data_path.endswith(".jsonl"):
            self._log(
                "[!] DPO requires a .jsonl file format.\n"
                "    Each line: "
                '{"prompt": "...", "chosen": "...", "rejected": "..."}')
            return

        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Fine-tune epochs must be 1-1000.")
            return

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return

        beta_val = 0.1

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 DPO TRAINING...")

        self._log("=" * 40)
        self._log("  DPO Training")
        self._log("=" * 40)
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _dpo_train():
            try:
                import json as _json
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.tokenizer import get_tokenizer

                # Load preference data
                self._log(f"Loading preference data: {Path(data_path).name}")
                pref_data: list[dict] = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = _json.loads(line)
                            if all(k in item for k in
                                   ("prompt", "chosen", "rejected")):
                                pref_data.append(item)
                        except _json.JSONDecodeError:
                            continue

                if not pref_data:
                    self._log("[!] No valid preference pairs found.")
                    self.training_active = False
                    return

                self._log(f"Found {len(pref_data)} preference pairs")

                # Load student model
                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")
                self._log(f"Loading student: {Path(student_path).stem}")

                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location=device)

                # Read model config — prefer model_config, fall back to config,
                # skip if it looks like a TrainingConfig
                cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
                config = ForgeConfig(**cfg_dict)
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                model.load_state_dict(state_dict)
                model = model.to(device)

                tokenizer = get_tokenizer("auto")
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                # Create trainer
                from enigma_engine.core.training import (
                    Trainer, TrainingConfig)
                forge_params = self._read_forge_train_params()
                train_config = TrainingConfig(
                    epochs=epochs,
                    learning_rate=lr,
                    batch_size=1,  # DPO uses single preference pairs
                    max_grad_accumulation=forge_params["max_grad_accumulation"],
                    use_gradient_checkpointing=forge_params["use_gradient_checkpointing"],
                )
                trainer = Trainer(model, tokenizer, train_config)

                def on_progress(pct: int, msg: str):
                    self._update_forge_progress(pct, msg)
                    self.after(0, lambda: self._log(f"[{pct}%] {msg}"))

                def on_loss(loss: float):
                    self.after(
                        0, lambda l=loss: self._log(f"  loss: {l:.4f}"))

                trainer.on_progress = on_progress
                trainer.on_loss = on_loss

                self._log(
                    f"Starting DPO: {epochs} epochs, lr={lr}, "
                    f"beta={beta_val}")
                state = trainer.train_dpo(pref_data, beta=beta_val)

                # Save updated model
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch if hasattr(state, 'epoch') else epochs,
                        "best_loss": state.best_loss if hasattr(state, 'best_loss') else 0.0,
                    },
                }, student_path)
                self._log(f"Model saved to {Path(student_path).name}")
                best = state.best_loss if hasattr(state, 'best_loss') else 0.0
                self._log(
                    f"DPO complete — best loss: {best:.4f}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "DPO", Path(student_path).stem,
                    epochs, best)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- DPO TRAINING STOPPED ---")
            except Exception as e:
                self._log(f"[ERROR] DPO training failed: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_dpo_train, daemon=True).start()

    # ================================================================
    # Vision training (image-text pairs)
    # ================================================================

    def _start_vision_training(self):
        """Train STUDENT to understand images using image-text pairs.

        Uses the VisionEncoder + model's vision_projection layer.
        Data comes from a directory of image+caption pairs scanned
        by scan_vision_data (image.png + image.txt or JSONL).
        """
        if self.training_active:
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        # Get vision data directory from UI
        vision_dir_var = getattr(self, "forge_vision_dir_var", None)
        vision_dir = (vision_dir_var.get()
                      if vision_dir_var else str(DATA_DIR / "images"))
        if not vision_dir or not Path(vision_dir).exists():
            self._log(
                "[!] Vision data directory not found.\n"
                f"    Expected: {vision_dir}\n"
                "    Create image.png + image.txt pairs in the folder.")
            return

        # Scan for image-text pairs
        from enigma_engine.gui.scanners import scan_vision_data
        pairs = scan_vision_data(vision_dir)
        if not pairs:
            self._log(
                "[!] No image-text pairs found.\n"
                f"    Scanned: {vision_dir}\n"
                "    Format 1: image.png + image.txt (same name)\n"
                "    Format 2: captions.jsonl with image+text fields")
            return

        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Epochs must be 1-1000.")
            return

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return

        # Get encoder preset from UI
        preset_var = getattr(self, "forge_vision_preset_var", None)
        preset = preset_var.get() if preset_var else "small"

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 VISION TRAINING...")

        # Read focus field from UI
        focus_field = ""
        ff_widget = getattr(self, "forge_focus_field", None)
        if ff_widget is not None:
            focus_field = ff_widget.get().strip()

        self._log("=" * 40)
        self._log("  Vision Training")
        self._log("=" * 40)
        self._log(f"Data dir : {vision_dir}")
        self._log(f"Pairs    : {len(pairs)}")
        self._log(f"Encoder  : {preset}")
        if focus_field:
            self._log(f"Focus   : {focus_field}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _vision_train():
            losses: list[float] = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.vision_encoder import (
                    VisionEncoder, VISION_PRESETS)

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")
                self._log(f"Loading student: {Path(student_path).stem}")

                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location=device)

                cfg_dict = (checkpoint.get("model_config")
                            or checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})

                # Set vision_hidden_size to match encoder preset dim
                v_preset = VISION_PRESETS.get(preset, VISION_PRESETS["small"])
                cfg_dict["vision_hidden_size"] = v_preset.dim

                config = ForgeConfig(**cfg_dict)
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                # Load with strict=False since we may have added vision_projection
                model.load_state_dict(state_dict, strict=False)
                model = model.to(device)

                tokenizer = get_tokenizer("auto")
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                # Create vision encoder
                v_encoder = VisionEncoder(v_preset).to(device)
                v_params = v_encoder.param_count()
                self._log(f"Vision encoder: {v_params:,} params ({preset})")

                # Convert scanned pairs to training data format
                # scan_vision_data returns {"image": path_str, "text": caption}
                # train_vision expects {"image": PIL or path, "text": caption}
                vision_data = [
                    {"image": p["image"], "text": p["text"]}
                    for p in pairs
                ]

                forge_params = self._read_forge_train_params()
                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=1,  # Vision trains per-pair
                    learning_rate=lr,
                    max_grad_accumulation=forge_params["max_grad_accumulation"],
                    use_gradient_checkpointing=forge_params["use_gradient_checkpointing"],
                    save_every=max(1, epochs // 5),
                    checkpoint_dir=str(MODELS_DIR / "checkpoints"),
                    use_amp=torch.cuda.is_available())

                trainer = Trainer(model, tokenizer, train_config)

                def on_progress(pct: int, msg: str):
                    self._update_forge_progress(pct, msg)
                    self.after(0, lambda: self._log(f"[{pct}%] {msg}"))

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    losses.append(loss)
                    pct = int(epoch / epochs * 100)
                    self._update_forge_progress(
                        pct, f"Epoch {epoch}/{epochs}")
                    self._log(
                        f"  Epoch {epoch:>3d}  |  "
                        f"loss {loss:.4f}")

                trainer.on_progress = on_progress
                trainer.on_epoch_complete = on_epoch

                self._log("Training vision encoder...\n")
                state = trainer.train_vision(
                    vision_encoder=v_encoder,
                    data=vision_data)

                # Save model with vision projection weights
                from enigma_engine.core.safe_save import atomic_torch_save
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "model_config": config.to_dict(),
                    "vision_encoder_state": v_encoder.state_dict(),
                    "vision_encoder_config": v_preset.to_dict(),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }
                atomic_torch_save(save_dict, student_path)

                self._log("\n--- VISION TRAINING COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Saved to  : {Path(student_path).name}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Vision", Path(student_path).stem,
                    epochs, state.best_loss)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- VISION TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(f"\n[!] Vision training failed: {exc}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_vision_train, daemon=True).start()

    # ================================================================
    # LoRA training (low-rank adapter fine-tuning)
    # ================================================================

    def _start_lora_training(self):
        """Fine-tune STUDENT using LoRA adapters.

        Trains small adapter weights instead of the full model.
        Uses much less VRAM while achieving comparable quality.
        """
        if self.training_active:
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log(
                "[!] No training data selected.\n"
                "    LoRA needs a data file to train on.")
            return

        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Epochs must be 1-1000.")
            return

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return

        # Read LoRA config from UI
        rank_var = getattr(self, "forge_lora_rank_var", None)
        alpha_var = getattr(self, "forge_lora_alpha_var", None)
        try:
            lora_rank = int(rank_var.get()) if rank_var else 8
            if lora_rank < 1 or lora_rank > 128:
                raise ValueError
        except (ValueError, TypeError):
            lora_rank = 8

        try:
            lora_alpha = int(alpha_var.get()) if alpha_var else 16
            if lora_alpha < 1 or lora_alpha > 256:
                raise ValueError
        except (ValueError, TypeError):
            lora_alpha = 16

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 LoRA TRAINING...")

        # Read focus field from UI
        focus_field = ""
        ff_widget = getattr(self, "forge_focus_field", None)
        if ff_widget is not None:
            focus_field = ff_widget.get().strip()

        model_name = Path(student_path).stem
        self._log("=" * 40)
        self._log("  LoRA Training")
        self._log("=" * 40)
        self._log(f"Student : {model_name}")
        self._log(f"Rank    : {lora_rank}  |  Alpha: {lora_alpha}")
        if focus_field:
            self._log(f"Focus   : {focus_field}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _lora_train():
            losses: list[float] = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.training import (
                    Trainer, TrainingConfig)

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location=device)

                cfg_dict = (checkpoint.get("model_config")
                            or checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})

                config = ForgeConfig(**cfg_dict)
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                model.load_state_dict(state_dict)
                model = model.to(device)

                tokenizer = get_tokenizer("auto")
                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                ppl_before = None
                ppl_after = None
                # Try PEFT LoRA first, fall back to manual
                try:
                    from enigma_engine.core.lora_utils import (
                        LoraConfig as EngineLoraConfig,
                        LoraTrainer)
                    lora_cfg = EngineLoraConfig(
                        rank=lora_rank, alpha=lora_alpha)
                    lora_trainer = LoraTrainer(
                        model, tokenizer, lora_config=lora_cfg)
                    lora_trainer.config.epochs = epochs
                    lora_trainer.config.learning_rate = lr

                    text = Path(data_path).read_text(encoding="utf-8")
                    self._log(f"Data    : {len(text):,} chars loaded")

                    trainable = sum(
                        p.numel() for p in model.parameters()
                        if p.requires_grad)
                    self._log(f"LoRA trainable params: {trainable:,}")

                    def on_epoch(epoch, loss):
                        if not self.training_active:
                            raise KeyboardInterrupt("Stopped")
                        losses.append(loss)
                        self._log(
                            f"  Epoch {epoch:>3d}  |  "
                            f"loss {loss:.4f}")

                    lora_trainer.on_epoch_complete = on_epoch
                    self._log("Training LoRA adapters...\n")
                    lora_trainer.train(text)

                    # Save adapter weights alongside model
                    adapter_path = (MODELS_DIR / "checkpoints"
                                    / f"{model_name}_lora.pth")
                    adapter_path.parent.mkdir(
                        parents=True, exist_ok=True)
                    lora_trainer.save_adapter(str(adapter_path))
                    self._log(
                        f"Adapter saved: {adapter_path.name}")

                except ImportError:
                    # Fall back to standard fine-tuning with frozen layers
                    self._log(
                        "PEFT not installed — using manual "
                        "partial freeze instead.")
                    self._log("pip install peft for full LoRA support")

                    # Freeze all but last N layers (approximate LoRA)
                    n_layers = len(model.layers)
                    unfreeze = max(1, n_layers // 4)
                    for param in model.parameters():
                        param.requires_grad = False
                    for layer in model.layers[-unfreeze:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    for param in model.output.parameters():
                        param.requires_grad = True

                    trainable = sum(
                        p.numel() for p in model.parameters()
                        if p.requires_grad)
                    self._log(f"Unfrozen last {unfreeze} layers")
                    self._log(f"Trainable: {trainable:,} / {pc:,}")

                    text = Path(data_path).read_text(encoding="utf-8")
                    self._log(f"Data    : {len(text):,} chars loaded")

                    forge_params = self._read_forge_train_params()
                    train_config = TrainingConfig(
                        epochs=epochs,
                        batch_size=forge_params["batch_size"],
                        learning_rate=lr,
                        max_grad_accumulation=forge_params["max_grad_accumulation"],
                        use_gradient_checkpointing=forge_params["use_gradient_checkpointing"],
                        save_every=max(1, epochs // 5),
                        checkpoint_dir=str(MODELS_DIR / "checkpoints"),
                        use_amp=torch.cuda.is_available(),
                        run_evaluation=True)  # Enable before/after evaluation

                    trainer = Trainer(model, tokenizer, train_config)

                    def on_epoch(epoch, loss):
                        if not self.training_active:
                            raise KeyboardInterrupt("Stopped")
                        losses.append(loss)
                        self._log(
                            f"  Epoch {epoch:>3d}  |  "
                            f"loss {loss:.4f}")
                    trainer.on_epoch_complete = on_epoch

                    self._log("Training (partial freeze)...\n")
                    state = trainer.train(text)

                    # Log evaluation results if available
                    if hasattr(state, "before_eval") and hasattr(state, "after_eval"):
                        before = state.before_eval
                        after = state.after_eval
                        ppl_before = before["perplexity"]
                        ppl_after = after["perplexity"]
                        improvement = ppl_before - ppl_after
                        improvement_pct = (improvement / ppl_before * 100) if ppl_before > 0 else 0
                        self._log("\n--- EVALUATION RESULTS ---")
                        self._log(f"Before: perplexity = {ppl_before:.2f}")
                        self._log(f"After:  perplexity = {ppl_after:.2f}")
                        self._log(f"Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")

                    # Re-enable all parameters after training
                    for param in model.parameters():
                        param.requires_grad = True

                # Save updated model
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": min(losses) if losses else 0.0,
                    },
                }, student_path)

                self._log(f"\nModel saved to {Path(student_path).name}")
                best_loss = min(losses) if losses else 0.0
                if losses:
                    self._log(f"Best loss : {best_loss:.4f}")
                self._log("--- LoRA TRAINING COMPLETE ---")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "LoRA", model_name, epochs, best_loss,
                    before_perplexity=ppl_before,
                    after_perplexity=ppl_after)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- LoRA TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(f"\n[!] LoRA training failed: {exc}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_lora_train, daemon=True).start()

