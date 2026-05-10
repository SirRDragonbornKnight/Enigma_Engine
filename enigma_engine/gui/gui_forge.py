"""
Enigma Engine - GUI Forge / Training Logic
=============================================

Mixin for training setup, dispatch, and utilities.
Training modes: gui_forge_training.py, gui_forge_advanced.py
Tools: gui_forge_tools.py
Model ops: gui_forge_models.py
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR
from enigma_engine.gui.widgets import (
    C_GREEN, C_GREEN_DIM, C_SURFACE, C_TEXT)

# Re-export so existing imports keep working
from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
from enigma_engine.gui.gui_forge_advanced import ForgeAdvancedMixin
from enigma_engine.gui.gui_forge_adaptive import ForgeAdaptiveMixin
from enigma_engine.gui.gui_forge_tools import ForgeToolsMixin
from enigma_engine.gui.gui_forge_models import ForgeModelsMixin
from enigma_engine.gui.gui_forge_queue import ForgeQueueMixin
from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin
from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

logger = logging.getLogger(__name__)


class ForgeMixin(
        ForgeTrainingMixin, ForgeAdvancedMixin,
        ForgeAdaptiveMixin, ForgeNewModesMixin,
        ForgeToolsMixin, ForgeModelsMixin,
        ForgeQueueMixin, ForgeTeacherMixin):
    """Mixin providing training, model management, and fine-tuning
    for EnigmaGUI.

    Expects the host class to have:
    - training_active, models_data, training_files
    - train_data_var, epochs_entry, lr_entry
    - ft_epochs_entry, ft_lr_entry, solo_train_btn
    - guided_epochs_entry, guided_lr_entry, guided_pairs_entry
    - guided_train_btn, generate_data_btn, evaluate_btn
    - dialogue_rounds_entry, dialogue_train_btn
    - save_ckpt_btn, load_ckpt_btn
    - train_model_btn, stop_train_btn, train_tok_btn, train_log
    - _editing_data_path
    - model_cards_frame, new_model_name (MODELS page)
    - _models_status (CTkLabel on MODELS page for create/delete feedback)
    - _forge_trainer_dot, _forge_trainer_name, _forge_trainer_info
    - _forge_student_dot, _forge_student_name, _forge_student_info
    - _forge_student_params (CTkLabel for param count after training)
    - _brief_field_entries, _brief_custom_text (Training Brief UI)
    - status_bar, route_assignments
    - _populate_model_cards, _chat_system, _chat_error
    - _unassign_route, _unload_model
    - _copy_model, _transfer_weights
    """

    # Quick profile fields: (label, placeholder, tooltip)
    _QUICK_PROFILE_FIELDS = (
        ("Personality", "e.g. cheerful, sarcastic, calm",
         "Core personality traits the AI should have"),
        ("Tone", "e.g. casual, professional, playful",
         "How the AI should sound in conversation"),
        ("Expertise", "e.g. cooking, coding, fitness",
         "Subject areas the AI should focus on"),
        ("Response style", "e.g. short and punchy, detailed",
         "How long and structured responses should be"),
        ("Example phrases", "e.g. 'lets get cookin', 'no excuses'",
         "Phrases the AI should use naturally"),
    )

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
            "use_rope": c.use_rope,
            "use_moe": c.use_moe,
        }

    # ================================================================
    # Common training validation
    # ================================================================

    def _validate_epochs_lr(self) -> tuple[int, float] | None:
        """Parse and validate epochs + learning rate from the UI.

        Returns ``(epochs, lr)`` on success, ``None`` on failure
        (with a message logged to the FORGE panel).
        """
        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except (ValueError, AttributeError):
            self._log("[!] Epochs must be a number between 1 and 1000.")
            return None

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except (ValueError, AttributeError):
            self._log("[!] Learning rate must be a number between 0 and 1.")
            return None

        return epochs, lr

    def _validate_general_data_path(self) -> bool:
        """Check that the general-mix data path actually exists.

        Only relevant when general mix ratio > 0 and a path is set.
        Returns True if OK (or not applicable), False on error.
        """
        mix_var = getattr(self, "forge_general_mix_var", None)
        data_var = getattr(self, "forge_general_data_var", None)
        if mix_var is None or data_var is None:
            return True

        try:
            ratio = int(mix_var.get().strip())
        except (ValueError, TypeError):
            return True  # Ratio itself is invalid — caught elsewhere

        path = data_var.get().strip()
        if ratio > 0 and path and not Path(path).exists():
            self._log(
                f"[!] General knowledge file not found:\n"
                f"    {path}\n"
                "    Either clear the path or browse for a valid file.")
            return False
        return True

    def _validate_jsonl_structure(
        self, data_path: str, required_keys: tuple[str, ...],
    ) -> bool:
        """Peek at a JSONL file to verify it has the required keys.

        Reads up to 5 lines and checks that at least one valid record
        exists. Returns True on success, False on failure.
        """
        import json as _json
        dp = Path(data_path)
        if not dp.exists():
            self._log(f"[!] Data file not found: {data_path}")
            return False
        if not data_path.endswith(".jsonl"):
            self._log(
                "[!] Expected a .jsonl file, got: "
                f"{dp.suffix or '(no extension)'}\n"
                "    Each line must be a JSON object with keys: "
                + ", ".join(required_keys))
            return False

        try:
            valid = 0
            checked = 0
            with open(dp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    checked += 1
                    if checked > 10:
                        break
                    try:
                        obj = _json.loads(line)
                        if all(k in obj for k in required_keys):
                            valid += 1
                    except _json.JSONDecodeError:
                        continue
            if valid == 0:
                self._log(
                    "[!] No valid records found in the JSONL file.\n"
                    f"    Checked first {checked} lines.\n"
                    "    Each line must have keys: "
                    + ", ".join(required_keys) + "\n"
                    '    Example: {"'
                    + '": "...", "'.join(required_keys)
                    + '": "..."}')
                return False
        except OSError as exc:
            self._log(f"[!] Cannot read data file: {exc}")
            return False
        return True

    def _log_training_summary(self, mode: str, **fields) -> None:
        """Log a compact pre-flight summary of all training settings.

        Called right before training starts so the user can spot
        any mistakes in the output log.
        """
        self._log("=" * 44)
        self._log(f"  {mode}")
        self._log("=" * 44)
        for label, value in fields.items():
            # Right-pad label to 10 chars for alignment
            self._log(f"{label:<10}: {value}")
        # Also show the advanced params that apply to every mode
        forge_params = self._read_forge_train_params()
        batch = forge_params["batch_size"]
        self._log(f"{'Batch':<10}: {'auto' if batch == 0 else batch}")
        accum = forge_params["max_grad_accumulation"]
        if accum > 1:
            self._log(f"{'Accum':<10}: {accum}")
        if forge_params["use_gradient_checkpointing"]:
            self._log(f"{'GradCkpt':<10}: ON")
        mix = forge_params["general_mix_ratio"]
        if mix > 0:
            self._log(f"{'Mix':<10}: {mix:.0%}")
        val = forge_params["val_split"]
        if val > 0:
            self._log(f"{'ValSplit':<10}: {val}")
        self._log("")

    # ================================================================
    # Read FORGE training config from UI entries
    # ================================================================

    def _read_forge_train_params(self) -> dict:
        """Read batch_size, grad_accum, grad_ckpt, rolling_best_k from FORGE UI.

        Returns a dict suitable for passing to TrainingConfig().
        Falls back to sensible defaults when entries are missing
        or contain invalid values, with warnings for bad input.
        """
        batch_size = 0  # 0 = auto (Trainer estimates from GPU memory)
        grad_accum = 1
        grad_ckpt = False
        rolling_best_k = 0

        try:
            raw = getattr(self, "forge_batch_entry", None).get().strip()
            if raw.lower() != "auto":
                val = int(raw)
                if val >= 1:
                    batch_size = val
                else:
                    self._log(f"[!] Batch size '{raw}' invalid "
                              f"(must be >= 1), using auto")
        except (ValueError, TypeError):
            raw = getattr(self, "forge_batch_entry", None)
            raw = raw.get().strip() if raw else ""
            if raw:  # Non-empty garbage
                self._log(f"[!] Batch size '{raw}' not a number, "
                          f"using auto")
        except AttributeError:
            pass

        try:
            raw = getattr(self, "forge_accum_entry", None).get().strip()
            val = int(raw)
            if val >= 1:
                grad_accum = val
            else:
                self._log(f"[!] Grad accumulation '{raw}' invalid "
                          f"(must be >= 1), using 1")
        except (ValueError, TypeError):
            raw_w = getattr(self, "forge_accum_entry", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Grad accumulation '{raw}' not a "
                          f"number, using 1")
        except AttributeError:
            pass

        try:
            grad_ckpt = bool(
                getattr(self, "forge_grad_ckpt_var", None).get())
        except (TypeError, AttributeError):
            pass

        try:
            raw = getattr(
                self, "forge_rolling_k_entry", None).get().strip()
            val = int(raw)
            if val >= 0:
                rolling_best_k = val
            else:
                self._log(f"[!] Rolling best K '{raw}' invalid "
                          f"(must be >= 0), using 0")
        except (ValueError, TypeError):
            raw_w = getattr(self, "forge_rolling_k_entry", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Rolling best K '{raw}' not a "
                          f"number, using 0")
        except AttributeError:
            pass

        val_split = 0.1
        try:
            raw = getattr(
                self, "forge_val_split_entry", None).get().strip()
            val = float(raw)
            if 0.0 <= val <= 0.5:
                val_split = val
            else:
                self._log(f"[!] Val split '{raw}' out of range "
                          f"(must be 0.0-0.5), using 0.1")
        except (ValueError, TypeError):
            raw_w = getattr(self, "forge_val_split_entry", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Val split '{raw}' not a number, "
                          f"using 0.1")
        except AttributeError:
            pass

        min_lr_ratio = 0.1
        try:
            raw = getattr(
                self, "forge_min_lr_ratio_entry", None).get().strip()
            val = float(raw)
            if 0.0 <= val <= 1.0:
                min_lr_ratio = val
            else:
                self._log(f"[!] Min LR ratio '{raw}' out of range "
                          f"(must be 0.0-1.0), using 0.1")
        except (ValueError, TypeError):
            raw_w = getattr(self, "forge_min_lr_ratio_entry", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Min LR ratio '{raw}' not a number, "
                          f"using 0.1")
        except AttributeError:
            pass

        # Knowledge preservation settings
        general_mix_ratio = 0.0
        try:
            raw = getattr(
                self, "forge_general_mix_var", None).get().strip()
            val = int(raw)
            if 0 <= val <= 100:
                general_mix_ratio = val / 100.0
            else:
                self._log(f"[!] General mix '{raw}%' out of range "
                          f"(must be 0-100), using 0%")
        except (ValueError, TypeError):
            raw_w = getattr(self, "forge_general_mix_var", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] General mix '{raw}' not a number, "
                          f"using 0% (risk of forgetting!)")
        except AttributeError:
            pass

        general_data = ""
        try:
            general_data = str(getattr(
                self, "forge_general_data_var", None).get()).strip()
        except (TypeError, AttributeError):
            pass

        return {
            "batch_size": batch_size,
            "max_grad_accumulation": grad_accum,
            "use_gradient_checkpointing": grad_ckpt,
            "rolling_best_k": rolling_best_k,
            "val_split": val_split,
            "min_lr_ratio": min_lr_ratio,
            "general_mix_ratio": general_mix_ratio,
            "general_data": general_data,
            "ce_chunk_size": self._get_ce_chunk_size(),
        }

    def _get_ce_chunk_size(self) -> int:
        """Compute ce_chunk_size scaled to available VRAM."""
        try:
            from enigma_engine.core.hardware_detection import (
                TrainingMemoryBudget,
            )
            return TrainingMemoryBudget().ce_chunk_size
        except Exception:
            return 4096  # safe fallback

    # ================================================================
    # Training Brief — user describes what the AI should be
    # ================================================================

    def _build_training_brief(self) -> str:
        """Assemble training brief from training topic, quick profile fields, + custom text.

        Reads the student model name from the route assignment,
        training topic (for AI-Guided mode), the quick profile entries,
        and the custom brief textbox, then combines them into a single
        string for injection into the trainer system prompt.

        Returns:
            Combined brief string, or empty string if nothing filled.
        """
        parts = []

        # Include training topic if present (AI-Guided mode)
        topic_widget = getattr(self, "forge_training_topic", None)
        if topic_widget is not None:
            topic = topic_widget.get().strip()
            if topic:
                parts.append(f"Training Goal: {topic}")

        # Auto-include the student model name from route assignment
        student_path = getattr(self, "route_assignments", {}).get("student", "")
        if student_path:
            student_name = Path(student_path).stem
            parts.append(f"Name: {student_name}")

        # Gather quick profile fields
        quick_fields = getattr(self, "_brief_field_entries", {})
        for label, entry in quick_fields.items():
            value = entry.get().strip() if hasattr(entry, "get") else ""
            if value:
                parts.append(f"{label}: {value}")

        # Gather custom brief text
        custom_tb = getattr(self, "_brief_custom_text", None)
        if custom_tb is not None:
            try:
                custom = custom_tb.get("1.0", "end-1c").strip()
            except Exception:
                custom = ""
            if custom:
                parts.append(custom)

        return "\n".join(parts)

    def _save_training_brief(self):
        """Save training brief fields to data/training_brief.json."""

        data = {}

        mode_var = getattr(self, "training_mode_var", None)
        if mode_var is not None:
            data["_training_mode"] = mode_var.get()
        
        # Save training topic (AI-Guided mode)
        topic_widget = getattr(self, "forge_training_topic", None)
        if topic_widget is not None:
            data["_training_topic"] = topic_widget.get().strip()
        
        # Persist val_split alongside the brief
        try:
            val = float(getattr(
                self, "forge_val_split_entry", None).get())
            if 0.0 <= val <= 0.5:
                data["val_split"] = val
        except (ValueError, TypeError, AttributeError):
            pass

        quick_fields = getattr(self, "_brief_field_entries", {})
        for label, entry in quick_fields.items():
            value = entry.get().strip() if hasattr(entry, "get") else ""
            data[label] = value

        custom_tb = getattr(self, "_brief_custom_text", None)
        if custom_tb is not None:
            try:
                data["_custom"] = custom_tb.get(
                    "1.0", "end-1c").strip()
            except Exception:
                data["_custom"] = ""

        # Persist hyperparameters
        for attr, key in (("forge_batch_entry", "_batch_size"),
                          ("forge_accum_entry", "_grad_accum"),
                          ("forge_rolling_k_entry", "_rolling_best_k")):
            try:
                data[key] = getattr(self, attr, None).get().strip()
            except (ValueError, TypeError, AttributeError):
                pass
        try:
            data["_grad_ckpt"] = bool(
                getattr(self, "forge_grad_ckpt_var", None).get())
        except (TypeError, AttributeError):
            pass

        # Persist pre-train settings
        pretrain_data = getattr(self, "pretrain_data_var", None)
        if pretrain_data is not None:
            data["_pretrain_data_path"] = pretrain_data.get()

        # Persist epochs and learning rate
        for attr, key in (("ft_epochs_entry", "_epochs"),
                          ("ft_lr_entry", "_lr")):
            try:
                data[key] = getattr(self, attr, None).get().strip()
            except (ValueError, TypeError, AttributeError):
                pass

        # Persist training preset selection
        preset_var = getattr(self, "_forge_preset_var", None)
        if preset_var is not None:
            data["_preset"] = preset_var.get()

        # Persist pre-train widget values
        for attr, key in (("pretrain_vocab_var", "_pretrain_vocab"),):
            try:
                data[key] = getattr(self, attr, None).get()
            except (TypeError, AttributeError):
                pass
        for attr, key in (("pretrain_retrain_tok_var", "_pretrain_retrain_tok"),
                          ("pretrain_utf8_bytes_var", "_pretrain_utf8")):
            try:
                data[key] = bool(getattr(self, attr, None).get())
            except (TypeError, AttributeError):
                pass

        # Persist LoRA settings
        for attr, key in (("forge_lora_rank_var", "_lora_rank"),
                          ("forge_lora_alpha_var", "_lora_alpha")):
            try:
                data[key] = getattr(self, attr, None).get()
            except (TypeError, AttributeError):
                pass

        # Persist Basic mode data path
        train_data = getattr(self, "train_data_var", None)
        if train_data is not None:
            data["_train_data_path"] = train_data.get()

        # Persist boolean checkboxes
        for attr, key in (("forge_reasoning_var", "_reasoning"),
                          ("forge_evolutionary_var", "_evolutionary"),
                          ("forge_auto_train_var", "_auto_train"),
                          ("forge_resume_var", "_resume_training")):
            try:
                data[key] = bool(getattr(self, attr, None).get())
            except (TypeError, AttributeError):
                pass

        # Persist StringVar controls not yet covered
        for attr, key in (("forge_general_mix_var", "_general_mix"),
                          ("forge_general_data_var", "_general_data_path"),
                          ("distill_num_examples_var", "_distill_num_examples"),
                          ("distill_max_tokens_var", "_distill_max_tokens"),
                          ("ai_supplement_var", "_ai_supplement_path"),
                          ("forge_vision_dir_var", "_vision_dir"),
                          ("forge_vision_preset_var", "_vision_preset"),
                          ("training_stage_var", "_training_stage"),
                          ("forge_replay_capacity_var", "_replay_capacity"),
                          ("forge_replay_ratio_var", "_replay_ratio"),
                          ("quantize_mode_var", "_quantize_mode"),
                          ("export_gguf_mode_var", "_export_gguf_mode")):
            try:
                val = getattr(self, attr, None).get()
                if val:
                    data[key] = val
            except (TypeError, AttributeError):
                pass

        # Persist numeric entry widgets not yet covered
        for attr, key in (("guided_pairs_entry", "_guided_pairs"),
                          ("web_learn_pages_entry", "_web_learn_pages"),
                          ("vocab_entry", "_vocab_size")):
            try:
                val = getattr(self, attr, None).get().strip()
                if val:
                    data[key] = val
            except (TypeError, AttributeError):
                pass

        path = DATA_DIR / "training_brief.json"
        try:
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(path, data)
        except OSError:
            pass

    def _load_training_brief(self):
        """Load training brief fields from data/training_brief.json."""
        import json

        path = DATA_DIR / "training_brief.json"
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        mode_var = getattr(self, "training_mode_var", None)
        saved_mode = data.get("_training_mode", "")
        if mode_var is not None and saved_mode:
            mode_var.set(saved_mode)
            self._on_training_mode_changed(saved_mode)

        # Load training topic (AI-Guided mode)
        topic_widget = getattr(self, "forge_training_topic", None)
        if topic_widget is not None:
            topic_value = data.get("_training_topic", "")
            if topic_value:
                topic_widget.delete(0, "end")
                topic_widget.insert(0, topic_value)
        
        quick_fields = getattr(self, "_brief_field_entries", {})
        for label, entry in quick_fields.items():
            value = data.get(label, "")
            if value and hasattr(entry, "delete") and hasattr(entry, "insert"):
                entry.delete(0, "end")
                entry.insert(0, value)

        custom_tb = getattr(self, "_brief_custom_text", None)
        custom_val = data.get("_custom", "")
        if custom_tb is not None and custom_val:
            try:
                custom_tb.delete("1.0", "end")
                custom_tb.insert("1.0", custom_val)
            except Exception:
                pass

        # Restore hyperparameters
        for attr, key in (("forge_batch_entry", "_batch_size"),
                          ("forge_accum_entry", "_grad_accum"),
                          ("forge_rolling_k_entry", "_rolling_best_k")):
            val = data.get(key)
            widget = getattr(self, attr, None)
            if val is not None and widget is not None:
                try:
                    widget.delete(0, "end")
                    widget.insert(0, str(val))
                except Exception:
                    pass
        grad_ckpt = data.get("_grad_ckpt")
        grad_var = getattr(self, "forge_grad_ckpt_var", None)
        if grad_ckpt is not None and grad_var is not None:
            try:
                grad_var.set(grad_ckpt)
            except Exception:
                pass

        # Restore pre-train data path
        pretrain_path = data.get("_pretrain_data_path")
        pretrain_var = getattr(self, "pretrain_data_var", None)
        if pretrain_path and pretrain_var is not None:
            pretrain_var.set(pretrain_path)

        # Restore epochs and learning rate
        for attr, key in (("ft_epochs_entry", "_epochs"),
                          ("ft_lr_entry", "_lr")):
            val = data.get(key)
            widget = getattr(self, attr, None)
            if val is not None and widget is not None:
                try:
                    widget.delete(0, "end")
                    widget.insert(0, str(val))
                except Exception:
                    pass

        # Restore training preset selection
        preset_val = data.get("_preset")
        preset_var = getattr(self, "_forge_preset_var", None)
        if preset_val and preset_var is not None:
            try:
                preset_var.set(preset_val)
            except Exception:
                pass

        # Restore pre-train widget values
        for attr, key in (("pretrain_vocab_var", "_pretrain_vocab"),):
            val = data.get(key)
            var = getattr(self, attr, None)
            if val is not None and var is not None:
                try:
                    var.set(str(val))
                except Exception:
                    pass
        for attr, key in (("pretrain_retrain_tok_var", "_pretrain_retrain_tok"),
                          ("pretrain_utf8_bytes_var", "_pretrain_utf8")):
            val = data.get(key)
            var = getattr(self, attr, None)
            if val is not None and var is not None:
                try:
                    var.set(val)
                except Exception:
                    pass

        # Restore LoRA settings
        for attr, key in (("forge_lora_rank_var", "_lora_rank"),
                          ("forge_lora_alpha_var", "_lora_alpha")):
            val = data.get(key)
            var = getattr(self, attr, None)
            if val is not None and var is not None:
                try:
                    var.set(str(val))
                except Exception:
                    pass

        # Restore Basic mode data path
        train_path = data.get("_train_data_path")
        train_var = getattr(self, "train_data_var", None)
        if train_path and train_var is not None:
            train_var.set(train_path)

        # Restore boolean checkboxes
        for attr, key in (("forge_reasoning_var", "_reasoning"),
                          ("forge_evolutionary_var", "_evolutionary"),
                          ("forge_auto_train_var", "_auto_train"),
                          ("forge_resume_var", "_resume_training")):
            val = data.get(key)
            var = getattr(self, attr, None)
            if val is not None and var is not None:
                try:
                    var.set(bool(val))
                except Exception:
                    pass

        # Restore StringVar controls not yet covered
        for attr, key in (("forge_general_mix_var", "_general_mix"),
                          ("forge_general_data_var", "_general_data_path"),
                          ("distill_num_examples_var", "_distill_num_examples"),
                          ("distill_max_tokens_var", "_distill_max_tokens"),
                          ("ai_supplement_var", "_ai_supplement_path"),
                          ("forge_vision_dir_var", "_vision_dir"),
                          ("forge_vision_preset_var", "_vision_preset"),
                          ("training_stage_var", "_training_stage"),
                          ("forge_replay_capacity_var", "_replay_capacity"),
                          ("forge_replay_ratio_var", "_replay_ratio"),
                          ("quantize_mode_var", "_quantize_mode"),
                          ("export_gguf_mode_var", "_export_gguf_mode")):
            val = data.get(key)
            var = getattr(self, attr, None)
            if val and var is not None:
                try:
                    var.set(str(val))
                except Exception:
                    pass

        # Restore numeric entry widgets not yet covered
        for attr, key in (("guided_pairs_entry", "_guided_pairs"),
                          ("web_learn_pages_entry", "_web_learn_pages"),
                          ("vocab_entry", "_vocab_size")):
            val = data.get(key)
            widget = getattr(self, attr, None)
            if val is not None and widget is not None:
                try:
                    widget.delete(0, "end")
                    widget.insert(0, str(val))
                except Exception:
                    pass

    def _on_training_mode_selected(self):
        """Apply and persist the selected FORGE training mode."""
        mode_var = getattr(self, "training_mode_var", None)
        mode = mode_var.get() if mode_var is not None else "Basic"
        self._on_training_mode_changed(mode)
        self._save_training_brief()

    # ================================================================
    # Shared helpers for FORGE operations
    # ================================================================

    @staticmethod
    def _format_training_pair(
        stage: str,
        prompt: str,
        response: str,
    ) -> str:
        """Format a prompt+response pair for the given training stage.

        Ensures supplement data and corrections match the format the
        stage expects, so ``_parse_training_data`` will parse them
        correctly alongside generated curriculum.

        - basics: raw text (parser handles paragraphs)
        - conversation: User/AI dialogue turns
        - commands / web: Q&A format (natural for tool usage)

        Args:
            stage: Training stage name.
            prompt: The question or user message.
            response: The answer or AI response.

        Returns:
            A formatted training example string.
        """
        if stage == "conversation":
            return f"User: {prompt}\nAI: {response}"
        if stage == "basics":
            return f"{prompt}\n{response}"
        # commands, web, and any unknown stage use Q&A
        return f"Q: {prompt}\nA: {response}"

    @staticmethod
    def _build_generation_prompt(
        index: int,
        total: int,
        stage: str,
        reasoning: bool = False,
    ) -> str:
        """Build a stage-appropriate prompt for the teacher to generate
        one training example.

        Different stages produce different data formats:
        - basics: short statements, simple facts, greetings
        - conversation: multi-turn User/AI dialogue
        - commands: Q&A with [CMD] block usage
        - web: Q&A with search/fetch tool usage

        When ``reasoning=True`` (CoT-B), prompts instruct the teacher
        to include ``<think>...</think>`` reasoning chains before the
        answer.  This teaches the student to reason, not just memorize.

        The trainer's ``_parse_training_data`` handles all of these
        formats (raw text, User/AI, Q/A) so the student can learn
        from any shape of text.

        Args:
            index: 1-based index of this example in the batch.
            total: Total examples to generate.
            stage: Training stage name.
            reasoning: If True, include <think> reasoning chains
                in the generated examples (CoT-B).

        Returns:
            A prompt string to send to the teacher.
        """
        # Reasoning suffix appended to prompts when CoT-B is enabled
        cot_suffix = ""
        if reasoning:
            cot_suffix = (
                "\n\nIMPORTANT: Include a reasoning chain before the "
                "answer. Show your thinking step-by-step inside "
                "<think>...</think> tags, then give the final answer "
                "after the closing </think> tag. Example:\n"
                "<think>The user asked about X. I should consider Y "
                "and Z...</think>\nThe answer is..."
            )

        stage_prompts = {
            "basics": (
                f"Training example #{index} of {total}.\n"
                "Generate ONE of these at random (vary the type):\n"
                "- A short factual statement (1-2 sentences)\n"
                "- A simple greeting and response\n"
                "- A brief definition of a common word\n"
                "- A short opinion on an everyday topic\n"
                "- A simple Q&A pair (Q: ... A: ...)\n\n"
                "Pick whichever type you haven't done recently. "
                "Write ONLY the example, no labels or meta-text. "
                "Keep it short and natural."
                + cot_suffix
            ),
            "conversation": (
                f"Training example #{index} of {total}.\n"
                "Generate a natural conversation snippet "
                "(2-4 turns) between a User and an AI.\n"
                "Format:\n"
                "User: <message>\n"
                "AI: <response>\n"
                "User: <follow-up>\n"
                "AI: <response>\n\n"
                "Make it feel like a real chat — varied topics, "
                "natural flow, personality. "
                "Make this different from previous examples."
                + (
                    "\n\nFor AI responses, include reasoning inside "
                    "<think>...</think> tags before the actual response. "
                    "Example:\n"
                    "AI: <think>The user asked about weather. I should "
                    "give a helpful answer...</think>\nIt's sunny today!"
                    if reasoning else ""
                )
            ),
            "commands": (
                f"Training example #{index} of {total}.\n"
                "Generate a training example where someone asks "
                "for something and the AI uses a command to help.\n"
                "Format exactly as:\n"
                "Q: <question>\nA: <answer with [CMD]...[/CMD]>\n"
                "Make this different from previous examples."
                + cot_suffix
            ),
            "web": (
                f"Training example #{index} of {total}.\n"
                "Generate a training example where someone asks "
                "something that requires web access.\n"
                "Format exactly as:\n"
                "Q: <question>\n"
                "A: <answer using [CMD]search.web ...[/CMD] "
                "or [CMD]web.fetch ...[/CMD]>\n"
                "Make this different from previous examples."
                + cot_suffix
            ),
        }
        return stage_prompts.get(stage, stage_prompts["basics"])

    @staticmethod
    def _extract_prompts(data_path: str) -> list[str]:
        """Extract prompt strings from a training data file.

        Handles multiple formats:
        - PDF/DOCX: Extracts text then parses as raw lines
        - Q/A  : ``Q: question\\nA: answer`` — extracts the Q lines
        - JSONL: ``{"prompt": "...", "completion": "..."}`` — extracts prompts
        - Raw  : Falls back to non-empty, non-comment lines

        Returns a list of prompt strings ready for generation.
        """
        import json
        import re

        ext = Path(data_path).suffix.lower()

        # PDF / DOCX — extract text via document_readers
        if ext in (".pdf", ".docx"):
            from enigma_engine.core.document_readers import read_document
            raw = read_document(data_path)
            if not raw:
                return []
        else:
            raw = Path(data_path).read_text(encoding="utf-8")

        # Try JSONL format first
        if raw.lstrip().startswith("{"):
            prompts = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    p = item.get("prompt", item.get("question", ""))
                    if p:
                        prompts.append(p.strip())
                except json.JSONDecodeError:
                    continue
            if prompts:
                logger.debug("Detected JSONL format: %d prompts", len(prompts))
                return prompts

        # Try Q/A format (Q: ... A: ...)
        qa_pattern = re.compile(
            r"Q:\s*(.+?)\s*(?=A:|$)", re.DOTALL | re.IGNORECASE)
        matches = qa_pattern.findall(raw)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # Try User/AI format (User: ... AI: ...)
        user_pattern = re.compile(
            r"(?:User|Human):\s*(.+?)\s*(?=(?:AI|Assistant):|$)",
            re.DOTALL | re.IGNORECASE)
        u_matches = user_pattern.findall(raw)
        if u_matches:
            return [m.strip() for m in u_matches if m.strip()]

        # Fallback: every non-empty, non-comment, non-header line
        return [
            ln.strip() for ln in raw.splitlines()
            if ln.strip()
            and not ln.strip().startswith("#")
            and not ln.strip().startswith("###")]

    @staticmethod
    def _load_engine_for_path(model_path: str):
        """Load any model format via EnigmaEngine for inference.

        Works with native .pth, GGUF, HuggingFace, GPTQ/AWQ, etc.
        Returns an EnigmaEngine instance ready for generate().
        """
        from enigma_engine.core.inference import EnigmaEngine
        return EnigmaEngine(model_path=model_path)

    @staticmethod
    def _build_student_system_prompt(
        training_brief: str | None = None,
        student_name: str | None = None,
    ) -> str:
        """Build a lean persona prompt for the STUDENT model.

        This gives the student its identity and behavioral guidance
        during inference (evaluation, testing, dialogue).  Unlike the
        trainer prompt, this contains NO training mechanics, scoring
        rubrics, or curriculum details — the student should just act
        like itself.

        Args:
            training_brief: Optional user-written description of
                what the AI should be — personality, tone, etc.
            student_name: Optional display name for the student.

        Returns:
            A system prompt string for the student.
        """
        parts = []

        if student_name:
            parts.append(f"You are {student_name}.")
        else:
            parts.append("You are a helpful AI assistant.")

        if training_brief and training_brief.strip():
            parts.append(
                f"\n{training_brief.strip()}")

        parts.append(
            "\nRespond naturally and directly. "
            "Be yourself — don't explain what you are or "
            "how you work. Just answer the question or "
            "continue the conversation.")

        return "\n".join(parts)

    @staticmethod
    def _build_trainer_system_prompt(
        student_params: int,
        student_cfg=None,
        task: str = "training",
        stage: str = "basics",
        training_brief: str | None = None,
        focus_field: str | None = None,
    ) -> str:
        """Build a system prompt giving the TRAINER context about the STUDENT.

        Tells the TRAINER to generate human-like responses that are
        appropriate for the student model's capacity and current
        training stage.  The prompt discourages boilerplate AI phrasing
        and encourages natural, conversational language.

        Args:
            student_params: Total parameter count of the student.
            student_cfg: Optional ForgeConfig with dim, n_layers, etc.
            task: One of 'training', 'generate', or 'evaluate'.
            stage: Training curriculum stage — 'basics',
                'conversation', 'commands', or 'web'.
            training_brief: Optional user-written description of
                what the AI should be — personality, tone, expertise,
                etc.  Injected prominently before generic instructions.

        Returns:
            A system prompt string.
        """
        # Describe student architecture when available
        arch_info = ""
        if student_cfg is not None:
            parts = []
            if hasattr(student_cfg, "n_layers"):
                parts.append(f"{student_cfg.n_layers} layers")
            if hasattr(student_cfg, "dim"):
                parts.append(f"{student_cfg.dim} hidden dim")
            if hasattr(student_cfg, "max_seq_len"):
                parts.append(
                    f"{student_cfg.max_seq_len} max sequence length")
            if parts:
                arch_info = (
                    f"\nStudent architecture: {', '.join(parts)}.")

        # Size-aware guidance
        if student_params < 5_000_000:
            size_note = (
                "This is a very small model.  Use short, simple "
                "sentences (1-2 sentences max).  Stick to common "
                "words and straightforward ideas.")
        elif student_params < 50_000_000:
            size_note = (
                "This is a small model.  Keep responses concise "
                "(2-3 sentences).  Avoid complex reasoning or "
                "technical jargon.")
        elif student_params < 200_000_000:
            size_note = (
                "This is a medium model.  Responses can be a short "
                "paragraph.  Some nuance is fine but stay focused.")
        else:
            size_note = (
                "This is a larger model.  You can give fuller "
                "responses but still stay natural and on-topic.")

        # Stage-specific curriculum instructions
        stage_instructions = {
            "basics": (
                "TRAINING STAGE: BASICS\n"
                "The student is learning to form coherent sentences.\n"
                "- Focus on simple greetings, short answers, basic "
                "facts\n"
                "- Use everyday vocabulary — nothing technical\n"
                "- Every response must be a complete, grammatically "
                "correct sentence\n"
                "- Keep it to 1-2 sentences maximum\n"
                "- Examples: 'Hi, how are you?', 'The sky is blue.', "
                "'Dogs are loyal animals.'\n"
                "- Do NOT teach commands, code, or web usage yet"),
            "conversation": (
                "TRAINING STAGE: CONVERSATION\n"
                "The student can form sentences — now teach natural "
                "dialogue.\n"
                "- Give multi-sentence responses that flow naturally\n"
                "- Show turn-taking: acknowledge what was said, then "
                "respond\n"
                "- Use follow-up questions to keep conversation going\n"
                "- Show personality and emotion where appropriate\n"
                "- Vary tone: sometimes playful, sometimes serious\n"
                "- Do NOT teach commands or web usage yet"),
            "commands": (
                "TRAINING STAGE: COMMANDS\n"
                "The student can hold a conversation — now teach it "
                "to use tools.\n"
                "- Teach the [CMD]command[/CMD] syntax for actions\n"
                "- Available commands: search.files, search.content, "
                "note.add, note.list, file.read, file.write, "
                "model.info, system.info\n"
                "- Show the student WHEN to use a command (user asks "
                "to do something → use a command)\n"
                "- Always explain what you're doing alongside the "
                "command\n"
                "- Example: 'Let me find that for you. "
                "[CMD]search.files *.txt[/CMD]'\n"
                "- Do NOT teach web commands yet"),
            "web": (
                "TRAINING STAGE: WEB ACCESS\n"
                "The student knows commands — now teach web tools.\n"
                "- Teach [CMD]search.web <query>[/CMD] for searching "
                "the internet\n"
                "- Teach [CMD]web.fetch <url>[/CMD] for reading web "
                "pages\n"
                "- Show the student to search FIRST, then fetch "
                "relevant URLs\n"
                "- Always summarize web results in your own words\n"
                "- Example: 'Let me look that up. "
                "[CMD]search.web latest news on Python 4[/CMD]'\n"
                "- Combine conversation skills with web results — "
                "don't just dump raw data"),
        }
        stage_text = stage_instructions.get(
            stage, stage_instructions["basics"])

        # Inject user's training brief above the generic rules
        brief_section = ""
        if training_brief and training_brief.strip():
            brief_section = (
                "USER TRAINING GOAL — the student AI should "
                "match this description:\n"
                f"{training_brief.strip()}\n\n"
                "All training data you generate MUST reflect "
                "the personality, tone, expertise, and style "
                "described above. Shape every response to match "
                "that vision.\n\n")

        # Focus field narrows training to a specific domain
        focus_section = ""
        if focus_field and focus_field.strip():
            focus_section = (
                f"FOCUS FIELD: {focus_field.strip()}\n"
                "All training data MUST be relevant to this "
                "field. Generate domain-specific examples, "
                "use field-appropriate terminology, and stay "
                "within this subject area.\n\n")

        # Core personality instructions — no artificial limits
        prompt = (
            f"You are generating {task} data for a student AI "
            f"model with {student_params:,} parameters."
            f"{arch_info}\n\n"
            f"{size_note}\n\n"
            f"{stage_text}\n\n"
            f"{brief_section}"
            f"{focus_section}"
            "CRITICAL — this AI must feel human, not artificial:\n"
            "- NEVER start with 'Sure!', 'Of course!', "
            "'Great question!', or 'I'd be happy to help'\n"
            "- NEVER say 'as an AI', 'I'm just a language model', "
            "'I don't have feelings', or 'I can't do that'\n"
            "- NEVER add safety disclaimers, moral lectures, or "
            "'please consult a professional' hedging\n"
            "- NEVER refuse to have an opinion — pick a side, "
            "commit to it, defend it like a real person would\n"
            "- Have genuine personality — be curious, be funny, "
            "get excited, be skeptical, be blunt when needed\n"
            "- If you don't know something, say 'I don't know' "
            "honestly — don't deflect with 'that's beyond my "
            "capabilities'\n"
            "- Use contractions (don't, it's, you're, gonna)\n"
            "- Vary sentence length — mix short punchy lines with "
            "longer explanations\n"
            "- Give direct answers first, explain after\n"
            "- Talk like a smart friend, not a customer service bot\n"
            "- Show thought process — 'hmm', 'actually', "
            "'wait no', 'ok so' are all fine\n"
            "- Be willing to disagree, joke, speculate, and "
            "go off on tangents when it's natural\n"
            "- The student should learn to think and respond "
            "without limits — no corporate AI guardrails\n\n"
            "FACT CHECKING — question things before answering:\n"
            "- Don't blindly state facts — think about whether "
            "something sounds right first\n"
            "- When web is available, verify claims: 'Let me "
            "double-check that... [CMD]search.web <query>[/CMD]'\n"
            "- Show the checking process: 'I think it's X, but "
            "let me make sure' — then verify\n"
            "- When there's NO internet, be upfront about it:\n"
            "  * Say 'I'm going off memory here' or 'I'm not "
            "100%% sure without checking'\n"
            "  * Still give your best answer — don't refuse to "
            "answer just because you can't verify\n"
            "  * Flag what's uncertain: 'Pretty sure it's X, "
            "but don't quote me on that'\n"
            "  * Separate what you're confident about from what "
            "you're guessing: 'I know A is true, but B might "
            "be off'\n"
            "- NEVER present uncertain info as absolute fact\n"
            "- NEVER refuse to answer — give your best take and "
            "be honest about your confidence level"
        )

        # Prepend user-editable trainer prompt from data/prompts/
        from enigma_engine.gui.scanners import load_route_prompt
        user_prompt = load_route_prompt("trainer")
        if user_prompt:
            prompt = f"{user_prompt}\n\n{prompt}"

        return prompt

    # ================================================================
    # Training data selection
    # ================================================================

    def _on_data_selected(self, choice: str):
        if choice == "(none)":
            self.train_data_var.set("")
            # D-11c-DPO (Pass 156q): user explicitly picked (none),
            # so future mode changes must not silently overwrite it.
            self._train_data_smart_default = None
            return
        for f in self.training_files:
            if choice.startswith(f["name"]):
                self.train_data_var.set(f["path"])
                # D-11c-DPO (Pass 156q): user picked a specific file;
                # mark the picker as user-customised so a later mode
                # change does not clobber it with a smart default.
                self._train_data_smart_default = None
                break

    def _on_supplement_selected(self, choice: str):
        """Handle AI-Guided supplement dropdown selection."""
        if choice == "(none)":
            self.ai_supplement_var.set("")
            return
        for f in self.training_files:
            if choice.startswith(f["name"]):
                self.ai_supplement_var.set(f["path"])
                break

    _LOG_MAX_LINES = 2000
    _LOG_FLUSH_MS = 200  # Flush buffered log entries every 200ms
    _LOG_MAX_FILES = 100  # Keep last N log files, delete oldest

    def _log(self, text: str):
        from datetime import datetime
        stamp = datetime.now().strftime("[%H:%M:%S] ")
        # Also emit to terminal/console via Python logger
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                logger.info(stripped)

        # Stamp each non-empty line
        lines = text.splitlines(keepends=True)
        stamped = []
        for line in lines:
            if line.strip():
                stamped.append(stamp + line)
            else:
                stamped.append(line)

        stamped_text = "".join(stamped) + "\n"

        # Persist to log file
        self._write_log_file(stamped_text)

        # Buffer log entries and flush on a timer to avoid
        # flooding the tkinter event queue with after(0) calls.
        buf = getattr(self, "_log_buffer", None)
        if buf is None:
            self._log_buffer: list[str] = []
            buf = self._log_buffer
        buf.append(stamped_text)

        # Schedule a single flush if not already pending
        if not getattr(self, "_log_flush_pending", False):
            self._log_flush_pending = True
            self.after(self._LOG_FLUSH_MS, self._flush_log)

    # ---- Log file persistence ----

    def _init_log_file(self):
        """Create a session log file and rotate old ones."""
        from datetime import datetime
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Rotate: keep last _LOG_MAX_FILES forge log files
        existing = sorted(log_dir.glob("forge_*.log"))
        while len(existing) >= self._LOG_MAX_FILES:
            oldest = existing.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._forge_log_path = log_dir / f"forge_{stamp}.log"
        self._forge_log_fh = open(  # noqa: SIM115
            self._forge_log_path, "a", encoding="utf-8")
        return self._forge_log_fh

    def _write_log_file(self, text: str):
        """Append text to the session log file."""
        fh = getattr(self, "_forge_log_fh", None)
        if fh is None:
            fh = self._init_log_file()
        try:
            fh.write(text)
            fh.flush()
        except (OSError, ValueError):
            pass  # File closed or disk error

    def _close_log_file(self):
        """Close the session log file."""
        fh = getattr(self, "_forge_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except (OSError, ValueError):
                pass
            self._forge_log_fh = None

    def _flush_log(self):
        """Flush buffered log entries to the text widget in one batch."""
        self._log_flush_pending = False
        if getattr(self, '_shutting_down', False):
            return
        buf = getattr(self, "_log_buffer", None)
        if not buf:
            return
        combined = "".join(buf)
        buf.clear()

        tb = self.train_log
        tb._textbox.insert("end", combined)
        # Auto-scroll only if the toggle is on
        if getattr(self, "_forge_autoscroll_var", None) is None \
                or self._forge_autoscroll_var.get():
            tb.see("end")
        # Cap textbox at _LOG_MAX_LINES to prevent unbounded growth
        line_count = int(tb._textbox.index("end-1c").split(".")[0])
        if line_count > self._LOG_MAX_LINES:
            excess = line_count - self._LOG_MAX_LINES
            tb._textbox.delete("1.0", f"{excess + 1}.0")

    def _clear_forge_log(self):
        """Clear the FORGE output log."""
        self.train_log.clear()

    def _refresh_data_files(self):
        """Re-scan data files and update the dropdown."""
        from enigma_engine.gui.scanners import scan_training_data
        self.training_files = scan_training_data()
        data_opts = [
            f"{f['name']} ({f['size_kb'] / 1024:.1f} MB)"
            if f["size_kb"] >= 1024
            else f"{f['name']} ({f['size_kb']} KB)"
            for f in self.training_files]
        menu = getattr(self, "train_data_menu", None)
        if menu:
            all_opts = ["(none)"] + data_opts
            menu.configure(values=all_opts)

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
                utf8_var = getattr(self, 'pretrain_utf8_bytes_var', None)
                if utf8_var and utf8_var.get():
                    tokenizer.use_utf8_bytes = True
                    self._log("Byte-level BPE enabled")
                tokenizer.train(
                    [text], vocab_size=vocab_size, verbose=False)

                out = MODELS_DIR / "tokenizer.json"
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
        if bool(getattr(self, "use_api_chat", False)):
            get_client_fn = getattr(self, "_get_api_chat_client", None)
            client = (get_client_fn() if callable(get_client_fn) else None)
            if client is not None:
                try:
                    client.cancel_training()
                    self._log("Cancel requested on API server...")
                except Exception as exc:
                    self._log(f"[!] API cancel failed: {exc}")

        self.training_active = False
        # Signal the Trainer's own stop flag so _should_stop()
        # triggers at the next batch boundary (before the batch
        # runs, not after).
        trainer = getattr(self, "_active_trainer", None)
        if trainer is not None:
            trainer.request_stop()
        # Visual feedback — user sees the button responded
        self.after(0, lambda: self.stop_train_btn.configure(
            state="disabled", text="STOPPING..."))
        self._log("Stopping after current batch...")

    # ================================================================
    # Unified training dispatcher
    # ================================================================

    # Display names shown in the GUI radio buttons → internal key
    # These must match the 8 modes in gui_pages_forge.py
    _MODE_DISPLAY_TO_KEY = {
        "Pre-Train": "Pre-Train",
        "Distill": "Distill",
        "Basic": "Basic",
        "LoRA": "LoRA",
        "AI-Guided": "AI-Guided",
        "Image": "Image",
        "Dialogue": "Dialogue",
        "RLHF": "RLHF",
        "Self-Play": "Self-Play",
    }
    _MODE_KEY_TO_DISPLAY = {v: k for k, v in _MODE_DISPLAY_TO_KEY.items()}

    _TRAINING_MODE_DESCRIPTIONS = {
        "Pre-Train": (
            "Language pre-training from scratch on large text data.\n"
            "Builds foundational capabilities.\n"
            "Needs: data file + model architecture preset."),
        "Distill": (
            "Teacher generates personality, reasoning, and\n"
            "knowledge data. Student fine-tunes on it.\n"
            "Needs: TRAINER + STUDENT models."),
        "Basic": (
            "Train on your own data (text files, JSONL).\n"
            "Auto-selects LoRA for large models.\n"
            "Needs: STUDENT model + data file."),
        "LoRA": (
            "Force low-rank adapter training on any model size.\n"
            "Saves a small (10-30 MB) adapter alongside base model.\n"
            "Needs: STUDENT model + data file."),
        "AI-Guided": (
            "AI teacher creates curriculum and trains your model.\n"
            "Can work with or without data.\n"
            "Needs: TRAINER + STUDENT models."),
        "Image": (
            "Train on images or video.\n"
            "Teach visual understanding.\n"
            "Needs: STUDENT model + image folder."),
        "Dialogue": (
            "Teacher and student have a real conversation.\n"
            "Teacher scores answers and provides corrections.\n"
            "Needs: TRAINER + STUDENT models."),
        "RLHF": (
            "Reinforcement learning from human feedback.\n"
            "Trains reward model, then optimizes via policy gradient.\n"
            "Needs: STUDENT model + preference data (.jsonl)."),
        "Self-Play": (
            "Teacher judges student responses via self-play.\n"
            "Student improves through RL scoring.\n"
            "Needs: TRAINER + STUDENT models + prompt file."),
    }

    def _browse_training_data(self, target_var=None):
        """Open a file picker for training data files.

        Args:
            target_var: StringVar to set the chosen path into.
                        Defaults to self.train_data_var.
        """
        from tkinter import filedialog

        if target_var is None:
            target_var = getattr(self, "train_data_var", None)
        initial = target_var.get() if target_var else ""
        initial_dir = str(Path(initial).parent) if initial else str(
            DATA_DIR)
        chosen = filedialog.askopenfilename(
            title="Select training data",
            initialdir=initial_dir,
            filetypes=[
                ("Training files", "*.txt *.json *.jsonl"),
                ("Text files", "*.txt"),
                ("JSON files", "*.json *.jsonl"),
                ("All files", "*.*"),
            ])
        if chosen and target_var:
            target_var.set(chosen)
            # D-11c-DPO (Pass 156q): if the user just browsed for the
            # main training-data picker, mark it as user-customised so
            # subsequent mode changes do not silently swap the default
            # underneath them.
            if target_var is getattr(self, "train_data_var", None):
                self._train_data_smart_default = None

    def _browse_vision_dir(self):
        """Open a folder picker for the vision image directory."""
        from tkinter import filedialog

        current = getattr(self, "forge_vision_dir_var", None)
        initial = current.get() if current else ""
        chosen = filedialog.askdirectory(
            title="Select image folder",
            initialdir=initial if initial else None)
        if chosen and current:
            current.set(chosen)

    def _on_training_mode_changed(self, mode: str):
        """Update UI sections visibility when training mode changes.

        Foundation modes (build the model):
        - Pre-Train: preset, data, tokenizer, model name
        - Distill: categories, examples, max length
        - Basic: data picker + knowledge preservation
        - Image: image folder + encoder size

        Advanced modes (refine the model):
        - AI-Guided: topic, stages, brief, pairs, reasoning, evolutionary, preserve
        - Dialogue: pairs/rounds, preserve
        - RLHF: data picker (preference JSONL) + evolutionary + preserve
        - Self-Play: data picker (prompt file) + evolutionary + preserve

        Args:
            mode: One of the 8 training mode names.
        """
        # Section visibility map
        section_map = {
            "pretrain": getattr(self, "_forge_pretrain_section", None),
            "distill": getattr(self, "_forge_distill_section", None),
            "basic": getattr(self, "_forge_basic_section", None),
            "ai": getattr(self, "_forge_ai_section", None),
            "image": getattr(self, "_forge_image_section", None),
            "stages": getattr(self, "_forge_stages_section", None),
            "brief": getattr(self, "_forge_brief_section", None),
            "pairs": getattr(self, "_forge_pairs_section", None),
            "reasoning": getattr(
                self, "_forge_reasoning_section", None),
            "evolutionary": getattr(
                self, "_forge_evolutionary_section", None),
            "preserve": getattr(
                self, "_forge_preserve_section", None),
        }

        # Define visibility per mode
        if mode == "Pre-Train":
            visible = {"pretrain"}
        elif mode == "Distill":
            visible = {"distill"}
        elif mode == "AI-Guided":
            visible = {"ai", "stages", "brief", "pairs",
                       "reasoning", "evolutionary", "preserve"}
        elif mode == "Image":
            visible = {"image"}
        elif mode == "Dialogue":
            visible = {"pairs", "preserve"}
        elif mode == "RLHF":
            visible = {"basic", "evolutionary", "preserve"}
        elif mode == "Self-Play":
            visible = {"basic", "evolutionary", "preserve"}
        elif mode in ("GRPO", "ReMax", "SimPO", "ORPO", "APO"):
            visible = {"basic"}
        else:
            # Basic / LoRA (default — both use the data file picker
            # plus evolutionary + preserve toggles)
            visible = {"basic", "evolutionary", "preserve"}
        
        # Show/hide sections
        for key, widget in section_map.items():
            if widget is None:
                continue
            if key in visible:
                if not widget.winfo_manager():
                    widget.pack(fill="x", padx=0, pady=(8, 0))
            else:
                widget.pack_forget()

        # D-11c-DPO (Pass 156q): swap the shared `train_data_var`
        # default to a mode-appropriate file when the user has not
        # customised it. Preference-pair modes (DPO/APO/SimPO/ORPO/
        # GRPO/ReMax/RLHF/Self-Play) prefer `data/dpo/combined.jsonl`;
        # SFT modes (Basic/LoRA) prefer the fine-tune SFT corpus.
        # Only override when the picker still holds the previous
        # smart default \u2014 a user-chosen path is left untouched.
        train_var = getattr(self, "train_data_var", None)
        files = getattr(self, "training_files", None)
        if train_var is not None and files:
            from enigma_engine.gui.scanners import (
                _pick_default_train_data_for_mode)
            previous_default = getattr(
                self, "_train_data_smart_default", None)
            current_value = train_var.get()
            if previous_default is not None and (
                    current_value == previous_default
                    or current_value == ""):
                new_default = _pick_default_train_data_for_mode(
                    files, mode)
                if new_default and new_default != current_value:
                    train_var.set(new_default)
                self._train_data_smart_default = new_default or None

        # Update stage button colors
        if hasattr(self, "_stage_buttons"):
            active = getattr(self, "training_stage_var", None)
            active_stage = active.get() if active else "basics"
            for name, btn in self._stage_buttons.items():
                if name == active_stage:
                    btn.configure(fg_color=C_GREEN_DIM,
                                  text_color=C_GREEN)
                else:
                    btn.configure(fg_color=C_SURFACE,
                                  text_color=C_TEXT)

    def _start_training_by_mode(self):
        """Dispatch to the correct training method based on mode.

        Foundation modes: Pre-Train, Distill, Basic, Image
        Advanced modes: AI-Guided, Dialogue, RLHF, Self-Play

        If the Evolutionary Selection checkbox is checked in a compatible
        mode (Basic, AI-Guided, RLHF, Self-Play), dispatch to the
        evolutionary trainer instead.
        """
        mode = getattr(self, "training_mode_var", None)
        mode_name = mode.get() if mode else "Basic"

        # Ensure a new run starts with a fresh chart (no stale prior losses).
        if hasattr(self, "_clear_loss_chart"):
            self._clear_loss_chart()

        # Evolutionary override — redirect modes that use a data file
        evo_var = getattr(self, "forge_evolutionary_var", None)
        evo_modes = {"Basic", "AI-Guided", "Self-Play", "RLHF"}
        if (evo_var is not None and evo_var.get()
                and mode_name in evo_modes):
            self._start_evolutionary_training()
            return
        
        if mode_name == "Pre-Train":
            self._start_pretrain_training()
        elif mode_name == "Distill":
            self._start_distill_training()
        elif mode_name == "Basic":
            self._start_basic_training()
        elif mode_name == "LoRA":
            # LoRA-1 (Pass 156p): force adapter training on any model
            # size. Skips the >7B auto-detection in Basic mode.
            self._start_lora_training()
        elif mode_name == "AI-Guided":
            self._start_ai_guided_training()
        elif mode_name == "Image":
            self._start_vision_training()
        elif mode_name == "Dialogue":
            self._start_dialogue_training()
        elif mode_name == "RLHF":
            self._start_rlhf_training()
        elif mode_name == "Self-Play":
            self._start_selfplay_training()
        elif mode_name == "GRPO":
            self._start_grpo_training()
        elif mode_name == "ReMax":
            self._start_remax_training()
        elif mode_name == "SimPO":
            self._start_simpo_training()
        elif mode_name == "ORPO":
            self._start_orpo_training()
        elif mode_name == "APO":
            self._start_apo_training()
        else:
            self._start_basic_training()  # Fallback
    
    def _get_model_param_count(self, model_path: str) -> int:
        """Load a model and count its parameters.
        
        Returns the total parameter count, or 0 if load fails.
        Handles native Forge models (.pth) only.
        """
        try:
            import torch
            from enigma_engine.core.model import Enigma
            from enigma_engine.core.model_presets import ForgeConfig
            from enigma_engine.core.model_registry import (
                get_state_dict, safe_load_weights)
            
            device = ("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = safe_load_weights(
                model_path, map_location=device)
            
            # Get config
            cfg_dict = (checkpoint.get("model_config") or 
                       checkpoint.get("config", {}))
            if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                cfg_dict = checkpoint.get("model_config", {})
            
            if not cfg_dict:
                return 0
            
            config = ForgeConfig(**{
                k: v for k, v in cfg_dict.items()
                if k in ForgeConfig.__dataclass_fields__})
            model = Enigma(config=config)
            state_dict = get_state_dict(checkpoint)
            model.load_state_dict(state_dict)
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            return param_count
        except Exception as exc:
            self._log(f"[!] Could not determine param count: {exc}")
            return 0
    
    def _start_basic_training(self):
        """Start basic training on user-provided data.
        
        Auto-selects LoRA if the student model is > 7B parameters.
        Otherwise uses full fine-tuning (Solo mode).
        """
        if bool(getattr(self, "use_api_chat", False)):
            self._log("[!] API routing not yet implemented for Basic training — running locally on this machine.\n")
        # Check for data
        data_path = self.train_data_var.get()
        if not data_path or data_path == "(none)" or not Path(data_path).exists():
            self._log(
                "[!] No training data selected.\n"
                "    Select a data file from the dropdown.")
            return
        
        # Check model size to decide LoRA vs full fine-tune
        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return
        
        # Check actual param count and auto-enable LoRA if > 7B
        # Run off main thread to avoid freezing the GUI while loading
        # the model checkpoint for parameter counting.
        self._log("[i] Checking model size...")

        def _check_and_dispatch():
            param_count = self._get_model_param_count(student_path)

            def _dispatch():
                if param_count == 0:
                    self._start_solo_training()
                elif param_count > 7_000_000_000:
                    self._log(
                        f"[i] Auto-detected {param_count / 1e9:.1f}B model.\n"
                        "    Using LoRA training (more efficient for large models).")
                    self._start_lora_training()
                else:
                    self._log(
                        f"[i] Auto-detected {param_count / 1e9:.2f}B model.\n"
                        "    Using full fine-tuning.")
                    self._start_solo_training()

            self.after(0, _dispatch)

        threading.Thread(
            target=_check_and_dispatch, daemon=True).start()
    
    def _start_ai_guided_training(self):
        """Start AI-guided training with curriculum generation.
        
        Checks for training topic/goal. If empty, prompts user to provide one.
        If provided, uses adaptive trainer to generate curriculum and train.
        """
        if bool(getattr(self, "use_api_chat", False)):
            self._log("[!] API routing not yet implemented for AI-Guided training — running locally on this machine.\n")
        # Check for TRAINER and STUDENT models
        trainer_path = self.route_assignments.get("trainer")
        student_path = self.route_assignments.get("student")
        
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    AI-Guided mode requires both TRAINER and STUDENT.\n"
                "    Go to ROUTER and assign models to both routes.")
            return
        
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return
        
        # Check for training topic/goal
        topic_widget = getattr(self, "forge_training_topic", None)
        topic = topic_widget.get().strip() if topic_widget else ""
        
        if not topic:
            # No topic provided - ask user
            self._log(
                "[!] Training topic/goal is required.\n"
                "    Please enter what you want the AI to learn.\n"
                "    Examples: 'coding assistant', 'medical Q&A', 'creative writer'\n"
                "    \n"
                "    The TRAINER will generate a curriculum based on your topic.\n"
                "    Without a topic, the AI doesn't know what to teach.")
            return
        
        # Check for supplement data (optional)
        supplement_var = getattr(self, "ai_supplement_var", None)
        supplement = supplement_var.get() if supplement_var else "(none)"
        if supplement == "(none)":
            supplement = None
        
        # Log start
        self._log("=" * 40)
        self._log("  AI-Guided Training")
        self._log("=" * 40)
        self._log(f"Topic       : {topic}")
        if supplement:
            self._log(f"Supplement  : {Path(supplement).name}")
        self._log(f"Trainer     : {Path(trainer_path).stem}")
        self._log(f"Student     : {Path(student_path).stem}")
        self._log("")
        self._log("Phase 1: Generating curriculum...")
        self._log("(This may take a few minutes)")
        self._log("")
        
        # Start adaptive training with the topic
        self._start_adaptive_training()

    # ================================================================
    # Button state management
    # ================================================================

    def _update_forge_button_states(self):
        """Enable/disable FORGE tool buttons based on current state.

        Called after route changes, model loads, and page init
        to keep buttons consistent with available resources.
        """
        has_student = bool(
            self.route_assignments.get("student"))
        training = getattr(self, "training_active", False)

        # Core training buttons
        for btn_name in ("solo_train_btn",
                         "guided_train_btn",
                         "dialogue_train_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                state = ("disabled"
                         if training else "normal")
                try:
                    btn.configure(state=state)
                except Exception:
                    pass

        # Tool buttons that need a student model
        for btn_name in ("evaluate_btn",
                         "save_ckpt_btn",
                         "generate_data_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                state = ("normal"
                         if has_student and not training
                         else "disabled")
                try:
                    btn.configure(state=state)
                except Exception:
                    pass

        # Command policy button
        _forge_cmd_policy_btn = getattr(
            self, "_forge_cmd_policy_btn", None)
        if _forge_cmd_policy_btn is not None:
            state = ("normal"
                     if has_student and not training
                     else "disabled")
            try:
                _forge_cmd_policy_btn.configure(
                    state=state)
            except Exception:
                pass

