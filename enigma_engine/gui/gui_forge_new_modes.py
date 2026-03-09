"""
Enigma Engine - Forge New Training Modes
==========================================

Training mode implementations: RLHF, Self-Play.
Split into its own mixin to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path



logger = logging.getLogger(__name__)


class ForgeNewModesMixin:
    """Training mode implementations: RLHF, Self-Play.

    Expects the host class to have ForgeMixin setup attributes.
    """

    # ================================================================
    # RLHF TRAINING (RL-B)
    # ================================================================

    def _start_rlhf_training(self):
        """Train STUDENT with RLHF: reward model + policy gradient.

        Workflow:
        1. Load preference data (.jsonl with prompt/chosen/rejected)
        2. Train a small reward model on that data
        3. Use the reward model to score STUDENT responses
        4. Policy gradient to improve STUDENT
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
                "[!] No data file selected.\n"
                "    RLHF needs a .jsonl preference file\n"
                "    (prompt/chosen/rejected per line).")
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

        model_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 RLHF TRAINING...")

        self._log("=" * 40)
        self._log("  RLHF Training (Reward + Policy)")
        self._log("=" * 40)
        self._log(f"Student : {model_name}")
        self._log(f"Data    : {Path(data_path).name}")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _rlhf_train():
            try:
                import json
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import get_state_dict
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.rl_training import (
                    RewardModel, RewardTrainer, RewardTrainerConfig,
                    RLHFTrainer, RLHFConfig,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._log(f"Device  : {device.upper()}")

                # Load student model
                self._log(f"Loading {model_name}...")
                from enigma_engine.core.model_registry import (
                    safe_load_weights)
                checkpoint = safe_load_weights(
                    student_path, map_location=device)

                cfg_dict = checkpoint.get(
                    "model_config",
                    checkpoint.get("config", {}))
                if "epochs" in cfg_dict:
                    cfg_dict = {}
                config = ForgeConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in ForgeConfig.__dataclass_fields__
                })

                model = Enigma(config=config).to(device)
                state = get_state_dict(checkpoint)
                model.load_state_dict(state, strict=False)

                tokenizer = get_tokenizer()
                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                # Load preference data
                self._log("Loading preference data...")
                pref_data = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if all(k in item for k in ("prompt", "chosen", "rejected")):
                                pref_data.append(item)
                        except json.JSONDecodeError:
                            continue

                if not pref_data:
                    self._log("[!] No valid preference pairs found in file.")
                    return

                self._log(f"Loaded {len(pref_data)} preference pairs")

                # Phase 1: Train reward model
                self._log("\n--- Phase 1: Training Reward Model ---")
                reward_model = RewardModel(model, freeze_base=True).to(device)
                reward_cfg = RewardTrainerConfig(
                    epochs=min(epochs, 5),
                    learning_rate=lr * 10,
                )
                reward_trainer = RewardTrainer(
                    reward_model, tokenizer, reward_cfg)
                reward_trainer.on_progress = lambda p, m: self.after(
                    0, lambda _m=m: self._log(f"  {_m}"))

                result = reward_trainer.train(pref_data)
                self._log(f"Reward model trained: loss={result['final_loss']:.4f}")

                if self._forge_stop_requested():
                    return

                # Phase 2: RLHF policy training
                self._log("\n--- Phase 2: RLHF Policy Training ---")
                prompts = [item["prompt"] for item in pref_data]

                rlhf_cfg = RLHFConfig(
                    epochs=epochs,
                    learning_rate=lr,
                )
                rlhf_trainer = RLHFTrainer(
                    model, tokenizer, reward_model, rlhf_cfg)

                def _rlhf_progress(p, m):
                    self.after(
                        0, lambda _m=m: self._log(f"  {_m}"))
                    self.after(
                        0, lambda _p=p: self._update_forge_progress(
                            _p, f"RLHF {_p}%"))

                rlhf_trainer.on_progress = _rlhf_progress
                rl_result = rlhf_trainer.train(prompts)

                self._log(f"\nFinal reward: {rl_result.get('final_reward', 0):.4f}")

                # Save model
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "model_config": config.__dict__,
                    "config": config.__dict__,
                }, student_path)

                self._log(f"Model saved to {Path(student_path).name}")
                self._log("--- RLHF TRAINING COMPLETE ---")
                self._update_forge_progress(100, "Complete")
                self.after(0, lambda _pc=pc: self._update_forge_param_count(_pc))
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- RLHF TRAINING STOPPED ---")
            except Exception as exc:
                self._log(f"\n[!] RLHF training failed: {exc}")
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

        threading.Thread(target=_rlhf_train, daemon=True).start()

    # ================================================================
    # SELF-PLAY TRAINING (RL-C)
    # ================================================================

    def _start_selfplay_training(self):
        """Train STUDENT via self-play: TRAINER scores responses."""
        if self.training_active:
            return

        student_path = self.route_assignments.get("student")
        trainer_path = self.route_assignments.get("trainer")

        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Self-play needs a TRAINER to score responses.")
            return

        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log(
                "[!] No data file selected.\n"
                "    Provide a text file with prompts "
                "(one per line).")
            return

        try:
            epochs = int(self.ft_epochs_entry.get())
        except ValueError:
            self._log("[!] Epochs must be a number.")
            return

        try:
            lr = float(self.ft_lr_entry.get())
        except ValueError:
            self._log("[!] LR must be a number.")
            return

        model_name = Path(student_path).stem
        trainer_name = Path(trainer_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 SELF-PLAY TRAINING...")

        self._log("=" * 40)
        self._log("  Self-Play Training")
        self._log("=" * 40)
        self._log(f"Student : {model_name}")
        self._log(f"Trainer : {trainer_name}")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _selfplay_train():
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import get_state_dict
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.inference import EnigmaEngine
                from enigma_engine.core.rl_training import (
                    SelfPlayTrainer, SelfPlayConfig,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load student
                self._log(f"Loading student: {model_name}...")
                from enigma_engine.core.model_registry import (
                    safe_load_weights)
                ckpt = safe_load_weights(
                    student_path, map_location=device)
                cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
                if "epochs" in cfg_dict:
                    cfg_dict = {}
                config = ForgeConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in ForgeConfig.__dataclass_fields__
                })
                student = Enigma(config=config).to(device)
                student.load_state_dict(
                    get_state_dict(ckpt), strict=False)

                tokenizer = get_tokenizer()

                # Load trainer as engine
                self._log(f"Loading trainer: {trainer_name}...")
                trainer_engine = EnigmaEngine(
                    model_path=trainer_path,
                    device=device)

                # Read prompts from data file
                prompts = Path(data_path).read_text(
                    encoding="utf-8").strip().splitlines()
                prompts = [p.strip() for p in prompts if p.strip()]
                self._log(f"Prompts : {len(prompts)}")

                sp_cfg = SelfPlayConfig(
                    epochs=epochs,
                    learning_rate=lr,
                )
                sp_trainer = SelfPlayTrainer(
                    student, tokenizer, trainer_engine, sp_cfg)

                def _sp_progress(p, m):
                    self.after(0, lambda _m=m: self._log(f"  {_m}"))
                    self.after(0, lambda _p=p: self._update_forge_progress(
                        _p, f"Self-play {_p}%"))

                sp_trainer.on_progress = _sp_progress
                result = sp_trainer.train(prompts)

                self._log(
                    f"\nFinal score: "
                    f"{result.get('final_score', 0):.2f}/10")

                # Save
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": student.state_dict(),
                    "model_config": config.__dict__,
                    "config": config.__dict__,
                }, student_path)

                self._log(f"Model saved to {Path(student_path).name}")
                self._log("--- SELF-PLAY COMPLETE ---")
                self._update_forge_progress(100, "Complete")
                self.after(0, self._refresh_models)

            except Exception as exc:
                self._log(f"\n[!] Self-play failed: {exc}")
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

        threading.Thread(target=_selfplay_train, daemon=True).start()

