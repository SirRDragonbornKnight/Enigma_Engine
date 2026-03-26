"""
Enigma Engine - Forge Training Modes (Advanced)
===================================================

Training mode implementations: Guided, Dialogue.
Split from gui_forge.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ForgeAdvancedMixin:
    """Training mode implementations: Guided, Dialogue.

    Expects the host class to have ForgeMixin setup attributes.
    """

    # ================================================================
    # AI-assisted training (TRAINER teaches STUDENT)
    # ================================================================

    def _guided_validate_inputs(self) -> dict | None:
        """Validate inputs for AI-assisted training.

        Returns a dict of validated params, or None on failure
        (with error logged to the FORGE panel).
        """
        if self.training_active:
            return None

        trainer_path = self.route_assignments.get("trainer")
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Go to ROUTER and assign the teacher model.")
            return None

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return None

        data_path = self.train_data_var.get()
        has_data = data_path and Path(data_path).exists()

        try:
            epochs = int(self.guided_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Epochs must be 1-1000.")
            return None

        try:
            lr = float(self.guided_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return None

        try:
            num_pairs = int(self.guided_pairs_entry.get())
            if num_pairs < 1 or num_pairs > 500:
                raise ValueError
        except (ValueError, AttributeError):
            num_pairs = 20

        return {
            "trainer_path": trainer_path,
            "student_path": student_path,
            "data_path": data_path,
            "has_data": has_data,
            "epochs": epochs,
            "lr": lr,
            "num_pairs": num_pairs,
        }

    def _guided_load_models(self, trainer_path: str,
                            student_path: str) -> tuple:
        """Load trainer engine + student model for guided training.

        Returns (teacher_engine, student, s_cfg, tokenizer,
                 s_params, t_params).
        """
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.model_presets import ForgeConfig

        trainer_name = Path(trainer_path).stem
        student_name = Path(student_path).stem

        # Load TRAINER via EnigmaEngine (handles any format)
        self._log(f"Loading trainer: {trainer_name}...")
        teacher_engine = self._load_engine_for_path(trainer_path)

        # Count trainer params
        t_params = 0
        if (hasattr(teacher_engine, "model")
                and teacher_engine.model is not None):
            if hasattr(teacher_engine.model, "parameters"):
                try:
                    t_params = sum(
                        p.numel()
                        for p in
                        teacher_engine.model.parameters())
                except Exception:
                    pass
            if t_params == 0 and hasattr(
                    teacher_engine.model, "config"):
                cfg = teacher_engine.model.config
                dim = getattr(cfg, "dim", 0)
                layers = getattr(cfg, "n_layers", 0)
                vocab = getattr(cfg, "vocab_size", 0)
                if dim and layers:
                    t_params = (
                        vocab * dim
                        + layers * (
                            4 * dim * dim + 3 * dim * dim)
                        + vocab * dim)
        if t_params == 0:
            tp = Path(trainer_path)
            if tp.is_dir():
                files = list(tp.glob("*.gguf"))
                if files:
                    tp = files[0]
            if tp.is_file():
                size_gb = tp.stat().st_size / (1024**3)
                t_params = int(size_gb * 2e9)
        t_label = (f"{t_params / 1e9:.1f}B"
                   if t_params >= 1e9
                   else f"{t_params:,}") if t_params else "N/A (external)"
        self._log(f"Trainer : ~{t_label} params")

        # Load STUDENT model on CPU
        self._log(f"Loading student: {student_name}...")
        from enigma_engine.core.model_registry import (
            get_state_dict, safe_load_weights)
        s_ckpt = safe_load_weights(
            student_path, map_location="cpu")
        s_cfg_dict = s_ckpt.get("model_config") or s_ckpt.get("config", {})
        if isinstance(s_cfg_dict, dict) and "epochs" in s_cfg_dict:
            s_cfg_dict = s_ckpt.get("model_config", {})
        s_cfg = ForgeConfig(**s_cfg_dict)
        student = Enigma(config=s_cfg)
        student.load_state_dict(get_state_dict(s_ckpt))
        s_params = sum(p.numel() for p in student.parameters())
        self._log(f"Student : {s_params:,} params")

        tokenizer = get_tokenizer("auto")

        return (teacher_engine, student, s_cfg, tokenizer,
                s_params, t_params)

    def _start_guided_training(self):
        """AI-assisted training: TRAINER creates curriculum,
        trains STUDENT, then interactively tests readiness.

        The TRAINER autonomously:
        1. Generates training material for the current stage
        2. STUDENT is trained on that material
        3. TRAINER tests STUDENT by asking questions and judging
        4. Reports readiness to advance to the next stage

        Data file is optional — supplements TRAINER curriculum.
        Activated via the 'Train with AI' toggle on any mode.
        """
        params = self._guided_validate_inputs()
        if params is None:
            return

        trainer_path = params["trainer_path"]
        student_path = params["student_path"]
        data_path = params["data_path"]
        has_data = params["has_data"]
        epochs = params["epochs"]
        lr = params["lr"]
        num_pairs = params["num_pairs"]

        trainer_name = Path(trainer_path).stem
        student_name = Path(student_path).stem
        self.training_active = True
        self.guided_train_btn.configure(state="disabled",
                                        text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(
            "\u2692 AI-ASSISTED TRAINING...")

        self._log("--- AI-ASSISTED TRAINING INITIATED ---")
        self._log(f"Trainer : {trainer_name} (teacher)")
        self._log(f"Student : {student_name} (learner)")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._log(f"Pairs   : {num_pairs}")
        if has_data:
            self._log(f"Bonus   : {Path(data_path).name}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _guided():
            losses = []
            try:
                import torch

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                (teacher_engine, student, s_cfg, tokenizer,
                 s_params, t_params) = self._guided_load_models(
                    trainer_path, student_path)

                # Read training stage from UI
                stage = getattr(self, 'training_stage_var', None)
                stage = stage.get() if stage else "basics"

                # Read training brief from UI
                training_brief = self._build_training_brief()
                self._save_training_brief()

                # Read focus field from UI
                focus_field = ""
                ff_widget = getattr(self, "forge_focus_field", None)
                if ff_widget is not None:
                    focus_field = ff_widget.get().strip()

                # Build system prompt so TRAINER knows the STUDENT
                trainer_sys = self._build_trainer_system_prompt(
                    student_params=s_params,
                    student_cfg=s_cfg,
                    task="training",
                    stage=stage,
                    training_brief=training_brief,
                    focus_field=focus_field)
                self._log(
                    f"Trainer context: {s_params:,} param student, "
                    f"{s_cfg.n_layers} layers, {s_cfg.dim} dim")
                self._log(f"Stage   : {stage.upper()}")
                if focus_field:
                    self._log(f"Focus   : {focus_field}")
                if training_brief:
                    self._log(f"Brief   : {len(training_brief)} chars")

                # === Phase 1: TRAINER generates curriculum ===
                pairs, curriculum_path = self._guided_generate_curriculum(
                    teacher_engine=teacher_engine,
                    trainer_sys=trainer_sys,
                    num_pairs=num_pairs,
                    stage=stage,
                    has_data=has_data,
                    data_path=data_path,
                    trainer_name=trainer_name,
                    student_name=student_name,
                )

                if not pairs:
                    self._log(
                        "[!] Trainer produced no training "
                        "material.")
                    return

                combined = "\n\n".join(pairs)

                # Free trainer memory before training student
                del teacher_engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Phase 2: Train student ===
                student, state = self._guided_train_student(
                    student=student,
                    combined=combined,
                    tokenizer=tokenizer,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    student_path=student_path,
                    losses=losses,
                )

                # Free training model memory
                del student
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Phase 3: TRAINER tests student ===
                self._guided_test_student(
                    trainer_path=trainer_path,
                    student_path=student_path,
                    trainer_sys=trainer_sys,
                    training_brief=training_brief,
                    student_name=student_name,
                    stage=stage,
                    curriculum_path=curriculum_path,
                )

                self._log(
                    "\n--- AI-ASSISTED TRAINING COMPLETE ---")
                best_loss = min(losses) if losses else 0.0
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "AI-Assisted", student_name, epochs, best_loss)
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, lambda sp=s_params: self._update_forge_param_count(sp))
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- AI-ASSISTED TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(
                    f"\n[!] AI-assisted training failed: {exc}")
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(
                    0, lambda: self.guided_train_btn.configure(
                        state="normal", text="TRAIN"))
                self.after(
                    0, lambda: self.stop_train_btn.configure(
                        state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_guided, daemon=True).start()

    def _guided_generate_curriculum(
        self, *, teacher_engine, trainer_sys, num_pairs,
        stage, has_data, data_path, trainer_name, student_name,
    ):
        """Phase 1: TRAINER generates training curriculum.

        Returns (pairs, curriculum_path) where pairs is a list of
        training examples and curriculum_path is the saved file.
        """
        self._log(
            "\n--- Phase 1: GENERATING CURRICULUM ---")
        self._log(
            f"TRAINER is creating {num_pairs} training "
            f"examples...")
        pairs = []
        for i in range(num_pairs):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            use_reasoning = getattr(
                self, "forge_reasoning_var", None)
            reasoning_on = (
                use_reasoning.get()
                if use_reasoning is not None else False)
            msg = self._build_generation_prompt(
                i + 1, num_pairs, stage,
                reasoning=reasoning_on)
            result = teacher_engine.chat(
                msg,
                system_prompt=trainer_sys,
                max_gen=256,
                temperature=0.8).strip()
            if result:
                pairs.append(result)
            if (i + 1) % max(1, num_pairs // 5) == 0:
                self._log(
                    f"  Generated {i + 1}/{num_pairs}")

        # Supplement with data file if provided
        if has_data:
            extras = self._extract_prompts(data_path)
            self._log(
                f"Adding {len(extras)} bonus prompts "
                f"from data file...")
            for prompt in extras:
                if not self.training_active:
                    raise KeyboardInterrupt("Stopped")
                response = teacher_engine.chat(
                    prompt,
                    system_prompt=trainer_sys,
                    max_gen=128,
                    temperature=0.7).strip()
                if response:
                    pairs.append(
                        self._format_training_pair(
                            stage, prompt, response))

        if not pairs:
            return pairs, None

        combined = "\n\n".join(pairs)
        self._log(
            f"Curriculum: {len(pairs)} examples "
            f"({len(combined):,} chars)")

        # Save curriculum to data/ for review on DOCS page
        from datetime import datetime
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S")
        curriculum_name = (
            f"guided_{student_name}"
            f"_{stage}_{timestamp}.txt")
        curriculum_path = DATA_DIR / curriculum_name
        curriculum_header = [
            "# AI-Assisted Training Curriculum",
            f"# Trainer: {trainer_name}",
            f"# Student: {student_name}",
            f"# Stage: {stage}",
            f"# Pairs: {len(pairs)}",
            f"# Date: {datetime.now().isoformat()}",
            ""]
        from enigma_engine.core.safe_save import atomic_write_text
        atomic_write_text(
            curriculum_path,
            "\n".join(curriculum_header)
            + combined + "\n")
        self._log(
            f"Saved     : {curriculum_name}")
        self.after(0, self._refresh_data_files)

        # Add to curated dataset for review
        add_fn = getattr(
            self, "_add_to_curated_dataset", None)
        if add_fn is not None:
            for pair in pairs:
                add_fn(pair, source="guided",
                       stage=stage)
            self._log(
                f"  Added {len(pairs)} entries"
                f" to curated dataset")

        return pairs, curriculum_path

    def _guided_train_student(
        self, *, student, combined, tokenizer, device,
        epochs, lr, student_path, losses,
    ):
        """Phase 2: Train student on generated curriculum.

        Returns (student, training_state).
        """
        import torch
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.safe_save import atomic_torch_save

        # Now move student to GPU for training
        student = student.to(device)
        self._log(f"Student moved to {device.upper()}")

        self._log(
            "\n--- Phase 2: TRAINING STUDENT ---\n")
        forge_params = self._read_forge_train_params()
        train_config = TrainingConfig(
            epochs=epochs,
            batch_size=forge_params["batch_size"],
            learning_rate=lr,
            max_grad_accumulation=forge_params["max_grad_accumulation"],
            use_gradient_checkpointing=forge_params["use_gradient_checkpointing"],
            val_split=forge_params["val_split"],
            save_every=max(1, epochs // 5),
            checkpoint_dir=str(MODELS_DIR / "checkpoints"),
            use_amp=torch.cuda.is_available())

        trainer_obj = Trainer(
            student, tokenizer, train_config)

        def on_epoch(epoch, loss):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            losses.append(loss)
            pct = 50 + int(epoch / epochs * 50)
            self._update_forge_progress(
                pct, f"Epoch {epoch}/{epochs}")
            self._log(
                f"  Epoch {epoch:>3d}  |  "
                f"loss {loss:.4f}")
        trainer_obj.on_epoch_complete = on_epoch

        state = trainer_obj.train(combined)

        # Save trained student
        out = Path(student_path)
        atomic_torch_save({
            "model_state_dict": student.state_dict(),
            "config": self._model_config_dict(student),
            "training_state": {
                "epochs": state.epoch,
                "best_loss": state.best_loss,
            },
        }, out)
        self._log(f"\nBest loss : {state.best_loss:.4f}")
        self._log(f"Saved to  : {out}")

        return student, state

    def _guided_test_student(
        self, *, trainer_path, student_path, trainer_sys,
        training_brief, student_name, stage, curriculum_path,
    ):
        """Phase 3: TRAINER tests student and assesses readiness."""
        if not self.training_active:
            raise KeyboardInterrupt("Stopped")
        self._log(
            "\n--- Phase 3: TESTING STUDENT ---")
        self._log(
            "Reloading models for testing...\n")
        teacher_engine = self._load_engine_for_path(
            trainer_path)
        student_engine = self._load_engine_for_path(
            student_path)

        # Build lean persona prompt for student
        student_sys = self._build_student_system_prompt(
            training_brief=training_brief,
            student_name=student_name)

        test_scores = []
        num_tests = 10
        for t in range(num_tests):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            # TRAINER asks a test question
            test_q = teacher_engine.chat(
                f"Test #{t + 1}: Generate a "
                f"{stage}-level question to test "
                f"the student. Write ONLY the "
                f"question, nothing else.",
                system_prompt=trainer_sys,
                max_gen=100,
                temperature=0.9).strip()
            if not test_q:
                continue
            # STUDENT answers (with persona prompt)
            s_answer = student_engine.chat(
                test_q,
                system_prompt=student_sys,
                max_gen=256,
                temperature=0.7).strip()
            # TRAINER judges the answer
            judge_msg = (
                f'Question: "{test_q}"\n'
                f'Student answered: '
                f'"{s_answer}"\n\n'
                "Score 1-10:\n"
                "1-3 = poor  4-6 = developing  "
                "7-8 = good  9-10 = excellent\n"
                "Reply: SCORE: <n> | <feedback>")
            judgment = teacher_engine.chat(
                judge_msg,
                system_prompt=trainer_sys,
                max_gen=128,
                temperature=0.3).strip()

            # Parse score from judgment
            score = 5
            feedback = judgment
            for line in judgment.splitlines():
                ln = line.strip()
                if ln.upper().startswith("SCORE:"):
                    rest = ln.split(":", 1)[1]
                    parts = rest.strip().split(
                        "|", 1)
                    try:
                        score = int(
                            parts[0].strip()
                            .split()[0])
                        score = max(
                            1, min(10, score))
                    except (
                        ValueError, IndexError):
                        pass
                    if len(parts) > 1:
                        feedback = (
                            parts[1].strip())
                    break
            test_scores.append(score)
            q_s = test_q.replace("\n", " ")
            a_s = s_answer.replace("\n", " ")
            self._log(
                f"  Test {t + 1:>2d}  |  "
                f"Score: {score}/10")
            self._log(f"    Q: {q_s}")
            self._log(f"    A: {a_s}")
            if feedback and feedback != judgment:
                self._log(
                    f"    \u2192 {feedback}")

        # Readiness assessment
        if test_scores:
            avg = (sum(test_scores)
                   / len(test_scores))
            stages_list = [
                "basics", "conversation",
                "commands", "web"]
            s_idx = (
                stages_list.index(stage)
                if stage in stages_list
                else 0)
            next_s = (
                stages_list[s_idx + 1]
                if s_idx < len(stages_list) - 1
                else None)
            self._log(
                f"\nAverage : {avg:.1f} / 10")
            if avg >= 7:
                self._log(
                    "Result  : READY")
                if next_s:
                    self._log(
                        f"Next    : advance to "
                        f"'{next_s}' stage")
            elif avg >= 5:
                self._log(
                    "Result  : PROGRESSING")
                self._log(
                    "Next    : continue at "
                    "this stage")
            else:
                self._log(
                    "Result  : NEEDS WORK")
                self._log(
                    "Next    : more training "
                    "needed")

        # Append test results to curriculum file
        if curriculum_path and curriculum_path.exists():
            test_lines = [
                "",
                "# --- TEST RESULTS ---",
                f"# Tests: {len(test_scores)}",
            ]
            if test_scores:
                test_lines.append(
                    f"# Average: {avg:.1f}/10")
            for t_idx in range(len(test_scores)):
                test_lines.append(
                    f"# Test {t_idx + 1}: "
                    f"{test_scores[t_idx]}/10")
            with open(
                curriculum_path, "a",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(test_lines) + "\n")

    # ================================================================
    # Dialogue training (TRAINER ↔ STUDENT conversation)
    # ================================================================

    def _start_dialogue_training(self):
        """Interactive dialogue training: TRAINER and STUDENT have
        a real conversation, and the STUDENT learns from corrections.

        Each round:
        1. TRAINER asks a question
        2. STUDENT answers
        3. TRAINER scores the answer and provides a corrected version
        4. The correction is collected as training data
        After all rounds, the STUDENT is fine-tuned on the corrections.

        This lets the two AIs talk directly — the TRAINER sees what
        the STUDENT actually says and tailors its teaching to the
        STUDENT's real weaknesses.
        """
        if self.training_active:
            return

        # Require both routes
        trainer_path = self.route_assignments.get("trainer")
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Go to ROUTER and assign the teacher model.")
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        try:
            num_rounds = int(self.dialogue_rounds_entry.get())
            if num_rounds < 1 or num_rounds > 200:
                raise ValueError
        except (ValueError, AttributeError):
            num_rounds = 10

        try:
            epochs = int(self.guided_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            epochs = 5

        try:
            lr = float(self.guided_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            lr = 0.00005

        trainer_name = Path(trainer_path).stem
        student_name = Path(student_path).stem
        self.training_active = True
        self.dialogue_train_btn.configure(state="disabled",
                                          text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(
            "\u2692 DIALOGUE TRAINING...")

        self._log("--- DIALOGUE TRAINING INITIATED ---")
        self._log(f"Trainer : {trainer_name} (teacher)")
        self._log(f"Student : {student_name} (learner)")
        self._log(f"Rounds  : {num_rounds}  |  "
                   f"Epochs: {epochs}  |  LR: {lr}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _dialogue():
            losses = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Load TRAINER via EnigmaEngine (any format)
                self._log(f"Loading trainer: {trainer_name}...")
                teacher_engine = self._load_engine_for_path(
                    trainer_path)

                # Load STUDENT via EnigmaEngine for inference
                self._log(f"Loading student: {student_name}...")
                student_engine = self._load_engine_for_path(
                    student_path)

                # Read training stage from UI
                stage = getattr(self, 'training_stage_var', None)
                stage = stage.get() if stage else "basics"

                # Count student params for system prompt
                s_params = 0
                s_cfg = None
                try:
                    from enigma_engine.core.model_registry import (
                        safe_load_weights)
                    s_ckpt = safe_load_weights(
                        student_path, map_location="cpu")
                    # Prefer model_config, fall back to config, skip TrainingConfig
                    s_cfg_d = s_ckpt.get("model_config") or s_ckpt.get("config", {})
                    if isinstance(s_cfg_d, dict) and "epochs" in s_cfg_d:
                        s_cfg_d = s_ckpt.get("model_config", {})
                    s_cfg = ForgeConfig(**s_cfg_d)
                    s_params = sum(
                        p.numel()
                        for p in Enigma(config=s_cfg).parameters())
                    del s_ckpt
                except Exception:
                    pass

                # Read training brief from UI
                training_brief = self._build_training_brief()
                self._save_training_brief()

                # Read focus field from UI
                focus_field = ""
                ff_widget = getattr(self, "forge_focus_field", None)
                if ff_widget is not None:
                    focus_field = ff_widget.get().strip()

                # Build system prompt for TRAINER
                trainer_sys = self._build_trainer_system_prompt(
                    student_params=s_params,
                    student_cfg=s_cfg,
                    task="training",
                    stage=stage,
                    training_brief=training_brief,
                    focus_field=focus_field)
                self._log(f"Stage   : {stage.upper()}")
                if focus_field:
                    self._log(f"Focus   : {focus_field}")
                if training_brief:
                    self._log(f"Brief   : {len(training_brief)} chars")

                # Build lean persona prompt for student
                student_sys = self._build_student_system_prompt(
                    training_brief=training_brief,
                    student_name=student_name)

                # === Conversation loop ===
                corrections, scores, transcript_name = (
                    self._dialogue_conversation_loop(
                        teacher_engine=teacher_engine,
                        student_engine=student_engine,
                        trainer_sys=trainer_sys,
                        student_sys=student_sys,
                        num_rounds=num_rounds,
                        stage=stage,
                        trainer_name=trainer_name,
                        student_name=student_name,
                    ))

                if not corrections:
                    self._log(
                        "[!] No corrections generated — "
                        "nothing to train on.")
                    return

                # Free inference models before training
                del teacher_engine, student_engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Train STUDENT on corrections ===
                self._dialogue_train_on_corrections(
                    corrections=corrections,
                    student_path=student_path,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    losses=losses,
                    ForgeConfig=ForgeConfig,
                    Enigma=Enigma,
                )

                self._log(
                    "\n--- DIALOGUE TRAINING COMPLETE ---")
                self._log(
                    "Run again to continue building on "
                    "what the student learned.")
                if transcript_name:
                    self._log(
                        f"Review transcript: {transcript_name}")
                best_loss = min(losses) if losses else 0.0
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Dialogue", student_name, epochs, best_loss)
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, lambda sp=s_params: self._update_forge_param_count(sp))
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log(
                    "\n--- DIALOGUE TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(
                    f"\n[!] Dialogue training failed: {exc}")
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(
                    0, lambda: self.dialogue_train_btn.configure(
                        state="normal", text="TRAIN"))
                self.after(
                    0, lambda: self.stop_train_btn.configure(
                        state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_dialogue, daemon=True).start()

    def _dialogue_conversation_loop(
        self, *, teacher_engine, student_engine, trainer_sys,
        student_sys, num_rounds, stage, trainer_name, student_name,
    ):
        """Run the multi-turn conversation between TRAINER and STUDENT.

        Returns (corrections, scores, transcript_name).
        """
        self._log(
            f"\n--- DIALOGUE: {num_rounds} ROUNDS ---\n")
        corrections = []
        scores = []
        conversation_history = []

        for r in range(num_rounds):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")

            # Build context from recent conversation
            context = ""
            if conversation_history:
                recent = conversation_history[-6:]
                context = (
                    "Recent conversation:\n"
                    + "\n".join(recent)
                    + "\n\n")

            # TRAINER asks a question
            ask_msg = (
                f"{context}"
                f"Round {r + 1} of {num_rounds}.\n"
                f"Ask the student a {stage}-level "
                f"question. Make it build on previous "
                f"conversation if possible. Write ONLY "
                f"the question.")
            question = teacher_engine.chat(
                ask_msg,
                system_prompt=trainer_sys,
                max_gen=128,
                temperature=0.8).strip()
            if not question:
                continue

            # STUDENT answers (with persona prompt)
            student_answer = student_engine.chat(
                question,
                system_prompt=student_sys,
                max_gen=256,
                temperature=0.7).strip()
            if not student_answer:
                student_answer = "(no response)"

            # Track conversation history
            conversation_history.append(
                f"TRAINER: {question}")
            conversation_history.append(
                f"STUDENT: {student_answer}")

            # TRAINER scores and provides correction
            judge_msg = (
                f"You asked: \"{question}\"\n"
                f"The student answered: "
                f"\"{student_answer}\"\n\n"
                f"If the answer deserves less than 8/10, "
                f"you MUST provide a CORRECTED: version.\n\n"
                f"1. Score the answer 1-10\n"
                f"2. Write the CORRECTED ideal answer "
                f"(how the student SHOULD have "
                f"responded)\n\n"
                f"Format:\n"
                f"SCORE: <n>\n"
                f"CORRECTED: <ideal answer>")
            judgment = teacher_engine.chat(
                judge_msg,
                system_prompt=trainer_sys,
                max_gen=300,
                temperature=0.3).strip()

            # Parse score and corrected answer
            score = 5
            corrected = ""
            for line in judgment.splitlines():
                ln = line.strip()
                if ln.upper().startswith("SCORE:"):
                    rest = ln.split(":", 1)[1].strip()
                    try:
                        score = int(
                            rest.split()[0])
                        score = max(1, min(10, score))
                    except (ValueError, IndexError):
                        pass
                elif ln.upper().startswith("CORRECTED:"):
                    corrected = ln.split(":", 1)[1].strip()

            # If no CORRECTED line, use full judgment
            # as the correction (minus the SCORE line)
            if not corrected:
                corrected = "\n".join(
                    ln for ln in judgment.splitlines()
                    if not ln.strip().upper()
                    .startswith("SCORE:")).strip()

            scores.append(score)

            # High-scoring answers: reinforce what
            # the student already got right
            if score >= 8 and student_answer:
                corrections.append(
                    self._format_training_pair(
                        stage, question,
                        student_answer))
            elif corrected:
                # Low-scoring: use TRAINER's correction
                corrections.append(
                    self._format_training_pair(
                        stage, question, corrected))

            # Log the turn
            q_s = question.replace("\n", " ")
            a_s = student_answer.replace("\n", " ")
            c_s = corrected.replace("\n", " ")
            self._log(
                f"  Round {r + 1:>3d}  |  "
                f"Score: {score}/10")
            pct = int((r + 1) / num_rounds * 50)
            self._update_forge_progress(
                pct, f"Round {r + 1}/{num_rounds}")
            self._log(f"    TRAINER: {q_s}")
            self._log(f"    STUDENT: {a_s}")
            if corrected and score < 8:
                self._log(f"    CORRECTED: {c_s}")

        # Conversation summary
        avg = 0.0
        if scores:
            avg = sum(scores) / len(scores)
            self._log(f"\n  Average score: {avg:.1f}/10")
            improvement = ""
            if len(scores) >= 4:
                first_half = (
                    sum(scores[:len(scores) // 2])
                    / max(1, len(scores) // 2))
                second_half = (
                    sum(scores[len(scores) // 2:])
                    / max(1, len(scores)
                          - len(scores) // 2))
                diff = second_half - first_half
                if diff > 0.5:
                    improvement = (
                        f"  Improvement: +{diff:.1f} "
                        f"(first half {first_half:.1f} "
                        f"→ second half "
                        f"{second_half:.1f})")
                elif diff < -0.5:
                    improvement = (
                        f"  Decline: {diff:.1f} "
                        f"(first half {first_half:.1f} "
                        f"→ second half "
                        f"{second_half:.1f})")
            if improvement:
                self._log(improvement)

        if not corrections:
            return corrections, scores, ""

        self._log(
            f"\nCollected {len(corrections)} "
            f"training pairs "
            f"({len([s for s in scores if s >= 8])} "
            f"reinforced, "
            f"{len([s for s in scores if s < 8])} "
            f"corrected).")

        # Save full transcript for review
        from datetime import datetime
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S")
        transcript_name = (
            f"dialogue_{student_name}"
            f"_{timestamp}.txt")
        transcript_path = DATA_DIR / transcript_name
        transcript_lines = [
            "# Dialogue Training Transcript",
            f"# Trainer: {trainer_name}",
            f"# Student: {student_name}",
            f"# Stage: {stage}",
            f"# Rounds: {len(scores)}",
            f"# Average: {avg:.1f}/10" if scores
            else "# Average: N/A",
            f"# Date: {datetime.now().isoformat()}",
            ""]
        for i, entry in enumerate(
                conversation_history):
            transcript_lines.append(entry)
            # Add score after each STUDENT line
            if entry.startswith("STUDENT:"):
                idx = i // 2
                if idx < len(scores):
                    transcript_lines.append(
                        f"  [Score: {scores[idx]}/10]")
        transcript_lines.append("")
        transcript_lines.append(
            "# --- Training Data Used ---")
        transcript_lines.extend(corrections)
        from enigma_engine.core.safe_save import atomic_write_text
        atomic_write_text(
            transcript_path,
            "\n".join(transcript_lines) + "\n")
        self._log(
            f"Transcript: {transcript_path.name}")
        self.after(0, self._refresh_data_files)

        return corrections, scores, transcript_name

    def _dialogue_train_on_corrections(
        self, *, corrections, student_path, device,
        epochs, lr, losses, ForgeConfig, Enigma,
    ):
        """Train STUDENT model on collected corrections."""
        import torch
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.safe_save import atomic_torch_save

        self._log(
            "\n--- TRAINING ON CORRECTIONS ---\n")
        combined = "\n\n".join(corrections)

        # Load STUDENT for gradient training
        from enigma_engine.core.model_registry import (
            safe_load_weights)
        s_ckpt = safe_load_weights(
            student_path, map_location=device)
        s_cfg_dict = (s_ckpt.get("model_config")
                      or s_ckpt.get("config", {}))
        if isinstance(s_cfg_dict, dict) and "epochs" in s_cfg_dict:
            s_cfg_dict = s_ckpt.get("model_config", {})
        s_cfg = ForgeConfig(**s_cfg_dict)
        student = Enigma(config=s_cfg)
        student.load_state_dict(
            s_ckpt["model_state_dict"])
        student = student.to(device)

        tokenizer = get_tokenizer("auto")

        forge_params = self._read_forge_train_params()
        train_config = TrainingConfig(
            epochs=epochs,
            batch_size=forge_params["batch_size"],
            learning_rate=lr,
            max_grad_accumulation=forge_params["max_grad_accumulation"],
            use_gradient_checkpointing=forge_params["use_gradient_checkpointing"],
            val_split=forge_params["val_split"],
            save_every=max(1, epochs // 5),
            checkpoint_dir=str(MODELS_DIR / "checkpoints"),
            use_amp=torch.cuda.is_available())

        trainer_obj = Trainer(
            student, tokenizer, train_config)

        def on_epoch(epoch, loss):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            losses.append(loss)
            pct = 50 + int(epoch / epochs * 50)
            self._update_forge_progress(
                pct, f"Epoch {epoch}/{epochs}")
            self._log(
                f"  Epoch {epoch:>3d}  |  "
                f"loss {loss:.4f}")
        trainer_obj.on_epoch_complete = on_epoch

        state = trainer_obj.train(combined)

        # Save trained STUDENT
        out = Path(student_path)
        atomic_torch_save({
            "model_state_dict": student.state_dict(),
            "config": self._model_config_dict(student),
            "training_state": {
                "epochs": state.epoch,
                "best_loss": state.best_loss,
            },
        }, out)
        self._log(f"\nBest loss : {state.best_loss:.4f}")
        self._log(f"Saved to  : {out}")

