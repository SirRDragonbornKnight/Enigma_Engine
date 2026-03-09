"""
Enigma Engine - Forge Adaptive Pipeline
====================================================

Adaptive training pipeline combining:
- TC-C3: Continuous adaptive loop (probe → adjust difficulty)
- SA-B: Auto-chain all 4 stages with phase 3 testing
- SA-C: Saveable JSON training plan with resume support

Split from gui_forge_advanced.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ForgeAdaptiveMixin:
    """Adaptive training pipeline for EnigmaGUI.

    Expects the host class to have ForgeMixin setup attributes
    plus ForgeAdvancedMixin helper methods (_build_trainer_system_prompt,
    _load_engine_for_path, _extract_prompts, _format_training_pair, etc.).
    """

    # ================================================================
    # Adaptive Pipeline (TC-C3 + SA-B + SA-C)
    # ================================================================

    def _start_adaptive_training(self):
        """Autonomous adaptive training pipeline.

        Combines:
        - TC-C3: Continuous adaptive loop — probes student, adjusts
          difficulty per stage based on real ability.
        - SA-B: Auto-chains BASICS → CONVERSATION → COMMANDS → WEB
          with Phase 3 tests deciding advance/retry/simplify.
        - SA-C: Saves progress as a JSON training plan that can
          resume if interrupted.

        One button runs the entire pipeline. The TRAINER evaluates
        the STUDENT at each stage and adapts difficulty. Progress
        is saved after every stage so training can resume on crash.
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
            epochs = int(self.guided_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except ValueError:
            self._log("[!] Epochs must be 1-1000.")
            return

        try:
            lr = float(self.guided_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except ValueError:
            self._log("[!] Learning rate must be 0 to 1.")
            return

        try:
            num_pairs = int(self.guided_pairs_entry.get())
            if num_pairs < 1 or num_pairs > 500:
                raise ValueError
        except (ValueError, AttributeError):
            num_pairs = 20

        # Data file is optional — supplements curriculum
        data_path = self.train_data_var.get()
        has_data = data_path and Path(data_path).exists()

        trainer_name = Path(trainer_path).stem
        student_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(
            state="disabled", text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(
            "\u2692 ADAPTIVE PIPELINE...")

        self._log("--- ADAPTIVE PIPELINE INITIATED ---")
        self._log(f"Trainer : {trainer_name}")
        self._log(f"Student : {student_name}")
        self._log(f"Epochs  : {epochs}/stage  |  LR: {lr}")
        self._log(f"Pairs   : {num_pairs}/stage")
        if has_data:
            self._log(f"Bonus   : {Path(data_path).name}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # Build training brief and focus
        training_brief = self._build_training_brief()
        self._save_training_brief()
        focus_field = ""
        ff_widget = getattr(self, "forge_focus_field", None)
        if ff_widget is not None:
            focus_field = ff_widget.get().strip()

        def _adaptive():
            all_losses = []
            plan = None
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.tokenizer import (
                    get_tokenizer)
                from enigma_engine.core.model_presets import (
                    ForgeConfig)
                from enigma_engine.core.adaptive_trainer import (
                    TrainingPlan)

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Check for existing plan to resume
                plan_path = (
                    DATA_DIR / f"plan_{student_name}.json")
                if plan_path.exists():
                    try:
                        plan = TrainingPlan.load(plan_path)
                        if plan.is_complete:
                            self._log(
                                "Previous plan is complete. "
                                "Starting fresh.")
                            plan = None
                        else:
                            self._log(
                                f"Resuming plan: stage "
                                f"{plan.current_stage_idx + 1}"
                                f"/{len(plan.stages)} "
                                f"({plan.current_stage})")
                    except Exception as exc:
                        self._log(
                            f"Could not load plan: {exc}")
                        plan = None

                if plan is None:
                    plan = TrainingPlan(
                        student_path=str(student_path),
                        trainer_path=str(trainer_path),
                        student_name=student_name,
                        trainer_name=trainer_name,
                        epochs_per_stage=epochs,
                        pairs_per_stage=num_pairs,
                        learning_rate=lr,
                        training_brief=training_brief,
                        focus_field=focus_field)
                plan.status = "running"
                plan.save(plan_path)

                # Load TRAINER
                self._log(
                    f"Loading trainer: {trainer_name}...")
                teacher_engine = self._load_engine_for_path(
                    trainer_path)

                # Load STUDENT on CPU for initial probe
                self._log(
                    f"Loading student: {student_name}...")
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                s_ckpt = safe_load_weights(
                    student_path, map_location="cpu")
                s_cfg_dict = (s_ckpt.get("model_config")
                              or s_ckpt.get("config", {}))
                if (isinstance(s_cfg_dict, dict)
                        and "epochs" in s_cfg_dict):
                    s_cfg_dict = (
                        s_ckpt.get("model_config", {}))
                s_cfg = ForgeConfig(**s_cfg_dict)
                student = Enigma(config=s_cfg)
                student.load_state_dict(
                    get_state_dict(s_ckpt))
                del s_ckpt
                s_params = sum(
                    p.numel()
                    for p in student.parameters())
                self._log(f"Student : {s_params:,} params")

                tokenizer = get_tokenizer("auto")

                # Start at simple difficulty (default)
                plan.current_difficulty = "simple"
                self._log(
                    f"Difficulty: {plan.current_difficulty}")

                # === Stage loop (SA-B auto-chain) ===
                self._run_adaptive_stages(
                    plan, student, teacher_engine,
                    trainer_path, student_path,
                    trainer_name, student_name,
                    s_params, s_cfg, tokenizer,
                    epochs, num_pairs, lr, has_data,
                    data_path, training_brief,
                    focus_field, all_losses, device)

                # === Pipeline complete ===
                plan_path = (
                    DATA_DIR / f"plan_{student_name}.json")
                if plan.is_complete:
                    plan.status = "completed"
                else:
                    plan.status = "paused"
                plan.save(plan_path)

                self._log(
                    "\n--- ADAPTIVE PIPELINE COMPLETE ---")
                self._log(plan.summary())
                total_stages = len(plan.stages)
                best_loss = (min(all_losses)
                             if all_losses else 0.0)
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Adaptive", student_name,
                    epochs * total_stages, best_loss)
                if all_losses:
                    self._display_loss_curve(all_losses)
                self.after(
                    0, lambda sp=s_params:
                    self._update_forge_param_count(sp))
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log(
                    "\n--- ADAPTIVE PIPELINE STOPPED ---")
                if plan is not None:
                    plan.status = "paused"
                    pp = (
                        DATA_DIR
                        / f"plan_{student_name}.json")
                    plan.save(pp)
                    self._log(
                        "Plan saved — resume with "
                        "Adaptive Pipeline mode.")
                if all_losses:
                    self._display_loss_curve(all_losses)
            except Exception as exc:
                self._log(
                    f"\n[!] Adaptive pipeline failed: {exc}")
                if plan is not None:
                    plan.status = "failed"
                    pp = (
                        DATA_DIR
                        / f"plan_{student_name}.json")
                    plan.save(pp)
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(
                    0, lambda: self.solo_train_btn.configure(
                        state="normal", text="TRAIN"))
                self.after(
                    0, lambda: self.stop_train_btn.configure(
                        state="disabled"))
                self.after(
                    0, lambda: self.status_bar.set_left(
                        "\u26a1 READY"))

        threading.Thread(
            target=_adaptive, daemon=True).start()

    def _run_adaptive_stages(
            self, plan, student, teacher_engine,
            trainer_path, student_path,
            trainer_name, student_name,
            s_params, s_cfg, tokenizer,
            epochs, num_pairs, lr, has_data,
            data_path, training_brief,
            focus_field, all_losses, device):
        """Execute the stage loop for adaptive training.

        Separated from _start_adaptive_training to keep method
        length manageable.
        """
        import torch
        from enigma_engine.core.adaptive_trainer import (
            StageResult)

        total_stages = len(plan.stages)
        while (plan.current_stage is not None
               and self.training_active):
            stage = plan.current_stage
            stage_idx = plan.current_stage_idx
            attempt = plan.current_attempt + 1
            difficulty = plan.current_difficulty

            self._log(f"\n{'=' * 50}")
            self._log(
                f"STAGE {stage_idx + 1}/{total_stages}: "
                f"{stage.upper()} "
                f"(attempt {attempt}, "
                f"difficulty: {difficulty})")
            self._log(f"{'=' * 50}")

            # Update progress bar
            stage_pct_base = int(
                stage_idx / total_stages * 100)
            self._update_forge_progress(
                stage_pct_base,
                f"Stage: {stage.upper()}")

            # Build system prompt for trainer
            trainer_sys = self._build_trainer_system_prompt(
                student_params=s_params,
                student_cfg=s_cfg,
                task="training",
                stage=stage,
                training_brief=training_brief,
                focus_field=focus_field)

            # --- Phase 1: Generate curriculum ---
            pairs = self._adaptive_phase1_generate(
                teacher_engine, trainer_sys,
                num_pairs, stage, difficulty,
                has_data, data_path)

            if not pairs:
                self._log(
                    "[!] No training material generated.")
                break

            combined = "\n\n".join(pairs)
            self._log(
                f"Curriculum: {len(pairs)} examples")

            # Save curriculum to data/
            from datetime import datetime
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S")
            self._save_adaptive_curriculum(
                combined, stage, difficulty,
                trainer_name, student_name,
                len(pairs), attempt, timestamp)

            # Free trainer memory for student training
            del teacher_engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Phase 2: Train student ---
            stage_losses = self._adaptive_phase2_train(
                student, tokenizer, combined,
                epochs, lr, stage, stage_pct_base,
                total_stages, all_losses, student_path,
                device)

            # Free student for testing
            student = student.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Phase 3: Test student ---
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")

            teacher_engine = self._load_engine_for_path(
                trainer_path)
            test_scores = self._adaptive_phase3_test(
                teacher_engine, student_path,
                trainer_sys, stage, difficulty)

            # Record result and decide action
            avg_score = (sum(test_scores)
                         / max(1, len(test_scores)))
            self._log(f"\nAverage : {avg_score:.1f}/10")

            result = StageResult(
                stage=stage,
                attempt=attempt,
                difficulty=difficulty,
                scores=[float(s) for s in test_scores],
                avg_score=avg_score,
                status="pending",
                epochs_trained=epochs,
                pairs_generated=len(pairs),
                best_loss=(min(stage_losses)
                           if stage_losses
                           else float("inf")),
                started_at=timestamp,
                completed_at=datetime.now().isoformat())

            self._adaptive_decide_action(
                plan, result, avg_score, attempt, stage)

            # Save plan after every stage attempt
            plan_path = (
                DATA_DIR / f"plan_{student_name}.json")
            plan.save(plan_path)

            # Reload student for next iteration
            if plan.current_stage is not None:
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import (
                    ForgeConfig)
                s_ckpt = safe_load_weights(
                    student_path, map_location="cpu")
                sc = (s_ckpt.get("model_config")
                      or s_ckpt.get("config", {}))
                if isinstance(sc, dict) and "epochs" in sc:
                    sc = s_ckpt.get("model_config", {})
                s_cfg = ForgeConfig(**sc)
                student = Enigma(config=s_cfg)
                student.load_state_dict(
                    get_state_dict(s_ckpt))
                del s_ckpt

    def _adaptive_phase1_generate(
            self, teacher_engine, trainer_sys,
            num_pairs, stage, difficulty,
            has_data, data_path):
        """Phase 1: Generate curriculum using adaptive prompts."""
        from enigma_engine.core.adaptive_trainer import (
            build_adaptive_prompt)

        self._log(
            "\n--- Phase 1: GENERATING CURRICULUM ---")
        pairs = []
        for i in range(num_pairs):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            msg = build_adaptive_prompt(
                i + 1, num_pairs, stage, difficulty)
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

        # Supplement with data file if available
        if has_data:
            extras = self._extract_prompts(data_path)
            self._log(
                f"Adding {len(extras)} bonus prompts...")
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
        return pairs

    def _adaptive_phase2_train(
            self, student, tokenizer, combined,
            epochs, lr, stage, stage_pct_base,
            total_stages, all_losses, student_path,
            device):
        """Phase 2: Train student on generated curriculum."""
        import torch
        from enigma_engine.core.training import (
            Trainer, TrainingConfig)
        from enigma_engine.core.safe_save import (
            atomic_torch_save)

        self._log(
            "\n--- Phase 2: TRAINING STUDENT ---")
        student = student.to(device)

        forge_params = self._read_forge_train_params()
        train_config = TrainingConfig(
            epochs=epochs,
            batch_size=forge_params["batch_size"],
            learning_rate=lr,
            max_grad_accumulation=forge_params[
                "max_grad_accumulation"],
            use_gradient_checkpointing=forge_params[
                "use_gradient_checkpointing"],
            save_every=max(1, epochs // 5),
            checkpoint_dir=str(
                MODELS_DIR / "checkpoints"),
            use_amp=torch.cuda.is_available())

        trainer_obj = Trainer(
            student, tokenizer, train_config)
        stage_losses = []

        def on_epoch(epoch, loss, _sl=stage_losses):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            _sl.append(loss)
            all_losses.append(loss)
            pct = stage_pct_base + int(
                epoch / epochs
                * (100 // total_stages) * 0.5)
            self._update_forge_progress(
                min(99, pct),
                f"{stage.upper()} "
                f"Epoch {epoch}/{epochs}")
            self._log(
                f"  Epoch {epoch:>3d}  |  "
                f"loss {loss:.4f}")
        trainer_obj.on_epoch_complete = on_epoch

        state = trainer_obj.train(combined)

        # Save student checkpoint
        out = Path(student_path)
        atomic_torch_save({
            "model_state_dict": student.state_dict(),
            "config": self._model_config_dict(student),
            "training_state": {
                "epochs": state.epoch,
                "best_loss": state.best_loss,
            },
        }, out)
        self._log(f"Best loss : {state.best_loss:.4f}")

        del trainer_obj
        return stage_losses

    def _adaptive_phase3_test(
            self, teacher_engine, student_path,
            trainer_sys, stage, difficulty):
        """Phase 3: Test student with TRAINER-generated questions."""
        import torch

        self._log(
            "\n--- Phase 3: TESTING STUDENT ---")
        student_engine = self._load_engine_for_path(
            student_path)

        test_scores = []
        num_tests = 10
        for t in range(num_tests):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            test_q = teacher_engine.chat(
                f"Test #{t + 1}: Generate a "
                f"{stage}-level ({difficulty} difficulty) "
                f"question. Write ONLY the question.",
                system_prompt=trainer_sys,
                max_gen=64,
                temperature=0.9).strip()
            if not test_q:
                continue
            s_answer = student_engine.generate(
                test_q, max_gen=128,
                temperature=0.7).strip()
            judge_msg = (
                f'Question: "{test_q}"\n'
                f'Answer: "{s_answer}"\n\n'
                "Score 1-10:\n"
                "Reply: SCORE: <n> | <feedback>")
            judgment = teacher_engine.chat(
                judge_msg,
                system_prompt=trainer_sys,
                max_gen=64,
                temperature=0.3).strip()

            score = self._parse_test_score(judgment)
            test_scores.append(score)
            q_short = (test_q[:40] + "..."
                       if len(test_q) > 40
                       else test_q)
            self._log(
                f"  Test {t + 1:>2d}  |  "
                f"Score: {score}/10  Q: {q_short}")

        # Free test engine
        del student_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return test_scores

    @staticmethod
    def _parse_test_score(judgment: str) -> int:
        """Extract numeric score from TRAINER judgment text."""
        score = 5
        for line in judgment.splitlines():
            ln = line.strip()
            if ln.upper().startswith("SCORE:"):
                rest = ln.split(":", 1)[1]
                parts = rest.strip().split("|", 1)
                try:
                    score = int(
                        parts[0].strip().split()[0])
                    score = max(1, min(10, score))
                except (ValueError, IndexError):
                    pass
                break
        return score

    def _adaptive_decide_action(
            self, plan, result, avg_score,
            attempt, stage):
        """Decide and execute adaptive action based on stage completion."""
        action = plan.decide_action(avg_score)
        self._log(f"Action  : {action.upper()}")

        if action in ("advance", "complete"):
            result.status = "passed"
            plan.record_result(result)
            has_next = plan.advance_stage()
            if has_next:
                self._log(
                    f"Advancing to: "
                    f"{plan.current_stage.upper()}")
            else:
                self._log("All stages complete!")

    def _save_adaptive_curriculum(
            self, combined, stage, difficulty,
            trainer_name, student_name,
            num_pairs, attempt, timestamp):
        """Save generated curriculum to data/ directory."""
        from datetime import datetime
        curr_name = (
            f"adaptive_{student_name}"
            f"_{stage}_{timestamp}.txt")
        curr_path = DATA_DIR / curr_name
        header_lines = [
            "# Adaptive Pipeline Curriculum",
            f"# Stage: {stage} ({difficulty})",
            f"# Trainer: {trainer_name}",
            f"# Student: {student_name}",
            f"# Pairs: {num_pairs}",
            f"# Attempt: {attempt}",
            f"# Date: {datetime.now().isoformat()}",
            ""]
        curr_path.write_text(
            "\n".join(header_lines)
            + combined + "\n",
            encoding="utf-8")
        self._log(f"Saved     : {curr_name}")
        self.after(0, self._refresh_data_files)

    def _resume_training_plan(
            self, plan_path: str | Path):
        """Resume an interrupted adaptive training plan.

        Loads the plan from JSON, sets appropriate mode, and starts
        the adaptive pipeline from where it left off.

        Args:
            plan_path: Path to the plan JSON file.
        """
        try:
            from enigma_engine.core.adaptive_trainer import (
                TrainingPlan)
            plan = TrainingPlan.load(plan_path)
            if plan.is_complete:
                self._log(
                    "[!] This training plan is already "
                    "complete.")
                return
            self._log(f"Resuming plan: {plan.summary()}")
            # Set routes from plan
            if plan.trainer_path:
                self.route_assignments["trainer"] = (
                    plan.trainer_path)
            if plan.student_path:
                self.route_assignments["student"] = (
                    plan.student_path)
            # Start adaptive training — it detects the plan
            self._start_adaptive_training()
        except Exception as exc:
            self._log(
                f"[!] Could not resume plan: {exc}")
