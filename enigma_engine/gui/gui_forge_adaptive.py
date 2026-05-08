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

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        try:
            raw_pairs = self.guided_pairs_entry.get().strip()
            num_pairs = int(raw_pairs)
            if num_pairs < 1 or num_pairs > 500:
                self._log(f"[!] Pairs '{raw_pairs}' out of range "
                          f"(1-500), using 20")
                num_pairs = 20
        except (ValueError, AttributeError):
            raw_w = getattr(self, "guided_pairs_entry", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Pairs '{raw}' not a number, "
                          f"using 20")
            num_pairs = 20

        # Data file is optional — supplements curriculum
        # In AI-Guided mode, read from the supplement picker
        _mode_var = getattr(self, "training_mode_var", None)
        _current_mode = _mode_var.get() if _mode_var else "Basic"
        if _current_mode == "AI-Guided":
            _suppl_var = getattr(self, "ai_supplement_var", None)
            _suppl = _suppl_var.get() if _suppl_var else ""
            data_path = "" if (not _suppl or _suppl == "(none)") else _suppl
        else:
            data_path = self.train_data_var.get()
        has_data = bool(data_path) and Path(data_path).exists()

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

        # Build training brief
        training_brief = self._build_training_brief()
        self._save_training_brief()
        # Focus field widget was removed in 3-mode FORGE.
        focus_field = ""
        # Start from selected stage — "start here, then continue forward"
        _stage_var = getattr(self, "training_stage_var", None)
        _selected_stage = _stage_var.get() if _stage_var else "basics"
        from enigma_engine.core.adaptive_trainer import ALL_STAGES as _ALL_STAGES
        _start_idx = (_ALL_STAGES.index(_selected_stage)
                      if _selected_stage in _ALL_STAGES else 0)
        if _selected_stage not in _ALL_STAGES:
            logger.warning(
                "Invalid stage '%s', defaulting to '%s'",
                _selected_stage, _ALL_STAGES[0])
        plan_stages = _ALL_STAGES[_start_idx:]

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
                        stages=list(plan_stages),
                        training_brief=training_brief)
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

                _bpe_path = MODELS_DIR / "tokenizer.json"
                if _bpe_path.exists():
                    try:
                        from enigma_engine.core.bpe_tokenizer import BPETokenizer
                        tokenizer = BPETokenizer(_bpe_path)
                    except Exception:
                        tokenizer = get_tokenizer("auto")
                else:
                    tokenizer = get_tokenizer("auto")

                # Only reset difficulty for a brand-new plan;
                # resumed plans keep their saved difficulty.
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
                self._notify_training_complete()

                # Training report card + next steps
                self._log_training_report(
                    plan, all_losses, student_name)

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
                import traceback
                tb = traceback.format_exc()
                self._log(
                    f"\n[!] Adaptive pipeline failed: {exc}")
                self._log(tb)
                if plan is not None:
                    plan.status = "failed"
                    plan.reset_difficulty()
                    pp = (
                        DATA_DIR
                        / f"plan_{student_name}.json")
                    plan.save(pp)
            finally:
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(
                    0, lambda: self.solo_train_btn.configure(
                        state="normal", text="TRAIN"))
                self.after(
                    0, lambda: self.stop_train_btn.configure(
                        state="disabled", text="STOP"))
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
        # Accumulate curriculum per stage across retry
        # attempts so the student trains on all prior data
        # plus new data each time.
        accumulated: dict[str, list[str]] = {}

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

            if not pairs and stage not in accumulated:
                self._log(
                    "[!] No training material generated.")
                break

            # Accumulate: add new pairs to prior data
            # for this stage (resets when stage advances)
            if stage not in accumulated:
                accumulated[stage] = []
            accumulated[stage].extend(pairs)
            all_pairs = accumulated[stage]

            combined = "\n\n".join(all_pairs)
            self._log(
                f"Curriculum: {len(pairs)} new, "
                f"{len(all_pairs)} total examples")

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
            # Build lean persona prompt for STUDENT
            student_sys = self._build_student_system_prompt(
                training_brief=training_brief,
                student_name=student_name)
            test_scores = self._adaptive_phase3_test(
                teacher_engine, student_path,
                trainer_sys, stage, difficulty,
                student_sys=student_sys)

            # Record result and decide action
            best_loss = (min(stage_losses)
                         if stage_losses
                         else float("inf"))
            if test_scores:
                avg_score = (sum(test_scores)
                             / len(test_scores))
            else:
                # Fallback: use training loss as proxy
                # when testing produced no valid scores
                from enigma_engine.core.adaptive_trainer import (
                    loss_to_proxy_score)
                avg_score = float(
                    loss_to_proxy_score(best_loss))
                self._log(
                    "  (No test scores — using loss-based "
                    f"proxy: {avg_score:.0f}/10)")
            self._log(f"\nAverage : {avg_score:.1f}/10")

            result = StageResult(
                stage=stage,
                attempt=attempt,
                difficulty=difficulty,
                scores=[float(s) for s in test_scores],
                avg_score=avg_score,
                status="pending",
                epochs_trained=epochs,
                pairs_generated=len(all_pairs),
                best_loss=best_loss,
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
        """Phase 1: Generate curriculum using adaptive prompts.

        Retries up to 2 times per example when the teacher returns
        an empty response, so the final count stays close to the
        requested ``num_pairs``.
        """
        from enigma_engine.core.adaptive_trainer import (
            build_adaptive_prompt, clean_example,
            validate_example, deduplicate_examples)

        self._log(
            "\n--- Phase 1: GENERATING CURRICULUM ---")
        pairs = []
        rejected = 0
        max_retries_per_example = 2
        for i in range(num_pairs):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")
            msg = build_adaptive_prompt(
                i + 1, num_pairs, stage, difficulty)
            result = ""
            for attempt in range(1 + max_retries_per_example):
                raw = teacher_engine.chat(
                    msg,
                    system_prompt=trainer_sys,
                    max_gen=512,
                    temperature=0.8).strip()
                if not raw:
                    continue
                cleaned = clean_example(raw)
                if validate_example(cleaned, stage):
                    result = cleaned
                    break
                # Invalid format — retry with a fresh generation
            if result:
                pairs.append(result)
                # Preview: show full example
                preview = result.replace("\n", " ")
                self._log(
                    f"  [{i + 1}/{num_pairs}] {preview}")
            else:
                rejected += 1
                self._log(
                    f"  [{i + 1}/{num_pairs}] "
                    f"(rejected after {max_retries_per_example} "
                    f"retries, last raw: {raw[:80]!r})"
                    if raw else
                    f"  [{i + 1}/{num_pairs}] "
                    f"(rejected, teacher returned empty)")
        if rejected:
            self._log(
                f"  Quality filter: {rejected} rejected, "
                f"{len(pairs)} accepted")

        # Deduplicate against already-accumulated examples
        pairs = deduplicate_examples(pairs)

        # Supplement with data file if available
        if has_data:
            extras = self._extract_prompts(data_path)
            self._log(
                f"Adding {len(extras)} bonus prompts...")
            for prompt in extras:
                if not self.training_active:
                    raise KeyboardInterrupt("Stopped")
                raw = teacher_engine.chat(
                    prompt,
                    system_prompt=trainer_sys,
                    max_gen=256,
                    temperature=0.7).strip()
                if raw:
                    cleaned = clean_example(raw)
                    pair = self._format_training_pair(
                        stage, prompt, cleaned or raw)
                    pairs.append(pair)
        return pairs

    def _adaptive_phase2_train(
            self, student, tokenizer, combined,
            epochs, lr, stage, stage_pct_base,
            total_stages, all_losses, student_path,
            device):
        """Phase 2: Train student on generated curriculum."""
        import torch
        from enigma_engine.training.training import (
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
            general_mix_ratio=forge_params["general_mix_ratio"],
            general_data=forge_params["general_data"],
            val_split=forge_params["val_split"],
            save_every=max(1, epochs // 5),
            checkpoint_dir=str(
                MODELS_DIR / "checkpoints"),
            use_amp=torch.cuda.is_available())

        trainer_obj = Trainer(
            student, tokenizer, train_config)
        stage_losses = []
        import time as _time_ad
        _ad_stage_start = [_time_ad.monotonic()]

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
            elapsed = (
                _time_ad.monotonic()
                - _ad_stage_start[0])
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            eta = ""
            if epoch > 0:
                remaining = (
                    (elapsed / epoch)
                    * (epochs - epoch))
                r_m = int(remaining // 60)
                r_s = int(remaining % 60)
                eta = (
                    f"  |  ETA "
                    f"{r_m}m {r_s:02d}s")
            self._log(
                f"  Epoch {epoch:>3d}/{epochs}  |  "
                f"loss {loss:.4f}  |  "
                f"{mins}m {secs:02d}s{eta}")
        trainer_obj.on_epoch_complete = on_epoch

        self._active_trainer = trainer_obj
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
            trainer_sys, stage, difficulty,
            student_sys=None):
        """Phase 3: Test student with TRAINER-generated questions.

        Retries empty test questions and empty judgments up to 2
        times each, matching the Phase 1 retry pattern.  Uses
        stage-specific test prompts so the teacher generates
        relevant questions (especially for COMMANDS stage).

        Args:
            student_sys: Optional lean persona prompt for the
                STUDENT model during inference (two-tier context).
        """
        import torch
        from enigma_engine.core.adaptive_trainer import (
            build_test_prompt)

        self._log(
            "\n--- Phase 3: TESTING STUDENT ---")
        student_engine = self._load_engine_for_path(
            student_path)

        test_scores = []
        num_tests = 10
        max_retries = 2
        for t in range(num_tests):
            if not self.training_active:
                raise KeyboardInterrupt("Stopped")

            # Generate test question with retries
            test_q = ""
            prompt = build_test_prompt(
                t + 1, stage, difficulty)
            for _attempt in range(1 + max_retries):
                test_q = teacher_engine.chat(
                    prompt,
                    system_prompt=trainer_sys,
                    max_gen=100,
                    temperature=0.9).strip()
                if test_q:
                    break
            if not test_q:
                self._log(
                    f"  Test {t + 1:>2d}  |  "
                    f"(no question generated, skipped)")
                continue

            s_answer = student_engine.chat(
                test_q,
                system_prompt=student_sys,
                max_gen=256,
                temperature=0.7).strip()
            judge_msg = (
                f'Question: "{test_q}"\n'
                f'Answer: "{s_answer}"\n\n'
                "Score 1-10:\n"
                "Reply: SCORE: <n> | <feedback>")

            # Judge with retries for empty responses
            judgment = ""
            for _attempt in range(1 + max_retries):
                judgment = teacher_engine.chat(
                    judge_msg,
                    system_prompt=trainer_sys,
                    max_gen=128,
                    temperature=0.3).strip()
                if judgment:
                    break

            score = self._parse_test_score(judgment)
            test_scores.append(score)
            q_short = test_q.replace("\n", " ")
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
        """Extract numeric score from TRAINER judgment text.

        Delegates to the robust ``parse_score()`` in
        ``adaptive_trainer`` which handles multiple LLM
        output formats (SCORE: N, N/10, bare number, etc.).
        """
        from enigma_engine.core.adaptive_trainer import (
            parse_score)
        return parse_score(judgment)

    def _adaptive_decide_action(
            self, plan, result, avg_score,
            attempt, stage):
        """Decide and execute adaptive action based on stage completion."""
        action = plan.decide_action(avg_score)
        self._log(f"Action  : {action.upper()}")

        if action == "retry":
            result.status = "retry"
            plan.record_result(result)
            self._log(
                f"Retrying {stage.upper()} at "
                f"{plan.current_difficulty} difficulty")
        elif action in ("advance", "complete"):
            result.status = "passed"
            plan.record_result(result)
            has_next = plan.advance_stage()
            if has_next:
                self._log(
                    f"Advancing to: "
                    f"{plan.current_stage.upper()}")
            else:
                self._log("All stages complete!")

    def _log_training_report(
            self, plan, all_losses, student_name):
        """Log a training report card with score and next steps."""
        # Gather scores from all stage results
        all_scores = []
        for r in plan.stage_results:
            avg = r.get("avg_score", 0)
            if avg > 0:
                all_scores.append(avg)

        overall = (sum(all_scores) / len(all_scores)
                   if all_scores else 0.0)
        has_valid_scores = bool(all_scores)
        best_loss = (min(all_losses)
                     if all_losses else float("inf"))
        worst_loss = (max(all_losses)
                      if all_losses else float("inf"))
        loss_improved = (
            all_losses[-1] < all_losses[0]
            if len(all_losses) >= 2 else False)

        # Grade
        if overall >= 8.0:
            grade = "A"
            verdict = "Excellent — model learned well"
        elif overall >= 7.0:
            grade = "B"
            verdict = "Good — solid foundation"
        elif overall >= 5.0:
            grade = "C"
            verdict = "Fair — some learning, needs more"
        elif overall >= 3.0:
            grade = "D"
            verdict = "Poor — struggling to learn"
        else:
            grade = "F"
            verdict = "Failed — not retaining material"

        self._log("\n" + "=" * 50)
        self._log("  TRAINING REPORT CARD")
        self._log("=" * 50)
        self._log(f"Student  : {student_name}")
        if has_valid_scores:
            self._log(f"Score    : {overall:.1f}/10  ({grade})")
            self._log(f"Verdict  : {verdict}")
        else:
            self._log("Score    : N/A (no valid test scores)")
            self._log(
                "Verdict  : Cannot assess — no tests "
                "completed successfully")
        self._log(f"Loss     : {best_loss:.4f} best"
                  f" / {worst_loss:.4f} worst")
        if len(all_losses) >= 2:
            self._log(
                f"Trend    : {'Improving' if loss_improved else 'Not improving'}")

        # Per-stage breakdown
        self._log("\nStage Breakdown:")
        for r in plan.stage_results:
            stage = r.get("stage", "?")
            avg = r.get("avg_score", 0)
            status = r.get("status", "?")
            bl = r.get("best_loss", 0)
            self._log(
                f"  {stage:<14s} {avg:.1f}/10  "
                f"loss={bl:.4f}  [{status}]")

        # Next step recommendations
        self._log("\n" + "-" * 50)
        self._log("  WHAT TO DO NEXT")
        self._log("-" * 50)

        if overall >= 8.0:
            self._log(
                "  [1] Train again with more epochs "
                "to refine further")
            self._log(
                "  [2] Try the next stage if you "
                "skipped any")
            self._log(
                "  [3] Test it in CORE — start chatting")
            self._log(
                "  [4] Create a checkpoint backup "
                "(MODELS page)")
        elif overall >= 5.0:
            self._log(
                "  [1] Run training again — it's "
                "learning, needs more reps")
            self._log(
                "  [2] Increase epochs (try 20-30) "
                "for deeper learning")
            self._log(
                "  [3] Add supplement data to give "
                "it more examples")
            self._log(
                "  [4] Use GENERATE DATA to create "
                "more training material")
        else:
            self._log(
                "  [1] Check that TRAINER model is "
                "capable (try it in CORE)")
            self._log(
                "  [2] Start with fewer, higher "
                "quality training examples")
            self._log(
                "  [3] Lower the learning rate "
                "(try 0.00001)")
            self._log(
                "  [4] Increase epochs significantly "
                "(30-50+)")
            self._log(
                "  [5] Try Basic mode with a curated "
                "data file instead")

        self._log("")

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
        from enigma_engine.core.safe_save import atomic_write_text
        atomic_write_text(
            curr_path,
            "\n".join(header_lines)
            + combined + "\n")
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
