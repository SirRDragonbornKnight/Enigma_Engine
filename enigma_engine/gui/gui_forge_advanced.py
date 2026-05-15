"""
Enigma Engine - Forge Training Modes (Advanced)
===================================================

Training mode implementations: Dialogue, Evolutionary.
Split from gui_forge.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ForgeAdvancedMixin:
    """Training mode implementations: Dialogue, Evolutionary.

    Expects the host class to have ForgeMixin setup attributes.
    """

    # ================================================================
    # NOTE: _start_guided_training was removed (S658).
    # AI-Guided mode in gui_forge.py covers the same use case
    # via _start_ai_guided_training + adaptive trainer.
    # ================================================================

    # ================================================================
    # Dialogue training (TRAINER ↔ STUDENT conversation)
    # ================================================================
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
        if getattr(self, "use_api_chat", False) is True:
            self._log("[!] API routing not yet implemented for Dialogue mode — running locally on this machine.\n")
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

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

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
            # Pre-bind so failure-path ``finally`` can surface the
            # rollback file regardless of when the exception fires.
            pre_dialogue_backup_path: str | None = None
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

                # Build system prompt for TRAINER
                trainer_sys = self._build_trainer_system_prompt(
                    student_params=s_params,
                    student_cfg=s_cfg,
                    task="training",
                    stage=stage,
                    training_brief=training_brief)
                self._log(f"Stage   : {stage.upper()}")
                if training_brief:
                    self._log(f"Brief   : {len(training_brief)} chars")

                # Build lean persona prompt for student
                student_sys = self._build_student_system_prompt(
                    training_brief=training_brief,
                    student_name=student_name)

                # ============================================
                # Conversation loop
                # ============================================
                self._log(
                    f"\n--- DIALOGUE: {num_rounds} "
                    f"ROUNDS ---\n")
                corrections = []
                scores = []
                conversation_history = []

                for r in range(num_rounds):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")

                    # Build context from recent
                    # conversation
                    context = ""
                    if conversation_history:
                        recent = (
                            conversation_history[-6:])
                        context = (
                            "Recent conversation:\n"
                            + "\n".join(recent)
                            + "\n\n")

                    # TRAINER asks a question
                    ask_msg = (
                        f"{context}"
                        f"Round {r + 1} of "
                        f"{num_rounds}.\n"
                        f"Ask the student a "
                        f"{stage}-level question. "
                        f"Make it build on previous "
                        f"conversation if possible. "
                        f"Write ONLY the question.")
                    question = teacher_engine.chat(
                        ask_msg,
                        system_prompt=trainer_sys,
                        max_gen=128,
                        temperature=0.8).strip()
                    if not question:
                        continue

                    # STUDENT answers
                    student_answer = (
                        student_engine.chat(
                            question,
                            system_prompt=student_sys,
                            max_gen=256,
                            temperature=0.7).strip())
                    if not student_answer:
                        student_answer = "(no response)"

                    # Track conversation history
                    conversation_history.append(
                        f"TRAINER: {question}")
                    conversation_history.append(
                        f"STUDENT: {student_answer}")

                    # TRAINER scores and corrects
                    judge_msg = (
                        f"You asked: \"{question}\"\n"
                        f"The student answered: "
                        f"\"{student_answer}\"\n\n"
                        f"If the answer deserves less "
                        f"than 8/10, you MUST provide "
                        f"a CORRECTED: version.\n\n"
                        f"1. Score the answer 1-10\n"
                        f"2. Write the CORRECTED ideal "
                        f"answer (how the student "
                        f"SHOULD have responded)\n\n"
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
                        if ln.upper().startswith(
                                "SCORE:"):
                            rest = ln.split(
                                ":", 1)[1].strip()
                            try:
                                score = int(
                                    rest.split()[0])
                                score = max(
                                    1, min(10, score))
                            except (
                                ValueError,
                                IndexError,
                            ):
                                pass
                        elif ln.upper().startswith(
                                "CORRECTED:"):
                            corrected = ln.split(
                                ":", 1)[1].strip()

                    # If no CORRECTED line, use full
                    # judgment minus the SCORE line
                    if not corrected:
                        corrected = "\n".join(
                            ln for ln in
                            judgment.splitlines()
                            if not ln.strip().upper()
                            .startswith("SCORE:")
                        ).strip()

                    scores.append(score)

                    # High-scoring: reinforce what
                    # the student already got right
                    if score >= 8 and student_answer:
                        corrections.append(
                            self._format_training_pair(
                                stage, question,
                                student_answer))
                    elif corrected:
                        # Low-scoring: use correction
                        corrections.append(
                            self._format_training_pair(
                                stage, question,
                                corrected))

                    # Log the turn
                    q_s = question.replace("\n", " ")
                    a_s = student_answer.replace(
                        "\n", " ")
                    c_s = corrected.replace("\n", " ")
                    self._log(
                        f"  Round {r + 1:>3d}  |  "
                        f"Score: {score}/10")
                    pct = int(
                        (r + 1) / num_rounds * 50)
                    self._update_forge_progress(
                        pct,
                        f"Round {r + 1}/"
                        f"{num_rounds}")
                    self._log(
                        f"    TRAINER: {q_s}")
                    self._log(
                        f"    STUDENT: {a_s}")
                    if corrected and score < 8:
                        self._log(
                            f"    CORRECTED: {c_s}")

                # Conversation summary
                avg = 0.0
                if scores:
                    avg = (sum(scores)
                           / len(scores))
                    self._log(
                        f"\n  Average score: "
                        f"{avg:.1f}/10")
                    improvement = ""
                    if len(scores) >= 4:
                        first_half = (
                            sum(scores[
                                :len(scores) // 2])
                            / max(
                                1, len(scores) // 2))
                        second_half = (
                            sum(scores[
                                len(scores) // 2:])
                            / max(
                                1, len(scores)
                                - len(scores) // 2))
                        diff = (
                            second_half - first_half)
                        if diff > 0.5:
                            improvement = (
                                f"  Improvement: "
                                f"+{diff:.1f} "
                                f"(first half "
                                f"{first_half:.1f} "
                                f"\u2192 second half "
                                f"{second_half:.1f})")
                        elif diff < -0.5:
                            improvement = (
                                f"  Decline: "
                                f"{diff:.1f} "
                                f"(first half "
                                f"{first_half:.1f} "
                                f"\u2192 second half "
                                f"{second_half:.1f})")
                    if improvement:
                        self._log(improvement)

                transcript_name = ""
                if not corrections:
                    self._log(
                        "[!] No corrections generated"
                        " — nothing to train on.")
                    return

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
                transcript_path = (
                    DATA_DIR / transcript_name)
                transcript_lines = [
                    "# Dialogue Training Transcript",
                    f"# Trainer: {trainer_name}",
                    f"# Student: {student_name}",
                    f"# Stage: {stage}",
                    f"# Rounds: {len(scores)}",
                    (f"# Average: {avg:.1f}/10"
                     if scores
                     else "# Average: N/A"),
                    f"# Date: "
                    f"{datetime.now().isoformat()}",
                    ""]
                for i, entry in enumerate(
                        conversation_history):
                    transcript_lines.append(entry)
                    if entry.startswith("STUDENT:"):
                        idx = i // 2
                        if idx < len(scores):
                            transcript_lines.append(
                                f"  [Score: "
                                f"{scores[idx]}/10]")
                transcript_lines.append("")
                transcript_lines.append(
                    "# --- Training Data Used ---")
                transcript_lines.extend(corrections)
                from enigma_engine.core.safe_save import (
                    atomic_write_text)
                atomic_write_text(
                    transcript_path,
                    "\n".join(transcript_lines)
                    + "\n")
                self._log(
                    f"Transcript: "
                    f"{transcript_path.name}")
                self.after(
                    0, self._refresh_data_files)

                # Free inference models before training
                del teacher_engine, student_engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ============================================
                # Train STUDENT on corrections
                # ============================================
                from enigma_engine.training.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.tokenizer import (
                    get_tokenizer as get_tok)
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)

                self._log(
                    "\n--- TRAINING ON CORRECTIONS "
                    "---\n")
                combined = "\n\n".join(corrections)

                # Load STUDENT for gradient training
                from enigma_engine.core.model_registry import (
                    safe_load_weights as slw)
                s_ckpt2 = slw(
                    student_path,
                    map_location=device)
                s_cfg_d2 = (
                    s_ckpt2.get("model_config")
                    or s_ckpt2.get("config", {}))
                if (isinstance(s_cfg_d2, dict)
                        and "epochs" in s_cfg_d2):
                    s_cfg_d2 = s_ckpt2.get(
                        "model_config", {})
                s_cfg2 = ForgeConfig(**s_cfg_d2)
                student_mdl = Enigma(config=s_cfg2)
                student_mdl.load_state_dict(
                    s_ckpt2["model_state_dict"])
                student_mdl = student_mdl.to(device)

                tokenizer2 = get_tok("auto")

                forge_params = (
                    self._read_forge_train_params())
                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=(
                        forge_params["batch_size"]),
                    learning_rate=lr,
                    max_grad_accumulation=(
                        forge_params[
                            "max_grad_accumulation"
                        ]),
                    use_gradient_checkpointing=(
                        forge_params[
                            "use_gradient_checkpointing"
                        ]),
                    use_sequence_packing=True,
                    ce_chunk_size=forge_params["ce_chunk_size"],
                    use_compile=True,
                    rolling_best_k=(
                        forge_params["rolling_best_k"]),
                    general_mix_ratio=(
                        forge_params[
                            "general_mix_ratio"]),
                    general_data=(
                        forge_params["general_data"]),
                    val_split=(
                        forge_params["val_split"]),
                    min_lr_ratio=forge_params["min_lr_ratio"],
                    save_every=max(1, epochs // 5),
                    checkpoint_dir=str(
                        MODELS_DIR / "checkpoints"),
                    use_amp=(
                        torch.cuda.is_available()),
                    run_evaluation=True)

                # Pass 156z9ar: pre-dialogue auto-checkpoint.
                # Dialogue training overwrites ``student_path``
                # in place at the end of the run.  The backup
                # is the only rollback path if the corrections
                # the trainer fed in drift the student.
                pre_dialogue_backup_path = (
                    self._pre_training_backup(
                        student_path,
                        suffix="pre_dialogue"))

                trainer_obj = Trainer(
                    student_mdl, tokenizer2,
                    train_config)

                import time as _time_dl
                _dl_train_start = [
                    _time_dl.monotonic()]

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt(
                            "Stopped")
                    losses.append(loss)
                    pct = 50 + int(
                        epoch / epochs * 50)
                    self._update_forge_progress(
                        pct,
                        f"Epoch {epoch}/{epochs}")
                    elapsed = (
                        _time_dl.monotonic()
                        - _dl_train_start[0])
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
                        f"  Epoch {epoch:>3d}/"
                        f"{epochs}  |  "
                        f"loss {loss:.4f}  |  "
                        f"{mins}m {secs:02d}s{eta}")
                trainer_obj.on_epoch_complete = (
                    on_epoch)

                self._active_trainer = trainer_obj
                state = trainer_obj.train(combined)

                # Save trained STUDENT
                out = Path(student_path)
                atomic_torch_save({
                    "model_state_dict": (
                        student_mdl.state_dict()),
                    "config": (
                        self._model_config_dict(
                            student_mdl)),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": (
                            state.best_loss),
                    },
                }, out)
                self._log(
                    f"\nBest loss : "
                    f"{state.best_loss:.4f}")
                self._log(f"Saved to  : {out}")

                self._log(
                    "\n--- DIALOGUE TRAINING "
                    "COMPLETE ---")
                self._log(
                    "Run again to continue building "
                    "on what the student learned.")
                if transcript_name:
                    self._log(
                        f"Review transcript: "
                        f"{transcript_name}")
                best_loss = (
                    min(losses) if losses else 0.0)
                self._update_forge_progress(
                    100, "Complete")
                self._save_training_run(
                    "Dialogue", student_name,
                    epochs, best_loss)
                self._notify_training_complete()
                if losses:
                    self._display_loss_curve(losses)
                self.after(
                    0, lambda sp=s_params:
                    self._update_forge_param_count(
                        sp))
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log(
                    "\n--- DIALOGUE TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                self._log(
                    f"\n[!] Dialogue training failed: {exc}")
                self._log(tb)
            finally:
                # Surface the rollback file on EVERY exit path
                # (success, user-stop, crash) so the user always
                # knows where the pre-training snapshot lives.
                if pre_dialogue_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_dialogue_backup_path).name}")
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(
                    0, lambda: self.dialogue_train_btn.configure(
                        state="normal", text="TRAIN"))
                self.after(
                    0, lambda: self.stop_train_btn.configure(
                        state="disabled", text="STOP"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_dialogue, daemon=True).start()

    # ================================================================
    # Evolutionary training
    # ================================================================

    def _start_evolutionary_training(self):
        """Evolutionary training: generate multiple answers per task,
        keep the best, train on winners. Repeats to improve.

        Uses evolutionary_training strategy:
        1. Load tasks from data file (one per line)
        2. For each task, generate N candidate answers
        3. Score all candidates, keep top-K
        4. Train STUDENT on the winners
        5. Repeat for multiple generations

        Needs: STUDENT model + task file.
        """
        if getattr(self, "use_api_chat", False) is True:
            self._log("[!] API routing not yet implemented for Evolutionary mode — running locally on this machine.\n")
        if self.training_active:
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model.")
            return

        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log(
                "[!] No data file selected.\n"
                "    Pick a task file (one task per line).")
            return

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        student_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(
            "\u2692 EVOLUTIONARY TRAINING...")
        self._log("--- EVOLUTIONARY TRAINING INITIATED ---")
        self._log(f"Student : {student_name}")
        self._log(f"Data    : {Path(data_path).name}")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _evo():
            losses = []
            # Pre-bind so failure-path ``finally`` can surface the
            # rollback file regardless of when the exception fires.
            pre_evolutionary_backup_path: str | None = None
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.tokenizer import (
                    get_tokenizer)
                from enigma_engine.core.model_presets import (
                    ForgeConfig)
                from enigma_engine.training.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)

                device = ("cuda"
                          if torch.cuda.is_available()
                          else "cpu")

                # Snapshot the student's current weights so a
                # failure mid-training has a rollback target.
                pre_evolutionary_backup_path = self._pre_training_backup(
                    student_path, suffix="pre_evolutionary")

                # Load tasks
                tasks = Path(data_path).read_text(
                    encoding="utf-8").strip().splitlines()
                tasks = [t.strip() for t in tasks if t.strip()]
                if not tasks:
                    self._log("[!] No tasks found in file.")
                    return
                self._log(f"Tasks   : {len(tasks)}")

                # Load student
                s_ckpt = safe_load_weights(
                    student_path, map_location="cpu")
                s_cfg_dict = (
                    s_ckpt.get("model_config")
                    or s_ckpt.get("config", {}))
                if (isinstance(s_cfg_dict, dict)
                        and "epochs" in s_cfg_dict):
                    s_cfg_dict = s_ckpt.get(
                        "model_config", {})
                s_cfg = ForgeConfig(**s_cfg_dict)
                student = Enigma(config=s_cfg)
                student.load_state_dict(
                    get_state_dict(s_ckpt))
                s_params = sum(
                    p.numel()
                    for p in student.parameters())
                self._log(f"Params  : {s_params:,}")

                _bpe_path = MODELS_DIR / "tokenizer.json"
                if _bpe_path.exists():
                    try:
                        from enigma_engine.core.bpe_tokenizer import BPETokenizer
                        tokenizer = BPETokenizer(_bpe_path)
                    except Exception:
                        tokenizer = get_tokenizer("auto")
                else:
                    tokenizer = get_tokenizer("auto")

                # Evolutionary loop: generate candidates,
                # keep best, train
                num_generations = 3
                candidates_per_task = 4
                keep_top_k = 2
                winners = []

                for gen in range(num_generations):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    self._log(
                        f"\n--- Generation {gen + 1}/"
                        f"{num_generations} ---")
                    gen_winners = []

                    student = student.to(device)
                    student.eval()

                    for ti, task in enumerate(tasks):
                        if not self.training_active:
                            raise KeyboardInterrupt(
                                "Stopped")
                        # Generate candidates
                        candidates = []
                        for _ in range(
                                candidates_per_task):
                            prompt_ids = (
                                tokenizer.encode(task))
                            input_t = torch.tensor(
                                [prompt_ids],
                                device=device)
                            with torch.no_grad():
                                out = student.generate(
                                    input_t,
                                    max_new_tokens=128,
                                    temperature=0.9,
                                    top_k=50)
                            text = tokenizer.decode(
                                out[0].tolist())
                            candidates.append(text)

                        # Score candidates by length
                        # and diversity (simple heuristic)
                        scored = []
                        for c in candidates:
                            resp = c[len(task):]
                            score = min(
                                len(resp.split()), 50)
                            unique_words = len(
                                set(resp.lower().split()))
                            score += unique_words
                            scored.append((score, c))
                        scored.sort(
                            key=lambda x: x[0],
                            reverse=True)
                        best = [
                            s[1]
                            for s in scored[:keep_top_k]]
                        gen_winners.extend(
                            f"User: {task}\n"
                            f"Assistant: {b}"
                            for b in best)

                        pct = int(
                            (gen * len(tasks) + ti + 1)
                            / (num_generations
                               * len(tasks)) * 50)
                        self._update_forge_progress(
                            pct,
                            f"Gen {gen + 1} "
                            f"Task {ti + 1}/"
                            f"{len(tasks)}")

                    winners = gen_winners
                    self._log(
                        f"  Winners: {len(winners)}")

                    # Train on winners
                    student = student.to(device)
                    student.train()
                    combined = "\n\n".join(winners)

                    forge_params = (
                        self._read_forge_train_params())
                    train_config = TrainingConfig(
                        epochs=max(1, epochs // num_generations),
                        batch_size=(
                            forge_params["batch_size"]),
                        learning_rate=lr,
                        max_grad_accumulation=(
                            forge_params[
                                "max_grad_accumulation"
                            ]),
                        use_gradient_checkpointing=(
                            forge_params[
                                "use_gradient_checkpointing"
                            ]),
                        use_sequence_packing=True,
                        ce_chunk_size=forge_params["ce_chunk_size"],
                        use_compile=True,
                        rolling_best_k=(
                            forge_params[
                                "rolling_best_k"]),
                        general_mix_ratio=(
                            forge_params[
                                "general_mix_ratio"]),
                        general_data=(
                            forge_params[
                                "general_data"]),
                        val_split=(
                            forge_params["val_split"]),
                        min_lr_ratio=forge_params["min_lr_ratio"],
                        save_every=999,
                        checkpoint_dir=str(
                            MODELS_DIR / "checkpoints"),
                        use_amp=(
                            torch.cuda.is_available()),
                        run_evaluation=True)

                    trainer_obj = Trainer(
                        student, tokenizer, train_config)

                    import time as _time_ev
                    _ev_gen_start = [
                        _time_ev.monotonic()]
                    _ev_gen_epochs = max(
                        1, epochs // num_generations)

                    def on_epoch(epoch, loss,
                                 _gen=gen,
                                 _start=_ev_gen_start,
                                 _ep=_ev_gen_epochs):
                        if not self.training_active:
                            raise KeyboardInterrupt(
                                "Stopped")
                        losses.append(loss)
                        elapsed = (
                            _time_ev.monotonic()
                            - _start[0])
                        mins = int(elapsed // 60)
                        secs = int(elapsed % 60)
                        eta = ""
                        if epoch > 0:
                            remaining = (
                                (elapsed / epoch)
                                * (_ep
                                   - epoch))
                            r_m = int(
                                remaining // 60)
                            r_s = int(
                                remaining % 60)
                            eta = (
                                f" | ETA "
                                f"{r_m}m {r_s:02d}s")
                        self._log(
                            f"  Gen {_gen + 1} "
                            f"Epoch {epoch} | "
                            f"loss {loss:.4f} | "
                            f"{mins}m {secs:02d}s"
                            f"{eta}")
                    trainer_obj.on_epoch_complete = (
                        on_epoch)

                    self._active_trainer = trainer_obj
                    trainer_obj.train(combined)

                # Save final model
                out = Path(student_path)
                atomic_torch_save({
                    "model_state_dict": (
                        student.state_dict()),
                    "config": (
                        self._model_config_dict(
                            student)),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": (
                            min(losses)
                            if losses else 0.0),
                    },
                }, out)
                best_loss = (
                    min(losses) if losses else 0.0)
                self._log(
                    f"\nBest loss : {best_loss:.4f}")
                self._log(f"Saved to  : {out}")
                self._log(
                    "\n--- EVOLUTIONARY TRAINING "
                    "COMPLETE ---")
                self._update_forge_progress(
                    100, "Complete")
                self._save_training_run(
                    "Evolutionary", student_name,
                    epochs, best_loss)
                if losses:
                    self._display_loss_curve(losses)
                self.after(
                    0, lambda sp=s_params:
                    self._update_forge_param_count(
                        sp))
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log(
                    "\n--- EVOLUTIONARY TRAINING "
                    "STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                self._log(
                    f"\n[!] Evolutionary training "
                    f"failed: {exc}")
                self._log(tb)
            finally:
                # Surface the rollback file on EVERY exit path
                # (success, user-stop, OOM, crash) so the user always
                # knows where the pre-training snapshot lives.
                if pre_evolutionary_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_evolutionary_backup_path).name}")
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(
                    0, lambda: self.solo_train_btn.configure(
                        state="normal", text="TRAIN"))
                self.after(
                    0, lambda: self.stop_train_btn.configure(
                        state="disabled", text="STOP"))
                self.after(0, lambda:
                    self.status_bar.set_left(
                        "\u26a1 READY"))

        threading.Thread(
            target=_evo, daemon=True).start()
