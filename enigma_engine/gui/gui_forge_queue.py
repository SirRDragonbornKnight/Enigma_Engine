"""
Enigma Engine - Forge Queue & Dataset Tools
===============================================

GUI callbacks for training queue (TS-B), overnight plan (TS-C),
and curated dataset review (DA-C).

Split from gui_forge_tools.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# Persistent paths
_QUEUE_PATH = DATA_DIR / "training_queue.json"
_DATASET_PATH = DATA_DIR / "curated_dataset.jsonl"

# Maps GUI display-mode names to dispatcher mode keys.
# Modes not listed here AND not in _QUEUE_UNSUPPORTED_MODES raise ValueError.
_QUEUE_MODE_MAP: dict[str, str] = {
    "Basic": "sft",
    "Pre-Train": "sft",
    "DPO": "dpo",
    "APO": "dpo",
    "Image": "vision",
    "GRPO": "grpo",
    "ReMax": "remax",
    "SimPO": "simpo",
    "ORPO": "orpo",
    "RLHF": "rlhf",
    "Self-Play": "self_play",
    "LoRA": "lora",
}

# GUI modes that use their own training paths and cannot be queued.
_QUEUE_UNSUPPORTED_MODES: frozenset[str] = frozenset({
    "Distill",
    "AI-Guided",
    "Dialogue",
})

# Queue display names that route through DPO must preserve algorithm choice.
_QUEUE_DPO_LOSS_TYPE_MAP: dict[str, str] = {
    "DPO": "dpo",
    "APO": "apo_zero",
}

# Modes that require preference-pair JSONL data (prompt/chosen/rejected).
_PREFERENCE_MODES = {"dpo", "simpo", "orpo", "reward_model"}
# Modes that require plain prompt lists.
_PROMPT_LIST_MODES = {"grpo", "remax", "rlhf", "self_play"}
# Experimental dispatcher modes that need allow_experimental=True.
_EXPERIMENTAL_MODES = {
    "simpo", "kto", "orpo", "rest", "reward_model",
    "rlhf", "self_play", "remax", "adaptive",
}


class ForgeQueueMixin:
    """Queue, overnight plan, and curated dataset GUI callbacks.

    Expects the host class to have:
    - _log, _update_forge_progress, _reset_forge_progress
    - status_bar, route_assignments
    - training_mode_var, training_stage_var, train_data_var
    - epochs_entry, lr_entry, forge_batch_entry
    - _read_forge_train_params, _start_training_by_mode
    """

    # ================================================================
    # Lazy singletons — avoids importing core at module level
    # ================================================================

    def _get_training_queue(self):
        """Return the shared TrainingQueue singleton (lazy init)."""
        queue = getattr(self, "_training_queue", None)
        if queue is None:
            from enigma_engine.training.training_queue import TrainingQueue
            queue = TrainingQueue(save_path=_QUEUE_PATH)
            queue.load_state()
            queue.on_progress = self._on_queue_progress
            queue.on_job_complete = self._on_queue_job_complete
            queue.on_job_failed = self._on_queue_job_failed
            queue.on_queue_complete = self._on_queue_complete
            queue.executor = self._execute_queue_job
            self._training_queue = queue
        return queue

    def _get_curated_dataset(self):
        """Return the shared CuratedDataset singleton (lazy init)."""
        ds = getattr(self, "_curated_dataset", None)
        if ds is None:
            from enigma_engine.core.curated_dataset import CuratedDataset
            ds = CuratedDataset(_DATASET_PATH)
            # Only set if another thread didn't beat us
            if getattr(self, "_curated_dataset", None) is None:
                self._curated_dataset = ds
            else:
                ds = self._curated_dataset
        return ds

    # ================================================================
    # TS-B: Training Queue
    # ================================================================

    def _add_to_training_queue(self):
        """Add the current FORGE settings as a job to the queue."""
        from enigma_engine.training.training_queue import TrainingJob

        student_path = self.route_assignments.get("student", "")
        if not student_path or not Path(student_path).exists():
            self._log("[!] No STUDENT model assigned — "
                      "cannot add to queue.")
            return

        # Read current UI settings
        mode_var = getattr(self, "training_mode_var", None)
        mode = mode_var.get() if mode_var else "Self Study"

        stage_var = getattr(self, "training_stage_var", None)
        stage = stage_var.get() if stage_var else "basics"

        data_path = self.train_data_var.get()

        try:
            epochs = int(self.epochs_entry.get())
            if epochs < 1:
                epochs = 10
        except (ValueError, AttributeError):
            epochs = 10

        try:
            lr = float(self.lr_entry.get())
        except (ValueError, AttributeError):
            lr = 1e-4

        params = self._read_forge_train_params()

        job = TrainingJob(
            mode=mode,
            model_path=str(student_path),
            data_path=data_path if data_path else "",
            stage=stage,
            epochs=epochs,
            learning_rate=lr,
            batch_size=params.get("batch_size", 4),
            extra_config={
                "rolling_best_k": params.get("rolling_best_k", 0),
                "use_gradient_checkpointing": params.get(
                    "use_gradient_checkpointing", False),
                "max_grad_accumulation": params.get(
                    "max_grad_accumulation", 1),
                "val_split": params.get("val_split", 0.1),
            },
        )

        queue = self._get_training_queue()
        queue.add_job(job)

        self._log(f"[Queue] Added job: {mode} "
                  f"({epochs} epochs, stage={stage})")
        self._log(f"  Queue now has {queue.pending_count} "
                  f"pending job(s)")
        self.status_bar.set_left(
            f"Job added to queue ({queue.pending_count} pending)")

    def _show_training_queue(self):
        """Display the current queue state in the forge log."""
        queue = self._get_training_queue()
        jobs = queue.jobs

        if not jobs:
            self._log("[Queue] Queue is empty. "
                      "Use ADD TO QUEUE to enqueue jobs.")
            return

        self._log("\n--- TRAINING QUEUE ---")
        self._log(queue.summary())
        self._log("")

    def _run_training_queue(self):
        """Start (or resume) the training queue."""
        queue = self._get_training_queue()

        if queue.is_running:
            # Already running — offer pause
            queue.pause()
            self._log("[Queue] Paused. Click RUN to resume.")
            # Propagate cancel to the daemon if a job is running via API.
            api_client = getattr(self, "_active_queue_api_client", None)
            if api_client is not None:
                try:
                    api_client.cancel_training()
                    self._log("[Queue] Sent cancel signal to daemon.")
                except Exception:
                    pass
            self.after(0, lambda: getattr(
                self, "_forge_run_queue_btn", None) and
                self._forge_run_queue_btn.configure(text="RUN"))
            return

        if queue.is_paused:
            queue.resume()
            self._log("[Queue] Resumed.")
            self.after(0, lambda: getattr(
                self, "_forge_run_queue_btn", None) and
                self._forge_run_queue_btn.configure(text="PAUSE"))
            return

        if queue.pending_count == 0:
            self._log("[Queue] No pending jobs to run.")
            return

        self._log("[Queue] Starting queue execution...")
        self.after(0, lambda: getattr(
            self, "_forge_run_queue_btn", None) and
            self._forge_run_queue_btn.configure(text="PAUSE"))
        queue.start()

    def _execute_queue_job(self, job):
        """Execute a single training job (called by the queue thread).

        Routes every job through ``build_dispatch_context`` +
        ``run_training`` so the queue honours all training modes
        supported by the dispatcher.  Runs synchronously — the queue
        thread blocks until this returns.

        Returns the best loss achieved.
        """
        # Resolve dispatcher mode key from display name.
        dispatch_mode = _QUEUE_MODE_MAP.get(job.mode)
        if dispatch_mode is None:
            if job.mode in _QUEUE_UNSUPPORTED_MODES:
                raise ValueError(
                    f"Mode '{job.mode}' cannot be run through the training "
                    "queue. Use the FORGE training page directly."
                )
            raise ValueError(
                f"Unknown training mode for queue: '{job.mode}'. "
                f"Known modes: {sorted(_QUEUE_MODE_MAP)}"
            )

        self._log(f"\n[Queue] Running job #{job.job_id}: "
                  f"{job.mode} ({dispatch_mode}) "
                  f"({job.epochs} epochs)")

        student_path = job.model_path
        if not student_path or not Path(student_path).exists():
            raise FileNotFoundError(
                f"Student model not found: {student_path}")

        # Load training data.
        if not job.data_path:
            raise ValueError("No training data path configured for this job")
        if not Path(job.data_path).exists():
            raise FileNotFoundError(
                f"Training data file not found: {job.data_path}")
        data_text = Path(job.data_path).read_text(encoding="utf-8")
        if not data_text.strip():
            raise ValueError(f"Training data file is empty: {job.data_path}")

        # API-mode queue execution: load model on daemon, submit one job,
        # then poll daemon status until completion.
        get_client = getattr(self, "_get_api_chat_client", None)
        client = None
        if bool(getattr(self, "use_api_chat", False)) and callable(get_client):
            client = get_client()

        if client is not None:
            import json
            import time

            self._log("  API mode enabled — loading model on daemon...")
            client.load_model(student_path)

            # Track active client so _run_training_queue can cancel the
            # daemon when the user pauses/stops the queue mid-job.
            self._active_queue_api_client = client

            payload: dict = {
                "mode": dispatch_mode,
                "training": {
                    "epochs": job.epochs,
                    "learning_rate": job.learning_rate,
                    "batch_size": job.batch_size,
                    "gradient_clip": job.extra_config.get("gradient_clip", 1.0),
                    "use_amp": False,
                    "rolling_best_k": job.extra_config.get("rolling_best_k", 0),
                    "use_gradient_checkpointing": job.extra_config.get(
                        "use_gradient_checkpointing", False),
                    "max_grad_accumulation": job.extra_config.get(
                        "max_grad_accumulation", 1),
                    "val_split": job.extra_config.get("val_split", 0.1),
                    "checkpoint_dir": str(MODELS_DIR / "checkpoints"),
                },
            }
            if dispatch_mode in _EXPERIMENTAL_MODES:
                payload["allow_experimental"] = True
            if dispatch_mode == "dpo":
                payload["dpo"] = {
                    "loss_type": _QUEUE_DPO_LOSS_TYPE_MAP.get(job.mode, "dpo"),
                }

            if dispatch_mode in _PREFERENCE_MODES:
                pref_rows = []
                for line in data_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    pref_rows.append(json.loads(stripped))
                payload["data"] = pref_rows
            elif dispatch_mode in _PROMPT_LIST_MODES:
                payload["data"] = [
                    ln.strip() for ln in data_text.splitlines() if ln.strip()]
            else:
                payload["data"] = data_text

            self._log(f"  API queue submit mode={dispatch_mode!r}...")
            client.train(payload)

            best_loss = float("inf")
            while True:
                status = client.training_status()
                pct = int(status.get("progress", 0) or 0)
                msg = str(status.get("message", ""))
                best = status.get("best_loss")
                if best is not None:
                    try:
                        best_loss = float(best)
                    except (TypeError, ValueError):
                        pass

                job.progress = pct
                job.message = msg
                q = self._get_training_queue()
                if q.on_progress:
                    q.on_progress(job, pct, msg)

                if not bool(status.get("active", False)):
                    abort_reason = str(status.get("abort_reason", "") or "")
                    if abort_reason == "cancel_requested":
                        # User-initiated stop — clean exit, not an error.
                        self._log("  Cancelled by user.")
                    elif abort_reason:
                        self._active_queue_api_client = None
                        raise RuntimeError(
                            f"API queue job aborted: {abort_reason}")
                    break
                time.sleep(1.0)

            self._active_queue_api_client = None
            self._log(f"  Completed (loss={best_loss:.4f})")
            return best_loss

        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_registry import (
            get_state_dict, safe_load_weights)
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.training.dispatch import (
            build_dispatch_context, run_training)

        # Load model + tokenizer.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = safe_load_weights(student_path, map_location=device)
        cfg_dict = (checkpoint.get("model_config")
                    or checkpoint.get("config", {}))
        if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
            cfg_dict = checkpoint.get("model_config", {})
        model_cfg = ForgeConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in ForgeConfig.__dataclass_fields__})
        model = Enigma(config=model_cfg)
        state_dict = get_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(device)

        _bpe_path = MODELS_DIR / "tokenizer.json"
        if _bpe_path.exists():
            try:
                from enigma_engine.core.bpe_tokenizer import BPETokenizer
                tokenizer = BPETokenizer(_bpe_path)
            except Exception:
                tokenizer = get_tokenizer("auto")
        else:
            tokenizer = get_tokenizer("auto")

        # Progress callback.
        def on_epoch(epoch, loss):
            pct = int((epoch / max(job.epochs, 1)) * 100)
            job.progress = pct
            job.message = (
                f"Epoch {epoch}/{job.epochs} "
                f"loss={loss:.4f}")
            q = self._get_training_queue()
            if q.on_progress:
                q.on_progress(job, pct, job.message)

        ctx = build_dispatch_context(
            model=model,
            tokenizer=tokenizer,
            on_epoch_complete=on_epoch,
        )

        # Build dispatcher payload.
        payload: dict = {
            "mode": dispatch_mode,
            "training": {
                "epochs": job.epochs,
                "learning_rate": job.learning_rate,
                "batch_size": job.batch_size,
                "gradient_clip": job.extra_config.get("gradient_clip", 1.0),
                "use_amp": torch.cuda.is_available(),
                "rolling_best_k": job.extra_config.get("rolling_best_k", 0),
                "use_gradient_checkpointing": job.extra_config.get(
                    "use_gradient_checkpointing", False),
                "max_grad_accumulation": job.extra_config.get(
                    "max_grad_accumulation", 1),
                "val_split": job.extra_config.get("val_split", 0.1),
                "checkpoint_dir": str(MODELS_DIR / "checkpoints"),
            },
        }
        if dispatch_mode in _EXPERIMENTAL_MODES:
            payload["allow_experimental"] = True

        if dispatch_mode == "dpo":
            payload["dpo"] = {
                "loss_type": _QUEUE_DPO_LOSS_TYPE_MAP.get(job.mode, "dpo"),
            }

        # Attach data under the key the dispatcher expects.
        if dispatch_mode in _PREFERENCE_MODES:
            import json

            pref_rows = []
            for line in data_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                pref_rows.append(json.loads(stripped))
            payload["data"] = pref_rows
        elif dispatch_mode in _PROMPT_LIST_MODES:
            payload["data"] = [ln.strip() for ln in data_text.splitlines() if ln.strip()]
        else:
            payload["data"] = data_text

        self._log(f"  Routing to dispatcher mode={dispatch_mode!r}...")
        result = run_training(payload, ctx)

        # Persist trained weights back to the original student checkpoint.
        from enigma_engine.core.safe_save import atomic_torch_save
        save_data = {
            "model_state_dict": model.state_dict(),
            "model_config": cfg_dict,
        }
        atomic_torch_save(save_data, student_path)
        self._log(f"  Saved trained model → {Path(student_path).name}")

        # Extract best_loss — result shape varies by mode.
        if hasattr(result, "best_loss"):
            best_loss = result.best_loss
        elif isinstance(result, dict):
            best_loss = (
                result.get("best_loss")
                or result.get("final_loss", float("inf"))
            )
        else:
            raise TypeError(
                f"run_training returned unexpected result type: {type(result).__name__}")
        self._log(f"  Completed (loss={best_loss:.4f})")
        return best_loss

    # Queue callbacks (called from background thread)

    def _on_queue_progress(self, job, pct, msg):
        """Update forge progress bar from queue."""
        self._update_forge_progress(pct, msg)

    def _on_queue_job_complete(self, job):
        """Handle a queue job completing."""
        self._log(f"[Queue] Job #{job.job_id} ({job.mode}) "
                  f"completed — loss={job.best_loss:.4f}")
        self._reset_forge_progress()

    def _on_queue_job_failed(self, job, error):
        """Handle a queue job failing."""
        self._log(f"[Queue] Job #{job.job_id} ({job.mode}) "
                  f"FAILED: {error}")
        self._reset_forge_progress()

    def _on_queue_complete(self):
        """Handle the queue finishing all jobs."""
        queue = self._get_training_queue()
        self._log("\n--- QUEUE COMPLETE ---")
        self._log(queue.summary())
        self.after(0, lambda: self.status_bar.set_left(
            "Training queue finished"))
        self.after(0, lambda: getattr(
            self, "_forge_run_queue_btn", None) and
            self._forge_run_queue_btn.configure(text="RUN"))
        self.after(0, self._refresh_models)

    # ================================================================
    # TS-C: Overnight Plan (save/load/resume)
    # ================================================================

    def _save_overnight_plan(self):
        """Save the current queue contents as an overnight plan."""
        from enigma_engine.training.training_queue import OvernightPlan

        queue = self._get_training_queue()
        jobs = queue.jobs

        if not jobs:
            self._log("[Plan] Queue is empty — nothing to save.\n"
                      "  Add jobs with ADD TO QUEUE first.")
            return

        # Build plan from queue jobs
        plan = OvernightPlan(
            name="Overnight Training",
            auto_checkpoint=True,
            auto_evaluate=False,
        )
        for job in jobs:
            if job.status in ("pending", "running"):
                plan.add_job_config(
                    mode=job.mode,
                    model_path=job.model_path,
                    data_path=job.data_path,
                    stage=job.stage,
                    epochs=job.epochs,
                    learning_rate=job.learning_rate,
                    batch_size=job.batch_size,
                    **job.extra_config,
                )

        if not plan.jobs:
            self._log("[Plan] No pending jobs to save.")
            return

        # Save via file dialog
        plan_dir = DATA_DIR
        plan_dir.mkdir(parents=True, exist_ok=True)
        dest = filedialog.asksaveasfilename(
            title="Save Overnight Plan",
            initialdir=str(plan_dir),
            defaultextension=".json",
            filetypes=[
                ("JSON plan files", "*.json"),
                ("All files", "*.*")],
            initialfile="overnight_plan.json")
        if not dest:
            return

        plan.save(dest)
        self._log(f"[Plan] Saved: {Path(dest).name}")
        self._log(f"  {plan.total_jobs} job(s) in plan")
        self.status_bar.set_left(
            f"Overnight plan saved ({plan.total_jobs} jobs)")

    def _load_overnight_plan(self):
        """Load an overnight plan and populate the queue."""
        from enigma_engine.training.training_queue import OvernightPlan

        plan_dir = DATA_DIR
        src = filedialog.askopenfilename(
            title="Load Overnight Plan",
            initialdir=str(plan_dir),
            filetypes=[
                ("JSON plan files", "*.json"),
                ("All files", "*.*")])
        if not src:
            return

        try:
            plan = OvernightPlan.load(src)
        except Exception as exc:
            self._log(f"[Plan] Failed to load: {exc}")
            return

        if plan.is_complete:
            self._log("[Plan] This plan is already complete.")
            self._log(plan.summary())
            return

        # Convert remaining plan jobs → queue jobs and add
        queue = self._get_training_queue()
        new_jobs = plan.to_queue_jobs()
        for job in new_jobs:
            queue.add_job(job)

        self._log(f"\n[Plan] Loaded: {plan.name}")
        self._log(f"  Added {len(new_jobs)} job(s) to queue")
        self._log(f"  Queue now has {queue.pending_count} "
                  f"pending job(s)")
        self._log(plan.summary())
        self.status_bar.set_left(
            f"Loaded plan: {len(new_jobs)} jobs added to queue")

    # ================================================================
    # DA-C: Curated Dataset Review
    # ================================================================

    def _review_curated_dataset(self):
        """Display curated dataset summary and pending entries."""
        ds = self._get_curated_dataset()

        if ds.count == 0:
            self._log("[Dataset] Curated dataset is empty.\n"
                      "  Data from training, generation, and "
                      "web learn will appear here.\n"
                      "  You can review and approve entries "
                      "before training.")
            return

        self._log("\n--- CURATED DATASET ---")
        self._log(ds.summary())

        # Show pending entries for review
        pending = ds.get_pending()
        if not pending:
            self._log("\nNo pending entries to review.")
            return

        self._log(f"\n--- PENDING ENTRIES ({len(pending)}) ---")
        for i, entry in enumerate(pending):
            idx = ds.entries.index(entry)
            # Show first 120 chars of text
            preview = entry.text.replace("\n", " ")
            if len(entry.text) > 120:
                preview += "..."
            source_tag = f" [{entry.source}]" if entry.source else ""
            stage_tag = f" ({entry.stage})" if entry.stage else ""
            self._log(f"  [{idx}]{source_tag}{stage_tag} {preview}")

            # Only show first 20 pending entries
            if i >= 19:
                remaining = len(pending) - 20
                if remaining > 0:
                    self._log(f"  ... and {remaining} more pending")
                break

        self._log("\nUse APPROVE ALL to approve all pending,")
        self._log("or reject individual entries on the DOCS page.")

    def _approve_all_dataset(self):
        """Approve all pending entries in the curated dataset."""
        ds = self._get_curated_dataset()

        pending = ds.pending_count
        if pending == 0:
            self._log("[Dataset] No pending entries to approve.")
            return

        count = ds.approve_all_pending()
        ds.save()

        self._log(f"[Dataset] Approved {count} entries.")
        self._log(f"  {ds.approved_count} total approved, "
                  f"ready for training.")
        self.status_bar.set_left(
            f"Approved {count} dataset entries")

    # ================================================================
    # Dataset integration helpers
    # ================================================================

    def _add_to_curated_dataset(
        self,
        text: str,
        source: str = "",
        stage: str = "",
    ) -> None:
        """Add a training text to the curated dataset.

        Called from data generation, guided training, web learn,
        and chat background training to accumulate training data.
        """
        if not text or not text.strip():
            return
        ds = self._get_curated_dataset()
        ds.add(text, source=source, stage=stage)
        ds.save()
        logger.debug("Added to curated dataset: %s (%d chars)",
                     source, len(text))
