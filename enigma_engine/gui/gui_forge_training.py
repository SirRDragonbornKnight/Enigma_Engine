"""
Enigma Engine - Forge Training Modes (Basic)
================================================

Training mode implementations: Solo, DPO, Vision, LoRA.
Split from gui_forge.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import random
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


def _load_hf_directory(model_dir: Path, device: str, log_fn):
    """Load a HuggingFace model directory into ForgeConfig + state_dict.

    Reads config.json, loads safetensors shards, maps weights to Forge format.
    Returns (ForgeConfig, forge_state_dict).
    """
    import json
    import torch
    from safetensors.torch import load_file
    from enigma_engine.core.model_presets import ForgeConfig

    # --- Config ---
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json in {model_dir}")
    with open(config_path, encoding="utf-8") as f:
        hf_cfg = json.load(f)

    # Build a simple namespace so convert_hf_config_to_forge works
    class _Ns:
        pass
    ns = _Ns()
    for k, v in hf_cfg.items():
        setattr(ns, k, v)

    from enigma_engine.core.huggingface_loader import (
        convert_hf_config_to_forge)
    forge_dict = convert_hf_config_to_forge(ns)
    config = ForgeConfig(**forge_dict)
    log_fn(f"  Config: dim={config.dim}, layers={config.n_layers}, "
           f"heads={config.n_heads}, kv_heads={config.n_kv_heads}, "
           f"vocab={config.vocab_size}")

    # --- Weights ---
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        # Sharded safetensors
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        log_fn(f"  Loading {len(shard_files)} safetensor shards...")
        raw_weights: dict[str, torch.Tensor] = {}
        for shard_name in shard_files:
            shard_path = model_dir / shard_name
            raw_weights.update(load_file(str(shard_path), device=device))
    else:
        single = model_dir / "model.safetensors"
        if single.exists():
            log_fn("  Loading model.safetensors...")
            raw_weights = load_file(str(single), device=device)
        else:
            raise FileNotFoundError(
                f"No safetensors files found in {model_dir}")

    log_fn(f"  Loaded {len(raw_weights)} raw weight tensors")

    # Map HF weight names → Forge names
    from enigma_engine.core.weight_mapping import WeightMapper
    model_type = hf_cfg.get("model_type", "llama")
    mapper = WeightMapper()
    forge_weights = mapper.map_huggingface_to_forge(
        raw_weights, model_type=model_type)
    log_fn(f"  Mapped {len(forge_weights)} weights to Forge format")

    return config, forge_weights


def _load_hf_tokenizer(model_dir: Path, log_fn):
    """Try to load the HuggingFace tokenizer from a model directory.

    Returns an AutoTokenizer if transformers is available and the directory
    contains tokenizer files, otherwise returns None.
    """
    tok_file = model_dir / "tokenizer.json"
    tok_config = model_dir / "tokenizer_config.json"
    if not (tok_file.exists() or tok_config.exists()):
        return None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=False)
        log_fn(f"  HF tokenizer loaded (vocab {tokenizer.vocab_size})")
        # Ensure pad_token_id is set (some HF tokenizers leave it None)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        return tokenizer
    except Exception as exc:
        log_fn(f"  [!] Could not load HF tokenizer: {exc}")
        return None


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
        if not data_path:
            # Nothing selected — try a sensible default
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
        elif not Path(data_path).exists():
            self._log(
                f"[!] Training file not found:\n"
                f"    {data_path}\n"
                f"    Browse for a valid file or pick from Quick select.")
            return

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        model_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 SOLO TRAINING...")

        self._log_training_summary(
            "Solo Training",
            Student=model_name,
            Data=Path(data_path).name,
            Epochs=epochs,
            LR=lr,
        )
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _finetune():
            losses = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.tokenizer import get_tokenizer
                # ARCH-1d: when the GUI is in API-chat mode, route training
                # through the daemon via EnigmaClient instead of running it
                # in-process.  The server loads the model, trains, and saves
                # back to its model file.  The GUI just ships the config dict
                # and polls status until done.
                if bool(getattr(self, "use_api_chat", False)):
                    get_client_fn = getattr(
                        self, "_get_api_chat_client", None)
                    client = (get_client_fn()
                              if callable(get_client_fn) else None)
                    if client is not None:
                        try:
                            text = Path(data_path).read_text(
                                encoding="utf-8")
                            forge_params = self._read_forge_train_params()
                            api_config = {
                                "mode": "sft",
                                "data": text,
                                "training": {
                                    "epochs": epochs,
                                    "batch_size":
                                        forge_params["batch_size"],
                                    "learning_rate": lr,
                                    "max_grad_accumulation":
                                        forge_params[
                                            "max_grad_accumulation"],
                                    "use_gradient_checkpointing":
                                        forge_params[
                                            "use_gradient_checkpointing"],
                                    "use_sequence_packing": True,
                                    "ce_chunk_size":
                                        forge_params["ce_chunk_size"],
                                    "use_compile": True,
                                    "rolling_best_k":
                                        forge_params["rolling_best_k"],
                                    "general_mix_ratio":
                                        forge_params["general_mix_ratio"],
                                    "general_data":
                                        forge_params["general_data"] or "",
                                    "val_split":
                                        forge_params["val_split"],
                                    "save_every": max(1, epochs // 5),
                                    "run_evaluation": True,
                                },
                            }
                            self._log("Sending to API server...\n")
                            client.train(api_config)
                            self._poll_api_training_status(client, mode_label="SOLO")
                        except Exception as exc:
                            import traceback
                            self._log(
                                f"\n[!] API training failed: {exc}")
                            self._log(traceback.format_exc())
                        return  # finally block handles GUI cleanup

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Load existing model
                self._log(f"Loading {model_name}...")
                from enigma_engine.core.model_presets import ForgeConfig

                sp = Path(student_path)
                if sp.is_dir():
                    # HuggingFace model directory — convert on-the-fly
                    config, state_dict = _load_hf_directory(
                        sp, device, self._log)
                else:
                    from enigma_engine.core.model_registry import (
                        get_state_dict, safe_load_weights)
                    checkpoint = safe_load_weights(
                        student_path, map_location=device)
                    # Prefer model_config, fall back to config, skip TrainingConfig
                    cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})
                    if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                        cfg_dict = checkpoint.get("model_config", {})
                    config = ForgeConfig(**{
                        k: v for k, v in cfg_dict.items()
                        if k in ForgeConfig.__dataclass_fields__})
                    state_dict = get_state_dict(checkpoint)

                model = Enigma(config=config)
                model.load_state_dict(state_dict, strict=False)
                model = model.to(device)

                # Try HF tokenizer first for HuggingFace models
                sp = Path(student_path)
                tokenizer = None
                if sp.is_dir():
                    tokenizer = _load_hf_tokenizer(sp, self._log)
                # Try the project BPE tokenizer next (models/tokenizer.json)
                if tokenizer is None:
                    _bpe_path = MODELS_DIR / "tokenizer.json"
                    if _bpe_path.exists():
                        try:
                            from enigma_engine.core.bpe_tokenizer import BPETokenizer
                            tokenizer = BPETokenizer(_bpe_path)
                            self._log(
                                f"  Tokenizer: BPE from {_bpe_path.name} "
                                f"(vocab {tokenizer.vocab_size})")
                        except Exception as _tok_err:
                            self._log(f"  [!] BPE tokenizer load failed: {_tok_err}")
                            tokenizer = None
                # Fall back to auto (tiktoken etc.) only if nothing else worked
                if tokenizer is None:
                    tokenizer = get_tokenizer("auto")
                if tokenizer.vocab_size != config.vocab_size:
                    self._log(
                        f"  [!] Tokenizer vocab ({tokenizer.vocab_size}) "
                        f"!= model vocab ({config.vocab_size})")
                    if tokenizer.vocab_size > config.vocab_size:
                        raise ValueError(
                            f"Tokenizer vocab ({tokenizer.vocab_size}) exceeds "
                            f"model vocab ({config.vocab_size}) — token IDs "
                            f"will be out of range. Use a matching tokenizer.")
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                text = Path(data_path).read_text(encoding="utf-8")
                self._log(f"Data    : {len(text):,} chars loaded")

                forge_params = self._read_forge_train_params()
                if forge_params["general_data"]:
                    self._log(
                        f"General : {forge_params['general_mix_ratio']:.0%} mix")

                # --- Live training callbacks ---
                # Throttle: only log+update when percentage
                # changes to avoid flooding the GUI event queue.
                import time as _time
                _last_pct = [-1]
                _last_progress_t = [0.0]

                def on_progress(pct, msg):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _last_progress_t[0] >= 1.0:
                        self._update_forge_progress(pct, msg)
                        _last_progress_t[0] = now
                    if pct != _last_pct[0]:
                        _last_pct[0] = pct

                # Throughput: accumulate tokens for speed calc
                _throughput_tokens = [0]
                _throughput_time = [0.0]

                def on_throughput(tokens: int, step_time: float):
                    _throughput_tokens[0] += tokens
                    _throughput_time[0] += step_time

                # Loss: compact step line with loss + tok/s + VRAM
                # Throttled to 2 updates/sec to prevent GUI freeze.
                _last_loss_t = [0.0]
                _train_start = [_time.monotonic()]
                # Mutable holder populated by on_trainer_ready before
                # training starts — on_loss/on_epoch access trainer
                # internals through this reference.
                _trainer_ref: list = []

                def on_loss(loss: float):
                    now = _time.monotonic()
                    if now - _last_loss_t[0] < 0.5:
                        return
                    _last_loss_t[0] = now
                    import torch as _torch
                    tok_s = int(
                        _throughput_tokens[0]
                        / max(0.001, _throughput_time[0]))
                    _throughput_tokens[0] = 0
                    _throughput_time[0] = 0.0
                    vram = ""
                    if _torch.cuda.is_available():
                        used_gb = (_torch.cuda.memory_allocated()
                                   / 1e9)
                        total_gb = (
                            _torch.cuda
                            .get_device_properties(0)
                            .total_memory / 1e9)
                        vram = (f" | {used_gb:.1f}/"
                                f"{total_gb:.0f} GB")
                    t = _trainer_ref[0] if _trainer_ref else None
                    step = t.state.step if t else 0
                    lr_val = (t.optimizer.param_groups[0]['lr']
                              if t else 0.0)
                    # Batch-level ETA from total_steps
                    eta_str = ""
                    total = (
                        getattr(t, '_total_training_steps', 0)
                        if t else 0)
                    if total > 0 and step > 0:
                        elapsed = now - _train_start[0]
                        remaining = (elapsed / step) * (
                            total - step)
                        if remaining >= 3600:
                            h = int(remaining // 3600)
                            m = int((remaining % 3600) // 60)
                            eta_str = f" | ETA {h}h {m:02d}m"
                        else:
                            m = int(remaining // 60)
                            s = int(remaining % 60)
                            eta_str = f" | ETA {m}m {s:02d}s"
                    self._log(
                        f"  Step {step:>5d}/{total} | loss {loss:.4f}"
                        f" | lr {lr_val:.2e}"
                        f" | {tok_s:,} tok/s{vram}{eta_str}")
                    # Update progress bar at step-level
                    if total > 0 and step > 0:
                        step_pct = int(step / total * 100)
                        self._update_forge_progress(
                            step_pct,
                            f"Step {step:,}/{total:,}")

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    losses.append(loss)
                    pct = int(epoch / epochs * 100)
                    self._update_forge_progress(
                        pct, f"Epoch {epoch}/{epochs}")
                    elapsed = _time.monotonic() - _train_start[0]
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    trend = ""
                    if len(losses) >= 2:
                        delta = losses[-1] - losses[-2]
                        trend = f"  ({delta:+.4f})"
                    # ETA based on average epoch time
                    eta = ""
                    if epoch > 0:
                        remaining = (elapsed / epoch) * (
                            epochs - epoch)
                        r_m = int(remaining // 60)
                        r_s = int(remaining % 60)
                        eta = f"  |  ETA {r_m}m {r_s:02d}s"
                    t = _trainer_ref[0] if _trainer_ref else None
                    best = t.state.best_loss if t else float("inf")
                    import math as _math
                    best_str = (f"  |  best {best:.4f}"
                                if not _math.isinf(best) else "")
                    self._log(
                        f"  Epoch {epoch:>3d}/{epochs}  |  "
                        f"loss {loss:.4f}{trend}{best_str}  |  "
                        f"{mins}m {secs:02d}s{eta}")

                # I-14: Check for checkpoint resume
                from enigma_engine.training.training import Trainer as _T
                ckpt_dir = Path(str(MODELS_DIR / "checkpoints"))
                resume_enabled = getattr(
                    self, 'forge_resume_var', None)
                try_resume = (resume_enabled is None
                              or resume_enabled.get())
                resume_path = (
                    _T._find_latest_checkpoint(ckpt_dir)
                    if try_resume else None)
                if resume_path is not None:
                    self._log(
                        f"--- RESUMING from {resume_path.name} ---")
                elif not try_resume:
                    existing = _T._find_latest_checkpoint(ckpt_dir)
                    if existing is not None:
                        self._log(
                            "[i] Checkpoint exists but Resume is "
                            "unchecked — starting fresh.")

                def on_trainer_ready(t) -> None:
                    _trainer_ref.append(t)
                    self._log(f"Batch   : {t.config.batch_size}")
                    self._active_trainer = t

                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    on_progress=on_progress,
                    on_epoch_complete=on_epoch,
                    on_loss=on_loss,
                    on_throughput=on_throughput,
                    on_trainer_ready=on_trainer_ready,
                )
                config_dict = {
                    "mode": "sft",
                    "data": text,
                    "resume_from": (str(resume_path)
                                    if resume_path is not None
                                    else None),
                    "training": {
                        "epochs": epochs,
                        "batch_size": forge_params["batch_size"],
                        "learning_rate": lr,
                        "max_grad_accumulation": forge_params[
                            "max_grad_accumulation"],
                        "use_gradient_checkpointing": forge_params[
                            "use_gradient_checkpointing"],
                        "use_sequence_packing": True,
                        "ce_chunk_size": forge_params["ce_chunk_size"],
                        "use_compile": True,
                        "rolling_best_k": forge_params["rolling_best_k"],
                        "general_mix_ratio": forge_params[
                            "general_mix_ratio"],
                        "general_data": (
                            forge_params["general_data"] or ""),
                        "val_split": forge_params["val_split"],
                        "save_every": max(1, epochs // 5),
                        "checkpoint_dir": str(
                            MODELS_DIR / "checkpoints"),
                        "use_amp": torch.cuda.is_available(),
                        "run_evaluation": True,
                    },
                }
                self._log("Training...\n")
                state = run_training(config_dict, ctx)

                # Check for early termination (OOM / NaN / Inf)
                import math
                if math.isinf(state.best_loss) and not losses:
                    reason = getattr(state, 'abort_reason', '') or (
                        "likely OOM or NaN loss")
                    self._log(
                        f"\n[!] Training aborted on first batch "
                        f"({reason}).")
                    self._log(
                        "    Try LoRA training instead for large "
                        "models, or reduce batch size.")
                    return

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
                if out.is_dir():
                    # HF directory — save as a Forge .pth beside it
                    out = out.with_suffix(".pth")
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "model_config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }, out)

                self._log("\n--- SOLO TRAINING COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Saved to  : {out}")
                total = _time.monotonic() - _train_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
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
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log("\n--- SOLO TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                self._log(f"\n[!] Solo training failed: {exc}")
                self._log(tb)
            finally:
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled", text="STOP"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_finetune, daemon=True).start()

    # ================================================================
    # DPO training (preference optimization)
    # ================================================================

    def _poll_api_training_status(self, client, mode_label: str = "TRAINING") -> None:
        """Poll GET /api/training/status until complete, updating the forge log.

        Called from a background thread by the ARCH-1d API-routing branch of
        _start_solo_training (and future API-routing launchers).  Blocks until
        the server reports ``active=False`` or ``self.training_active`` is
        cleared by the STOP button.

        Args:
            mode_label: Used in completion log (e.g., "SOLO", "DPO").
        """
        import math
        import time as _time

        _last_step = -1
        _start_t = _time.monotonic()

        while self.training_active:
            try:
                status = client.training_status()
            except Exception as exc:
                self._log(f"[!] API status error: {exc}")
                _time.sleep(2.0)
                continue

            step = status.get("step", 0)
            total_steps = status.get("total_steps", 0)
            loss = status.get("loss", 0.0)
            lr_val = status.get("lr", 0.0)
            tok_s = status.get("tok_s", 0)
            pct = status.get("progress", 0)
            msg = status.get("message", "Training...")
            active = status.get("active", True)
            best = status.get("best_loss")
            if best is None:
                best = float("inf")
            abort_reason = status.get("abort_reason", "")
            output_path = status.get("output_path", "")

            # Log on each new step
            if step != _last_step and step > 0:
                _last_step = step
                tok_str = f" | {tok_s:,} tok/s" if tok_s > 0 else ""
                self._log(
                    f"  Step {step:>5d}/{total_steps}"
                    f" | loss {loss:.4f}"
                    f" | lr {lr_val:.2e}{tok_str}")

            self._update_forge_progress(pct, msg)

            if not active:
                elapsed = _time.monotonic() - _start_t
                t_m, t_s = int(elapsed // 60), int(elapsed % 60)
                if abort_reason:
                    self._log(f"\n[!] Training aborted: {abort_reason}")
                elif "failed" in msg.lower():
                    self._log(f"\n[!] Training failed: {msg}")
                else:
                    self._log(f"\n--- API {mode_label} TRAINING COMPLETE ---")
                    if not math.isinf(best):
                        self._log(f"Best loss : {best:.4f}")
                    if output_path:
                        self._log(f"Saved to  : {output_path}")
                    self._log(f"Duration  : {t_m}m {t_s:02d}s")
                    self._update_forge_progress(100, "Complete")
                self.after(0, self._refresh_models)
                self._notify_training_complete()
                break

            _time.sleep(1.0)

    def _start_apo_training(self):
        """Train STUDENT with Anchored Preference Optimization (zero).

        D-9b (Pass 156k): thin wrapper that delegates to the shared
        DPO training body with ``loss_type='apo_zero'``. Every other
        knob (data format, beta, scheduler, eval) is identical to DPO
        — only the loss math differs (chosen and rejected each
        anchored independently to the reference policy).
        """
        self._start_dpo_training(loss_type="apo_zero")

    def _start_dpo_training(self, loss_type: str = "dpo"):
        """Train STUDENT with Direct Preference Optimization.

        Requires a JSONL data file where each line has:
        {"prompt": "...", "chosen": "...", "rejected": "..."}

        Args:
            loss_type: ``"dpo"`` (default) or ``"apo_zero"``. Forwarded
                to ``trainer.train_dpo`` which dispatches via
                ``Trainer._resolve_preference_loss``. Pass 156k.
        """
        if self.training_active:
            return
        # Display label for log/status — "DPO" or "APO-ZERO".
        algo_label = "DPO" if loss_type == "dpo" else "APO-ZERO"
        algo_summary_label = (
            "DPO Training" if loss_type == "dpo"
            else "APO-Zero Training")

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        # DPO/APO require a JSONL file with preference pairs
        data_path = self.train_data_var.get()
        if not data_path:
            self._log(
                "[!] No data file selected.\n"
                f"    {algo_label} requires a JSONL file with:\n"
                '    {"prompt": "...", "chosen": "...", '
                '"rejected": "..."}')
            return
        if not Path(data_path).exists():
            self._log(
                f"[!] Training file not found:\n"
                f"    {data_path}\n"
                f"    Browse for a valid file or pick from Quick select.")
            return

        if not self._validate_jsonl_structure(
            data_path, ("prompt", "chosen", "rejected"),
        ):
            return

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        beta_val = 0.1

        model_name = Path(student_path).stem

        # ARCH-1d Slice 2: API routing for preference training
        if bool(getattr(self, "use_api_chat", False)):
            get_client_fn = getattr(self, "_get_api_chat_client", None)
            client = (get_client_fn() if callable(get_client_fn) else None)
            if client is not None:
                def _run_api():
                    try:
                        pref_data = Path(data_path).read_text(encoding="utf-8")
                        forge_params = self._read_forge_train_params()
                        api_config = {
                            "mode": "dpo",
                            "data": pref_data,
                            "training": {
                                "epochs": epochs,
                                "batch_size": forge_params["batch_size"],
                                "learning_rate": lr,
                                "max_grad_accumulation":
                                    forge_params["max_grad_accumulation"],
                                "use_gradient_checkpointing":
                                    forge_params["use_gradient_checkpointing"],
                                "use_sequence_packing": True,
                                "ce_chunk_size": forge_params["ce_chunk_size"],
                                "use_compile": True,
                                "rolling_best_k": forge_params["rolling_best_k"],
                                "general_mix_ratio":
                                    forge_params["general_mix_ratio"],
                                "general_data":
                                    forge_params["general_data"] or "",
                                "val_split": forge_params["val_split"],
                                "save_every": max(1, epochs // 5),
                                "run_evaluation": True,
                            },
                            "dpo": {
                                "beta": beta_val,
                                "loss_type": loss_type,
                            },
                        }
                        self._log("Sending preference training to API server...\n")
                        self.training_active = True
                        self.solo_train_btn.configure(state="disabled",
                                                      text="TRAINING...")
                        self.stop_train_btn.configure(state="normal")
                        self.status_bar.set_left(
                            f"\u2692 {algo_label} TRAINING...")
                        client.train(api_config)
                        self._poll_api_training_status(
                            client,
                            mode_label=algo_label)
                    except Exception as exc:
                        import traceback
                        self._log(f"\n[!] API training failed: {exc}")
                        self._log(traceback.format_exc())
                    finally:
                        self.training_active = False
                        self.after(0, lambda: self.solo_train_btn.configure(
                            state="normal", text="TRAIN"))
                        self.after(0, lambda: self.stop_train_btn.configure(
                            state="disabled", text="STOP"))
                        self.after(0, lambda: self.status_bar.set_left(
                            "\u26a1 READY"))
                import threading
                threading.Thread(target=_run_api, daemon=True).start()
                return

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(f"\u2692 {algo_label} TRAINING...")

        self._log_training_summary(
            algo_summary_label,
            Student=model_name,
            Data=Path(data_path).name,
            Epochs=epochs,
            LR=lr,
        )
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
                            self._log(
                                f"[!] Skipping malformed JSONL line "
                                f"{len(pref_data) + 1}")
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
                config = ForgeConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in ForgeConfig.__dataclass_fields__})
                model = Enigma(config=config)
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
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                forge_params = self._read_forge_train_params()

                # Pass 156z9as: pre-DPO/APO auto-checkpoint.
                # ``loss_type`` distinguishes the two algorithms in the
                # rollback file name so the user can tell DPO and APO
                # backups apart at a glance.
                _backup_suffix = f"pre_{loss_type}"
                pre_dpo_backup_path = (
                    self._pre_training_backup(
                        student_path, suffix=_backup_suffix))

                import time as _time
                _last_dpo_pct = [-1]
                _last_dpo_t = [0.0]
                _dpo_start = [_time.monotonic()]

                def on_progress(pct: int, msg: str):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _last_dpo_t[0] >= 1.0:
                        eta = ""
                        if pct > 0:
                            elapsed = now - _dpo_start[0]
                            remaining = (elapsed / pct) * (
                                100 - pct)
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            pct, f"{msg}{eta}")
                        _last_dpo_t[0] = now

                _last_dpo_loss_t = [0.0]
                _trainer_ref: list = []

                def on_loss(loss: float):
                    now = _time.monotonic()
                    if now - _last_dpo_loss_t[0] < 0.5:
                        return
                    _last_dpo_loss_t[0] = now
                    t = _trainer_ref[0] if _trainer_ref else None
                    step = t.state.step if t else 0
                    lr_now = (
                        t.optimizer.param_groups[0]['lr']
                        if t else 0.0)
                    self._log(
                        f"  Step {step:>5d} | loss {loss:.4f}"
                        f" | lr {lr_now:.2e}")

                def on_trainer_ready(t) -> None:
                    _trainer_ref.append(t)
                    self._active_trainer = t

                self._log(
                    f"Starting {algo_label}: {epochs} epochs, "
                    f"lr={lr}, beta={beta_val}")
                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    on_progress=on_progress,
                    on_loss=on_loss,
                    on_trainer_ready=on_trainer_ready,
                )
                config_dict = {
                    "mode": "dpo",
                    "data": pref_data,
                    "training": {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "batch_size": forge_params["batch_size"],
                        "max_grad_accumulation": forge_params[
                            "max_grad_accumulation"],
                        "use_gradient_checkpointing": forge_params[
                            "use_gradient_checkpointing"],
                        "use_sequence_packing": True,
                        "ce_chunk_size": forge_params["ce_chunk_size"],
                        "use_compile": True,
                        "rolling_best_k": forge_params["rolling_best_k"],
                        "general_mix_ratio": forge_params[
                            "general_mix_ratio"],
                        "general_data": (
                            forge_params["general_data"] or ""),
                        "val_split": forge_params["val_split"],
                        "save_every": max(1, epochs // 5),
                        "checkpoint_dir": str(
                            MODELS_DIR / "checkpoints"),
                        "use_amp": torch.cuda.is_available(),
                        "run_evaluation": True,
                    },
                    "dpo": {
                        "beta": beta_val,
                        "loss_type": loss_type,
                    },
                }
                state = run_training(config_dict, ctx)

                # Save updated model
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "model_config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch if hasattr(state, 'epoch') else epochs,
                        "best_loss": state.best_loss if hasattr(state, 'best_loss') else 0.0,
                    },
                }, student_path)
                self._log(f"Model saved to {Path(student_path).name}")
                best = state.best_loss if hasattr(state, 'best_loss') else 0.0
                self._log(
                    f"{algo_label} complete — best loss: {best:.4f}")
                total = _time.monotonic() - _dpo_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                if pre_dpo_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_dpo_backup_path).name}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    algo_label, Path(student_path).stem,
                    epochs, best)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log(f"\n--- {algo_label} TRAINING STOPPED ---")
            except Exception as e:
                self._log(f"[ERROR] {algo_label} training failed: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled", text="STOP"))
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

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        # Get encoder preset from UI
        preset_var = getattr(self, "forge_vision_preset_var", None)
        preset = preset_var.get() if preset_var else "small"
        preset = preset.split(" - ", 1)[0]

        # Code-6b (Pass 156r): unfreeze last N text layers (LLaVA
        # Stage-2 knob). Default 0 = projection-only Stage-1.
        # Range-clamp + bad-input guard mirror the LoRA rank parsing
        # pattern from `_start_lora_training`.
        unfreeze_var = getattr(self, "forge_vision_unfreeze_var", None)
        try:
            raw_unfreeze = unfreeze_var.get().strip() if unfreeze_var else ""
            unfreeze_text_layers = int(raw_unfreeze) if raw_unfreeze else 0
            if unfreeze_text_layers < 0:
                self._log(
                    f"[!] Unfreeze layers '{raw_unfreeze}' negative, "
                    f"using 0")
                unfreeze_text_layers = 0
            if unfreeze_text_layers > 64:
                self._log(
                    f"[!] Unfreeze layers '{raw_unfreeze}' "
                    f"unreasonably large, clamping to 64")
                unfreeze_text_layers = 64
        except (ValueError, TypeError):
            if unfreeze_var and unfreeze_var.get().strip():
                self._log(
                    f"[!] Unfreeze layers '{unfreeze_var.get().strip()}' "
                    f"not a number, using 0")
            unfreeze_text_layers = 0

        # V-4: Check for stale heartbeat from a previously interrupted
        # session. Heartbeat exists + status not clean + PID is dead =
        # OS OOM kill / crash with no traceback.
        import json as _json_stale
        _hb_check_path = Path("logs") / "training_heartbeat.json"
        if _hb_check_path.exists():
            try:
                _hb_prev = _json_stale.loads(
                    _hb_check_path.read_text(encoding="utf-8"))
                if _hb_prev.get("status") not in ("complete", "stopped"):
                    _hb_pid = _hb_prev.get("pid")
                    _hb_dead = True
                    if _hb_pid:
                        try:
                            import psutil as _psutil_hb
                            _hb_dead = not _psutil_hb.pid_exists(_hb_pid)
                        except ImportError:
                            _hb_dead = False
                    if _hb_dead:
                        self._log(
                            "\n[!] PREVIOUS VISION SESSION WAS "
                            "INTERRUPTED UNEXPECTEDLY\n"
                            f"    Model : {_hb_prev.get('model', '?')}"
                            f"  |  Phase: {_hb_prev.get('phase', '?')}"
                            f"  |  Step: {_hb_prev.get('step', '?')}\n"
                            f"    Last seen: "
                            f"{_hb_prev.get('timestamp', '?')}\n"
                            "    Likely cause: OS out-of-memory kill "
                            "(no Python traceback).\n"
                            "    Any saved checkpoint is intact in "
                            "models/checkpoints/.\n")
            except Exception:
                pass

        # ARCH-1d Slice 2: API routing for vision training
        if bool(getattr(self, "use_api_chat", False)):
            get_client_fn = getattr(self, "_get_api_chat_client", None)
            client = (get_client_fn() if callable(get_client_fn) else None)
            if client is not None:
                def _run_api():
                    try:
                        # Serialize vision pairs to JSONL-like format for API
                        import json as _json_vision
                        pairs_data = "\n".join(
                            _json_vision.dumps(p) for p in pairs)
                        forge_params = self._read_forge_train_params()
                        api_config = {
                            "mode": "vision",
                            "data": pairs_data,
                            "training": {
                                "epochs": epochs,
                                "batch_size": forge_params["batch_size"],
                                "learning_rate": lr,
                                "max_grad_accumulation":
                                    forge_params["max_grad_accumulation"],
                                "use_gradient_checkpointing":
                                    forge_params["use_gradient_checkpointing"],
                                "use_sequence_packing": True,
                                "ce_chunk_size": forge_params["ce_chunk_size"],
                                "use_compile": True,
                                "rolling_best_k": forge_params["rolling_best_k"],
                                "general_mix_ratio":
                                    forge_params["general_mix_ratio"],
                                "general_data":
                                    forge_params["general_data"] or "",
                                "val_split": forge_params["val_split"],
                                "save_every": max(1, epochs // 5),
                                "run_evaluation": True,
                            },
                            "vision": {
                                "preset": preset,
                                "unfreeze_text_layers": unfreeze_text_layers,
                            },
                        }
                        self._log("Sending vision training to API server...\n")
                        self.training_active = True
                        self.solo_train_btn.configure(state="disabled",
                                                      text="TRAINING...")
                        self.stop_train_btn.configure(state="normal")
                        self.status_bar.set_left("\u2692 VISION TRAINING...")
                        client.train(api_config)
                        self._poll_api_training_status(
                            client,
                            mode_label="VISION")
                    except Exception as exc:
                        import traceback
                        self._log(f"\n[!] API training failed: {exc}")
                        self._log(traceback.format_exc())
                    finally:
                        self.training_active = False
                        self.after(0, lambda: self.solo_train_btn.configure(
                            state="normal", text="TRAIN"))
                        self.after(0, lambda: self.stop_train_btn.configure(
                            state="disabled", text="STOP"))
                        self.after(0, lambda: self.status_bar.set_left(
                            "\u26a1 READY"))
                import threading
                threading.Thread(target=_run_api, daemon=True).start()
                return

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 VISION TRAINING...")

        summary_fields = {
            "Student": Path(student_path).stem,
            "Data dir": vision_dir,
            "Pairs": str(len(pairs)),
            "Encoder": preset,
            "Unfreeze": (f"{unfreeze_text_layers} text layers"
                         if unfreeze_text_layers > 0
                         else "projection only (Stage-1)"),
            "Epochs": epochs,
            "LR": lr,
        }
        self._log_training_summary("Vision Training", **summary_fields)
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _vision_train():
            losses: list[float] = []
            # V-4: heartbeat for silent-kill / OOM detection.
            import os as _os_hb
            import json as _json_hb
            import datetime as _dt_hb
            _hb_path = Path("logs") / "training_heartbeat.json"
            _hb_path.parent.mkdir(exist_ok=True)
            _safe_name = Path(student_path).stem

            def _write_hb(phase="data_load", step=None,
                          loss=None, status="running"):
                """Write heartbeat so silent kills are detectable."""
                try:
                    hb = {
                        "pid": _os_hb.getpid(),
                        "status": status,
                        "model": _safe_name,
                        "phase": phase,
                        "timestamp": _dt_hb.datetime.now()
                            .isoformat(timespec="seconds"),
                    }
                    if step is not None:
                        hb["step"] = step
                    if loss is not None:
                        hb["loss"] = round(float(loss), 4)
                    _hb_path.write_text(
                        _json_hb.dumps(hb, indent=2),
                        encoding="utf-8")
                except Exception:
                    pass  # never crash on heartbeat

            _write_hb("data_load")
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.tokenizer import get_tokenizer
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

                config = ForgeConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in ForgeConfig.__dataclass_fields__})
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                # Load with strict=False since we may have added vision_projection
                model.load_state_dict(state_dict, strict=False)
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
                _trainer_ref: list = []

                # Pass 156z9at: pre-vision auto-checkpoint, ONLY when
                # Stage-2 (``unfreeze_text_layers > 0``) will mutate
                # the text backbone.  Projection-only training keeps
                # the text weights frozen, so the rollback rail is
                # unnecessary in that path.
                pre_vision_backup_path = None
                if unfreeze_text_layers > 0:
                    pre_vision_backup_path = (
                        self._pre_training_backup(
                            student_path,
                            suffix="pre_vision_stage2"))

                import time as _time
                _last_vis_pct = [-1]
                _last_vis_t = [0.0]
                _vis_start = [_time.monotonic()]

                def on_progress(pct: int, msg: str):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _last_vis_t[0] >= 1.0:
                        self._update_forge_progress(pct, msg)
                        _last_vis_t[0] = now
                        # V-4: piggyback heartbeat on the same throttle.
                        _write_hb("training", step=pct)

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    losses.append(loss)
                    # V-4: per-epoch heartbeat with step + loss.
                    _write_hb("training", step=epoch, loss=loss)
                    pct = int(epoch / epochs * 100)
                    self._update_forge_progress(
                        pct, f"Epoch {epoch}/{epochs}")
                    elapsed = _time.monotonic() - _vis_start[0]
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    eta = ""
                    if epoch > 0:
                        remaining = (elapsed / epoch) * (
                            epochs - epoch)
                        r_m = int(remaining // 60)
                        r_s = int(remaining % 60)
                        eta = f"  |  ETA {r_m}m {r_s:02d}s"
                    t = _trainer_ref[0] if _trainer_ref else None
                    best = t.state.best_loss if t else float("inf")
                    import math as _math
                    best_str = (f"  |  best {best:.4f}"
                                if not _math.isinf(best) else "")
                    self._log(
                        f"  Epoch {epoch:>3d}/{epochs}  |  "
                        f"loss {loss:.4f}{best_str}  |  "
                        f"{mins}m {secs:02d}s{eta}")

                def on_trainer_ready(t) -> None:
                    _trainer_ref.append(t)
                    self._active_trainer = t

                self._log("Training vision encoder...\n")

                # V-6b: honor Forge val_split for vision training.
                # train_vision() backend gained val_data in V-6 (Pass 156g);
                # plumb the GUI-controlled fraction through here. Small
                # datasets fall back to no-val gracefully (need >=2 pairs
                # to leave at least one each side).
                #
                # Seed is currently not exposed in the Forge vision UI,
                # so hold-out split uses an unseeded local Random here.
                # This matches train_vision()'s own per-epoch shuffle
                # policy when no config seed is set.
                val_split_frac = float(
                    forge_params["val_split"] or 0.0)
                val_pairs_data = None
                train_pairs_data = vision_data
                if val_split_frac > 0 and len(vision_data) >= 2:
                    seed = None
                    rng = random.Random(seed) if seed is not None else random.Random()
                    _shuffled = list(vision_data)
                    rng.shuffle(_shuffled)
                    n_val = max(1, int(len(_shuffled) * val_split_frac))
                    n_val = min(n_val, len(_shuffled) - 1)
                    val_pairs_data = _shuffled[:n_val]
                    train_pairs_data = _shuffled[n_val:]
                    self._log(
                        f"Vision split: {len(train_pairs_data)} train / "
                        f"{len(val_pairs_data)} val "
                        f"(val_split={val_split_frac:.2f}, "
                        f"seed={seed})\n")

                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    vision_encoder=v_encoder,
                    on_progress=on_progress,
                    on_epoch_complete=on_epoch,
                    on_trainer_ready=on_trainer_ready,
                )
                config_dict = {
                    "mode": "vision",
                    "data": {
                        "train": train_pairs_data,
                        "val": val_pairs_data,
                    },
                    "training": {
                        "epochs": epochs,
                        "batch_size": 1,
                        "learning_rate": lr,
                        "max_grad_accumulation": forge_params[
                            "max_grad_accumulation"],
                        "use_gradient_checkpointing": forge_params[
                            "use_gradient_checkpointing"],
                        "use_sequence_packing": True,
                        "ce_chunk_size": forge_params["ce_chunk_size"],
                        "use_compile": True,
                        "rolling_best_k": forge_params["rolling_best_k"],
                        "general_mix_ratio": forge_params[
                            "general_mix_ratio"],
                        "general_data": (
                            forge_params["general_data"] or ""),
                        "val_split": forge_params["val_split"],
                        "save_every": max(1, epochs // 5),
                        "checkpoint_dir": str(
                            MODELS_DIR / "checkpoints"),
                        "use_amp": torch.cuda.is_available(),
                        "run_evaluation": True,
                    },
                    "vision": {
                        "unfreeze_text_layers": unfreeze_text_layers,
                    },
                }
                state = run_training(config_dict, ctx)

                # Save model with vision projection weights
                from enigma_engine.core.safe_save import atomic_torch_save
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "model_config": self._model_config_dict(model),
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
                total = _time.monotonic() - _vis_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                if pre_vision_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_vision_backup_path).name}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Vision", Path(student_path).stem,
                    epochs, state.best_loss)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)
                self._notify_training_complete()
                # V-4: clean exit — heartbeat marked complete.
                _write_hb("training", step=state.epoch,
                          loss=state.best_loss, status="complete")

            except KeyboardInterrupt:
                self._log("\n--- VISION TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
                # V-4: user stop — heartbeat marked stopped.
                _write_hb("training", status="stopped")
            except RuntimeError as exc:
                # V-4 audit fix: split RuntimeError from generic
                # Exception so OOM-friendly user advice does not fire
                # on PIL/NumPy errors that happen to mention 'memory'.
                # Taxonomy matches gui_forge_new_modes.py reference.
                import traceback
                tb = traceback.format_exc()
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda" in msg:
                    _write_hb("training", status="crashed_oom")
                    self._log(
                        f"\n[!] GPU out of memory: {exc}\n"
                        "    Try: smaller vision encoder preset, "
                        "fewer epochs, or smaller image size.")
                else:
                    _write_hb("training", status="crashed")
                    self._log(f"\n[!] Vision training failed: {exc}")
                self._log(tb)
            except Exception as exc:
                self._log(f"\n[!] Vision training failed: {exc}")
                import traceback
                self._log(traceback.format_exc())
                _write_hb("training", status="crashed")
            finally:
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled", text="STOP"))
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
        if not data_path:
            self._log(
                "[!] No training data selected.\n"
                "    LoRA needs a data file to train on.")
            return
        if not Path(data_path).exists():
            self._log(
                f"[!] Training file not found:\n"
                f"    {data_path}\n"
                f"    Browse for a valid file or pick from Quick select.")
            return

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        # Read LoRA config from UI
        rank_var = getattr(self, "forge_lora_rank_var", None)
        alpha_var = getattr(self, "forge_lora_alpha_var", None)
        try:
            raw_rank = rank_var.get().strip() if rank_var else ""
            lora_rank = int(raw_rank) if raw_rank else 8
            if lora_rank < 1 or lora_rank > 128:
                self._log(f"[!] LoRA rank '{raw_rank}' out of range "
                          f"(1-128), using 8")
                lora_rank = 8
        except (ValueError, TypeError):
            if rank_var and rank_var.get().strip():
                self._log(f"[!] LoRA rank '{rank_var.get().strip()}' "
                          f"not a number, using 8")
            lora_rank = 8

        try:
            raw_alpha = alpha_var.get().strip() if alpha_var else ""
            lora_alpha = int(raw_alpha) if raw_alpha else 16
            if lora_alpha < 1 or lora_alpha > 256:
                self._log(f"[!] LoRA alpha '{raw_alpha}' out of "
                          f"range (1-256), using 16")
                lora_alpha = 16
        except (ValueError, TypeError):
            if alpha_var and alpha_var.get().strip():
                self._log(f"[!] LoRA alpha '{alpha_var.get().strip()}'"
                          f" not a number, using 16")
            lora_alpha = 16

        # ARCH-1d Slice 2: API routing for LoRA training.
        if bool(getattr(self, "use_api_chat", False)):
            get_client_fn = getattr(self, "_get_api_chat_client", None)
            client = (get_client_fn() if callable(get_client_fn) else None)
            if client is not None:
                def _run_api():
                    try:
                        train_data = Path(data_path).read_text(encoding="utf-8")
                        forge_params = self._read_forge_train_params()
                        api_config = {
                            "mode": "lora",
                            "data": train_data,
                            "training": {
                                "epochs": epochs,
                                "batch_size": forge_params["batch_size"],
                                "learning_rate": lr,
                                "max_grad_accumulation":
                                    forge_params["max_grad_accumulation"],
                                "use_gradient_checkpointing":
                                    forge_params["use_gradient_checkpointing"],
                                "use_sequence_packing": True,
                                "ce_chunk_size": forge_params["ce_chunk_size"],
                                "use_compile": True,
                                "rolling_best_k": forge_params["rolling_best_k"],
                                "general_mix_ratio":
                                    forge_params["general_mix_ratio"],
                                "general_data":
                                    forge_params["general_data"] or "",
                                "val_split": forge_params["val_split"],
                                "save_every": max(1, epochs // 5),
                                "run_evaluation": True,
                            },
                            "lora": {
                                "rank": lora_rank,
                                "alpha": lora_alpha,
                            },
                        }
                        self._log("Sending LoRA training to API server...\n")
                        self.training_active = True
                        self.solo_train_btn.configure(state="disabled",
                                                      text="TRAINING...")
                        self.stop_train_btn.configure(state="normal")
                        self.status_bar.set_left("\u2692 LoRA TRAINING...")
                        client.train(api_config)
                        self._poll_api_training_status(client, mode_label="LoRA")
                    except Exception as exc:
                        import traceback
                        self._log(f"\n[!] API training failed: {exc}")
                        self._log(traceback.format_exc())
                    finally:
                        self.training_active = False
                        self.after(0, lambda: self.solo_train_btn.configure(
                            state="normal", text="TRAIN"))
                        self.after(0, lambda: self.stop_train_btn.configure(
                            state="disabled", text="STOP"))
                        self.after(0, lambda: self.status_bar.set_left(
                            "\u26a1 READY"))
                import threading
                threading.Thread(target=_run_api, daemon=True).start()
                return

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 LoRA TRAINING...")

        model_name = Path(student_path).stem
        summary_fields = {
            "Student": model_name,
            "Data": Path(data_path).name,
            "Rank": lora_rank,
            "Alpha": lora_alpha,
            "Epochs": epochs,
            "LR": lr,
        }
        self._log_training_summary("LoRA Training", **summary_fields)
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _lora_train():
            losses: list[float] = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.tokenizer import get_tokenizer

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                sp = Path(student_path)
                if sp.is_dir():
                    # HuggingFace model directory
                    config, state_dict = _load_hf_directory(
                        sp, device, self._log)
                else:
                    from enigma_engine.core.model_registry import (
                        get_state_dict, safe_load_weights)
                    checkpoint = safe_load_weights(
                        student_path, map_location=device)

                    cfg_dict = (checkpoint.get("model_config")
                                or checkpoint.get("config", {}))
                    if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                        cfg_dict = checkpoint.get("model_config", {})

                    config = ForgeConfig(**{
                        k: v for k, v in cfg_dict.items()
                        if k in ForgeConfig.__dataclass_fields__})
                    state_dict = get_state_dict(checkpoint)

                model = Enigma(config=config)
                model.load_state_dict(state_dict, strict=False)
                model = model.to(device)

                # Try HF tokenizer for HuggingFace models
                tokenizer = None
                if sp.is_dir():
                    tokenizer = _load_hf_tokenizer(sp, self._log)
                if tokenizer is None:
                    _bpe_path = MODELS_DIR / "tokenizer.json"
                    if _bpe_path.exists():
                        try:
                            from enigma_engine.core.bpe_tokenizer import BPETokenizer
                            tokenizer = BPETokenizer(_bpe_path)
                        except Exception:
                            tokenizer = get_tokenizer("auto")
                    else:
                        tokenizer = get_tokenizer("auto")
                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                ppl_before = None
                ppl_after = None
                import time as _time
                _lora_start_common = [_time.monotonic()]
                forge_params = self._read_forge_train_params()
                # Try PEFT LoRA first, fall back to manual
                try:
                    text = Path(data_path).read_text(encoding="utf-8")
                    self._log(f"Data    : {len(text):,} chars loaded")

                    def on_loss(loss: float) -> None:
                        if not self.training_active:
                            raise KeyboardInterrupt("Stopped")
                        losses.append(loss)

                    def on_trainer_ready(t) -> None:
                        self._active_trainer = t

                    from enigma_engine.training.dispatch import (
                        build_dispatch_context, run_training)
                    ctx = build_dispatch_context(
                        model=model,
                        tokenizer=tokenizer,
                        on_loss=on_loss,
                        on_trainer_ready=on_trainer_ready,
                    )
                    lora_batch_size = forge_params["batch_size"]
                    if lora_batch_size <= 0:
                        lora_batch_size = 4

                    config_dict = {
                        "mode": "lora",
                        "data": text,
                        "lora": {
                            "rank": lora_rank,
                            "alpha": lora_alpha,
                            "output_dir": str(MODELS_DIR / "checkpoints"),
                            "learning_rate": lr,
                            "batch_size": lora_batch_size,
                            "epochs": epochs,
                            "gradient_accumulation_steps": forge_params[
                                "max_grad_accumulation"],
                            "max_length": 512,
                        },
                    }

                    self._log("Training LoRA adapters...\n")
                    lora_result = run_training(config_dict, ctx)
                    adapter_path_raw = str(
                        lora_result.get("adapter_path", ""))
                    if adapter_path_raw:
                        adapter_path = Path(adapter_path_raw)
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

                    _best_track: list[float] = []

                    def on_loss_fallback(loss: float) -> None:
                        if not self.training_active:
                            raise KeyboardInterrupt("Stopped")
                        losses.append(loss)
                        if not _best_track or loss < _best_track[0]:
                            if _best_track:
                                _best_track[0] = loss
                            else:
                                _best_track.append(loss)

                    def on_epoch_fallback(epoch, loss):
                        if not self.training_active:
                            raise KeyboardInterrupt("Stopped")
                        elapsed = _time.monotonic() - _lora_start_common[0]
                        mins = int(elapsed // 60)
                        secs = int(elapsed % 60)
                        eta = ""
                        if epoch > 0:
                            remaining = (elapsed / epoch) * (
                                epochs - epoch)
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f"  |  ETA {r_m}m {r_s:02d}s"
                        import math as _math
                        best = _best_track[0] if _best_track else loss
                        best_str = (f"  |  best {best:.4f}"
                                    if not _math.isinf(best) else "")
                        self._log(
                            f"  Epoch {epoch:>3d}/{epochs}  |  "
                            f"loss {loss:.4f}{best_str}  |  "
                            f"{mins}m {secs:02d}s{eta}")

                    def on_trainer_ready_fallback(t) -> None:
                        self._active_trainer = t

                    from enigma_engine.training.dispatch import (
                        build_dispatch_context as _bdc,
                        run_training as _rt)
                    _fb_ctx = _bdc(
                        model=model,
                        tokenizer=tokenizer,
                        on_loss=on_loss_fallback,
                        on_epoch_complete=on_epoch_fallback,
                        on_trainer_ready=on_trainer_ready_fallback,
                    )
                    _fb_payload = {
                        "mode": "sft",
                        "data": text,
                        "training": {
                            "epochs": epochs,
                            "batch_size": forge_params["batch_size"],
                            "learning_rate": lr,
                            "max_grad_accumulation": forge_params["max_grad_accumulation"],
                            "use_gradient_checkpointing": forge_params["use_gradient_checkpointing"],
                            "use_sequence_packing": True,
                            "ce_chunk_size": forge_params["ce_chunk_size"],
                            "use_compile": True,
                            "rolling_best_k": forge_params["rolling_best_k"],
                            "general_mix_ratio": forge_params["general_mix_ratio"],
                            "general_data": forge_params["general_data"],
                            "val_split": forge_params["val_split"],
                            "save_every": max(1, epochs // 5),
                            "checkpoint_dir": str(MODELS_DIR / "checkpoints"),
                            "use_amp": torch.cuda.is_available(),
                            "run_evaluation": True,
                        },
                    }
                    self._log("Training (partial freeze)...\n")
                    state = _rt(_fb_payload, _fb_ctx)

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
                save_path = Path(student_path)
                if save_path.is_dir():
                    save_path = save_path.with_suffix(".pth")
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "model_config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": min(losses) if losses else 0.0,
                    },
                }, save_path)

                self._log(f"\nModel saved to {save_path.name}")
                best_loss = min(losses) if losses else 0.0
                if losses:
                    self._log(f"Best loss : {best_loss:.4f}")
                self._log("--- LoRA TRAINING COMPLETE ---")
                total = _time.monotonic() - _lora_start_common[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "LoRA", model_name, epochs, best_loss,
                    before_perplexity=ppl_before,
                    after_perplexity=ppl_after)
                self.after(0, lambda pc=pc: self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log("\n--- LoRA TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(f"\n[!] LoRA training failed: {exc}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self._active_trainer = None
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled", text="STOP"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_lora_train, daemon=True).start()

