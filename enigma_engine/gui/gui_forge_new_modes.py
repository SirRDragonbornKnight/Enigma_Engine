"""
Enigma Engine - Forge New Training Modes
==========================================

Training mode implementations: Pre-Train, Distill, RLHF, Self-Play.
Split into its own mixin to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path



logger = logging.getLogger(__name__)


class ForgeNewModesMixin:
    """Training mode implementations: Pre-Train, Distill, RLHF, Self-Play.

    Expects the host class to have ForgeMixin setup attributes.
    """

    def _forge_stop_requested(self) -> bool:
        """Check if the user requested training to stop."""
        return not self.training_active

    def _read_forge_rl_params(self) -> dict:
        """Read RL-specific parameters from FORGE UI widgets.

        Returns a dict with replay_capacity and replay_ratio,
        falling back to defaults when widgets are missing or invalid.
        """
        replay_capacity = 256
        replay_ratio = 0.25

        try:
            raw = getattr(
                self, "forge_replay_capacity_var", None).get().strip()
            val = int(raw)
            if val >= 0:
                replay_capacity = val
            else:
                self._log(f"[!] Replay capacity '{raw}' invalid "
                          f"(must be >= 0), using 256")
        except (ValueError, TypeError):
            raw_w = getattr(
                self, "forge_replay_capacity_var", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Replay capacity '{raw}' not a "
                          f"number, using 256")
        except AttributeError:
            pass

        try:
            raw = getattr(
                self, "forge_replay_ratio_var", None).get().strip()
            val = float(raw)
            if 0.0 <= val <= 1.0:
                replay_ratio = val
            else:
                self._log(f"[!] Replay ratio '{raw}' out of range "
                          f"(must be 0.0-1.0), using 0.25")
        except (ValueError, TypeError):
            raw_w = getattr(
                self, "forge_replay_ratio_var", None)
            raw = raw_w.get().strip() if raw_w else ""
            if raw:
                self._log(f"[!] Replay ratio '{raw}' not a "
                          f"number, using 0.25")
        except AttributeError:
            pass

        return {
            "replay_capacity": replay_capacity,
            "replay_ratio": replay_ratio,
        }

    # ================================================================
    # PRE-TRAINING (Phase 1a)
    # ================================================================

    def _pretrain_validate_inputs(self) -> dict | None:
        """Validate inputs for pre-training.

        Requires a STUDENT model assigned in the Router.
        Create models in the Models tab first.

        Returns a dict of validated params, or None on failure
        (with error logged to the FORGE panel).
        """
        if self.training_active:
            return None

        # STUDENT model is required - create in Models tab first
        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No STUDENT model assigned.\n"
                "    1. Go to MODELS and create a model\n"
                "    2. Go to ROUTER and assign it to STUDENT\n"
                "    3. Return here to pre-train it")
            return None

        data_var = getattr(self, "pretrain_data_var", None)
        data_path = data_var.get().strip() if data_var else ""

        retrain_tok_var = getattr(
            self, "pretrain_retrain_tok_var", None)
        retrain_tok = (retrain_tok_var.get()
                       if retrain_tok_var else True)

        vocab_var = getattr(self, "pretrain_vocab_var", None)
        try:
            vocab_size = (int(vocab_var.get())
                          if vocab_var else 32000)
            if vocab_size < 256 or vocab_size > 100000:
                raise ValueError
        except (ValueError, TypeError):
            self._log("[!] Vocab size must be 256-100000.")
            return None

        if not data_path:
            self._log(
                "[!] No pre-training data selected.\n"
                "    Browse for a text file or directory.")
            return None
        data_source = Path(data_path)
        if not data_source.exists():
            self._log(
                f"[!] Data path not found: {data_path}\n"
                "    Browse for a valid file or directory.")
            return None

        result = self._validate_epochs_lr()
        if result is None:
            return None
        epochs, lr = result

        return {
            "student_path": student_path,
            "data_source": data_source,
            "retrain_tok": retrain_tok,
            "vocab_size": vocab_size,
            "epochs": epochs,
            "lr": lr,
        }

    def _start_pretrain_training(self):
        """Pre-train a model on large text data.

        Requires a STUDENT model assigned in the Router.
        Create models in the Models tab first.

        Workflow:
        1. Load STUDENT model from Router
        2. Optionally retrain BPE tokenizer on the data
        3. Process and clean the pre-training corpus
        4. Run standard causal LM training with pre-training defaults

        Pre-training uses higher LR, no general mix (this IS the
        general knowledge), and longer warmup than fine-tuning.
        """
        if bool(getattr(self, "use_api_chat", False)):
            self._log("[!] API routing not yet implemented for Pre-Train mode — running locally on this machine.\n")
        params = self._pretrain_validate_inputs()
        if params is None:
            return

        student_path = params["student_path"]
        data_source = params["data_source"]
        retrain_tok = params["retrain_tok"]
        vocab_size = params["vocab_size"]
        epochs = params["epochs"]
        lr = params["lr"]

        out_path = Path(student_path)
        safe_name = out_path.stem

        # Check for stale heartbeat from a previously interrupted session.
        # Heartbeat exists + status not clean + PID is dead = OOM/crash kill.
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
                            "\n[!] PREVIOUS SESSION WAS INTERRUPTED "
                            "UNEXPECTEDLY\n"
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

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="PRE-TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 PRE-TRAINING...")

        self._log_training_summary(
            "Pre-Training",
            Model=safe_name,
            Data=str(data_source),
            Vocab=vocab_size,
            Epochs=epochs,
            LR=lr,
        )
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._log(f"Vocab   : {vocab_size}  |  Retrain tok: {retrain_tok}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _pretrain():
            losses = []
            import math as _math
            import os as _os_hb
            import json as _json_hb
            import datetime as _dt_hb
            _hb_path = Path("logs") / "training_heartbeat.json"
            _hb_path.parent.mkdir(exist_ok=True)
            _last_hb_t = [0.0]

            def _write_hb(phase="data_load", step=None,
                          loss=None, status="running"):
                """Write a heartbeat file so silent kills are detectable."""
                try:
                    hb = {
                        "pid": _os_hb.getpid(),
                        "status": status,
                        "model": safe_name,
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
                from enigma_engine.training.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.dataset import (
                    iter_text_chunks)

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # I-12/I-14: Check for existing checkpoints to resume
                ckpt_dir = (
                    out_path.parent / "checkpoints"
                    / out_path.stem)
                resume_enabled = getattr(
                    self, 'forge_resume_var', None)
                try_resume = (resume_enabled is None
                              or resume_enabled.get())
                resume_path = (
                    Trainer._find_latest_checkpoint(ckpt_dir)
                    if try_resume else None)
                do_retrain_tok = retrain_tok
                if resume_path is not None:
                    self._log(
                        f"\n--- RESUMING from {resume_path.name} ---")
                    self._log(
                        "Skipping tokenizer retraining (using "
                        "tokenizer from checkpoint)")
                    self._log(
                        "Training will continue from the saved "
                        "step/epoch.")
                    self.after(0, lambda:
                        self.solo_train_btn.configure(
                            text="RESUMING..."))
                    do_retrain_tok = False
                elif not try_resume:
                    existing = Trainer._find_latest_checkpoint(
                        ckpt_dir)
                    if existing is not None:
                        self._log(
                            "[i] Checkpoint exists but Resume is "
                            "unchecked - starting fresh.")

                # Step 1: Process pre-training data (streaming)
                #
                # Instead of loading all data into RAM at once,
                # stream through chunks one at a time.  Two passes:
                #   Pass 1: Count chars + collect tokenizer samples
                #   Pass 2: Split into sequences + write to JSONL
                # Peak RAM: ~200 MB (one chunk) + ~2 GB (tok samples)
                self._log("Scanning data...")
                import time as _time
                import json as _json

                def _ram_str():
                    try:
                        import psutil
                        m = psutil.virtual_memory()
                        return (f"RAM: {m.used / 1073741824:.1f}/"
                                f"{m.total / 1073741824:.0f} GB "
                                f"({m.percent}%)")
                    except Exception:
                        return ""

                _phase_t0 = _time.monotonic()
                _ram_warned = [False]

                def _load_progress(pct, msg):
                    self._log(f"  [{pct:>3d}%] {msg}")
                    # Warn once when RAM crosses 80% during load -
                    # this is the window where OOM kills happen.
                    if not _ram_warned[0] and pct >= 50:
                        try:
                            import psutil as _ps
                            _m = _ps.virtual_memory()
                            if _m.percent >= 80:
                                _avail = _m.available / 1_073_741_824
                                self._log(
                                    f"\n[!] RAM WARNING: {_m.percent:.0f}%"
                                    f" used ({_avail:.1f} GB free).\n"
                                    f"    OS may kill the process during"
                                    f" model init or torch.compile.\n"
                                    f"    Close other apps now.")
                                _ram_warned[0] = True
                        except ImportError:
                            pass

                # -- Pass 1: scan for total chars + tok samples --
                self._log("")
                self._log("=== Phase 1/5: Loading Data ===")
                self._log("  Reading the full dataset into memory.")
                self._log("  Large files (80+ GB) take 15-25 min.")
                self._log("  RAM usage will climb - this is normal.")
                self._log("")
                total_chars = 0
                calibration_sample = None
                tok_samples: list[str] = []
                _tok_sample_chars = 0
                try:
                    from enigma_engine.core.hardware_detection import (
                        TrainingMemoryBudget,
                    )
                    _TOK_SAMPLE_CAP = TrainingMemoryBudget().tok_sample_cap
                except Exception:
                    _TOK_SAMPLE_CAP = 2_000_000_000  # 2 GB fallback

                for chunk_text in iter_text_chunks(
                        data_source, text_key="text",
                        on_progress=_load_progress):
                    total_chars += len(chunk_text)
                    if calibration_sample is None:
                        calibration_sample = chunk_text[:10_000]
                    # Collect tokenizer training samples
                    if (do_retrain_tok
                            and _tok_sample_chars < _TOK_SAMPLE_CAP):
                        tok_samples.append(chunk_text)
                        _tok_sample_chars += len(chunk_text)

                if total_chars < 100:
                    self._log(
                        "[!] Not enough text data. Need at least "
                        "100 characters of clean text.")
                    return

                est_tokens = max(1, total_chars // 4)
                _elapsed = _time.monotonic() - _phase_t0
                self._log(
                    f"Data    : {total_chars:,} chars "
                    f"(~{est_tokens:,} tokens) "
                    f"[{_elapsed:.1f}s]")
                self._log(f"          {_ram_str()}")

                # -- C-2: Warn before tokenizer retrain destroys
                # existing weights (vocab size change makes all
                # embedding/output weights incompatible).
                if do_retrain_tok:
                    self._log(
                        "[i] Retraining tokenizer will change "
                        "vocab size. All model weights will be "
                        "randomly re-initialized (training from "
                        "scratch).")

                # Step 2: Optionally retrain tokenizer

                if do_retrain_tok:
                    _phase_t0 = _time.monotonic()
                    self._log("")
                    self._log("=== Phase 2/5: Training Tokenizer ===")
                    _sample_mb = _tok_sample_chars // 1_000_000
                    self._log(
                        f"  Building {vocab_size:,}-token BPE "
                        f"vocabulary from {_sample_mb:,} MB of text.")
                    self._log(
                        "  This is CPU-bound and can take "
                        "1-6 hours depending on data size.")
                    self._log(
                        "  The GUI may feel sluggish - "
                        "this is normal. Progress updates "
                        "every 100 merges.")
                    self._log(
                        "  The GPU is idle during this phase.")
                    self._log("")
                    self._log(
                        f"Training BPE tokenizer "
                        f"(vocab {vocab_size})...")
                    from enigma_engine.core.bpe_tokenizer import (
                        BPETokenizer)
                    tokenizer = BPETokenizer()
                    utf8_var = getattr(
                        self, 'pretrain_utf8_bytes_var', None)
                    if utf8_var and utf8_var.get():
                        tokenizer.use_utf8_bytes = True
                        self._log("Byte-level BPE enabled")

                    # Split tok_samples into 500K-char pieces
                    # for BPE training.
                    chunk_size = 500_000
                    train_texts = []
                    self._log(
                        f"  Tokenizer samples: "
                        f"{_tok_sample_chars // 1_000_000:,} MB "
                        f"from {len(tok_samples)} chunks")
                    for tc in tok_samples:
                        start = 0
                        while start < len(tc):
                            end = min(
                                start + chunk_size, len(tc))
                            if end < len(tc):
                                ws = tc.rfind(
                                    ' ', start, end)
                                nl = tc.rfind(
                                    '\n', start, end)
                                boundary = max(ws, nl)
                                if boundary > start:
                                    end = boundary + 1
                            train_texts.append(
                                tc[start:end])
                            start = end
                    del tok_samples

                    def _tok_progress(pct, msg):
                        self._log(f"  Tokenizer [{pct:>3d}%] {msg}")

                    tokenizer.train(
                        train_texts,
                        vocab_size=vocab_size,
                        verbose=False,
                        on_progress=_tok_progress)
                    del train_texts
                    # Save tokenizer
                    tok_dir = (Path(__file__).parent.parent
                               / "vocab_model")
                    tok_dir.mkdir(exist_ok=True)
                    tokenizer.save(tok_dir / "tokenizer.json")
                    # Also save to checkpoint dir so resume can
                    # recover the tokenizer after a crash
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    tokenizer.save(ckpt_dir / "tokenizer.json")
                    self._log(
                        f"Tokenizer trained: "
                        f"{tokenizer.vocab_size} tokens "
                        f"[{_time.monotonic() - _phase_t0:.1f}s]")
                else:
                    # Not retraining - free the tok samples
                    del tok_samples
                    # Try to load tokenizer from checkpoint dir
                    # first (resume), then bundled in student
                    # checkpoint (C-5), then auto-detection.
                    tok_loaded = False
                    _ckpt_tok_path = ckpt_dir / "tokenizer.json"
                    if _ckpt_tok_path.exists():
                        try:
                            from enigma_engine.core.bpe_tokenizer \
                                import BPETokenizer as _BPE
                            tokenizer = _BPE()
                            tokenizer.load(_ckpt_tok_path)
                            tok_loaded = True
                            self._log(
                                f"Tokenizer: BPETokenizer "
                                f"(from checkpoint dir, vocab "
                                f"{tokenizer.vocab_size})")
                        except Exception:
                            pass
                    if not tok_loaded:
                        try:
                            from enigma_engine.core.model_registry \
                                import safe_load_weights as _slw
                            _ckpt = _slw(
                                student_path, map_location="cpu")
                            tok_data = _ckpt.get("tokenizer_data")
                            if tok_data and isinstance(
                                    tok_data, dict):
                                from enigma_engine.core \
                                    .bpe_tokenizer import (
                                    BPETokenizer)
                                tokenizer = BPETokenizer()
                                tokenizer.token_to_id = (
                                    tok_data["token_to_id"])
                                tokenizer.id_to_token = {
                                    v: k for k, v
                                    in tokenizer.token_to_id
                                    .items()}
                                tokenizer.merges = [
                                    tuple(m)
                                    for m in tok_data["merges"]]
                                tokenizer.merge_ranks = {
                                    tuple(m): i for i, m
                                    in enumerate(
                                        tokenizer.merges)}
                                tokenizer.special_tokens = (
                                    tok_data.get(
                                        "special_tokens",
                                        tokenizer
                                        .special_tokens))
                                tokenizer._sync_special_ids()
                                tokenizer.use_utf8_bytes = (
                                    tok_data.get(
                                        "use_utf8_bytes",
                                        False))
                                tokenizer.vocab_size = len(
                                    tokenizer.token_to_id)
                                tok_loaded = True
                                self._log(
                                    f"Tokenizer: BPETokenizer "
                                    f"(from checkpoint, vocab "
                                    f"{tokenizer.vocab_size})")
                            del _ckpt
                        except Exception:
                            pass
                    if not tok_loaded:
                        from enigma_engine.gui.scanners import (
                            MODELS_DIR as _MODELS_DIR)
                        _bpe_path = _MODELS_DIR / "tokenizer.json"
                        if _bpe_path.exists():
                            try:
                                from enigma_engine.core.bpe_tokenizer import BPETokenizer
                                tokenizer = BPETokenizer(_bpe_path)
                                tok_loaded = True
                            except Exception:
                                pass
                        if not tok_loaded:
                            tokenizer = get_tokenizer("auto")
                        self._log(
                            f"Tokenizer: "
                            f"{type(tokenizer).__name__} "
                            f"(vocab {tokenizer.vocab_size})")

                # Step 3: Load STUDENT model
                _phase_t0 = _time.monotonic()
                self._log("")
                self._log("=== Phase 3/5: Loading Model ===")
                self._log(
                    "  Creating model architecture and loading "
                    "weights. Quick (under 1 min).")
                self._log("")
                self._log(
                    f"Loading STUDENT model: "
                    f"{Path(student_path).stem}")
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                from enigma_engine.core.model_presets import (
                    ForgeConfig)
                checkpoint = safe_load_weights(
                    student_path, map_location=device)
                cfg_dict = (checkpoint.get("model_config")
                            or checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
                if isinstance(cfg_dict, dict) and "dim" in cfg_dict:
                    cfg_dict["vocab_size"] = tokenizer.vocab_size
                    config = ForgeConfig(**{
                        k: v for k, v in cfg_dict.items()
                        if k in ForgeConfig.__dataclass_fields__})
                else:
                    self._log(
                        "[!] Could not read model config from "
                        "checkpoint. Re-create the model in "
                        "the MODELS tab.")
                    return
                model = Enigma(config=config)
                state_dict = get_state_dict(checkpoint)
                if state_dict:
                    try:
                        model.load_state_dict(
                            state_dict, strict=False)
                        self._log("Loaded existing weights")
                    except Exception as e:
                        self._log(
                            f"Fresh init (weights incompatible:"
                            f" {e})")
                model = model.to(device)

                pc = sum(p.numel() for p in model.parameters())
                self._log(
                    f"Params  : {pc:,} "
                    f"[{_time.monotonic() - _phase_t0:.1f}s]")
                self._log(
                    f"Arch    : dim={config.dim}, "
                    f"layers={config.n_layers}, "
                    f"heads={config.n_heads}, "
                    f"seq={config.max_seq_len}")
                self._log(f"          {_ram_str()}")

                # -- C-1: Data/model ratio guard.
                # Chinchilla scaling: need ~20 tokens per param.
                # Warn when data is insufficient and recommend a
                # preset that fits the available data.
                tokens_per_param = est_tokens / max(1, pc)
                if tokens_per_param < 1.0:
                    from enigma_engine.core.model_presets import (
                        recommend_preset_for_tokens,
                        MODEL_DESCRIPTIONS)
                    rec_name, rec_params = (
                        recommend_preset_for_tokens(
                            est_tokens, tokenizer.vocab_size))
                    rec_desc = MODEL_DESCRIPTIONS.get(
                        rec_name, "")
                    tpp = f"{tokens_per_param:.1f}"
                    self._log(
                        f"\n[!] WARNING: Only {tpp} tokens per "
                        f"parameter "
                        f"({est_tokens:,} tokens / {pc:,} "
                        f"params).\n"
                        f"    Chinchilla scaling recommends "
                        f"~20 tokens per param.\n"
                        f"    This model WILL memorize noise "
                        f"rather than learn language.\n"
                        f"    Recommended size for your data: "
                        f"'{rec_name}' (~{rec_params:,} params"
                        f") - {rec_desc}\n"
                        f"    Create a smaller model in the "
                        f"MODELS tab, or collect more data "
                        f"with: python "
                        f"collect_pretraining_data.py "
                        f"--all-sources\n")
                elif tokens_per_param < 5.0:
                    tpp = f"{tokens_per_param:.1f}"
                    self._log(
                        f"[i] Data is thin: {tpp} tokens/param "
                        f"(20 is optimal). "
                        f"Multi-epoch + BPE-dropout will help.")

                # Step 3b: Stream text into context-window-sized
                # sequences and write directly to JSONL on disk.
                # Peak RAM: one ~200 MB chunk at a time.
                max_seq = config.max_seq_len

                # -- C-4: Calibrate chars_per_token from the
                # actual tokenizer instead of guessing.
                if calibration_sample:
                    sample_ids = tokenizer.encode(
                        calibration_sample)
                    if len(sample_ids) > 0:
                        chars_per_token = (
                            len(calibration_sample)
                            / len(sample_ids))
                    else:
                        chars_per_token = 4.0
                else:
                    chars_per_token = 4.0
                del calibration_sample
                chars_per_chunk = max(
                    1000, int(max_seq * chars_per_token))
                self._log("")
                self._log("=== Phase 4/5: Streaming Sequences ===")
                self._log(
                    "  Splitting text into model-sized chunks "
                    "and writing to disk.")
                self._log(
                    "  Re-reads the full dataset. "
                    "Similar time to Phase 1.")
                self._log(
                    "  RAM stays stable - sequences go "
                    "straight to disk.")
                self._log("")
                self._log(
                    f"Streaming pass 2: splitting into "
                    f"~{max_seq}-token sequences "
                    f"({chars_per_token:.1f} chars/token)...")
                _phase_t0 = _time.monotonic()

                # Write sequences to JSONL on disk as we go
                _seq_path = (
                    out_path.parent / "checkpoints"
                    / out_path.stem
                    / "_pretrain_sequences.jsonl")
                _seq_path.parent.mkdir(
                    parents=True, exist_ok=True)
                _seq_offsets: list[int] = []
                _n_seqs = 0

                with open(_seq_path, "wb") as _sf:
                    for chunk_text in iter_text_chunks(
                            data_source, text_key="text",
                            on_progress=_load_progress):
                        pos = 0
                        tc_len = len(chunk_text)
                        while pos < tc_len:
                            end = min(
                                pos + chars_per_chunk, tc_len)
                            if end < tc_len:
                                boundary = chunk_text.rfind(
                                    ' ', pos, end)
                                nl = chunk_text.rfind(
                                    '\n', pos, end)
                                boundary = max(boundary, nl)
                                if boundary > pos:
                                    end = boundary + 1
                            seq = chunk_text[pos:end].strip()
                            if len(seq) > 50:
                                _seq_offsets.append(_sf.tell())
                                _sf.write(
                                    (_json.dumps(seq) + "\n")
                                    .encode("utf-8"))
                                _n_seqs += 1
                            pos = end

                _time.sleep(0)
                self._log(
                    f"Sequences: {_n_seqs:,} written to disk "
                    f"(~{chars_per_chunk} chars each) "
                    f"[{_time.monotonic() - _phase_t0:.1f}s]")
                self._log(f"          {_ram_str()}")

                if _n_seqs == 0:
                    self._log(
                        "[!] No sequences produced. Data may "
                        "be too short or empty.")
                    return

                # Step 4: Train with pre-training defaults
                forge_params = self._read_forge_train_params()

                # Use general mix to prevent catastrophic
                # forgetting on existing model weights.
                mix = forge_params["general_mix_ratio"]
                if mix == 0.0:
                    mix = 0.1  # Safe default for existing models
                    # Update the widget so user can see the override
                    mix_w = getattr(
                        self, "forge_general_mix_var", None)
                    if mix_w is not None:
                        self.after(0, lambda: mix_w.set("10"))
                    self._log(
                        "[!] General mix was 0% - raised to 10% "
                        "to prevent catastrophic forgetting")
                self._log(
                    f"General data mix: {mix:.0%} "
                    f"(prevents forgetting)")
                general_mix = mix
                general_data = forge_params["general_data"]

                # -- C-3: Auto-enable pre-training optimizations.
                # These have no downsides for pre-training and
                # significantly improve memory/throughput.

                # Warmup steps: ~1% of total steps is standard
                # for pre-training. Estimate from sequences.
                _est_total = max(
                    1,
                    (_n_seqs
                     // max(1, forge_params["batch_size"]))
                    * epochs)
                _warmup = max(10, _est_total // 100)

                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=forge_params["batch_size"],
                    learning_rate=lr,
                    max_grad_accumulation=forge_params[
                        "max_grad_accumulation"],
                    # Always enable for pre-training (large data)
                    use_gradient_checkpointing=True,
                    # Pack short sequences to eliminate padding
                    use_sequence_packing=True,
                    # Chunk CE to avoid huge logit tensors
                    ce_chunk_size=forge_params["ce_chunk_size"],
                    use_compile=True,
                    rolling_best_k=forge_params["rolling_best_k"],
                    # WSD (warmup-stable-decay) is optimal for
                    # pre-training - cosine wastes LR budget
                    schedule_type="wsd",
                    warmup_steps=_warmup,
                    general_mix_ratio=general_mix,
                    general_data=general_data,
                    val_split=forge_params["val_split"],
                    eval_every=max(100, _est_total // 20),
                    save_every=max(1, epochs // 5),
                    save_every_steps=max(500, _est_total // 20),
                    checkpoint_dir=str(
                        out_path.parent / "checkpoints"
                        / out_path.stem),
                    use_amp=torch.cuda.is_available(),
                    run_evaluation=True)

                self._log("")
                self._log("=== Phase 5/5: Training ===")
                self._log(
                    f"  {_n_seqs:,} sequences, {epochs} epoch(s).")
                self._log(
                    "  GPU will be fully utilized. "
                    "Loss chart updates each batch.")
                self._log(
                    "  Checkpoints save periodically - "
                    "safe to stop anytime.")
                self._log("")
                self._log("--- Auto-Optimizations ---")
                self._log(
                    "  GradCheckpoint: ON  |  SeqPacking: ON  "
                    "|  Compile: ON")
                self._log(
                    f"  CEChunk: 4096  |  Schedule: wsd  "
                    f"|  AMP: {'ON' if train_config.use_amp else 'OFF'}")
                self._log(
                    f"  Warmup: {_warmup} steps (~1% of "
                    f"{_est_total})")
                self._log(
                    f"  SaveEvery: "
                    f"{train_config.save_every_steps} steps"
                    f"  |  EvalEvery: "
                    f"{train_config.eval_every} steps"
                    f"  |  Eval: ON")

                trainer = Trainer(model, tokenizer, train_config)
                self._log(f"Batch   : {trainer.config.batch_size}")

                # --- Live training callbacks ---
                # Throttle: only log+update when percentage
                # changes to avoid flooding the GUI event queue.
                _last_pct = [-1]
                _last_progress_t = [0.0]

                def on_progress(pct, msg):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    # Update progress bar at most 1x/sec
                    if now - _last_progress_t[0] >= 1.0:
                        self._update_forge_progress(pct, msg)
                        _last_progress_t[0] = now
                trainer.on_progress = on_progress

                # Throughput: accumulate tokens for speed calc
                _throughput_tokens = [0]
                _throughput_time = [0.0]

                def on_throughput(tokens: int, step_time: float):
                    _throughput_tokens[0] += tokens
                    _throughput_time[0] += step_time
                trainer.on_throughput = on_throughput

                # Loss: compact step line with loss + tok/s + VRAM
                # Throttled to 2 updates/sec to avoid flooding
                # the GUI event queue (which freezes the window).
                _last_loss_t = [0.0]

                def on_loss(loss: float):
                    # Catch diverged training immediately
                    if _math.isnan(loss) or _math.isinf(loss):
                        label = "NaN" if _math.isnan(loss) else "Inf"
                        self._log(
                            f"\n[!] Loss is {label} - training has"
                            f" diverged.\n"
                            f"    Try: lower learning rate, check data"
                            f" for corrupt sequences, reduce batch size.")
                        _write_hb("training",
                                  step=trainer.state.step,
                                  status="crashed_nan")
                        return
                    now = _time.monotonic()
                    if now - _last_loss_t[0] < 0.5:
                        return
                    _last_loss_t[0] = now
                    # Heartbeat: write every 30s so a silent kill
                    # leaves a record of the last known step.
                    if now - _last_hb_t[0] >= 30.0:
                        _last_hb_t[0] = now
                        _write_hb("training",
                                  step=trainer.state.step,
                                  loss=loss)
                    tok_s = int(
                        _throughput_tokens[0]
                        / max(0.001, _throughput_time[0]))
                    _throughput_tokens[0] = 0
                    _throughput_time[0] = 0.0
                    vram = ""
                    if torch.cuda.is_available():
                        used_gb = (torch.cuda.memory_allocated()
                                   / 1e9)
                        total_gb = (
                            torch.cuda
                            .get_device_properties(0)
                            .total_memory / 1e9)
                        vram = (f" | {used_gb:.1f}/"
                                f"{total_gb:.0f} GB")
                    step = trainer.state.step
                    lr = trainer.optimizer.param_groups[0]['lr']
                    self._log(
                        f"  Step {step:>5d} | loss {loss:.4f}"
                        f" | lr {lr:.2e}"
                        f" | {tok_s:,} tok/s{vram}")
                trainer.on_loss = on_loss

                _train_start = [_time.monotonic()]

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
                    best = trainer.state.best_loss
                    import math as _math
                    best_str = (f"  |  best {best:.4f}"
                                if not _math.isinf(best) else "")
                    self._log(
                        f"  Epoch {epoch:>3d}/{epochs}  |  "
                        f"loss {loss:.4f}{trend}{best_str}  |  "
                        f"{mins}m {secs:02d}s{eta}")
                trainer.on_epoch_complete = on_epoch

                def on_warning(msg: str):
                    self._log(f"[!] WARNING: {msg}")
                trainer.on_warning = on_warning

                # -- Resource check: warn if system RAM is low
                # (silent OS kills leave no traceback)
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    # Warn when less than 10% of total RAM is free
                    _low_ram_bytes = max(
                        int(mem.total * 0.10),
                        2 * 1073741824,  # at least 2 GB absolute floor
                    )
                    if mem.available < _low_ram_bytes:
                        avail_gb = mem.available / 1073741824
                        self._log(
                            f"[!] WARNING: Only {avail_gb:.1f} "
                            f"GB RAM free. Training may be "
                            f"killed by the OS.\n"
                            f"    Close other apps or create a "
                            f"smaller model (lower Memory GB).")
                except ImportError:
                    pass

                _write_hb("training", step=0)
                self._log("Pre-training...\n")
                self._active_trainer = trainer
                state = trainer.train(
                    data_path=str(_seq_path),
                    data_offsets=_seq_offsets,
                    resume_from=str(resume_path)
                    if resume_path is not None else None)

                # Check for early termination
                import math
                if math.isinf(state.best_loss) and not losses:
                    reason = getattr(state, 'abort_reason', '') or (
                        "likely OOM or NaN loss")
                    self._log(
                        f"\n[!] Pre-training aborted - {reason}.\n"
                        "    Try a smaller model (lower Memory "
                        "GB in MODELS tab) or reduce "
                        "batch size.")
                    return

                # Log evaluation results
                ppl_before = None
                ppl_after = None
                if (hasattr(state, "before_eval")
                        and hasattr(state, "after_eval")):
                    before = state.before_eval
                    after = state.after_eval
                    ppl_before = before["perplexity"]
                    ppl_after = after["perplexity"]
                    self._log("\n--- EVALUATION ---")
                    self._log(
                        f"Perplexity: {ppl_after:.2f} "
                        f"(from {ppl_before:.2f})")

                # Save model
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)

                # -- C-5: Bundle tokenizer data in the checkpoint
                # so the model is self-contained. If the tokenizer
                # file is moved or the vocab_model dir cleared,
                # the checkpoint can reconstruct it.
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }
                # Save BPE tokenizer data if available
                from enigma_engine.core.bpe_tokenizer import (
                    BPETokenizer)
                if isinstance(tokenizer, BPETokenizer):
                    save_dict["tokenizer_data"] = {
                        "token_to_id": tokenizer.token_to_id,
                        "merges": tokenizer.merges,
                        "special_tokens": tokenizer.special_tokens,
                        "use_utf8_bytes": tokenizer.use_utf8_bytes,
                    }
                atomic_torch_save(save_dict, out_path)

                _write_hb("training",
                          step=getattr(state, 'step', None),
                          loss=state.best_loss,
                          status="complete")
                self._log("\n--- PRE-TRAINING COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Saved to  : {out_path}")
                total = _time.monotonic() - _train_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Pre-Train", safe_name, epochs,
                    state.best_loss,
                    before_perplexity=ppl_before,
                    after_perplexity=ppl_after)
                self.after(0, lambda pc=pc:
                           self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                _write_hb("training", status="stopped")
                self._log("\n--- PRE-TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except RuntimeError as exc:
                import traceback
                tb = traceback.format_exc()
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda" in msg:
                    _write_hb("training", status="crashed_oom")
                    self._log(
                        f"\n[!] GPU out of memory: {exc}\n"
                        "    Try: smaller model (lower Memory "
                        "GB in MODELS tab), reduce batch "
                        "size, or reduce sequence length.")
                else:
                    self._log(
                        f"\n[!] Pre-training failed: {exc}")
                self._log(tb)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                _write_hb("training", status="crashed")
                self._log(f"\n[!] Pre-training failed: {exc}")
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

        _pretrain_thread = threading.Thread(
            target=_pretrain, daemon=True)
        _pretrain_thread.start()

        # Watchdog: detect silent thread death (OS OOM kill)
        def _check_pretrain_alive():
            if not _pretrain_thread.is_alive() \
                    and self.training_active:
                self._log(
                    "\n[!] Training thread died unexpectedly "
                    "(process may have been killed by the OS "
                    "due to low memory).\n"
                    "    Close other programs and try again "
                    "with fewer resources in use.")
                self.training_active = False
                self._reset_forge_progress()
                self.solo_train_btn.configure(
                    state="normal", text="TRAIN")
                self.stop_train_btn.configure(
                    state="disabled", text="STOP")
                self.status_bar.set_left("\u26a1 READY")
            elif self.training_active:
                self.after(5000, _check_pretrain_alive)
        self.after(10000, _check_pretrain_alive)

    # ================================================================
    # DISTILLATION (Step 1b - Teacher -> Student)
    # ================================================================

    def _pre_training_backup(
        self,
        model_path: str,
        *,
        suffix: str = "pre_train",
    ) -> str | None:
        """P5-pre-2 helper (Pass 156z9ar): copy the on-disk model to
        ``models/checkpoints/{stem}_{suffix}_{ts}{ext}`` BEFORE any
        weight-mutating training begins, so the user has a one-step
        rollback if training drifts identity, destroys reasoning,
        or otherwise corrupts the model.

        Returns the backup path as a string on success, or ``None``
        when the source file does not exist yet (caller passed an
        unsaved path) or when the copy itself fails (loud `[!]` log
        but the run still proceeds - backup is a safety rail, not
        a precondition).

        ``suffix`` lets each entry point name its rollback files
        distinctly (``pre_distill``, ``pre_dialogue``, ``pre_dpo``,
        etc.) so a `models/checkpoints/` listing is self-explanatory.
        """
        try:
            src_path = Path(model_path)
            if not (src_path.exists() and src_path.is_file()):
                self._log(
                    f"Pre-{suffix} backup skipped: source file "
                    f"does not exist yet.")
                return None
            from datetime import datetime as _dt
            import shutil as _shutil
            from enigma_engine.gui.scanners import (
                MODELS_DIR as _MODELS_DIR)
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            ckpt_dir = _MODELS_DIR / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            backup_name = (
                f"{src_path.stem}_{suffix}_{_ts}{src_path.suffix}")
            backup_path = ckpt_dir / backup_name
            _shutil.copy2(src_path, backup_path)
            self._log(f"Pre-{suffix} backup: {backup_path.name}")
            return str(backup_path)
        except Exception as backup_exc:
            # Loud but non-fatal - the user knows the rollback rail
            # is missing for this run.
            self._log(
                f"[!] Pre-{suffix} backup FAILED: "
                f"{backup_exc}.  Run will proceed WITHOUT a "
                f"rollback checkpoint.")
            return None

    def _run_identity_probe(
        self,
        model,
        tokenizer,
        device: str,
        prompts: list[str],
        *,
        max_new_tokens: int = 64,
    ) -> dict:
        """P5-pre-3 (Pass 156z9aq): run identity-probe prompts against
        the student model and return ``{prompt: response_text}``.

        Used pre+post distillation to detect drift toward the teacher's
        identity (e.g. responses to "Who are you?" leaking "I am Qwen").

        Failure of any single probe is swallowed with an empty-string
        response so the surrounding training run never crashes on a
        diagnostic - the probe is observability, not flow control.
        """
        import torch as _torch
        results: dict[str, str] = {}
        was_training = model.training
        model.eval()
        try:
            with _torch.no_grad():
                for p in prompts:
                    try:
                        prompt_text = (
                            f"User: {p}\nAssistant:")
                        ids = tokenizer.encode(prompt_text)
                        input_ids = _torch.tensor(
                            [ids], dtype=_torch.long, device=device)
                        out = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.9,
                            repetition_penalty=1.1)
                        # ``generate`` may return a tensor or
                        # (tensor, logits) depending on flags; the
                        # default path returns a bare tensor.
                        gen_ids = (out[0] if isinstance(out, tuple)
                                   else out)
                        new_ids = gen_ids[0, input_ids.shape[1]:]
                        text = tokenizer.decode(new_ids.tolist())
                        results[p] = text.strip()
                    except Exception as probe_exc:
                        results[p] = ""
                        self._log(
                            f"  [!] Probe failed for "
                            f"{p!r}: {probe_exc}")
        finally:
            if was_training:
                model.train()
        return results

    def _distill_validate_inputs(self) -> dict | None:
        """Validate inputs for distillation training.

        Returns a dict of validated params, or None on failure
        (with error logged to the FORGE panel).
        """
        if self.training_active:
            return None

        trainer_path = self.route_assignments.get("trainer")
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Go to ROUTER and assign the teacher model\n"
                "    (e.g. Qwen3-8B GGUF).")
            return None

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return None

        # Gather selected categories
        cats = getattr(self, "distill_categories", {})
        selected = [k for k, v in cats.items() if v.get()]
        if not selected:
            self._log("[!] Select at least one distillation category.")
            return None

        # Num examples per category
        num_var = getattr(self, "distill_num_examples_var", None)
        try:
            num_examples = int(num_var.get()) if num_var else 50
            if num_examples < 1 or num_examples > 500:
                raise ValueError
        except (ValueError, TypeError):
            self._log("[!] Examples per category must be 1-500.")
            return None

        # Max tokens per teacher response
        tok_var = getattr(self, "distill_max_tokens_var", None)
        try:
            max_tokens = int(tok_var.get()) if tok_var else 512
            if max_tokens < 32 or max_tokens > 8192:
                raise ValueError
        except (ValueError, TypeError):
            self._log("[!] Max response length must be 32-8192.")
            return None

        result = self._validate_epochs_lr()
        if result is None:
            return None
        epochs, lr = result

        return {
            "trainer_path": trainer_path,
            "student_path": student_path,
            "categories": selected,
            "num_examples": num_examples,
            "max_tokens": max_tokens,
            "epochs": epochs,
            "lr": lr,
        }

    def _start_distill_training(self):
        """Distillation: TRAINER generates targeted data, STUDENT trains.

        Workflow:
        1. Load TRAINER (teacher) as engine, STUDENT as model
        2. For each selected category, generate training examples
           by prompting the teacher with category-specific prompts
        3. Collect all generated data into a training corpus
        4. Train the student on the distilled data
        5. Save the improved student model

        This is Step 1b of the Emotional AI roadmap:
        teacher (Qwen3-8B) generates personality, reasoning,
        conversation styles that the student then learns.
        """
        if bool(getattr(self, "use_api_chat", False)):
            self._log("[!] API routing not yet implemented for Distill mode — running locally on this machine.\n")
        params = self._distill_validate_inputs()
        if params is None:
            return

        trainer_path = params["trainer_path"]
        student_path = params["student_path"]
        categories = params["categories"]
        num_examples = params["num_examples"]
        max_tokens = params["max_tokens"]
        epochs = params["epochs"]
        lr = params["lr"]

        student_name = Path(student_path).stem
        trainer_name = Path(trainer_path).stem

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="DISTILLING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 DISTILLING...")

        self._log_training_summary(
            "Knowledge Distillation",
            Teacher=trainer_name,
            Student=student_name,
            Categories=", ".join(categories),
            Examples=f"{num_examples} x {len(categories)} = "
                     f"{num_examples * len(categories)}",
            Epochs=epochs,
            LR=lr,
        )
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # Category-specific prompts for the teacher.
        # P5-pre-1 (Pass 156z9am): the ``personality`` pool moved to
        # :mod:`enigma_engine.core.personality_data` (50 prompts across
        # 10 themes) so it can be unit-tested without the GUI and so
        # quality/identity filters can be applied to teacher outputs.
        from enigma_engine.core.personality_data import (
            PERSONALITY_PROMPTS as _PERSONALITY_PROMPTS,
        )
        category_prompts = {
            "personality": list(_PERSONALITY_PROMPTS),
            "reasoning": [
                "Solve this step by step: If a train travels 60 mph "
                "for 2.5 hours, how far does it go?",
                "Compare the pros and cons of working from home "
                "versus working in an office.",
                "Explain why the sky appears blue using simple terms.",
                "A recipe calls for 3 eggs for 4 servings. How many "
                "eggs are needed for 10 servings?",
                "What would happen if the Earth suddenly had no moon? "
                "Think through the consequences.",
            ],
            "knowledge": [
                "Explain how photosynthesis works in simple terms.",
                "What are the main differences between Python "
                "and JavaScript?",
                "Describe the water cycle and why it matters.",
                "What is machine learning and how does it work?",
                "Explain the basics of how the internet works.",
            ],
            "conversation": [
                # Pass 156z9ao audit (F-B): rewrote from raw
                # ``User: ...\nAssistant:`` prefixed prompts to direct
                # imperatives.  The distill loop wraps each prompt as
                # ``f"User: {prompt}\nAssistant: {response}"`` so the
                # old form double-wrapped into malformed training data
                # (``User: User: Hey...\nAssistant:\nAssistant: ...``).
                "Respond casually to a friend saying 'hey, what's up?'",
                "A friend asks for ideas about a new hobby they "
                "could pick up. Suggest something with personality.",
                "Someone asks 'tell me something interesting you've "
                "learned recently.' Pick something genuine.",
                "A friend just got a new puppy and is excited. "
                "Match their energy in the response.",
                "A friend asks for help planning a weekend trip. "
                "Reply naturally with two or three concrete ideas.",
            ],
            "commands": [
                "The user wants to find all Python files in their "
                "project. Show how to use [CMD]search.files *.py[/CMD] "
                "and explain the results.",
                "The user wants to save a note. Show how to use "
                "[CMD]note.add Remember to update docs[/CMD] and "
                "confirm the action.",
                "The user asks what model is loaded. Use "
                "[CMD]model.info[/CMD] and explain what it shows.",
                "The user wants to check system resources. Use "
                "[CMD]system.info[/CMD] and summarize.",
                "The user wants to read a file. Show how to use "
                "[CMD]file.read config.txt[/CMD] appropriately.",
            ],
            "creativity": [
                "Write a very short story (3-5 sentences) about "
                "a robot discovering music for the first time.",
                "Create a haiku about artificial intelligence.",
                "Describe a sunset over the ocean in vivid, "
                "poetic language.",
                "Write a brief dialogue between a cat and a dog "
                "who are unlikely friends.",
                "Invent a creative name and description for a "
                "fictional cafe that serves unusual drinks.",
            ],
        }

        def _distill():
            losses = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.training.training import (
                    Trainer, TrainingConfig)

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Step 1: Load teacher engine
                self._log("\n--- Loading Models ---")
                teacher_engine = self._load_engine_for_path(
                    trainer_path)
                self._log(f"Teacher loaded: {trainer_name}")

                # Load student model
                self._log(f"Loading student: {student_name}...")
                s_ckpt = safe_load_weights(
                    student_path, map_location="cpu")
                s_cfg_dict = (s_ckpt.get("model_config")
                              or s_ckpt.get("config", {}))
                if isinstance(s_cfg_dict, dict):
                    if "epochs" in s_cfg_dict:
                        s_cfg_dict = s_ckpt.get("model_config", {})
                s_cfg = ForgeConfig(**{
                    k: v for k, v in s_cfg_dict.items()
                    if k in ForgeConfig.__dataclass_fields__
                })
                student = Enigma(config=s_cfg).to(device)
                student.load_state_dict(
                    get_state_dict(s_ckpt), strict=False)

                s_params = sum(
                    p.numel() for p in student.parameters())
                self._log(f"Student : {s_params:,} params")

                from enigma_engine.gui.scanners import (
                    MODELS_DIR as _MODELS_DIR)
                _bpe_path = _MODELS_DIR / "tokenizer.json"
                if _bpe_path.exists():
                    try:
                        from enigma_engine.core.bpe_tokenizer import BPETokenizer
                        tokenizer = BPETokenizer(_bpe_path)
                    except Exception:
                        tokenizer = get_tokenizer("auto")
                else:
                    tokenizer = get_tokenizer("auto")
                if tokenizer.vocab_size != s_cfg.vocab_size:
                    self._log(
                        f"  [!] Tokenizer vocab ({tokenizer.vocab_size}) "
                        f"!= model vocab ({s_cfg.vocab_size})")
                    if tokenizer.vocab_size > s_cfg.vocab_size:
                        raise ValueError(
                            f"Tokenizer vocab ({tokenizer.vocab_size}) exceeds "
                            f"model vocab ({s_cfg.vocab_size}) - token IDs "
                            f"will be out of range. Use a matching tokenizer.")
                self._log(
                    f"Tokenizer: {type(tokenizer).__name__} "
                    f"(vocab {tokenizer.vocab_size})")

                # Get training brief for personality context
                brief = ""
                brief_fields = getattr(
                    self, "_brief_field_entries", {})
                for entry in brief_fields.values():
                    val = entry.get().strip() if hasattr(
                        entry, "get") else ""
                    if val:
                        brief += val + " "
                custom_tb = getattr(
                    self, "_brief_custom_text", None)
                if custom_tb is not None:
                    try:
                        custom = custom_tb.get(
                            "1.0", "end-1c").strip()
                        if custom:
                            brief += custom
                    except Exception:
                        pass
                brief = brief.strip()

                # Build teacher system prompt with student context
                teacher_sys = self._build_trainer_system_prompt(
                    student_params=s_params,
                    student_cfg=s_cfg,
                    task="generate",
                    stage="conversation",
                    training_brief=brief or None,
                )

                # Include reasoning if enabled
                reasoning = getattr(
                    self, "forge_reasoning_var", None)
                use_reasoning = (reasoning.get()
                                 if reasoning else False)
                if use_reasoning:
                    teacher_sys += (
                        "\n\nIMPORTANT: Include reasoning in your "
                        "responses. Show your thinking process inside "
                        "<think>...</think> tags before giving your "
                        "answer.")

                # Step 2: Generate distillation data
                self._log("\n--- Generating Distillation Data ---")
                all_examples = []
                total_to_gen = num_examples * len(categories)
                generated = 0
                # P5-pre-1: per-category reject counters for the
                # personality filter (identity / quality / duplicate).
                # Other categories don't filter today; counters stay 0.
                personality_reject_counts = {
                    "identity": 0,
                    "quality": 0,
                    "duplicate": 0,
                }
                # Track accepted personality responses for near-dup
                # detection within this run.
                personality_accepted_responses: list[str] = []

                for cat in categories:
                    if not self.training_active:
                        break

                    seed_prompts = category_prompts.get(cat, [])
                    if not seed_prompts:
                        continue

                    self._log(f"\nCategory: {cat} "
                              f"({num_examples} examples)")

                    for i in range(num_examples):
                        if not self.training_active:
                            break

                        # Cycle through seed prompts
                        seed = seed_prompts[i % len(seed_prompts)]

                        # Vary the prompt for diversity
                        if i >= len(seed_prompts):
                            prompt = (
                                f"Generate a unique {cat} example "
                                f"(variation {i + 1}). "
                                f"Similar style to: {seed}")
                        else:
                            prompt = seed

                        try:
                            response = teacher_engine.chat(
                                prompt,
                                system_prompt=teacher_sys,
                                max_tokens=max_tokens,
                                temperature=0.8,
                            )
                            clean_response = response.strip() if response else ""
                            # P5-pre-1: personality category gets
                            # identity-leak + quality + dedup filters.
                            # Other categories keep the legacy 20-char
                            # minimum (out of scope this slice).
                            if cat == "personality":
                                from enigma_engine.core.personality_data import (
                                    is_near_duplicate,
                                    passes_identity_filter,
                                    passes_quality_filter,
                                )
                                # Pass 156z9ao (F-A audit): empty
                                # response buckets as ``quality``
                                # rather than silently logging
                                # "too short: 0 chars" with no
                                # counter increment.  Keeps the
                                # personality_reject_counts total
                                # equal to the visible reject log.
                                accept = True
                                reject_reason = ""
                                if not clean_response:
                                    accept = False
                                    reject_reason = "quality"
                                    personality_reject_counts[
                                        "quality"] += 1
                                if accept and not passes_identity_filter(
                                        clean_response):
                                    accept = False
                                    reject_reason = "identity-leak"
                                    personality_reject_counts[
                                        "identity"] += 1
                                if accept and not passes_quality_filter(
                                        clean_response):
                                    accept = False
                                    reject_reason = "quality"
                                    personality_reject_counts[
                                        "quality"] += 1
                                if accept and is_near_duplicate(
                                        clean_response,
                                        personality_accepted_responses):
                                    accept = False
                                    reject_reason = "duplicate"
                                    personality_reject_counts[
                                        "duplicate"] += 1
                                if accept:
                                    personality_accepted_responses.append(
                                        clean_response)
                            else:
                                accept = bool(clean_response) and len(
                                    clean_response) > 20
                                reject_reason = (
                                    "too-short" if not accept else "")

                            if accept:
                                # Format as training data
                                example = (
                                    f"User: {prompt}\n"
                                    f"Assistant: {clean_response}")
                                all_examples.append(example)
                                generated += 1

                                # Log full accepted example so the
                                # Forge panel shows both conversation sides.
                                self._log(
                                    f"  [{generated}/{total_to_gen}] "
                                    f"User: {prompt}\n"
                                    f"Assistant: {clean_response}")
                            else:
                                short_len = len(clean_response)
                                # P5-pre-1: surface filter reason in
                                # log so the user can tell teacher
                                # output drift apart from short noise.
                                if reject_reason and reject_reason != "too-short":
                                    self._log(
                                        f"  [{generated}/{total_to_gen}] "
                                        f"Skipped ({reject_reason}: "
                                        f"{short_len} chars)")
                                else:
                                    self._log(
                                        f"  [{generated}/{total_to_gen}] "
                                        f"Skipped (too short: "
                                        f"{short_len} chars)")

                        except Exception as exc:
                            self._log(
                                f"  Generation error: {exc}")
                            continue

                        pct = int(generated / total_to_gen * 50)
                        self._update_forge_progress(
                            pct,
                            f"Generating {generated}/{total_to_gen}")

                if not self.training_active:
                    self._log("\n--- DISTILLATION STOPPED ---")
                    return

                if not all_examples:
                    self._log(
                        "\n[!] No distillation data generated.\n"
                        "    Check that the TRAINER model is loaded "
                        "and responsive.")
                    return

                self._log(
                    f"\nGenerated {len(all_examples)} training examples")

                # P5-pre-1: surface personality filter rejections so
                # the user knows when teacher drift is high.
                if "personality" in categories:
                    pr = personality_reject_counts
                    total_rejected = (
                        pr["identity"] + pr["quality"] + pr["duplicate"])
                    if total_rejected > 0:
                        self._log(
                            f"  Personality filters rejected "
                            f"{total_rejected} response(s): "
                            f"identity={pr['identity']}, "
                            f"quality={pr['quality']}, "
                            f"duplicate={pr['duplicate']}")

                # Personality-5 BUILD: quick-profile fields already
                # steer the teacher prompt, but without direct student
                # examples the requested voice survives only indirectly
                # through teacher generations. Add a small deterministic
                # profile-scoped example set on the same SFT path.
                if "personality" in categories:
                    from enigma_engine.core.personality_data import (
                        build_profile_consistency_examples,
                    )
                    profile_fields = {}
                    for label, entry in brief_fields.items():
                        value = entry.get().strip() if hasattr(entry, "get") else ""
                        if value:
                            profile_fields[label] = value
                    profile_examples = build_profile_consistency_examples(
                        profile_fields,
                        student_name=student_name,
                    )
                    if profile_examples:
                        all_examples.extend(profile_examples)
                        self._log(
                            f"  Personality profile anchors: +{len(profile_examples)} example(s)")

                # Save generated data for reference
                from enigma_engine.gui.scanners import (
                    DATA_DIR, MODELS_DIR)
                data_file = DATA_DIR / f"distilled_{student_name}.txt"
                training_text = "\n\n".join(all_examples)
                from enigma_engine.core.safe_save import atomic_write_text
                atomic_write_text(data_file, training_text)
                self._log(f"Data saved: {data_file.name}")

                # Step 3: Train student on distilled data
                self._log("\n--- Training Student ---")
                forge_params = self._read_forge_train_params()

                # P5-pre-2: pre-SFT auto-checkpoint (Pass 156z9ap,
                # refactored Pass 156z9ar to use the shared
                # ``_pre_training_backup`` helper).  Distillation
                # overwrites ``student_path`` in place; without a
                # backup, drift toward teacher identity or loss of
                # general capability has no rollback.
                pre_distill_backup_path = self._pre_training_backup(
                    student_path, suffix="pre_distill")

                # P5-pre-2: anchor-mix gate.  When the personality
                # category is selected we mix curated general
                # examples (math/code/knowledge - see
                # ``data/anchor_examples.jsonl``) into the SFT batch
                # at a default 30% ratio to mitigate catastrophic
                # forgetting of base skills.  Other distill
                # categories keep ``general_mix_ratio=0`` (status
                # quo); they don't have the same identity-drift
                # pressure.  When the anchor file resolves to None
                # (no override + no repo default) the mix is
                # silently disabled and the user is informed.
                _mix_ratio = 0.0
                _mix_data: str | None = None
                if "personality" in categories:
                    try:
                        from enigma_engine.gui.scanners import (
                            _resolve_anchor_path,
                        )
                        _saved = self._read_gui_str_setting(
                            "anchor_data_path", "")
                        _anchor = _resolve_anchor_path(_saved)
                        if _anchor is not None and _anchor.exists():
                            _mix_ratio = 0.3
                            _mix_data = str(_anchor)
                            self._log(
                                f"Personality anchor mix: "
                                f"{_anchor.name} "
                                f"(ratio {int(_mix_ratio*100)}%)")
                        else:
                            self._log(
                                "Personality anchor mix: "
                                "DISABLED (no anchor file). "
                                "Catastrophic-forgetting risk "
                                "is HIGHER for this run.")
                    except Exception as anchor_exc:
                        self._log(
                            f"[!] Anchor mix resolution failed: "
                            f"{anchor_exc}.  Continuing without "
                            f"mix.")

                # P5-pre-3: identity-guard probe (pre-training).
                # Only fires when the personality category is in
                # play - that's the only category that can drift
                # student identity toward the teacher.  Stored for
                # post-training comparison after `trainer.train`
                # returns.  Probe failures are non-fatal.
                pre_probe_responses: dict | None = None
                if "personality" in categories:
                    try:
                        from enigma_engine.core.personality_data import (
                            IDENTITY_PROBE_PROMPTS,
                        )
                        self._log(
                            "Identity probe (pre-training)...")
                        pre_probe_responses = (
                            self._run_identity_probe(
                                student, tokenizer, device,
                                list(IDENTITY_PROBE_PROMPTS)))
                        # Surface unsafe pre-probe lines so the user
                        # knows the BASELINE leak rate before any
                        # training has occurred.
                        from enigma_engine.core.personality_data import (
                            passes_identity_filter,
                        )
                        leaked_pre = [
                            p for p, r in pre_probe_responses.items()
                            if not passes_identity_filter(r)]
                        if leaked_pre:
                            self._log(
                                f"  Pre-probe identity leaks: "
                                f"{len(leaked_pre)}/"
                                f"{len(pre_probe_responses)} prompts")
                    except Exception as probe_exc:
                        self._log(
                            f"[!] Pre-probe failed: {probe_exc}.  "
                            f"Post-probe drift comparison will be "
                            f"skipped.")
                        pre_probe_responses = None

                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=forge_params["batch_size"],
                    learning_rate=lr,
                    max_grad_accumulation=forge_params[
                        "max_grad_accumulation"],
                    use_gradient_checkpointing=forge_params[
                        "use_gradient_checkpointing"],
                    use_sequence_packing=True,
                    ce_chunk_size=forge_params["ce_chunk_size"],
                    use_compile=True,
                    rolling_best_k=forge_params["rolling_best_k"],
                    general_mix_ratio=_mix_ratio,
                    general_data=_mix_data,
                    val_split=forge_params["val_split"],
                    save_every=max(1, epochs // 5),
                    checkpoint_dir=str(
                        MODELS_DIR / "checkpoints"),
                    use_amp=torch.cuda.is_available(),
                    run_evaluation=True)

                trainer = Trainer(student, tokenizer, train_config)

                import time as _time_d
                _distill_start = [_time_d.monotonic()]

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    losses.append(loss)
                    pct = 50 + int(epoch / epochs * 50)
                    self._update_forge_progress(
                        pct, f"Training {epoch}/{epochs}")
                    elapsed = _time_d.monotonic() - _distill_start[0]
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    eta = ""
                    if epoch > 0:
                        remaining = (elapsed / epoch) * (
                            epochs - epoch)
                        r_m = int(remaining // 60)
                        r_s = int(remaining % 60)
                        eta = f"  |  ETA {r_m}m {r_s:02d}s"
                    self._log(
                        f"  Epoch {epoch:>3d}/{epochs}  |  "
                        f"loss {loss:.4f}  |  "
                        f"{mins}m {secs:02d}s{eta}")
                trainer.on_epoch_complete = on_epoch

                def on_warning_solo(msg: str):
                    self._log(f"[!] WARNING: {msg}")
                trainer.on_warning = on_warning_solo

                self._log(f"Training on {len(all_examples)} "
                          f"examples for {epochs} epochs...\n")
                self._active_trainer = trainer
                state = trainer.train(training_text)

                # Check for failure
                import math
                if math.isinf(state.best_loss) and not losses:
                    reason = getattr(state, 'abort_reason', '') or (
                        "likely OOM or NaN loss")
                    self._log(
                        f"\n[!] Training aborted - {reason}.\n"
                        "    Try reducing batch size.")
                    return

                # P5-pre-3: identity-guard probe (post-training).
                # Compare against `pre_probe_responses` and flag any
                # prompts that drifted from safe ? leaking.  This is
                # the regression signal personality SFT must avoid.
                if (pre_probe_responses is not None
                        and "personality" in categories):
                    try:
                        from enigma_engine.core.personality_data import (
                            IDENTITY_PROBE_PROMPTS,
                            summarize_identity_probe,
                        )
                        self._log(
                            "Identity probe (post-training)...")
                        post_probe_responses = (
                            self._run_identity_probe(
                                student, tokenizer, device,
                                list(IDENTITY_PROBE_PROMPTS)))
                        summary = summarize_identity_probe(
                            pre_probe_responses,
                            post_probe_responses)
                        self._log(
                            f"  Identity safety: "
                            f"{summary['pre_safe']}/{summary['total']} "
                            f"pre  ?  "
                            f"{summary['post_safe']}/{summary['total']} "
                            f"post")
                        if summary["drifted"]:
                            self._log(
                                f"  [!] IDENTITY DRIFT on "
                                f"{len(summary['drifted'])} prompt(s):")
                            for p in summary["drifted"]:
                                self._log(f"      - {p!r}")
                            self._log(
                                f"      Rollback available: "
                                f"{Path(pre_distill_backup_path).name}"
                                if pre_distill_backup_path else
                                "      [!] NO rollback "
                                "checkpoint exists for this run.")
                        if summary["recovered"]:
                            self._log(
                                f"  Identity recovered on "
                                f"{len(summary['recovered'])} "
                                f"prompt(s)")
                    except Exception as probe_exc:
                        self._log(
                            f"[!] Post-probe failed: {probe_exc}")

                # Save model
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)
                atomic_torch_save({
                    "model_state_dict": student.state_dict(),
                    "config": self._model_config_dict(student),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }, student_path)

                self._log("\n--- DISTILLATION COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Examples  : {len(all_examples)}")
                self._log(f"Saved to  : {Path(student_path).name}")
                if pre_distill_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_distill_backup_path).name}")
                total = _time_d.monotonic() - _distill_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Distill", student_name, epochs,
                    state.best_loss)
                self.after(0, lambda pc=s_params:
                           self._update_forge_param_count(pc))
                if losses:
                    self._display_loss_curve(losses)
                self.after(0, self._refresh_models)

            except KeyboardInterrupt:
                self._log("\n--- DISTILLATION STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(f"\n[!] Distillation failed: {exc}")
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

        threading.Thread(target=_distill, daemon=True).start()

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

        model_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 RLHF TRAINING...")

        self._log_training_summary(
            "RLHF Training (Reward + Policy)",
            Student=model_name,
            Data=Path(data_path).name,
            Epochs=epochs,
            LR=lr,
        )
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # ARCH-1d Slice 3: API routing for RLHF
        client = self._get_api_chat_client() if bool(getattr(self, "use_api_chat", False)) else None
        if client is not None:
            def _run_api():
                try:
                    pref_data = Path(data_path).read_text(encoding="utf-8")
                    forge_params = self._read_forge_train_params()
                    api_config = {
                        "mode": "rlhf",
                        "allow_experimental": True,
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
                            "val_split": forge_params["val_split"],
                            "save_every": max(1, epochs // 5),
                            "run_evaluation": True,
                        },
                    }
                    self._log("Sending RLHF training to API server...\n")
                    self.training_active = True
                    self.solo_train_btn.configure(state="disabled",
                                                  text="TRAINING...")
                    self.stop_train_btn.configure(state="normal")
                    self.status_bar.set_left(
                        "\u2692 RLHF TRAINING...")
                    client.train(api_config)
                    self._poll_api_training_status(
                        client,
                        mode_label="RLHF")
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

        def _rlhf_train():
            try:
                import json
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import get_state_dict
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.rl_training import (
                    RewardModel,
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
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
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
                import time as _time
                _rlhf_start = [_time.monotonic()]
                _reward_last_t = [0.0]
                def _reward_progress(p, m):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _reward_last_t[0] >= 1.0:
                        eta = ""
                        if p > 0:
                            elapsed = now - _rlhf_start[0]
                            remaining = (elapsed / p) * (100 - p)
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            p, f"{m}{eta}")
                        _reward_last_t[0] = now
                def on_reward_trainer_ready(t) -> None:
                    self._active_trainer = t

                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                reward_ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    reward_model=reward_model,
                    on_progress=_reward_progress,
                    on_trainer_ready=on_reward_trainer_ready,
                )
                reward_config = {
                    "mode": "reward_model",
                    "allow_experimental": True,
                    "data": pref_data,
                    "training": {
                        "epochs": min(epochs, 5),
                        "learning_rate": lr * 10,
                        "use_amp": torch.cuda.is_available(),
                    },
                }
                result = run_training(reward_config, reward_ctx)
                self._log(f"Reward model trained: loss={result['final_loss']:.4f}")

                if self._forge_stop_requested():
                    return

                # Phase 2: RLHF policy training
                self._log("\n--- Phase 2: RLHF Policy Training ---")
                prompts = [item["prompt"] for item in pref_data]

                _rlhf_phase2_start = [_time.monotonic()]

                def _rlhf_progress(p, m):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _reward_last_t[0] >= 1.0:
                        eta = ""
                        if p > 0:
                            elapsed = now - _rlhf_phase2_start[0]
                            remaining = (elapsed / p) * (100 - p)
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            p, f"RLHF {p}%{eta}")
                        _reward_last_t[0] = now

                def on_trainer_ready(t) -> None:
                    self._active_trainer = t

                ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    reward_model=reward_model,
                    on_progress=_rlhf_progress,
                    on_trainer_ready=on_trainer_ready,
                )
                config_dict = {
                    "mode": "rlhf",
                    "allow_experimental": True,
                    "data": prompts,
                    "training": {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "use_amp": torch.cuda.is_available(),
                    },
                }
                rl_result = run_training(config_dict, ctx)

                self._log(f"\nFinal reward: {rl_result.get('final_reward', 0):.4f}")

                # Save model
                final_reward = rl_result.get('final_reward', 0)
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": final_reward,
                    },
                }, student_path)

                self._log(f"Model saved to {Path(student_path).name}")
                self._log("--- RLHF TRAINING COMPLETE ---")
                total = _time.monotonic() - _rlhf_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "RLHF", model_name, epochs, final_reward)
                self.after(0, lambda _pc=pc: self._update_forge_param_count(_pc))
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log("\n--- RLHF TRAINING STOPPED ---")
            except Exception as exc:
                self._log(f"\n[!] RLHF training failed: {exc}")
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

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        if not self._validate_general_data_path():
            return

        model_name = Path(student_path).stem
        trainer_name = Path(trainer_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 SELF-PLAY TRAINING...")

        self._log_training_summary(
            "Self-Play Training",
            Student=model_name,
            Trainer=trainer_name,
            Data=Path(data_path).name,
            Epochs=epochs,
            LR=lr,
        )
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # ARCH-1d Slice 3: API routing for Self-Play
        client = self._get_api_chat_client() if bool(getattr(self, "use_api_chat", False)) else None
        if client is not None:
            def _run_api():
                try:
                    # Read prompt data from text file
                    prompt_data = Path(data_path).read_text(encoding="utf-8")
                    forge_params = self._read_forge_train_params()
                    api_config = {
                        "mode": "self_play",
                        "allow_experimental": True,
                        "data": prompt_data,
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
                            "val_split": forge_params["val_split"],
                            "save_every": max(1, epochs // 5),
                            "run_evaluation": True,
                        },
                        "self_play": {
                            "trainer_path": trainer_path,
                        },
                    }
                    self._log("Sending Self-Play training to API server...\n")
                    self.training_active = True
                    self.solo_train_btn.configure(state="disabled",
                                                  text="TRAINING...")
                    self.stop_train_btn.configure(state="normal")
                    self.status_bar.set_left(
                        "\u2692 SELF-PLAY TRAINING...")
                    client.train(api_config)
                    self._poll_api_training_status(
                        client,
                        mode_label="SELF-PLAY")
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

        def _selfplay_train():
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import get_state_dict
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.inference import EnigmaEngine

                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load student
                self._log(f"Loading student: {model_name}...")
                from enigma_engine.core.model_registry import (
                    safe_load_weights)
                ckpt = safe_load_weights(
                    student_path, map_location=device)
                cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = ckpt.get("model_config", {})
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

                import time as _time
                _sp_last_t = [0.0]
                _sp_start = [_time.monotonic()]

                def _sp_progress(p, m):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _sp_last_t[0] >= 1.0:
                        eta = ""
                        if p > 0:
                            elapsed = now - _sp_start[0]
                            remaining = (elapsed / p) * (100 - p)
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            p, f"Self-play {p}%{eta}")
                        _sp_last_t[0] = now

                def on_trainer_ready(t) -> None:
                    self._active_trainer = t

                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                ctx = build_dispatch_context(
                    model=student,
                    tokenizer=tokenizer,
                    trainer_engine=trainer_engine,
                    on_progress=_sp_progress,
                    on_trainer_ready=on_trainer_ready,
                )
                config_dict = {
                    "mode": "self_play",
                    "allow_experimental": True,
                    "data": prompts,
                    "training": {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "use_amp": torch.cuda.is_available(),
                    },
                }
                result = run_training(config_dict, ctx)

                self._log(
                    f"\nFinal score: "
                    f"{result.get('final_score', 0):.2f}/10")

                # Save
                final_score = result.get('final_score', 0)
                from enigma_engine.core.safe_save import atomic_torch_save
                atomic_torch_save({
                    "model_state_dict": student.state_dict(),
                    "config": self._model_config_dict(student),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": final_score,
                    },
                }, student_path)

                self._log(f"Model saved to {Path(student_path).name}")
                self._log("--- SELF-PLAY COMPLETE ---")
                total = _time.monotonic() - _sp_start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    "Self-Play", model_name, epochs, final_score)
                sp_params = sum(
                    p.numel() for p in student.parameters())
                self.after(0, lambda _pc=sp_params:
                           self._update_forge_param_count(_pc))
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log("\n--- SELF-PLAY STOPPED ---")
            except Exception as exc:
                self._log(f"\n[!] Self-play failed: {exc}")
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

        threading.Thread(target=_selfplay_train, daemon=True).start()

    # ================================================================
    # GRPO TRAINING (N-11)
    # ================================================================

    def _start_grpo_training(self):
        """Train STUDENT with Group Relative Policy Optimization."""
        self._start_rl_variant_training("GRPO")

    # ================================================================
    # REMAX TRAINING (N-11)
    # ================================================================

    def _start_remax_training(self):
        """Train STUDENT with ReMax (REINFORCE + mean baseline)."""
        self._start_rl_variant_training("ReMax")

    def _start_rl_variant_training(self, algo: str):
        """Shared handler for GRPO and ReMax RL training."""
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
                f"[!] No data file selected.\n"
                f"    {algo} needs a .jsonl preference file\n"
                f"    (prompt/chosen/rejected per line).")
            return

        if not self._validate_jsonl_structure(
            data_path, ("prompt", "chosen", "rejected"),
        ):
            return

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        model_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(f"\u2692 {algo.upper()} TRAINING...")

        self._log_training_summary(
            f"{algo} Training (Reward + Policy)",
            Student=model_name,
            Data=Path(data_path).name,
            Epochs=epochs,
            LR=lr,
        )
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # ARCH-1d Slice 3: API routing for GRPO/ReMax
        client = self._get_api_chat_client() if bool(getattr(self, "use_api_chat", False)) else None
        if client is not None:
            def _run_api():
                try:
                    pref_data = Path(data_path).read_text(encoding="utf-8")
                    forge_params = self._read_forge_train_params()
                    algo_lower = algo.lower()
                    api_config = {
                        "mode": algo_lower,
                        "allow_experimental": True,
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
                            "val_split": forge_params["val_split"],
                            "save_every": max(1, epochs // 5),
                            "run_evaluation": True,
                        },
                    }
                    self._log(f"Sending {algo} training to API server...\n")
                    self.training_active = True
                    self.solo_train_btn.configure(state="disabled",
                                                  text="TRAINING...")
                    self.stop_train_btn.configure(state="normal")
                    self.status_bar.set_left(
                        f"\u2692 {algo.upper()} TRAINING...")
                    client.train(api_config)
                    self._poll_api_training_status(
                        client,
                        mode_label=algo.upper())
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

        def _rl_train():
            try:
                import json
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.rl_training import (
                    RewardModel,
                )
                from enigma_engine.core.reward_functions import (
                    reasoning_reward,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._log(f"Device  : {device.upper()}")

                # Load student model
                self._log(f"Loading {model_name}...")
                checkpoint = safe_load_weights(
                    student_path, map_location=device)
                cfg_dict = checkpoint.get(
                    "model_config",
                    checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
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
                            if all(k in item
                                   for k in ("prompt", "chosen",
                                             "rejected")):
                                pref_data.append(item)
                        except json.JSONDecodeError:
                            continue

                if not pref_data:
                    self._log(
                        "[!] No valid preference pairs found.")
                    return

                self._log(f"Loaded {len(pref_data)} preference pairs")

                # Pass 156z9as: pre-RL auto-checkpoint.  ``algo`` is
                # "GRPO" or "ReMax" and lower-cases into a sensible
                # rollback suffix.
                pre_rl_backup_path = (
                    self._pre_training_backup(
                        student_path,
                        suffix=f"pre_{algo.lower()}"))

                reward_model = None

                import time as _time
                _start = [_time.monotonic()]
                _last_t = [0.0]

                def _reward_progress(p, m):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _last_t[0] >= 1.0:
                        eta = ""
                        if p > 0:
                            elapsed = now - _start[0]
                            remaining = (
                                (elapsed / p) * (100 - p))
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            p, f"{m}{eta}")
                        _last_t[0] = now

                if algo == "GRPO":
                    self._log(
                        "\n--- Phase 1: Rule-based Reward (no neural reward model) ---")
                else:
                    # Phase 1: Train reward model
                    self._log(
                        "\n--- Phase 1: Training Reward Model ---")
                    reward_model = RewardModel(
                        model, freeze_base=True).to(device)
                    def on_reward_trainer_ready(t) -> None:
                        self._active_trainer = t

                    from enigma_engine.training.dispatch import (
                        build_dispatch_context, run_training)
                    reward_ctx = build_dispatch_context(
                        model=model,
                        tokenizer=tokenizer,
                        reward_model=reward_model,
                        on_progress=_reward_progress,
                        on_trainer_ready=on_reward_trainer_ready,
                    )
                    reward_config = {
                        "mode": "reward_model",
                        "allow_experimental": True,
                        "data": pref_data,
                        "training": {
                            "epochs": min(epochs, 5),
                            "learning_rate": lr * 10,
                            "use_amp": torch.cuda.is_available(),
                        },
                    }
                    result = run_training(reward_config, reward_ctx)
                    self._log(
                        f"Reward model trained: "
                        f"loss={result['final_loss']:.4f}")

                if self._forge_stop_requested():
                    return

                # Phase 2: RL policy training
                self._log(
                    f"\n--- Phase 2: {algo} Policy Training ---")
                prompts = [item["prompt"] for item in pref_data]

                if algo == "GRPO":
                    def reward_fn(prompt, response):
                        return reasoning_reward(prompt, response)
                else:
                    if reward_model is None:
                        raise RuntimeError(
                            "Reward model not initialized for ReMax training")

                    def reward_fn(prompt, response):
                        return reward_model.score(
                            prompt, response, tokenizer, device)

                _phase2_start = [_time.monotonic()]

                def _rl_progress(p, m):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _last_t[0] >= 1.0:
                        eta = ""
                        if p > 0:
                            elapsed = now - _phase2_start[0]
                            remaining = (
                                (elapsed / p) * (100 - p))
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            p, f"{algo} {p}%{eta}")
                        _last_t[0] = now

                def on_trainer_ready(t) -> None:
                    self._active_trainer = t

                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    on_progress=_rl_progress,
                    on_trainer_ready=on_trainer_ready,
                )
                mode_name = "grpo" if algo == "GRPO" else "remax"
                config_dict = {
                    "mode": mode_name,
                    "allow_experimental": algo == "ReMax",
                    "data": prompts,
                    "training": {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "use_amp": torch.cuda.is_available(),
                    },
                }
                rl_result = run_training(config_dict, ctx)

                final_reward = rl_result.get('final_reward', 0)
                self._log(
                    f"\nFinal reward: {final_reward:.4f}")

                # Save model
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": final_reward,
                    },
                }, student_path)

                self._log(
                    f"Model saved to {Path(student_path).name}")
                self._log(f"--- {algo.upper()} TRAINING COMPLETE ---")
                total = _time.monotonic() - _start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                if pre_rl_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_rl_backup_path).name}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    algo, model_name, epochs, final_reward)
                self.after(
                    0, lambda _pc=pc:
                    self._update_forge_param_count(_pc))
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log(
                    f"\n--- {algo.upper()} TRAINING STOPPED ---")
            except Exception as exc:
                self._log(
                    f"\n[!] {algo} training failed: {exc}")
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

        threading.Thread(target=_rl_train, daemon=True).start()

    # ================================================================
    # SimPO TRAINING (N-11)
    # ================================================================

    def _start_simpo_training(self):
        """Train with Simple Preference Optimization (no ref model)."""
        self._start_preference_variant_training("SimPO")

    # ================================================================
    # ORPO TRAINING (N-11)
    # ================================================================

    def _start_orpo_training(self):
        """Train with Odds Ratio Preference Optimization."""
        self._start_preference_variant_training("ORPO")

    def _start_preference_variant_training(self, algo: str):
        """Shared handler for SimPO and ORPO preference training."""
        if self.training_active:
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Go to ROUTER and assign the model to train.")
            return

        data_path = self.train_data_var.get()
        if not self._validate_jsonl_structure(
            data_path, ("prompt", "chosen", "rejected"),
        ):
            return

        result = self._validate_epochs_lr()
        if result is None:
            return
        epochs, lr = result

        model_name = Path(student_path).stem
        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left(
            f"\u2692 {algo.upper()} TRAINING...")

        self._log_training_summary(
            f"{algo} Preference Training",
            Student=model_name,
            Data=Path(data_path).name,
            Epochs=epochs,
            LR=lr,
        )
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # ARCH-1d Slice 3: API routing for SimPO/ORPO
        client = self._get_api_chat_client() if bool(getattr(self, "use_api_chat", False)) else None
        if client is not None:
            def _run_api():
                try:
                    pref_data = Path(data_path).read_text(encoding="utf-8")
                    forge_params = self._read_forge_train_params()
                    algo_lower = algo.lower()
                    api_config = {
                        "mode": algo_lower,
                        "allow_experimental": True,
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
                            "val_split": forge_params["val_split"],
                            "save_every": max(1, epochs // 5),
                            "run_evaluation": True,
                        },
                    }
                    self._log(f"Sending {algo} training to API server...\n")
                    self.training_active = True
                    self.solo_train_btn.configure(state="disabled",
                                                  text="TRAINING...")
                    self.stop_train_btn.configure(state="normal")
                    self.status_bar.set_left(
                        f"\u2692 {algo.upper()} TRAINING...")
                    client.train(api_config)
                    self._poll_api_training_status(
                        client,
                        mode_label=algo.upper())
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

        forge_params = self._read_forge_train_params()

        def _pref_train():
            try:
                import json
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import ForgeConfig
                from enigma_engine.core.model_registry import (
                    get_state_dict, safe_load_weights)
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.gui.scanners import MODELS_DIR

                device = (
                    "cuda" if torch.cuda.is_available()
                    else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Load student model
                self._log(f"Loading {model_name}...")
                checkpoint = safe_load_weights(
                    student_path, map_location=device)
                cfg_dict = checkpoint.get(
                    "model_config",
                    checkpoint.get("config", {}))
                if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
                    cfg_dict = checkpoint.get("model_config", {})
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
                            if all(k in item
                                   for k in ("prompt", "chosen",
                                             "rejected")):
                                pref_data.append(item)
                        except json.JSONDecodeError:
                            continue

                if not pref_data:
                    self._log(
                        "[!] No valid preference pairs found.")
                    return

                self._log(f"Loaded {len(pref_data)} preference pairs")

                # Pass 156z9as: pre-SimPO/ORPO auto-checkpoint.
                pre_pref_backup_path = (
                    self._pre_training_backup(
                        student_path,
                        suffix=f"pre_{algo.lower()}"))

                # Progress callback
                import time as _time
                _start = [_time.monotonic()]
                _last_t = [0.0]
                _trainer_ref: list = []

                def on_progress(p, m):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    now = _time.monotonic()
                    if now - _last_t[0] >= 1.0:
                        eta = ""
                        if p > 0:
                            elapsed = now - _start[0]
                            remaining = (
                                (elapsed / p) * (100 - p))
                            r_m = int(remaining // 60)
                            r_s = int(remaining % 60)
                            eta = f" | ETA {r_m}m {r_s:02d}s"
                        self._update_forge_progress(
                            p, f"{m}{eta}")
                        _last_t[0] = now

                def on_loss(loss):
                    now = _time.monotonic()
                    if now - _last_t[0] < 0.5:
                        return
                    _last_t[0] = now
                    step = (
                        _trainer_ref[0].state.step
                        if _trainer_ref else 0)
                    self._log(
                        f"  Step {step:>5d} | loss {loss:.4f}")

                def on_trainer_ready(t) -> None:
                    _trainer_ref.append(t)
                    self._active_trainer = t

                # ARCH-1.5c: route SimPO/ORPO through dispatcher.
                from enigma_engine.training.dispatch import (
                    build_dispatch_context, run_training)
                ctx = build_dispatch_context(
                    model=model,
                    tokenizer=tokenizer,
                    on_progress=on_progress,
                    on_loss=on_loss,
                    on_trainer_ready=on_trainer_ready,
                )
                mode_name = "simpo" if algo == "SimPO" else "orpo"
                config_dict: dict = {
                    "mode": mode_name,
                    "allow_experimental": True,
                    "data": pref_data,
                    "training": {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "batch_size": forge_params["batch_size"],
                        "use_gradient_checkpointing": forge_params[
                            "use_gradient_checkpointing"],
                        "ce_chunk_size": forge_params["ce_chunk_size"],
                        "use_compile": True,
                        "rolling_best_k": forge_params["rolling_best_k"],
                        "save_every": max(1, epochs // 5),
                        "checkpoint_dir": str(
                            MODELS_DIR / "checkpoints"),
                        "use_amp": True,
                        "run_evaluation": True,
                    },
                }
                state = run_training(config_dict, ctx)

                final_loss = state.best_loss
                self._log(
                    f"\nFinal loss: {final_loss:.4f}")

                # Save model
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": epochs,
                        "best_loss": final_loss,
                    },
                }, student_path)

                self._log(
                    f"Model saved to {Path(student_path).name}")
                self._log(
                    f"--- {algo.upper()} TRAINING COMPLETE ---")
                total = _time.monotonic() - _start[0]
                t_m, t_s = int(total // 60), int(total % 60)
                self._log(f"Duration  : {t_m}m {t_s:02d}s")
                if pre_pref_backup_path:
                    self._log(
                        f"Rollback  : "
                        f"{Path(pre_pref_backup_path).name}")
                self._update_forge_progress(100, "Complete")
                self._save_training_run(
                    algo, model_name, epochs, final_loss)
                self.after(
                    0, lambda _pc=pc:
                    self._update_forge_param_count(_pc))
                self.after(0, self._refresh_models)
                self._notify_training_complete()

            except KeyboardInterrupt:
                self._log(
                    f"\n--- {algo.upper()} TRAINING STOPPED ---")
            except Exception as exc:
                self._log(
                    f"\n[!] {algo} training failed: {exc}")
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

        threading.Thread(target=_pref_train, daemon=True).start()

