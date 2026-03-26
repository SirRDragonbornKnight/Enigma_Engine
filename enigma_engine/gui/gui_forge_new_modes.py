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

    # ================================================================
    # PRE-TRAINING (Phase 1a)
    # ================================================================

    def _pretrain_validate_inputs(self) -> dict | None:
        """Validate inputs for pre-training.

        Returns a dict of validated params, or None on failure
        (with error logged to the FORGE panel).
        """
        if self.training_active:
            return None

        preset_name = getattr(self, "pretrain_preset_var", None)
        preset_name = preset_name.get() if preset_name else "large"
        # Strip description suffix from display value (e.g. "large - RTX 3080+...")
        preset_name = preset_name.split(" - ", 1)[0]

        data_var = getattr(self, "pretrain_data_var", None)
        data_path = data_var.get().strip() if data_var else ""

        model_name_var = getattr(
            self, "pretrain_model_name_var", None)
        model_name = (model_name_var.get().strip()
                      if model_name_var else "")

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

        if not model_name:
            model_name = f"pretrained_{preset_name}"
        safe_name = "".join(
            c for c in model_name if c.isalnum() or c in "_-")
        if not safe_name:
            self._log("[!] Invalid model name.")
            return None

        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except (ValueError, AttributeError):
            self._log("[!] Epochs must be 1-1000.")
            return None

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except (ValueError, AttributeError):
            self._log("[!] Learning rate must be 0 to 1.")
            return None

        return {
            "preset_name": preset_name,
            "data_source": data_source,
            "safe_name": safe_name,
            "retrain_tok": retrain_tok,
            "vocab_size": vocab_size,
            "epochs": epochs,
            "lr": lr,
        }

    def _start_pretrain_training(self):
        """Pre-train a new model from scratch on large text data.

        Workflow:
        1. Create a fresh model from selected preset (random init)
        2. Optionally retrain BPE tokenizer on the data
        3. Process and clean the pre-training corpus
        4. Run standard causal LM training with pre-training defaults

        Pre-training uses higher LR, no general mix (this IS the
        general knowledge), and longer warmup than fine-tuning.
        """
        params = self._pretrain_validate_inputs()
        if params is None:
            return

        preset_name = params["preset_name"]
        data_source = params["data_source"]
        safe_name = params["safe_name"]
        retrain_tok = params["retrain_tok"]
        vocab_size = params["vocab_size"]
        epochs = params["epochs"]
        lr = params["lr"]

        from enigma_engine.gui.scanners import MODELS_DIR
        out_path = MODELS_DIR / f"{safe_name}.pth"

        self.training_active = True
        self.solo_train_btn.configure(state="disabled",
                                      text="PRE-TRAINING...")
        self.stop_train_btn.configure(state="normal")
        self.status_bar.set_left("\u2692 PRE-TRAINING...")

        self._log("--- PRE-TRAINING INITIATED ---")
        self._log(f"Preset  : {preset_name}")
        self._log(f"Data    : {data_source}")
        self._log(f"Output  : {out_path}")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._log(f"Vocab   : {vocab_size}  |  Retrain tok: {retrain_tok}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        def _pretrain():
            losses = []
            try:
                import torch
                from enigma_engine.core.model import Enigma
                from enigma_engine.core.model_presets import (
                    get_preset)
                from enigma_engine.core.training import (
                    Trainer, TrainingConfig)
                from enigma_engine.core.tokenizer import get_tokenizer
                from enigma_engine.core.dataset import (
                    process_text_corpus, estimate_token_count)

                device = ("cuda"
                          if torch.cuda.is_available() else "cpu")
                self._log(f"Device  : {device.upper()}")

                # Step 1: Process pre-training data
                self._log("Processing data...")
                text = process_text_corpus(
                    data_source, text_key="text")
                if not text or len(text) < 100:
                    self._log(
                        "[!] Not enough text data. Need at least "
                        "100 characters of clean text.")
                    return

                est_tokens = estimate_token_count(text)
                self._log(
                    f"Data    : {len(text):,} chars "
                    f"(~{est_tokens:,} tokens)")

                # Step 2: Optionally retrain tokenizer

                if retrain_tok:
                    self._log(
                        f"Training BPE tokenizer "
                        f"(vocab {vocab_size})...")
                    from enigma_engine.core.bpe_tokenizer import (
                        BPETokenizer)
                    tokenizer = BPETokenizer()
                    # Split into chunks for training
                    chunk_size = 500_000
                    train_texts = [
                        text[i:i + chunk_size]
                        for i in range(0, len(text), chunk_size)
                    ]
                    tokenizer.train(
                        train_texts,
                        vocab_size=vocab_size,
                        verbose=False)
                    # Save tokenizer
                    tok_dir = (Path(__file__).parent.parent
                               / "vocab_model")
                    tok_dir.mkdir(exist_ok=True)
                    tokenizer.save(str(tok_dir / "tokenizer.json"))
                    self._log(
                        f"Tokenizer trained: "
                        f"{tokenizer.vocab_size} tokens")
                else:
                    tokenizer = get_tokenizer("auto")
                    self._log(
                        f"Tokenizer: {type(tokenizer).__name__} "
                        f"(vocab {tokenizer.vocab_size})")

                # Step 3: Create fresh model from preset
                self._log(
                    f"Creating model from '{preset_name}' preset...")
                config = get_preset(
                    preset_name, vocab_size=tokenizer.vocab_size)
                model = Enigma(config=config)
                model = model.to(device)

                pc = sum(p.numel() for p in model.parameters())
                self._log(f"Params  : {pc:,}")

                # Step 4: Train with pre-training defaults
                forge_params = self._read_forge_train_params()
                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=forge_params["batch_size"],
                    learning_rate=lr,
                    max_grad_accumulation=forge_params[
                        "max_grad_accumulation"],
                    use_gradient_checkpointing=forge_params[
                        "use_gradient_checkpointing"],
                    general_mix_ratio=0.0,  # No mix — this IS general
                    val_split=forge_params["val_split"],
                    save_every=max(1, epochs // 5),
                    checkpoint_dir=str(MODELS_DIR / "checkpoints"),
                    use_amp=torch.cuda.is_available(),
                    run_evaluation=True)

                trainer = Trainer(model, tokenizer, train_config)

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

                self._log("Pre-training...\n")
                state = trainer.train(text)

                # Check for early termination
                import math
                if math.isinf(state.best_loss) and not losses:
                    self._log(
                        "\n[!] Pre-training aborted — likely OOM "
                        "or NaN loss.\n"
                        "    Try a smaller preset or reduce "
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
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "config": self._model_config_dict(model),
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }, out_path)

                self._log("\n--- PRE-TRAINING COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Saved to  : {out_path}")
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

            except KeyboardInterrupt:
                self._log("\n--- PRE-TRAINING STOPPED ---")
                if losses:
                    self._display_loss_curve(losses)
            except Exception as exc:
                self._log(f"\n[!] Pre-training failed: {exc}")
            finally:
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_pretrain, daemon=True).start()

    # ================================================================
    # DISTILLATION (Step 1b — Teacher → Student)
    # ================================================================

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

        try:
            epochs = int(self.ft_epochs_entry.get())
            if epochs < 1 or epochs > 1000:
                raise ValueError
        except (ValueError, AttributeError):
            self._log("[!] Epochs must be 1-1000.")
            return None

        try:
            lr = float(self.ft_lr_entry.get())
            if lr <= 0 or lr > 1:
                raise ValueError
        except (ValueError, AttributeError):
            self._log("[!] Learning rate must be 0 to 1.")
            return None

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

        self._log("=" * 40)
        self._log("  Knowledge Distillation")
        self._log("=" * 40)
        self._log(f"Teacher : {trainer_name}")
        self._log(f"Student : {student_name}")
        self._log(f"Categories: {', '.join(categories)}")
        self._log(f"Examples: {num_examples} per category "
                  f"({num_examples * len(categories)} total)")
        self._log(f"Epochs  : {epochs}  |  LR: {lr}")
        self._clear_forge_param_count()
        self._reset_forge_progress()

        # Category-specific prompts for the teacher
        category_prompts = {
            "personality": [
                "Introduce yourself with a warm, unique personality. "
                "Show character and genuine emotion in your response.",
                "Someone just complimented you. Respond naturally "
                "showing your personality.",
                "Express a strong opinion about something you care "
                "about. Be genuine and show feeling.",
                "Tell a short personal anecdote that reveals "
                "something about your character.",
                "Someone is having a bad day. Respond with empathy "
                "and your own emotional perspective.",
            ],
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
                "User: Hey, what's up?\nAssistant:",
                "User: I'm thinking about learning a new hobby. "
                "Any suggestions?\nAssistant:",
                "User: Tell me something interesting you've "
                "learned recently.\nAssistant:",
                "User: I just got a new puppy! I'm so excited!\n"
                "Assistant:",
                "User: Can you help me plan a weekend trip?\n"
                "Assistant:",
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
                "fictional café that serves unusual drinks.",
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
                from enigma_engine.core.training import (
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

                tokenizer = get_tokenizer("auto")

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
                            if response and len(response.strip()) > 20:
                                # Format as training data
                                example = (
                                    f"User: {prompt}\n"
                                    f"Assistant: {response.strip()}")
                                all_examples.append(example)
                                generated += 1

                                # Log periodically
                                if generated % 10 == 0 or generated == 1:
                                    preview = response.strip()
                                    if len(preview) > 100:
                                        preview = preview[:100] + "..."
                                    self._log(
                                        f"  [{generated}/{total_to_gen}] "
                                        f"{preview}")
                            else:
                                self._log(
                                    f"  [{generated}/{total_to_gen}] "
                                    f"Skipped (too short)")

                        except Exception as exc:
                            self._log(
                                f"  Generation error: {exc}")
                            continue

                        pct = int(generated / total_to_gen * 50)
                        self._update_forge_progress(
                            pct,
                            f"Generating {generated}/{total_to_gen}")

                if not all_examples:
                    self._log(
                        "\n[!] No distillation data generated.\n"
                        "    Check that the TRAINER model is loaded "
                        "and responsive.")
                    return

                self._log(
                    f"\nGenerated {len(all_examples)} training examples")

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
                train_config = TrainingConfig(
                    epochs=epochs,
                    batch_size=forge_params["batch_size"],
                    learning_rate=lr,
                    max_grad_accumulation=forge_params[
                        "max_grad_accumulation"],
                    use_gradient_checkpointing=forge_params[
                        "use_gradient_checkpointing"],
                    general_mix_ratio=0.0,  # Only distilled data
                    val_split=forge_params["val_split"],
                    save_every=max(1, epochs // 5),
                    checkpoint_dir=str(
                        MODELS_DIR / "checkpoints"),
                    use_amp=torch.cuda.is_available(),
                    run_evaluation=True)

                trainer = Trainer(student, tokenizer, train_config)

                def on_epoch(epoch, loss):
                    if not self.training_active:
                        raise KeyboardInterrupt("Stopped")
                    losses.append(loss)
                    pct = 50 + int(epoch / epochs * 50)
                    self._update_forge_progress(
                        pct, f"Training {epoch}/{epochs}")
                    self._log(
                        f"  Epoch {epoch:>3d}  |  "
                        f"loss {loss:.4f}")
                trainer.on_epoch_complete = on_epoch

                self._log(f"Training on {len(all_examples)} "
                          f"examples for {epochs} epochs...\n")
                state = trainer.train(training_text)

                # Check for failure
                import math
                if math.isinf(state.best_loss) and not losses:
                    self._log(
                        "\n[!] Training aborted — likely OOM "
                        "or NaN loss.\n"
                        "    Try reducing batch size.")
                    return

                # Save model
                from enigma_engine.core.safe_save import (
                    atomic_torch_save)
                atomic_torch_save({
                    "model_state_dict": student.state_dict(),
                    "model_config": s_cfg.__dict__,
                    "config": s_cfg.__dict__,
                    "training_state": {
                        "epochs": state.epoch,
                        "best_loss": state.best_loss,
                    },
                }, student_path)

                self._log("\n--- DISTILLATION COMPLETE ---")
                self._log(f"Best loss : {state.best_loss:.4f}")
                self._log(f"Examples  : {len(all_examples)}")
                self._log(f"Saved to  : {Path(student_path).name}")
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
                self.training_active = False
                self._reset_forge_progress()
                self.after(0, lambda: self.solo_train_btn.configure(
                    state="normal", text="TRAIN"))
                self.after(0, lambda: self.stop_train_btn.configure(
                    state="disabled"))
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

                rl_params = self._read_forge_rl_params()
                rlhf_cfg = RLHFConfig(
                    epochs=epochs,
                    learning_rate=lr,
                    replay_capacity=rl_params.get("replay_capacity", 256),
                    replay_ratio=rl_params.get("replay_ratio", 0.25),
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

                sp_rl_params = self._read_forge_rl_params()
                sp_cfg = SelfPlayConfig(
                    epochs=epochs,
                    learning_rate=lr,
                    replay_capacity=sp_rl_params.get("replay_capacity", 256),
                    replay_ratio=sp_rl_params.get("replay_ratio", 0.25),
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

