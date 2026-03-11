"""
Enigma Engine - Forge Tools
==============================

Tool implementations: data generation, evaluation, web learn,
checkpoints, loss curves, identity cards, presets, history.
Split from gui_forge.py to keep files under 800 lines.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR
from enigma_engine.gui.widgets import C_GREEN, C_TEXT_DIM

logger = logging.getLogger(__name__)


class ForgeToolsMixin:
    """Forge tool implementations.

    Expects the host class to have ForgeMixin setup attributes.
    """

    def _select_generated_data_for_current_mode(self, path: str) -> None:
        """Route generated data path to the active mode's data selector."""
        mode_var = getattr(self, "training_mode_var", None)
        mode_name = mode_var.get() if mode_var is not None else "Basic"
        if mode_name == "AI-Guided":
            sup_var = getattr(self, "ai_supplement_var", None)
            if sup_var is not None:
                sup_var.set(path)
                return
        self.train_data_var.set(path)

    # ================================================================
    # Generate training data (TRAINER produces synthetic data)
    # ================================================================

    def _generate_training_data(self):
        """TRAINER autonomously generates training data.

        The TRAINER creates Q/A pairs for the selected stage
        without needing a pre-made prompt file.  If a data file
        is selected, its prompts are used as supplementary input.
        Saves all pairs as a new data file.

        When Auto-train is checked, automatically selects the new
        file and starts training after generation completes.
        """
        trainer_path = self.route_assignments.get("trainer")
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Go to ROUTER and assign the teacher model.")
            return

        try:
            num_pairs = int(self.guided_pairs_entry.get())
            if num_pairs < 1 or num_pairs > 500:
                raise ValueError
        except (ValueError, AttributeError):
            num_pairs = 20

        # Data file is optional — supplements TRAINER output
        mode_var = getattr(self, "training_mode_var", None)
        mode_name = mode_var.get() if mode_var is not None else "Basic"
        if mode_name == "AI-Guided":
            sup_var = getattr(self, "ai_supplement_var", None)
            sup_value = sup_var.get() if sup_var is not None else "(none)"
            data_path = "" if sup_value == "(none)" else sup_value
        else:
            data_path = self.train_data_var.get()
        has_data = bool(data_path) and Path(data_path).exists()

        trainer_name = Path(trainer_path).stem
        self.generate_data_btn.configure(state="disabled")
        self.status_bar.set_left("Generating training data...")
        self._reset_forge_progress()

        self._log("--- GENERATING TRAINING DATA ---")
        self._log(f"Trainer : {trainer_name}")
        self._log(f"Pairs   : {num_pairs}")

        def _gen():
            try:
                # Load TRAINER via EnigmaEngine (any format)
                self._log(f"Loading {trainer_name}...")
                self._update_forge_progress(
                    5, "Loading trainer")
                engine = self._load_engine_for_path(trainer_path)

                # Read training stage from UI
                stage = getattr(self, 'training_stage_var', None)
                stage = stage.get() if stage else "basics"

                # Read training brief from UI
                training_brief = self._build_training_brief()
                self._save_training_brief()

                # Build system prompt
                gen_sys = self._build_trainer_system_prompt(
                    student_params=0,
                    task="generate",
                    stage=stage,
                    training_brief=training_brief)

                self._log(f"Stage   : {stage.upper()}")
                if training_brief:
                    self._log(f"Brief   : {len(training_brief)} chars")
                self._log(
                    f"TRAINER generating {num_pairs} "
                    f"examples...\n")
                self._update_forge_progress(
                    10, "Generating")
                results = []

                # TRAINER generates pairs autonomously
                for i in range(num_pairs):
                    use_reasoning = getattr(
                        self, "forge_reasoning_var", None)
                    reasoning_on = (
                        use_reasoning.get()
                        if use_reasoning is not None else False)
                    msg = self._build_generation_prompt(
                        i + 1, num_pairs, stage,
                        reasoning=reasoning_on)
                    result = engine.chat(
                        msg,
                        system_prompt=gen_sys,
                        max_gen=256,
                        temperature=0.8).strip()
                    if result:
                        results.append(result)

                    # Progress: 10-80% for main generation
                    pct = 10 + int(
                        (i + 1) / num_pairs * 70)
                    self._update_forge_progress(
                        pct, f"{i + 1}/{num_pairs}")
                    if (i + 1) % max(1, num_pairs // 5) == 0:
                        self._log(
                            f"  Generated {i + 1}/{num_pairs}")

                # Supplement from data file if selected
                if has_data:
                    extras = self._extract_prompts(data_path)
                    self._log(
                        f"Processing {len(extras)} "
                        f"supplementary prompts...")
                    self._update_forge_progress(
                        82, "Supplements")
                    for j, prompt in enumerate(extras):
                        response = engine.chat(
                            prompt,
                            system_prompt=gen_sys,
                            max_gen=256,
                            temperature=0.8).strip()
                        if response:
                            results.append(
                                self._format_training_pair(
                                    stage, prompt, response))
                        if extras:
                            pct = 82 + int(
                                (j + 1) / len(extras) * 10)
                            self._update_forge_progress(
                                pct, "Supplements")

                if not results:
                    self._log("[!] No valid data generated.")
                    return

                # Save to new data file
                self._update_forge_progress(95, "Saving")
                out_name = (
                    f"generated_{stage}_{trainer_name}.txt")
                out_path = DATA_DIR / out_name
                out_path.write_text(
                    "\n\n".join(results) + "\n",
                    encoding="utf-8")

                self._update_forge_progress(100, "Complete")
                self._log("\n--- DATA GENERATION COMPLETE ---")
                self._log(
                    f"Generated : {len(results)} pairs")
                self._log(f"Saved to  : {out_path}")
                self.after(0, self._refresh_data_files)

                # Add to curated dataset for review
                add_fn = getattr(
                    self, "_add_to_curated_dataset", None)
                if add_fn is not None:
                    for r in results:
                        add_fn(r, source="generated",
                               stage=stage)
                    self._log(
                        f"  Added {len(results)} entries"
                        f" to curated dataset")

                # Auto-train: select new file and start
                auto_var = getattr(
                    self, "forge_auto_train_var", None)
                if (auto_var is not None
                        and auto_var.get()):
                    self._log(
                        "Auto-train: starting training...")
                    self.after(0, lambda p=str(out_path): (
                        self._select_generated_data_for_current_mode(p),
                        self._start_training_by_mode()))

            except Exception as exc:
                self._log(
                    f"\n[!] Data generation failed: {exc}")
            finally:
                self.after(
                    0, lambda: self.generate_data_btn.configure(
                        state="normal"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_gen, daemon=True).start()

    # ================================================================
    # Evaluate student (TRAINER scores STUDENT responses)
    # ================================================================

    def _evaluate_student(self):
        """TRAINER interactively tests STUDENT and judges responses.

        The TRAINER generates questions appropriate for the current
        stage, the STUDENT answers, and the TRAINER scores each
        answer 1-10.  Determines readiness to advance.

        No data file required — TRAINER creates its own test.
        """
        trainer_path = self.route_assignments.get("trainer")
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Need both models for evaluation.")
            return

        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log(
                "[!] No model assigned to STUDENT route.\n"
                "    Need both models for evaluation.")
            return

        trainer_name = Path(trainer_path).stem
        student_name = Path(student_path).stem
        self.evaluate_btn.configure(state="disabled")
        self.status_bar.set_left("Evaluating...")

        self._log("--- EVALUATION STARTED ---")
        self._log(f"Trainer : {trainer_name} (examiner)")
        self._log(f"Student : {student_name} (tested)")

        def _eval():
            try:
                # Load both models via EnigmaEngine
                self._log(f"Loading {trainer_name}...")
                teacher_engine = self._load_engine_for_path(
                    trainer_path)

                self._log(f"Loading {student_name}...")
                student_engine = self._load_engine_for_path(
                    student_path)

                # Read training stage from UI
                stage = getattr(self, 'training_stage_var', None)
                stage = stage.get() if stage else "basics"

                # Read training brief from UI
                training_brief = self._build_training_brief()
                self._save_training_brief()

                # System prompt for TRAINER
                eval_sys = self._build_trainer_system_prompt(
                    student_params=0,
                    task="evaluate",
                    stage=stage,
                    training_brief=training_brief)

                num_tests = 15
                self._log(f"Stage   : {stage.upper()}")
                self._log(
                    f"\nTRAINER is testing STUDENT with "
                    f"{num_tests} questions...\n")
                scores = []

                for t in range(num_tests):
                    # TRAINER generates a test question
                    test_q = teacher_engine.chat(
                        f"Test question #{t + 1}: Generate a "
                        f"{stage}-level question to evaluate "
                        f"the student. Write ONLY the question, "
                        f"nothing else.",
                        system_prompt=eval_sys,
                        max_gen=64,
                        temperature=0.9).strip()
                    if not test_q:
                        continue

                    # STUDENT answers
                    s_answer = student_engine.generate(
                        test_q, max_gen=128,
                        temperature=0.7).strip()

                    # TRAINER judges the answer
                    judge_msg = (
                        f'Question: "{test_q}"\n'
                        f'Student answered: "{s_answer}"\n\n'
                        "Score this response 1-10:\n"
                        "1-3 = poor (incoherent/wrong)\n"
                        "4-6 = developing (partially right)\n"
                        "7-8 = good (solid response)\n"
                        "9-10 = excellent (natural and correct)"
                        "\n\nReply ONLY: SCORE: <n> | <feedback>")
                    judgment = teacher_engine.chat(
                        judge_msg,
                        system_prompt=eval_sys,
                        max_gen=64,
                        temperature=0.3).strip()

                    # Parse score from judgment
                    score = 5
                    feedback = judgment
                    for line in judgment.splitlines():
                        ln = line.strip()
                        if ln.upper().startswith("SCORE:"):
                            rest = ln.split(":", 1)[1].strip()
                            parts = rest.split("|", 1)
                            try:
                                score = int(
                                    parts[0].strip().split()[0])
                                score = max(1, min(10, score))
                            except (ValueError, IndexError):
                                pass
                            if len(parts) > 1:
                                feedback = parts[1].strip()
                            break

                    scores.append(score)
                    q_s = (test_q[:45] + "..."
                           if len(test_q) > 45 else test_q)
                    a_s = (s_answer[:45] + "..."
                           if len(s_answer) > 45
                           else s_answer)
                    self._log(
                        f"  Test {t + 1:>2d}  |  "
                        f"Score: {score}/10")
                    self._log(f"    Q: {q_s}")
                    self._log(f"    A: {a_s}")
                    if feedback and feedback != judgment:
                        self._log(f"    \u2192 {feedback}")

                # Readiness assessment
                if scores:
                    avg = sum(scores) / len(scores)
                    stages_list = [
                        "basics", "conversation",
                        "commands", "web"]
                    s_idx = (stages_list.index(stage)
                             if stage in stages_list else 0)
                    next_s = (stages_list[s_idx + 1]
                              if s_idx < len(stages_list) - 1
                              else None)

                    self._log("\n--- EVALUATION RESULTS ---")
                    self._log(f"Tests   : {len(scores)}")
                    self._log(f"Average : {avg:.1f} / 10")
                    if avg >= 7:
                        self._log("Result  : READY")
                        if next_s:
                            self._log(
                                f"Next    : advance to "
                                f"'{next_s}' stage")
                    elif avg >= 5:
                        self._log("Result  : PROGRESSING")
                        self._log(
                            "Next    : continue at this stage")
                    else:
                        self._log("Result  : NEEDS WORK")
                        self._log(
                            "Next    : more training needed")
                else:
                    self._log(
                        "\n[!] No test results to evaluate.")

            except Exception as exc:
                self._log(f"\n[!] Evaluation failed: {exc}")
            finally:
                self.after(
                    0, lambda: self.evaluate_btn.configure(
                        state="normal"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_eval, daemon=True).start()

    # ================================================================
    # Web Learn (TRAINER searches web, generates training data)
    # ================================================================

    def _web_learn(self):
        """Search the web for a topic, fetch pages, and generate
        training data from the content using the TRAINER model.

        Uses shared web_utils for search and page fetching.
        The TRAINER reads real web content and produces Q/A pairs
        that can be used to train the STUDENT.  Saves output as a
        new data file in the data/ directory.

        When Auto-train is checked, automatically selects the new
        file and starts training after generation completes.
        """
        trainer_path = self.route_assignments.get("trainer")
        if not trainer_path or not Path(trainer_path).exists():
            self._log(
                "[!] No model assigned to TRAINER route.\n"
                "    Go to ROUTER and assign the teacher model.")
            return

        topic = self.web_learn_topic.get().strip()
        if not topic:
            self._log("[!] Enter a topic to search for.")
            return

        try:
            max_pages = int(self.web_learn_pages_entry.get())
            if max_pages < 1 or max_pages > 10:
                raise ValueError
        except ValueError:
            self._log("[!] Max pages must be 1-10.")
            return

        # Read training stage
        stage = getattr(self, 'training_stage_var', None)
        stage = stage.get() if stage else "basics"

        trainer_name = Path(trainer_path).stem
        self.web_learn_btn.configure(state="disabled")
        self.status_bar.set_left("WEB LEARN: searching...")
        self._reset_forge_progress()

        self._log("--- WEB LEARN STARTED ---")
        self._log(f"Topic   : {topic}")
        self._log(f"Trainer : {trainer_name}")
        self._log(f"Pages   : {max_pages}")
        self._log(f"Stage   : {stage.upper()}")

        def _learn():
            try:
                from enigma_engine.core.web_utils import (
                    ddg_search, fetch_page_text)

                # Step 1: Search the web  (0-10%)
                self._log("\nSearching the web...")
                self._update_forge_progress(5, "Searching...")
                results = ddg_search(topic, max_results=max_pages)

                if not results:
                    self._log("[!] No search results found.")
                    return

                self._log(
                    f"Found {len(results)} results, "
                    f"fetching top {max_pages}...")
                self._update_forge_progress(10, "Fetching pages")

                # Step 2: Fetch page content  (10-30%)
                pages_content = []
                for i, result in enumerate(results[:max_pages]):
                    try:
                        self._log(
                            f"  Reading: "
                            f"{result['title'][:50]}...")
                        content = fetch_page_text(
                            result["url"], max_chars=3000)
                        if content.strip():
                            pages_content.append({
                                "title": result["title"],
                                "content": content.strip()})
                    except Exception as exc:
                        self._log(f"  [!] Failed: {exc}")
                    pct = 10 + int(
                        (i + 1) / max_pages * 20)
                    self._update_forge_progress(
                        pct, f"Fetched {i + 1}/{max_pages}")

                if not pages_content:
                    self._log(
                        "[!] Could not read any pages.")
                    return

                total_chars = sum(
                    len(p["content"]) for p in pages_content)
                self._log(
                    f"\nRead {len(pages_content)} pages "
                    f"({total_chars:,} chars total)")

                # Step 3: TRAINER generates Q/A pairs  (30-90%)
                self._log(f"Loading {trainer_name}...")
                self._update_forge_progress(
                    30, "Loading trainer")
                engine = self._load_engine_for_path(trainer_path)

                # Read training brief from UI
                training_brief = self._build_training_brief()

                # Use shared trainer system prompt
                sys_prompt = self._build_trainer_system_prompt(
                    student_params=0,
                    task="generate",
                    stage=stage,
                    training_brief=training_brief)

                self._log(
                    "Generating Q&A pairs from content...\n")
                all_pairs = []
                chunk_failures = 0

                # Count total chunks for progress
                total_chunks = 0
                for page in pages_content:
                    n = min(
                        5,
                        len(range(
                            0, len(page["content"]), 800)))
                    total_chunks += n
                done_chunks = 0

                for page in pages_content:
                    content = page["content"]
                    chunk_size = 800
                    chunks = [
                        content[i:i + chunk_size]
                        for i in range(
                            0, len(content), chunk_size)]
                    chunks = chunks[:5]

                    for chunk in chunks:
                        msg = (
                            f"Source: {page['title']}\n\n"
                            f"Content:\n{chunk}\n\n"
                            "Generate one Q&A pair from "
                            "this.")
                        try:
                            result = engine.chat(
                                msg,
                                system_prompt=sys_prompt,
                                max_gen=256,
                                temperature=0.7).strip()
                            if result and "Q:" in result:
                                all_pairs.append(result)
                        except Exception as e:
                            chunk_failures += 1
                            if chunk_failures == 1:
                                self._log(
                                    f"  [!] Generation "
                                    f"error: {e}")
                        done_chunks += 1
                        pct = 30 + int(
                            done_chunks / max(
                                1, total_chunks) * 60)
                        self._update_forge_progress(
                            pct, "Generating Q&A")

                    self._log(
                        f"  {page['title'][:40]}: "
                        f"{len(chunks)} chunks processed")

                if chunk_failures:
                    self._log(
                        f"  [!] {chunk_failures} chunk(s) "
                        f"failed during generation")

                if not all_pairs:
                    self._log(
                        "[!] TRAINER could not generate "
                        "any pairs.")
                    return

                # Step 4: Save as training data file  (90-100%)
                self._update_forge_progress(
                    92, "Saving data")
                safe_topic = "".join(
                    c if c.isalnum() or c in " _-" else ""
                    for c in topic
                ).strip().replace(" ", "_")
                out_name = f"web_{safe_topic}.txt"
                out_path = DATA_DIR / out_name
                out_path.write_text(
                    "\n\n".join(all_pairs) + "\n",
                    encoding="utf-8")

                self._update_forge_progress(
                    100, "Complete")
                self._log("\n--- WEB LEARN COMPLETE ---")
                self._log(
                    f"Generated : {len(all_pairs)} Q&A pairs")
                self._log(f"Saved to  : {out_path}")

                self.after(0, self._refresh_data_files)

                # Add to curated dataset for review
                add_fn = getattr(
                    self, "_add_to_curated_dataset", None)
                if add_fn is not None:
                    for pair in all_pairs:
                        add_fn(pair, source="web_learn",
                               stage=stage)
                    self._log(
                        f"  Added {len(all_pairs)} entries"
                        f" to curated dataset")

                # Auto-train: select new file and start
                auto_var = getattr(
                    self, "forge_auto_train_var", None)
                if (auto_var is not None
                        and auto_var.get()):
                    self._log(
                        "Auto-train: starting training...")
                    self.after(0, lambda p=str(out_path): (
                        self._select_generated_data_for_current_mode(p),
                        self._start_training_by_mode()))
                else:
                    self._log(
                        "Select this file as data source "
                        "to train your model on it.")

                # Free trainer memory
                del engine
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            except ImportError:
                self._log(
                    "[!] 'requests' library not installed.\n"
                    "    Run: pip install requests")
            except Exception as exc:
                self._log(f"\n[!] Web learn failed: {exc}")
            finally:
                self.after(
                    0, lambda: self.web_learn_btn.configure(
                        state="normal"))
                self.after(0, lambda: self.status_bar.set_left(
                    "\u26a1 READY"))

        threading.Thread(target=_learn, daemon=True).start()

    # ================================================================
    # Checkpoint save / load
    # ================================================================

    def _save_forge_checkpoint(self):
        """Save the current STUDENT model as an auto-named checkpoint.

        Uses the pattern: modelname_epoch_timestamp so no prompt
        is needed.
        """
        student_path = self.route_assignments.get("student")
        if not student_path or not Path(student_path).exists():
            self._log("[!] No STUDENT model to checkpoint.")
            return

        import shutil
        import time

        model_stem = Path(student_path).stem
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = f"{model_stem}_{ts}"

        ckpt_dir = MODELS_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dest = ckpt_dir / f"{safe}.pth"

        if dest.exists():
            self._log(f"[!] Checkpoint '{safe}' already exists.")
            return

        try:
            shutil.copy2(student_path, str(dest))
            size_mb = round(
                dest.stat().st_size / (1024 * 1024), 1)
            self._log(
                f"Checkpoint saved: {safe} ({size_mb} MB)")
        except OSError as exc:
            self._log(f"[!] Checkpoint save failed: {exc}")

    def _load_forge_checkpoint(self):
        """Load a checkpoint back into the STUDENT model slot."""
        student_path = self.route_assignments.get("student")
        if not student_path:
            self._log(
                "[!] Assign a STUDENT model first to load into.")
            return

        from tkinter import filedialog
        ckpt_dir = MODELS_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        src = filedialog.askopenfilename(
            title="Load Checkpoint",
            initialdir=str(ckpt_dir),
            filetypes=[
                ("PyTorch checkpoints", "*.pth"),
                ("All files", "*.*")])
        if not src:
            return

        import shutil
        try:
            shutil.copy2(src, student_path)
            name = Path(src).stem
            self._log(f"Checkpoint loaded: {name} → STUDENT")
            self.after(0, self._refresh_models)
        except OSError as exc:
            self._log(f"[!] Checkpoint load failed: {exc}")

    # ================================================================
    # Loss curve visualization
    # ================================================================

    def _display_loss_curve(self, losses: list[float]):
        """Render a text-based loss curve in the output log.

        Uses block characters (█) to draw a horizontal bar chart
        showing how loss decreased over training epochs.  Also
        updates the graphical loss chart canvas when available.
        """
        if not losses:
            return

        max_loss = max(losses)
        min_loss = min(losses)
        bar_width = 30

        self._log("\n  Loss Curve:")
        for i, loss in enumerate(losses):
            if max_loss > min_loss:
                ratio = (loss - min_loss) / (max_loss - min_loss)
            else:
                ratio = 1.0
            filled = int(ratio * bar_width)
            bar = "█" * max(1, filled)
            self._log(
                f"    E{i + 1:<3d} {loss:>7.4f}  {bar}")

        self._log(
            f"    Range: {min_loss:.4f} → {max_loss:.4f}")

        # Update graphical chart if canvas exists
        if getattr(self, "_loss_canvas", None) is not None:
            # Compute simple moving average
            window = max(1, len(losses) // 10)
            moving_avg: list[float] = []
            for i in range(len(losses)):
                start = max(0, i - window + 1)
                moving_avg.append(
                    sum(losses[start:i + 1]) / (i - start + 1))

            # Gather perplexities from training monitor if available
            perplexities: list[float] | None = None
            monitor = getattr(self, "_training_monitor", None)
            if monitor is not None:
                ppl = getattr(monitor, "epoch_perplexities", None)
                if ppl:
                    perplexities = list(ppl)

            self._update_loss_chart(losses, moving_avg, perplexities)

            # Auto-expand the loss chart panel
            panel = getattr(self, "_loss_chart_panel", None)
            if panel is not None and hasattr(panel, "expand"):
                self.after(0, panel.expand)

    # ================================================================
    # Update FORGE model status cards
    # ================================================================

    def _update_forge_cards(self):
        """Refresh the trainer/student status cards on the FORGE page.

        Reads the current route assignments and updates the card
        labels, dots, and info text.
        """
        # Trainer card
        trainer_path = self.route_assignments.get("trainer")
        t_dot = getattr(self, "_forge_trainer_dot", None)
        t_name = getattr(self, "_forge_trainer_name", None)
        t_info = getattr(self, "_forge_trainer_info", None)
        if t_dot and t_name:
            if trainer_path and Path(trainer_path).exists():
                name = Path(trainer_path).stem
                p = Path(trainer_path)
                if p.is_dir():
                    ext = "HF"
                    size_mb = round(
                        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                        / (1024 * 1024), 1)
                else:
                    ext = p.suffix[1:].upper()
                    size_mb = round(p.stat().st_size / (1024 * 1024), 1)
                t_dot.set_color(C_GREEN)
                t_name.configure(
                    text=name, text_color=C_GREEN)
                if t_info:
                    t_info.configure(
                        text=f"{ext} // {size_mb} MB",
                        text_color=C_TEXT_DIM)
            else:
                t_dot.set_color(C_TEXT_DIM)
                t_name.configure(
                    text="Not assigned", text_color=C_TEXT_DIM)
                if t_info:
                    t_info.configure(text="")

        # Student card
        student_path = self.route_assignments.get("student")
        s_dot = getattr(self, "_forge_student_dot", None)
        s_name = getattr(self, "_forge_student_name", None)
        s_info = getattr(self, "_forge_student_info", None)
        if s_dot and s_name:
            if student_path and Path(student_path).exists():
                name = Path(student_path).stem
                p = Path(student_path)
                if p.is_dir():
                    ext = "HF"
                    size_mb = round(
                        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                        / (1024 * 1024), 1)
                else:
                    ext = p.suffix[1:].upper()
                    size_mb = round(p.stat().st_size / (1024 * 1024), 1)
                s_dot.set_color(C_GREEN)
                s_name.configure(
                    text=name, text_color=C_GREEN)
                if s_info:
                    s_info.configure(
                        text=f"{ext} // {size_mb} MB",
                        text_color=C_TEXT_DIM)
            else:
                s_dot.set_color(C_TEXT_DIM)
                s_name.configure(
                    text="Not assigned", text_color=C_TEXT_DIM)
                if s_info:
                    s_info.configure(text="")

    # ================================================================
    # Forge param count — shown after training
    # ================================================================

    def _update_forge_param_count(self, param_count: int):
        """Display the student model's param count on the FORGE card.

        Called from the UI thread after each training session completes.
        Replaces any previous param count display.
        """
        label = getattr(self, "_forge_student_params", None)
        if label is None:
            return
        from enigma_engine.gui.scanners import _format_param_count
        text = f"{_format_param_count(param_count)} params"
        label.configure(text=text, text_color=C_GREEN)

    def _clear_forge_param_count(self):
        """Clear the param count label at the start of a new training run."""
        label = getattr(self, "_forge_student_params", None)
        if label is None:
            return
        label.configure(text="", text_color=C_TEXT_DIM)

    # ================================================================
    # Hyperparameter presets
    # ================================================================

    # Preset → (epochs, learning_rate, batch_size)
    _TRAINING_PRESETS = {
        "Quick": ("3", "0.0001", "4"),
        "Balanced": ("10", "0.00005", "4"),
        "Thorough": ("30", "0.00002", "2"),
    }

    def _on_preset_changed(self, choice: str):
        """Apply hyperparameter preset values to epoch/lr/batch fields."""
        preset = self._TRAINING_PRESETS.get(choice)
        if preset is None:
            # "Custom" — leave fields as-is
            return
        epochs, lr, batch = preset
        for entry, val in [
            (self.ft_epochs_entry, epochs),
            (self.ft_lr_entry, lr),
            (self.forge_batch_entry, batch),
        ]:
            entry.delete(0, "end")
            entry.insert(0, val)
        self._log(
            f"Preset '{choice}': {epochs} epochs, "
            f"lr={lr}, batch={batch}")

    # ================================================================
    # Training history
    # ================================================================

    _HISTORY_FILE = DATA_DIR / "training_history.json"

    def _show_training_history(self):
        """Display past training runs from training_history.json."""
        if not self._HISTORY_FILE.exists():
            self._log("[!] No training history yet.")
            return
        try:
            runs = json.loads(
                self._HISTORY_FILE.read_text(encoding="utf-8"))
            if not runs:
                self._log("[!] Training history is empty.")
                return
            self._log(f"--- Training History ({len(runs)} runs) ---")
            for run in runs[-20:]:  # Show last 20 runs
                mode = run.get("mode", "?")
                model = run.get("model", "?")
                epochs = run.get("epochs", "?")
                best_loss = run.get("best_loss", "?")
                timestamp = run.get("timestamp", "?")
                ppl_before = run.get("before_perplexity")
                ppl_after = run.get("after_perplexity")
                ppl_info = ""
                if ppl_before is not None and ppl_after is not None:
                    ppl_info = f"  ppl {ppl_before:.1f}→{ppl_after:.1f}"
                self._log(
                    f"  {timestamp}  {mode:>10}  "
                    f"{model}  epochs={epochs}  "
                    f"loss={best_loss}{ppl_info}")
            self._log("--- End of history ---")
        except Exception as exc:
            self._log(f"[!] Could not read history: {exc}")

    def _save_training_run(
        self, mode: str, model_name: str,
        epochs: int, best_loss: float,
        before_perplexity: float | None = None,
        after_perplexity: float | None = None,
    ):
        """Append a completed training run to training_history.json
        and the model's identity card."""
        import datetime
        entry = {
            "timestamp": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M"),
            "mode": mode,
            "model": model_name,
            "epochs": epochs,
            "best_loss": round(best_loss, 6),
        }
        if before_perplexity is not None:
            entry["before_perplexity"] = round(before_perplexity, 4)
        if after_perplexity is not None:
            entry["after_perplexity"] = round(after_perplexity, 4)
        try:
            runs: list = []
            if self._HISTORY_FILE.exists():
                runs = json.loads(
                    self._HISTORY_FILE.read_text(encoding="utf-8"))
            runs.append(entry)
            # Keep last 200 runs
            if len(runs) > 200:
                runs = runs[-200:]
            self._HISTORY_FILE.write_text(
                json.dumps(runs, indent=2), encoding="utf-8")
            logger.debug("Saved training run: %s", entry)
        except Exception as exc:
            logger.debug("Could not save training run: %s", exc)

        # Also record in the model's identity card
        ctx = getattr(self, "model_context", None)
        if ctx is not None:
            ctx.record_training_run(
                mode=mode, epochs=epochs, best_loss=best_loss,
                before_perplexity=before_perplexity,
                after_perplexity=after_perplexity)
            self._save_model_context()

    # ================================================================
    # Progress bar
    # ================================================================

    def _update_forge_progress(self, pct: int, msg: str):
        """Update the FORGE progress bar and label (thread-safe)."""
        def _do():
            bar = getattr(self, "_forge_progress_bar", None)
            label = getattr(self, "_forge_progress_label", None)
            if bar is not None:
                bar.set(max(0.0, min(1.0, pct / 100)))
            if label is not None:
                label.configure(text=f"{pct}%" if pct > 0 else "")
        self.after(0, _do)

    def _reset_forge_progress(self):
        """Reset the FORGE progress bar to zero (thread-safe)."""
        def _do():
            bar = getattr(self, "_forge_progress_bar", None)
            label = getattr(self, "_forge_progress_label", None)
            if bar is not None:
                bar.set(0)
            if label is not None:
                label.configure(text="")
        self.after(0, _do)

    def _clear_loss_chart(self):
        """Clear any previously rendered loss chart (thread-safe)."""
        def _do():
            canvas = getattr(self, "_loss_canvas", None)
            info_label = getattr(self, "_loss_chart_info", None)
            panel = getattr(self, "_loss_chart_panel", None)

            if canvas is not None:
                canvas.delete("all")
            if info_label is not None:
                info_label.configure(text="No data yet")
            if panel is not None and hasattr(panel, "collapse"):
                panel.collapse()

        self.after(0, _do)

    # ================================================================
    # Loss chart (EV-D)
    # ================================================================

    def _update_loss_chart(self, losses: list[float],
                           moving_avg: list[float] | None = None,
                           perplexities: list[float] | None = None):
        """Redraw the loss chart on the FORGE page (thread-safe).

        Draws a simple line chart of training loss over steps.
        Optionally overlays the moving average line.

        Args:
            losses: List of loss values (one per step or epoch).
            moving_avg: Optional smoothed loss values (same length).
            perplexities: Optional per-epoch perplexity values.
        """
        def _do():
            canvas = getattr(self, "_loss_canvas", None)
            info_label = getattr(self, "_loss_chart_info", None)
            if canvas is None:
                return
            if not losses:
                return

            # Import theme colors
            from enigma_engine.gui.widgets import (
                C_GREEN, C_SURFACE, C_TEXT_DIM, C_ACCENT_DIM,
            )

            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 20 or h < 20:
                # Widget not yet mapped — schedule retry
                self.after(100, lambda: self._update_loss_chart(
                    losses, moving_avg, perplexities))
                return

            # Chart margins
            margin_l = 50
            margin_r = 10
            margin_t = 10
            margin_b = 20
            cw = w - margin_l - margin_r
            ch = h - margin_t - margin_b

            if cw < 10 or ch < 10:
                return

            # Compute data range
            min_loss = min(losses)
            max_loss = max(losses)
            if max_loss == min_loss:
                max_loss = min_loss + 1.0

            n = len(losses)

            def to_x(i):
                return margin_l + (i / max(1, n - 1)) * cw

            def to_y(val):
                frac = (val - min_loss) / (max_loss - min_loss)
                return margin_t + (1 - frac) * ch

            # Draw grid lines (3 horizontal)
            for frac in (0.0, 0.5, 1.0):
                y = margin_t + (1 - frac) * ch
                val = min_loss + frac * (max_loss - min_loss)
                canvas.create_line(
                    margin_l, y, w - margin_r, y,
                    fill=C_SURFACE, dash=(2, 4))
                canvas.create_text(
                    margin_l - 4, y, text=f"{val:.2f}",
                    fill=C_TEXT_DIM, anchor="e",
                    font=("Consolas", 8))

            # Draw loss curve (green)
            if n > 1:
                points = []
                for i, loss in enumerate(losses):
                    points.extend([to_x(i), to_y(loss)])
                canvas.create_line(
                    *points, fill=C_GREEN, width=1,
                    smooth=True)

            # Draw moving average (accent, thicker)
            if moving_avg and len(moving_avg) > 1:
                ma_points = []
                for i, val in enumerate(moving_avg):
                    ma_points.extend([to_x(i), to_y(val)])
                canvas.create_line(
                    *ma_points, fill=C_ACCENT_DIM, width=2,
                    smooth=True)

            # Step axis labels
            if n > 0:
                canvas.create_text(
                    margin_l, h - 2, text="0",
                    fill=C_TEXT_DIM, anchor="w",
                    font=("Consolas", 8))
                canvas.create_text(
                    w - margin_r, h - 2, text=str(n),
                    fill=C_TEXT_DIM, anchor="e",
                    font=("Consolas", 8))

            # Update info label
            if info_label is not None:
                parts = [
                    f"Steps: {n}",
                    f"Loss: {losses[-1]:.4f}",
                    f"Best: {min_loss:.4f}",
                ]
                if perplexities:
                    parts.append(f"PPL: {perplexities[-1]:.1f}")
                info_label.configure(text="  |  ".join(parts))

        self.after(0, _do)
