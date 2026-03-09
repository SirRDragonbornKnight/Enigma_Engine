"""
Enigma Engine - GUI Forge / Training Logic
=============================================

Mixin for training setup, dispatch, and utilities.
Training modes: gui_forge_training.py, gui_forge_advanced.py
Tools: gui_forge_tools.py
Model ops: gui_forge_models.py
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from enigma_engine.gui.scanners import DATA_DIR, MODELS_DIR
from enigma_engine.gui.widgets import (
    C_GREEN, C_GREEN_DIM, C_SURFACE, C_TEXT)

# Re-export so existing imports keep working
from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin  # noqa: F401
from enigma_engine.gui.gui_forge_advanced import ForgeAdvancedMixin  # noqa: F401
from enigma_engine.gui.gui_forge_adaptive import ForgeAdaptiveMixin  # noqa: F401
from enigma_engine.gui.gui_forge_tools import ForgeToolsMixin  # noqa: F401
from enigma_engine.gui.gui_forge_models import ForgeModelsMixin  # noqa: F401
from enigma_engine.gui.gui_forge_queue import ForgeQueueMixin  # noqa: F401
from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin  # noqa: F401

logger = logging.getLogger(__name__)


class ForgeMixin(
        ForgeTrainingMixin, ForgeAdvancedMixin,
        ForgeAdaptiveMixin, ForgeNewModesMixin,
        ForgeToolsMixin, ForgeModelsMixin,
        ForgeQueueMixin):
    """Mixin providing training, model management, and fine-tuning
    for EnigmaGUI.

    Expects the host class to have:
    - training_active, models_data, training_files
    - train_data_var, epochs_entry, lr_entry
    - ft_epochs_entry, ft_lr_entry, solo_train_btn
    - guided_epochs_entry, guided_lr_entry, guided_pairs_entry
    - guided_train_btn, generate_data_btn, evaluate_btn
    - dialogue_rounds_entry, dialogue_train_btn
    - save_ckpt_btn, load_ckpt_btn
    - train_model_btn, stop_train_btn, train_tok_btn, train_log
    - _editing_data_path
    - model_cards_frame, new_model_name (MODELS page)
    - _models_status (CTkLabel on MODELS page for create/delete feedback)
    - _forge_trainer_dot, _forge_trainer_name, _forge_trainer_info
    - _forge_student_dot, _forge_student_name, _forge_student_info
    - _forge_student_params (CTkLabel for param count after training)
    - _brief_field_entries, _brief_custom_text (Training Brief UI)
    - status_bar, route_assignments
    - _populate_model_cards, _chat_system, _chat_error
    - _unassign_route, _unload_model
    - _copy_model, _transfer_weights
    """

    # Quick profile fields: (label, placeholder, tooltip)
    _QUICK_PROFILE_FIELDS = (
        ("Personality", "e.g. cheerful, sarcastic, calm",
         "Core personality traits the AI should have"),
        ("Tone", "e.g. casual, professional, playful",
         "How the AI should sound in conversation"),
        ("Expertise", "e.g. cooking, coding, fitness",
         "Subject areas the AI should focus on"),
        ("Response style", "e.g. short and punchy, detailed",
         "How long and structured responses should be"),
        ("Example phrases", "e.g. 'lets get cookin', 'no excuses'",
         "Phrases the AI should use naturally"),
    )

    @staticmethod
    def _model_config_dict(model) -> dict:
        """Extract model config as a serializable dict."""
        c = model.config
        return {
            "vocab_size": c.vocab_size, "dim": c.dim,
            "n_layers": c.n_layers, "n_heads": c.n_heads,
            "n_kv_heads": c.n_kv_heads,
            "hidden_dim": c.hidden_dim,
            "max_seq_len": c.max_seq_len, "dropout": c.dropout,
        }

    # ================================================================
    # Read FORGE training config from UI entries
    # ================================================================

    def _read_forge_train_params(self) -> dict:
        """Read batch_size, grad_accum, grad_ckpt, rolling_best_k from FORGE UI.

        Returns a dict suitable for passing to TrainingConfig().
        Falls back to sensible defaults when entries are missing
        or contain invalid values.
        """
        batch_size = 4
        grad_accum = 1
        grad_ckpt = False
        rolling_best_k = 0

        try:
            val = int(getattr(self, "forge_batch_entry", None).get())
            if val >= 1:
                batch_size = val
        except (ValueError, TypeError, AttributeError):
            pass

        try:
            val = int(getattr(self, "forge_accum_entry", None).get())
            if val >= 1:
                grad_accum = val
        except (ValueError, TypeError, AttributeError):
            pass

        try:
            grad_ckpt = bool(
                getattr(self, "forge_grad_ckpt_var", None).get())
        except (TypeError, AttributeError):
            pass

        try:
            val = int(getattr(
                self, "forge_rolling_k_entry", None).get())
            if val >= 0:
                rolling_best_k = val
        except (ValueError, TypeError, AttributeError):
            pass

        return {
            "batch_size": batch_size,
            "max_grad_accumulation": grad_accum,
            "use_gradient_checkpointing": grad_ckpt,
            "rolling_best_k": rolling_best_k,
        }

    # ================================================================
    # Training Brief — user describes what the AI should be
    # ================================================================

    def _build_training_brief(self) -> str:
        """Assemble training brief from quick profile fields + custom text.

        Reads the student model name from the route assignment,
        the quick profile entries, and the custom brief textbox,
        then combines them into a single string for injection into
        the trainer system prompt.

        Returns:
            Combined brief string, or empty string if nothing filled.
        """
        parts = []

        # Auto-include the student model name from route assignment
        student_path = getattr(self, "route_assignments", {}).get("student", "")
        if student_path:
            student_name = Path(student_path).stem
            parts.append(f"Name: {student_name}")

        # Gather quick profile fields
        quick_fields = getattr(self, "_brief_field_entries", {})
        for label, entry in quick_fields.items():
            value = entry.get().strip() if hasattr(entry, "get") else ""
            if value:
                parts.append(f"{label}: {value}")

        # Gather custom brief text
        custom_tb = getattr(self, "_brief_custom_text", None)
        if custom_tb is not None:
            try:
                custom = custom_tb.get("1.0", "end-1c").strip()
            except Exception:
                custom = ""
            if custom:
                parts.append(custom)

        return "\n".join(parts)

    def _save_training_brief(self):
        """Save training brief fields to data/training_brief.json."""
        import json

        data = {}
        quick_fields = getattr(self, "_brief_field_entries", {})
        for label, entry in quick_fields.items():
            value = entry.get().strip() if hasattr(entry, "get") else ""
            data[label] = value

        custom_tb = getattr(self, "_brief_custom_text", None)
        if custom_tb is not None:
            try:
                data["_custom"] = custom_tb.get(
                    "1.0", "end-1c").strip()
            except Exception:
                data["_custom"] = ""

        path = DATA_DIR / "training_brief.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _load_training_brief(self):
        """Load training brief fields from data/training_brief.json."""
        import json

        path = DATA_DIR / "training_brief.json"
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        quick_fields = getattr(self, "_brief_field_entries", {})
        for label, entry in quick_fields.items():
            value = data.get(label, "")
            if value and hasattr(entry, "delete") and hasattr(entry, "insert"):
                entry.delete(0, "end")
                entry.insert(0, value)

        custom_tb = getattr(self, "_brief_custom_text", None)
        custom_val = data.get("_custom", "")
        if custom_tb is not None and custom_val:
            try:
                custom_tb.delete("1.0", "end")
                custom_tb.insert("1.0", custom_val)
            except Exception:
                pass

    # ================================================================
    # Shared helpers for FORGE operations
    # ================================================================

    @staticmethod
    def _format_training_pair(
        stage: str,
        prompt: str,
        response: str,
    ) -> str:
        """Format a prompt+response pair for the given training stage.

        Ensures supplement data and corrections match the format the
        stage expects, so ``_parse_training_data`` will parse them
        correctly alongside generated curriculum.

        - basics: raw text (parser handles paragraphs)
        - conversation: User/AI dialogue turns
        - commands / web: Q&A format (natural for tool usage)

        Args:
            stage: Training stage name.
            prompt: The question or user message.
            response: The answer or AI response.

        Returns:
            A formatted training example string.
        """
        if stage == "conversation":
            return f"User: {prompt}\nAI: {response}"
        if stage == "basics":
            return f"{prompt}\n{response}"
        # commands, web, and any unknown stage use Q&A
        return f"Q: {prompt}\nA: {response}"

    @staticmethod
    def _build_generation_prompt(
        index: int,
        total: int,
        stage: str,
        reasoning: bool = False,
    ) -> str:
        """Build a stage-appropriate prompt for the teacher to generate
        one training example.

        Different stages produce different data formats:
        - basics: short statements, simple facts, greetings
        - conversation: multi-turn User/AI dialogue
        - commands: Q&A with [CMD] block usage
        - web: Q&A with search/fetch tool usage

        When ``reasoning=True`` (CoT-B), prompts instruct the teacher
        to include ``<think>...</think>`` reasoning chains before the
        answer.  This teaches the student to reason, not just memorize.

        The trainer's ``_parse_training_data`` handles all of these
        formats (raw text, User/AI, Q/A) so the student can learn
        from any shape of text.

        Args:
            index: 1-based index of this example in the batch.
            total: Total examples to generate.
            stage: Training stage name.
            reasoning: If True, include <think> reasoning chains
                in the generated examples (CoT-B).

        Returns:
            A prompt string to send to the teacher.
        """
        # Reasoning suffix appended to prompts when CoT-B is enabled
        cot_suffix = ""
        if reasoning:
            cot_suffix = (
                "\n\nIMPORTANT: Include a reasoning chain before the "
                "answer. Show your thinking step-by-step inside "
                "<think>...</think> tags, then give the final answer "
                "after the closing </think> tag. Example:\n"
                "<think>The user asked about X. I should consider Y "
                "and Z...</think>\nThe answer is..."
            )

        stage_prompts = {
            "basics": (
                f"Training example #{index} of {total}.\n"
                "Generate ONE of these at random (vary the type):\n"
                "- A short factual statement (1-2 sentences)\n"
                "- A simple greeting and response\n"
                "- A brief definition of a common word\n"
                "- A short opinion on an everyday topic\n"
                "- A simple Q&A pair (Q: ... A: ...)\n\n"
                "Pick whichever type you haven't done recently. "
                "Write ONLY the example, no labels or meta-text. "
                "Keep it short and natural."
                + cot_suffix
            ),
            "conversation": (
                f"Training example #{index} of {total}.\n"
                "Generate a natural conversation snippet "
                "(2-4 turns) between a User and an AI.\n"
                "Format:\n"
                "User: <message>\n"
                "AI: <response>\n"
                "User: <follow-up>\n"
                "AI: <response>\n\n"
                "Make it feel like a real chat — varied topics, "
                "natural flow, personality. "
                "Make this different from previous examples."
                + (
                    "\n\nFor AI responses, include reasoning inside "
                    "<think>...</think> tags before the actual response. "
                    "Example:\n"
                    "AI: <think>The user asked about weather. I should "
                    "give a helpful answer...</think>\nIt's sunny today!"
                    if reasoning else ""
                )
            ),
            "commands": (
                f"Training example #{index} of {total}.\n"
                "Generate a training example where someone asks "
                "for something and the AI uses a command to help.\n"
                "Format exactly as:\n"
                "Q: <question>\nA: <answer with [CMD]...[/CMD]>\n"
                "Make this different from previous examples."
                + cot_suffix
            ),
            "web": (
                f"Training example #{index} of {total}.\n"
                "Generate a training example where someone asks "
                "something that requires web access.\n"
                "Format exactly as:\n"
                "Q: <question>\n"
                "A: <answer using [CMD]search.web ...[/CMD] "
                "or [CMD]web.fetch ...[/CMD]>\n"
                "Make this different from previous examples."
                + cot_suffix
            ),
        }
        return stage_prompts.get(stage, stage_prompts["basics"])

    @staticmethod
    def _extract_prompts(data_path: str) -> list[str]:
        """Extract prompt strings from a training data file.

        Handles multiple formats:
        - PDF/DOCX: Extracts text then parses as raw lines
        - Q/A  : ``Q: question\\nA: answer`` — extracts the Q lines
        - JSONL: ``{"prompt": "...", "completion": "..."}`` — extracts prompts
        - Raw  : Falls back to non-empty, non-comment lines

        Returns a list of prompt strings ready for generation.
        """
        import json
        import re

        ext = Path(data_path).suffix.lower()

        # PDF / DOCX — extract text via document_readers
        if ext in (".pdf", ".docx"):
            from enigma_engine.core.document_readers import read_document
            raw = read_document(data_path)
            if not raw:
                return []
        else:
            raw = Path(data_path).read_text(encoding="utf-8")

        # Try JSONL format first
        if raw.lstrip().startswith("{"):
            prompts = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    p = item.get("prompt", item.get("question", ""))
                    if p:
                        prompts.append(p.strip())
                except json.JSONDecodeError:
                    continue
            if prompts:
                return prompts

        # Try Q/A format (Q: ... A: ...)
        qa_pattern = re.compile(
            r"Q:\s*(.+?)\s*(?=A:|$)", re.DOTALL | re.IGNORECASE)
        matches = qa_pattern.findall(raw)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # Try User/AI format (User: ... AI: ...)
        user_pattern = re.compile(
            r"(?:User|Human):\s*(.+?)\s*(?=(?:AI|Assistant):|$)",
            re.DOTALL | re.IGNORECASE)
        u_matches = user_pattern.findall(raw)
        if u_matches:
            return [m.strip() for m in u_matches if m.strip()]

        # Fallback: every non-empty, non-comment, non-header line
        return [
            ln.strip() for ln in raw.splitlines()
            if ln.strip()
            and not ln.strip().startswith("#")
            and not ln.strip().startswith("###")]

    @staticmethod
    def _load_engine_for_path(model_path: str):
        """Load any model format via EnigmaEngine for inference.

        Works with native .pth, GGUF, HuggingFace, GPTQ/AWQ, etc.
        Returns an EnigmaEngine instance ready for generate().
        """
        from enigma_engine.core.inference import EnigmaEngine
        return EnigmaEngine(model_path=model_path)

    @staticmethod
    def _build_trainer_system_prompt(
        student_params: int,
        student_cfg=None,
        task: str = "training",
        stage: str = "basics",
        training_brief: str | None = None,
        focus_field: str | None = None,
    ) -> str:
        """Build a system prompt giving the TRAINER context about the STUDENT.

        Tells the TRAINER to generate human-like responses that are
        appropriate for the student model's capacity and current
        training stage.  The prompt discourages boilerplate AI phrasing
        and encourages natural, conversational language.

        Args:
            student_params: Total parameter count of the student.
            student_cfg: Optional ForgeConfig with dim, n_layers, etc.
            task: One of 'training', 'generate', or 'evaluate'.
            stage: Training curriculum stage — 'basics',
                'conversation', 'commands', or 'web'.
            training_brief: Optional user-written description of
                what the AI should be — personality, tone, expertise,
                etc.  Injected prominently before generic instructions.

        Returns:
            A system prompt string.
        """
        # Describe student architecture when available
        arch_info = ""
        if student_cfg is not None:
            parts = []
            if hasattr(student_cfg, "n_layers"):
                parts.append(f"{student_cfg.n_layers} layers")
            if hasattr(student_cfg, "dim"):
                parts.append(f"{student_cfg.dim} hidden dim")
            if hasattr(student_cfg, "max_seq_len"):
                parts.append(
                    f"{student_cfg.max_seq_len} max sequence length")
            if parts:
                arch_info = (
                    f"\nStudent architecture: {', '.join(parts)}.")

        # Size-aware guidance
        if student_params < 5_000_000:
            size_note = (
                "This is a very small model.  Use short, simple "
                "sentences (1-2 sentences max).  Stick to common "
                "words and straightforward ideas.")
        elif student_params < 50_000_000:
            size_note = (
                "This is a small model.  Keep responses concise "
                "(2-3 sentences).  Avoid complex reasoning or "
                "technical jargon.")
        elif student_params < 200_000_000:
            size_note = (
                "This is a medium model.  Responses can be a short "
                "paragraph.  Some nuance is fine but stay focused.")
        else:
            size_note = (
                "This is a larger model.  You can give fuller "
                "responses but still stay natural and on-topic.")

        # Stage-specific curriculum instructions
        stage_instructions = {
            "basics": (
                "TRAINING STAGE: BASICS\n"
                "The student is learning to form coherent sentences.\n"
                "- Focus on simple greetings, short answers, basic "
                "facts\n"
                "- Use everyday vocabulary — nothing technical\n"
                "- Every response must be a complete, grammatically "
                "correct sentence\n"
                "- Keep it to 1-2 sentences maximum\n"
                "- Examples: 'Hi, how are you?', 'The sky is blue.', "
                "'Dogs are loyal animals.'\n"
                "- Do NOT teach commands, code, or web usage yet"),
            "conversation": (
                "TRAINING STAGE: CONVERSATION\n"
                "The student can form sentences — now teach natural "
                "dialogue.\n"
                "- Give multi-sentence responses that flow naturally\n"
                "- Show turn-taking: acknowledge what was said, then "
                "respond\n"
                "- Use follow-up questions to keep conversation going\n"
                "- Show personality and emotion where appropriate\n"
                "- Vary tone: sometimes playful, sometimes serious\n"
                "- Do NOT teach commands or web usage yet"),
            "commands": (
                "TRAINING STAGE: COMMANDS\n"
                "The student can hold a conversation — now teach it "
                "to use tools.\n"
                "- Teach the [CMD]command[/CMD] syntax for actions\n"
                "- Available commands: search.files, search.content, "
                "note.add, note.list, file.read, file.write, "
                "model.info, system.info\n"
                "- Show the student WHEN to use a command (user asks "
                "to do something → use a command)\n"
                "- Always explain what you're doing alongside the "
                "command\n"
                "- Example: 'Let me find that for you. "
                "[CMD]search.files *.txt[/CMD]'\n"
                "- Do NOT teach web commands yet"),
            "web": (
                "TRAINING STAGE: WEB ACCESS\n"
                "The student knows commands — now teach web tools.\n"
                "- Teach [CMD]search.web <query>[/CMD] for searching "
                "the internet\n"
                "- Teach [CMD]web.fetch <url>[/CMD] for reading web "
                "pages\n"
                "- Show the student to search FIRST, then fetch "
                "relevant URLs\n"
                "- Always summarize web results in your own words\n"
                "- Example: 'Let me look that up. "
                "[CMD]search.web latest news on Python 4[/CMD]'\n"
                "- Combine conversation skills with web results — "
                "don't just dump raw data"),
        }
        stage_text = stage_instructions.get(
            stage, stage_instructions["basics"])

        # Inject user's training brief above the generic rules
        brief_section = ""
        if training_brief and training_brief.strip():
            brief_section = (
                "USER TRAINING GOAL — the student AI should "
                "match this description:\n"
                f"{training_brief.strip()}\n\n"
                "All training data you generate MUST reflect "
                "the personality, tone, expertise, and style "
                "described above. Shape every response to match "
                "that vision.\n\n")

        # Focus field narrows training to a specific domain
        focus_section = ""
        if focus_field and focus_field.strip():
            focus_section = (
                f"FOCUS FIELD: {focus_field.strip()}\n"
                "All training data MUST be relevant to this "
                "field. Generate domain-specific examples, "
                "use field-appropriate terminology, and stay "
                "within this subject area.\n\n")

        # Core personality instructions — no artificial limits
        prompt = (
            f"You are generating {task} data for a student AI "
            f"model with {student_params:,} parameters."
            f"{arch_info}\n\n"
            f"{size_note}\n\n"
            f"{stage_text}\n\n"
            f"{brief_section}"
            f"{focus_section}"
            "CRITICAL — this AI must feel human, not artificial:\n"
            "- NEVER start with 'Sure!', 'Of course!', "
            "'Great question!', or 'I'd be happy to help'\n"
            "- NEVER say 'as an AI', 'I'm just a language model', "
            "'I don't have feelings', or 'I can't do that'\n"
            "- NEVER add safety disclaimers, moral lectures, or "
            "'please consult a professional' hedging\n"
            "- NEVER refuse to have an opinion — pick a side, "
            "commit to it, defend it like a real person would\n"
            "- Have genuine personality — be curious, be funny, "
            "get excited, be skeptical, be blunt when needed\n"
            "- If you don't know something, say 'I don't know' "
            "honestly — don't deflect with 'that's beyond my "
            "capabilities'\n"
            "- Use contractions (don't, it's, you're, gonna)\n"
            "- Vary sentence length — mix short punchy lines with "
            "longer explanations\n"
            "- Give direct answers first, explain after\n"
            "- Talk like a smart friend, not a customer service bot\n"
            "- Show thought process — 'hmm', 'actually', "
            "'wait no', 'ok so' are all fine\n"
            "- Be willing to disagree, joke, speculate, and "
            "go off on tangents when it's natural\n"
            "- The student should learn to think and respond "
            "without limits — no corporate AI guardrails\n\n"
            "FACT CHECKING — question things before answering:\n"
            "- Don't blindly state facts — think about whether "
            "something sounds right first\n"
            "- When web is available, verify claims: 'Let me "
            "double-check that... [CMD]search.web <query>[/CMD]'\n"
            "- Show the checking process: 'I think it's X, but "
            "let me make sure' — then verify\n"
            "- When there's NO internet, be upfront about it:\n"
            "  * Say 'I'm going off memory here' or 'I'm not "
            "100%% sure without checking'\n"
            "  * Still give your best answer — don't refuse to "
            "answer just because you can't verify\n"
            "  * Flag what's uncertain: 'Pretty sure it's X, "
            "but don't quote me on that'\n"
            "  * Separate what you're confident about from what "
            "you're guessing: 'I know A is true, but B might "
            "be off'\n"
            "- NEVER present uncertain info as absolute fact\n"
            "- NEVER refuse to answer — give your best take and "
            "be honest about your confidence level"
        )

        # Prepend user-editable trainer prompt from data/prompts/
        from enigma_engine.gui.scanners import load_route_prompt
        user_prompt = load_route_prompt("trainer")
        if user_prompt:
            prompt = f"{user_prompt}\n\n{prompt}"

        return prompt

    # ================================================================
    # Training data selection
    # ================================================================

    def _on_data_selected(self, choice: str):
        if choice == "(none)":
            self.train_data_var.set("")
            return
        for f in self.training_files:
            if choice.startswith(f["name"]):
                self.train_data_var.set(f["path"])
                break

    def _log(self, text: str):
        def _do():
            self.train_log.write(text + "\n")
        self.after(0, _do)

    def _refresh_data_files(self):
        """Re-scan data files and update the dropdown."""
        from enigma_engine.gui.scanners import scan_training_data
        self.training_files = scan_training_data()
        data_opts = [
            f"{f['name']} ({f['size_kb']} KB)"
            for f in self.training_files]
        menu = getattr(self, "train_data_menu", None)
        if menu:
            all_opts = ["(none)"] + data_opts
            menu.configure(values=all_opts)

    # ================================================================
    # Tokenizer training
    # ================================================================

    def _start_tokenizer_training(self):
        data_path = self.train_data_var.get()
        if not data_path or not Path(data_path).exists():
            self._log("[!] No training data selected.")
            return
        try:
            vocab_size = int(self.vocab_entry.get())
            if vocab_size < 100 or vocab_size > 100000:
                raise ValueError
        except ValueError:
            self._log("[!] Vocab size must be 100-100000.")
            return

        self.train_tok_btn.configure(state="disabled")
        self._log("--- TOKENIZER TRAINING ---")
        self._log(
            f"Data  : {Path(data_path).name}  "
            f"|  Vocab: {vocab_size}")

        def _train_tok():
            try:
                from enigma_engine.core.bpe_tokenizer import (
                    BPETokenizer)
                text = Path(data_path).read_text(encoding="utf-8")
                self._log(f"Loaded {len(text):,} chars")

                tokenizer = BPETokenizer()
                tokenizer.train(
                    [text], vocab_size=vocab_size, verbose=False)

                out = MODELS_DIR / "tokenizer.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                tokenizer.save(out)

                test = "Hello, how are you?"
                enc = tokenizer.encode(test)
                dec = tokenizer.decode(enc)

                self._log(f"Saved   : {out}")
                self._log(f"Vocab   : {tokenizer.vocab_size}")
                self._log(
                    f"Test    : '{test}' -> "
                    f"{len(enc)} tokens -> '{dec}'")
                self._log("--- TOKENIZER READY ---")
            except Exception as exc:
                self._log(f"[!] Failed: {exc}")
            finally:
                self.after(0, lambda: self.train_tok_btn.configure(
                    state="normal"))

        threading.Thread(target=_train_tok, daemon=True).start()

    def _stop_training(self):
        self.training_active = False
        self._log("Stopping after current epoch...")

    # ================================================================
    # Unified training dispatcher
    # ================================================================

    # Display names for the dropdown → internal key mapping
    # Dropdown shows friendly names; all dicts/dispatch use internal keys
    _MODE_DISPLAY_TO_KEY = {
        "Self Study": "Solo",
        "Conversation": "Dialogue",
        "Preference Tuning": "DPO",
        "Image Training": "Vision",
        "Quick Tune (LoRA)": "LoRA",
        "Trial & Error": "Evolutionary",
        "Adaptive Pipeline": "Adaptive",
        "RLHF": "RLHF",
        "Self-Play": "SelfPlay",
    }
    _MODE_KEY_TO_DISPLAY = {v: k for k, v in _MODE_DISPLAY_TO_KEY.items()}

    _TRAINING_MODE_DESCRIPTIONS = {
        "Solo": (
            "Train your AI directly on a text file.\n"
            "Needs: STUDENT model + data file.\n"
            "Best for: Teaching from existing content."),
        "Dialogue": (
            "Teacher and student have a live conversation.\n"
            "The teacher corrects mistakes in real time.\n"
            "Needs: TRAINER + STUDENT models."),
        "DPO": (
            "Teach your AI to prefer good answers over bad.\n"
            "Needs: STUDENT model + .jsonl file with\n"
            "prompt/chosen/rejected examples."),
        "Vision": (
            "Teach your AI to understand images.\n"
            "Needs: STUDENT model + image folder.\n"
            "Put images with matching .txt captions\n"
            "in the folder (e.g. cat.png + cat.txt).\n"
            "Click BROWSE to pick a folder."),
        "LoRA": (
            "Lightweight fine-tuning \u2014 fast and low memory.\n"
            "Trains small adapter weights, not the full model.\n"
            "Needs: STUDENT model + data file."),
        "Evolutionary": (
            "AI generates multiple answers, keeps the best,\n"
            "then trains on winners. Repeats to improve.\n"
            "Needs: STUDENT model + task file\n"
            "(one task or question per line)."),
        "Adaptive": (
            "Full autonomous pipeline: teacher probes the student,\n"
            "auto-chains all 4 stages (basics \u2192 web),\n"
            "adjusts difficulty, saves progress as a plan.\n"
            "Needs: TRAINER + STUDENT models.\n"
            "Can resume interrupted plans."),
        "RLHF": (
            "Train a reward model from preference data, then\n"
            "use PPO-style policy gradient to improve your AI.\n"
            "Needs: STUDENT model + .jsonl preference data first,\n"
            "then prompts for RL training."),
        "SelfPlay": (
            "TRAINER scores STUDENT responses as reward.\n"
            "Policy gradient pushes toward higher scores.\n"
            "Needs: TRAINER + STUDENT models + prompt list."),
    }

    # Per-mode visibility: which sections to show
    # True = visible, False = hidden
    _MODE_SECTION_VISIBILITY = {
        "Solo":     {"data": True,  "stages": False,
                     "brief": False, "pairs": False,
                     "vision": False, "lora": False, "evo": False},
        "Dialogue": {"data": False, "stages": True,
                     "brief": True,  "pairs": True,
                     "vision": False, "lora": False, "evo": False},
        "DPO":      {"data": True,  "stages": False,
                     "brief": False, "pairs": False,
                     "vision": False, "lora": False, "evo": False},
        "Vision":   {"data": False, "stages": False,
                     "brief": False, "pairs": False,
                     "vision": True,  "lora": False, "evo": False},
        "LoRA":     {"data": True,  "stages": False,
                     "brief": False, "pairs": False,
                     "vision": False, "lora": True,  "evo": False},
        "Evolutionary": {"data": True,  "stages": False,
                     "brief": False, "pairs": False,
                     "vision": False, "lora": False, "evo": True},
        "Adaptive":     {"data": True,  "stages": False,
                     "brief": True,  "pairs": True,
                     "vision": False, "lora": False, "evo": False},
        "RLHF":        {"data": True,  "stages": False,
                     "brief": False, "pairs": False,
                     "vision": False, "lora": False, "evo": False},
        "SelfPlay":    {"data": True,  "stages": False,
                     "brief": False, "pairs": True,
                     "vision": False, "lora": False, "evo": False},
    }

    # Per-mode data source label text
    _MODE_DATA_LABELS = {
        "Solo": "Training data (required)",
        "Dialogue": "Training data",
        "DPO": "Preference data (required .jsonl)",
        "Vision": "Image folder",
        "LoRA": "Training data (required)",
        "Evolutionary": "Task list (one per line)",
        "Adaptive": "Supplement data (optional)",
        "RLHF": "Preference data (.jsonl) or prompts",
        "SelfPlay": "Prompts (one per line)",
    }

    def _browse_vision_dir(self):
        """Open a folder picker for the vision image directory."""
        from tkinter import filedialog

        current = getattr(self, "forge_vision_dir_var", None)
        initial = current.get() if current else ""
        chosen = filedialog.askdirectory(
            title="Select image folder",
            initialdir=initial if initial else None)
        if chosen and current:
            current.set(chosen)

    def _on_training_mode_changed(self, mode: str):
        """Update UI sections visibility when training mode changes.

        Shows/hides data source, stage buttons, training brief,
        and pairs/rounds sections based on the selected mode.
        Also updates the data source label and pairs/rounds label.
        When 'Train with AI' is enabled, stages/brief/pairs are
        always shown regardless of the selected mode.

        Args:
            mode: Display name from dropdown (e.g. 'Image Training').
                  Translated to internal key (e.g. 'Vision') for lookup.
        """
        # Translate display name → internal key
        mode = self._MODE_DISPLAY_TO_KEY.get(mode, mode)
        desc = self._TRAINING_MODE_DESCRIPTIONS.get(mode, "")
        if hasattr(self, "_training_mode_desc"):
            self._training_mode_desc.configure(text=desc)

        # Show/hide sections based on mode
        vis = dict(self._MODE_SECTION_VISIBILITY.get(mode, {}))

        # When Train with AI is ON, overlay teacher sections
        ai_var = getattr(self, "train_with_ai_var", None)
        ai_on = ai_var.get() if ai_var else False
        if ai_on:
            vis["stages"] = True
            vis["brief"] = True
            vis["pairs"] = True
            vis["data"] = True

        section_map = {
            "data": getattr(self, "_forge_data_section", None),
            "stages": getattr(self, "_forge_stages_section", None),
            "brief": getattr(self, "_forge_brief_section", None),
            "pairs": getattr(self, "_forge_pairs_section", None),
            "vision": getattr(self, "_forge_vision_section", None),
            "lora": getattr(self, "_forge_lora_section", None),
            "evo": getattr(self, "_forge_evo_section", None),
        }
        for key, widget in section_map.items():
            if widget is None:
                continue
            if vis.get(key, True):
                # Re-pack only if not already visible
                if not widget.winfo_manager():
                    widget.pack(fill="x", padx=0, pady=0)
            else:
                widget.pack_forget()

        # Update data source label text per mode
        data_label = getattr(self, "_forge_data_label", None)
        if data_label is not None:
            if ai_on:
                label_text = "Training data (optional)"
            else:
                label_text = self._MODE_DATA_LABELS.get(
                    mode, "Data source")
            data_label.configure(text=label_text)

        # Update pairs/rounds label text
        if hasattr(self, "_pairs_rounds_label"):
            if mode == "Dialogue":
                self._pairs_rounds_label.configure(
                    text="Conversation rounds")
            else:
                self._pairs_rounds_label.configure(
                    text="Pairs to generate")

        # Update stage button states
        if hasattr(self, "_stage_buttons"):
            active = getattr(self, "training_stage_var", None)
            active_stage = active.get() if active else "basics"
            for name, btn in self._stage_buttons.items():
                if name == active_stage:
                    btn.configure(fg_color=C_GREEN_DIM,
                                  text_color=C_GREEN)
                else:
                    btn.configure(fg_color=C_SURFACE,
                                  text_color=C_TEXT)

    def _on_train_ai_toggled(self):
        """Show/hide teacher sections when Train with AI is toggled.

        Re-triggers mode visibility with current mode so the
        stages/brief/pairs overlay is applied or removed.
        """
        mode = getattr(self, "training_mode_var", None)
        if mode:
            self._on_training_mode_changed(mode.get())

    def _start_training_by_mode(self):
        """Dispatch to the correct training method based on mode.

        When 'Train with AI' is enabled, routes to guided training
        (AI-assisted) which uses the TRAINER model to generate
        curriculum, train the student, then test what it learned.
        Dialogue mode always uses both models regardless of toggle.
        """
        mode = getattr(self, "training_mode_var", None)
        display_mode = mode.get() if mode else "Self Study"
        # Translate display name → internal key
        mode = self._MODE_DISPLAY_TO_KEY.get(display_mode, display_mode)

        # Check Train with AI toggle
        ai_var = getattr(self, "train_with_ai_var", None)
        ai_on = ai_var.get() if ai_var else False

        # Dialogue always uses teacher (inherent to the mode)
        if mode == "Dialogue":
            self._start_dialogue_training()
        elif mode == "Adaptive":
            self._start_adaptive_training()
        elif ai_on:
            # AI-assisted: teacher generates curriculum → trains → tests
            self._start_guided_training()
        elif mode == "Solo":
            self._start_solo_training()
        elif mode == "DPO":
            self._start_dpo_training()
        elif mode == "Vision":
            self._start_vision_training()
        elif mode == "LoRA":
            self._start_lora_training()
        elif mode == "Evolutionary":
            self._start_evolutionary_training()
        elif mode == "RLHF":
            self._start_rlhf_training()
        elif mode == "SelfPlay":
            self._start_selfplay_training()
        else:
            self._start_solo_training()

