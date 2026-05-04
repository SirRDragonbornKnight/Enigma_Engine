"""
Enigma Engine - GUI Forge Page Builder
========================================

Mixin providing the FORGE page construction for EnigmaGUI.
Split from gui_pages.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT_DIM, C_BG, C_BORDER,
    C_GREEN, C_GREEN_DIM, C_INPUT,
    C_PANEL, C_RED, C_SURFACE, C_TEXT, C_TEXT_BRIGHT,
    C_TEXT_DIM,
    FONT_MONO, FONT_SECTION, FONT_SMALL, FONT_TINY,
    CollapsiblePanel, HUDFrame, SectionLabel, SelectableLabel,
    SelectableTextbox, StatusDot, Tooltip,
    themed_button, themed_dropdown, themed_entry, themed_numeric_entry,
    themed_scroll, wire_hotkeys,
)
from enigma_engine.gui.scanners import DATA_DIR

logger = logging.getLogger(__name__)


class ForgePageMixin:
    """Mixin providing the FORGE page builder for EnigmaGUI.

    Expects the host class to have:
    - _make_page, training_files, _QUICK_PROFILE_FIELDS
    - Various forge callback methods from ForgeMixin
    """

    def _build_mode_card(self, parent, mode_name, mode_desc):
        """Create a single radio-button mode card."""
        mode_card = HUDFrame(parent, glow_color=C_BORDER)
        mode_card.pack(fill="x", pady=(0, 4))

        card_inner = ctk.CTkFrame(
            mode_card, fg_color="transparent")
        card_inner.pack(fill="x", padx=8, pady=8)
        card_inner.grid_columnconfigure(1, weight=1)

        radio = ctk.CTkRadioButton(
            card_inner, text="", variable=self.training_mode_var,
            value=mode_name, font=FONT_SMALL,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM,
            command=self._on_training_mode_selected)
        radio.grid(row=0, column=0, rowspan=2, sticky="nw",
                   padx=(0, 8))

        SelectableLabel(
            card_inner, text=mode_name.upper(),
            font=FONT_SMALL, text_color=C_TEXT_BRIGHT,
            anchor="w"
        ).grid(row=0, column=1, sticky="w")

        desc_lbl = ctk.CTkLabel(
            card_inner, text=mode_desc,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            anchor="w", justify="left",
            wraplength=300)
        desc_lbl.grid(row=1, column=1, sticky="ew", pady=(2, 0))
        self._forge_mode_desc_labels.append(desc_lbl)

    # ================================================================
    # PAGE: FORGE - Training
    # ================================================================

    def _build_page_forge(self):
        page = self._make_page("FORGE")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, sticky="ew",
                 padx=10, pady=(8, 2))
        SectionLabel(top, "The Forge").pack(
            side="left", fill="x", expand=True)

        # Resizable paned layout: controls | log
        import tkinter as tk
        self._forge_pane = tk.PanedWindow(
            page, orient="horizontal", sashwidth=6,
            bg=C_BG, borderwidth=0, sashpad=0,
            opaqueresize=True)
        self._forge_pane.grid(row=1, column=0, sticky="nsew",
                              padx=0, pady=(2, 8))
        self._forge_pane.configure(sashcursor="sb_h_double_arrow")

        # Left: controls
        controls = HUDFrame(self._forge_pane, glow_color=C_BORDER)
        ctrl_scroll = themed_scroll(controls)
        ctrl_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Assigned models ---
        self._forge_heading(ctrl_scroll, "ASSIGNED MODELS")

        # Trainer status card
        self._forge_trainer_card = HUDFrame(
            ctrl_scroll, glow_color=C_BORDER)
        self._forge_trainer_card.pack(
            fill="x", padx=10, pady=(2, 3))
        tr_inner = ctk.CTkFrame(
            self._forge_trainer_card, fg_color="transparent")
        tr_inner.pack(fill="x", padx=8, pady=6)
        tr_inner.grid_columnconfigure(1, weight=1)
        self._forge_trainer_dot = StatusDot(
            tr_inner, color=C_TEXT_DIM)
        self._forge_trainer_dot.grid(
            row=0, column=0, rowspan=3, padx=(0, 6))
        SelectableLabel(
            tr_inner, text="TRAINER", font=FONT_SMALL,
            text_color=C_TEXT_BRIGHT, anchor="w"
        ).grid(row=0, column=1, sticky="w")
        self._forge_trainer_name = SelectableLabel(
            tr_inner, text="Not assigned", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._forge_trainer_name.grid(row=1, column=1, sticky="w")
        self._forge_trainer_info = SelectableLabel(
            tr_inner, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._forge_trainer_info.grid(
            row=2, column=1, sticky="w", pady=(1, 0))

        # Student status card
        self._forge_student_card = HUDFrame(
            ctrl_scroll, glow_color=C_BORDER)
        self._forge_student_card.pack(
            fill="x", padx=10, pady=(0, 6))
        st_inner = ctk.CTkFrame(
            self._forge_student_card, fg_color="transparent")
        st_inner.pack(fill="x", padx=8, pady=6)
        st_inner.grid_columnconfigure(1, weight=1)
        self._forge_student_dot = StatusDot(
            st_inner, color=C_TEXT_DIM)
        self._forge_student_dot.grid(
            row=0, column=0, rowspan=4, padx=(0, 6))
        SelectableLabel(
            st_inner, text="STUDENT", font=FONT_SMALL,
            text_color=C_TEXT_BRIGHT, anchor="w"
        ).grid(row=0, column=1, sticky="w")
        self._forge_student_name = SelectableLabel(
            st_inner, text="Not assigned", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._forge_student_name.grid(row=1, column=1, sticky="w")
        self._forge_student_info = SelectableLabel(
            st_inner, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._forge_student_info.grid(
            row=2, column=1, sticky="w", pady=(1, 0))

        # Param count label — updated after each training session
        self._forge_student_params = SelectableLabel(
            st_inner, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._forge_student_params.grid(
            row=3, column=1, sticky="w", pady=(2, 0))

        # Kept for backward compatibility
        self._trainer_route_label = self._forge_trainer_name
        self._student_route_label = self._forge_student_name

        # --- Training section ---
        self._forge_heading(ctrl_scroll, "TRAIN")

        # ===== NEW SIMPLIFIED TRAINING MODE SELECTOR =====
        # Instead of dropdown, use card-based selection
        self._forge_label(ctrl_scroll, "Training method")

        # Card container
        modes_frame = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        modes_frame.pack(fill="x", padx=10, pady=(0, 4))

        self.training_mode_var = ctk.StringVar(value="Basic")

        # Foundation modes — build the model
        foundation_modes = [
            ("Pre-Train", "Language pre-training from scratch on large text data"),
            ("Distill", "Teacher generates personality, reasoning, and knowledge data"),
            ("Basic", "Train on your own data (text files, JSONL). Auto-LoRA for large models"),
            ("LoRA", "Force low-rank adapter training on any model size. 10-30 MB adapter"),
            ("Image", "Train on images or video. Teach visual understanding"),
        ]

        # Advanced modes — refine the model
        advanced_modes = [
            ("AI-Guided", "AI teacher creates curriculum and trains your model adaptively"),
            ("Dialogue", "Teacher and student converse. Teacher scores and corrects responses"),
            ("RLHF", "Reinforcement learning from human feedback using preference data"),
            ("Self-Play", "Teacher judges student responses. Student improves via RL"),
        ]

        # Alignment modes — preference & RL alternatives
        alignment_modes = [
            ("GRPO", "Group Relative Policy Optimization. RL without a critic network"),
            ("ReMax", "REINFORCE with mean-reward baseline. Simpler than PPO"),
            ("SimPO", "Simple Preference Optimization. No reference model needed"),
            ("ORPO", "Odds Ratio Preference Optimization. SFT + alignment in one step"),
            ("APO", "Anchored Preference Optimization (zero). Both sides anchored to reference independently"),
        ]

        # Collect description labels for dynamic wraplength
        self._forge_mode_desc_labels: list[ctk.CTkLabel] = []

        # --- Foundation row ---
        SectionLabel(modes_frame, text="Foundation").pack(
            anchor="w", padx=4, pady=(0, 2))
        for mode_name, mode_desc in foundation_modes:
            self._build_mode_card(modes_frame, mode_name, mode_desc)

        # --- Advanced row ---
        SectionLabel(modes_frame, text="Advanced").pack(
            anchor="w", padx=4, pady=(8, 2))
        for mode_name, mode_desc in advanced_modes:
            self._build_mode_card(modes_frame, mode_name, mode_desc)

        # --- Alignment row ---
        SectionLabel(modes_frame, text="Alignment").pack(
            anchor="w", padx=4, pady=(8, 2))
        for mode_name, mode_desc in alignment_modes:
            self._build_mode_card(modes_frame, mode_name, mode_desc)

        # Dynamic wraplength: adjust desc labels when panel resizes
        def _on_modes_resize(event):
            w = event.width - 60  # account for radio + padding
            if w < 100:
                return
            for lbl in self._forge_mode_desc_labels:
                lbl.configure(wraplength=w)
        modes_frame.bind("<Configure>", _on_modes_resize)

        # Include reasoning (CoT-B) — generates <think> chains in data
        self._forge_reasoning_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_reasoning_section.pack(fill="x", padx=10, pady=(0, 6))
        self.forge_reasoning_var = ctk.BooleanVar(value=False)
        self._forge_reasoning_cb = ctk.CTkCheckBox(
            self._forge_reasoning_section, text="Include reasoning chains",
            variable=self.forge_reasoning_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        self._forge_reasoning_cb.pack(anchor="w")
        Tooltip(self._forge_reasoning_cb,
                "Generate training data with <think> reasoning\n"
                "chains. Teaches the model to reason step-by-step\n"
                "before answering, not just memorize answers.\n"
                "Applies to AI-Guided mode.")

        # Evolutionary selection — generate multiple answers, keep best
        self._forge_evolutionary_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_evolutionary_section.pack(
            fill="x", padx=10, pady=(0, 6))
        self.forge_evolutionary_var = ctk.BooleanVar(value=False)
        self._forge_evolutionary_cb = ctk.CTkCheckBox(
            self._forge_evolutionary_section,
            text="Evolutionary selection",
            variable=self.forge_evolutionary_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        self._forge_evolutionary_cb.pack(anchor="w")
        Tooltip(self._forge_evolutionary_cb,
                "For each task, generate multiple candidate answers,\n"
                "score them, keep the best, and train on winners.\n"
                "Repeats over generations to improve quality.\n"
                "Needs a task file (one task per line).\n"
                "Works with Basic, AI-Guided, RLHF, and Self-Play.")

        # === KNOWLEDGE PRESERVATION ===
        # The AI should be general-purpose. Training focuses it on a
        # topic without forgetting everything else.
        self._forge_preserve_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_preserve_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_preserve_section, "Knowledge preservation")

        # Focus balance — how much to focus vs stay general
        focus_row = ctk.CTkFrame(
            self._forge_preserve_section, fg_color="transparent")
        focus_row.pack(fill="x", padx=10, pady=(0, 4))
        SelectableLabel(
            focus_row, text="General mix:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.forge_general_mix_var = ctk.StringVar(value="20")
        general_mix_entry = themed_numeric_entry(
            focus_row, mode="int",
            textvariable=self.forge_general_mix_var,
            width=50)
        general_mix_entry.pack(side="left", padx=(0, 4))
        SelectableLabel(
            focus_row, text="%", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left")
        Tooltip(general_mix_entry,
                "Prevents the model from forgetting what it already knows.\n"
                "Mixes in general knowledge alongside your training data.\n"
                "0% = only your data (risky — can cause forgetting).\n"
                "10-20% = recommended for existing models.\n"
                "Only matters when training an existing model, not from scratch.")

        # General knowledge file
        gen_data_row = ctk.CTkFrame(
            self._forge_preserve_section, fg_color="transparent")
        gen_data_row.pack(fill="x", padx=10, pady=(0, 6))
        gen_data_row.grid_columnconfigure(0, weight=1)
        self.forge_general_data_var = ctk.StringVar(value="")
        self._forge_general_data_entry = themed_entry(
            gen_data_row,
            textvariable=self.forge_general_data_var,
            placeholder_text="General knowledge file (optional)")
        self._forge_general_data_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._forge_general_browse_btn = themed_button(
            gen_data_row, "Browse", style="secondary",
            width=70,
            font=FONT_SMALL,
            command=lambda: self._browse_training_data(
                target_var=self.forge_general_data_var))
        self._forge_general_browse_btn.grid(
            row=0, column=1, sticky="e")
        Tooltip(self._forge_general_data_entry,
                "Optional file with diverse/general knowledge.\n"
                "Mixed into training so the AI stays well-rounded.\n"
                "Good sources: previous training data, Q&A sets,\n"
                "general facts, conversation examples.\n"
                "Leave empty to skip mixing.")

        # === PRE-TRAIN OPTIONS ===
        self._forge_pretrain_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_pretrain_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_pretrain_section,
            "Pre-training data (required)")
        # D-11c wiring (Pass 156m): prefer data/pretrain/combined.txt or
        # combined_pretrain.txt when one of those has been collected, so
        # the picker surfaces the real corpus without manual nav.
        from enigma_engine.gui.scanners import _pick_default_pretrain_file
        self.pretrain_data_var = ctk.StringVar(
            value=_pick_default_pretrain_file(self.training_files))
        pt_data_row = ctk.CTkFrame(
            self._forge_pretrain_section, fg_color="transparent")
        pt_data_row.pack(fill="x", padx=10, pady=(0, 4))
        pt_data_row.grid_columnconfigure(0, weight=1)
        self._pretrain_data_entry = themed_entry(
            pt_data_row, textvariable=self.pretrain_data_var,
            placeholder_text="Path to text file or directory...")
        self._pretrain_data_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._pretrain_browse_btn = themed_button(
            pt_data_row, "Browse", style="secondary",
            width=70,
            font=FONT_SMALL,
            command=lambda: self._browse_training_data(
                target_var=self.pretrain_data_var))
        self._pretrain_browse_btn.grid(row=0, column=1, sticky="e")
        Tooltip(self._pretrain_data_entry,
                "Path to a large text file or directory of text files\n"
                "for pre-training. Supports .txt, .jsonl formats.\n"
                "Use collect_pretraining_data.py to build a dataset.")

        # Tokenizer vocab size
        tok_row = ctk.CTkFrame(
            self._forge_pretrain_section, fg_color="transparent")
        tok_row.pack(fill="x", padx=10, pady=(0, 4))
        SelectableLabel(
            tok_row, text="Vocab size:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.pretrain_vocab_var = ctk.StringVar(value="32000")
        pretrain_vocab_entry = themed_entry(
            tok_row, textvariable=self.pretrain_vocab_var, width=80)
        pretrain_vocab_entry.pack(side="left", padx=(0, 6))
        self.pretrain_retrain_tok_var = ctk.BooleanVar(value=True)
        retrain_tok_cb = ctk.CTkCheckBox(
            tok_row, text="Retrain tokenizer on data",
            variable=self.pretrain_retrain_tok_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        retrain_tok_cb.pack(side="left")
        self.pretrain_utf8_bytes_var = ctk.BooleanVar(value=False)
        utf8_cb = ctk.CTkCheckBox(
            tok_row, text="Byte-level BPE",
            variable=self.pretrain_utf8_bytes_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        utf8_cb.pack(side="left", padx=(10, 0))
        Tooltip(pretrain_vocab_entry,
                "Tokenizer vocabulary size for the new model.\n"
                "32K-50K is standard for pre-training.\n"
                "Larger vocab = better coverage, more params.")
        Tooltip(retrain_tok_cb,
                "Train a new BPE tokenizer as part of pre-training.\n"
                "Runs automatically before training starts.\n"
                "For standalone tokenizer training, use FORGE Tools.")
        Tooltip(utf8_cb,
                "Use byte-level BPE encoding.\n"
                "Handles any Unicode character (emoji, CJK, etc.)\n"
                "by encoding text as UTF-8 bytes before BPE merges.")

        # === DISTILLATION OPTIONS ===
        self._forge_distill_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_distill_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_distill_section,
            "Distillation settings")

        # Distillation categories — what kind of data to generate
        DISTILL_CATEGORIES = [
            ("personality", "Personality and tone"),
            ("reasoning", "Reasoning and problem-solving"),
            ("knowledge", "General knowledge Q&A"),
            ("conversation", "Natural conversation"),
            ("commands", "Tool and command usage"),
            ("creativity", "Creative writing and stories"),
        ]
        self.distill_categories = {}
        cat_frame = ctk.CTkFrame(
            self._forge_distill_section, fg_color="transparent")
        cat_frame.pack(fill="x", padx=10, pady=(0, 4))
        self._forge_label(cat_frame, "Data categories")
        for cat_key, cat_label in DISTILL_CATEGORIES:
            var = ctk.BooleanVar(value=True)
            self.distill_categories[cat_key] = var
            cb = ctk.CTkCheckBox(
                cat_frame, text=cat_label, variable=var,
                font=FONT_TINY, text_color=C_TEXT,
                fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
                border_color=C_ACCENT_DIM, corner_radius=2)
            cb.pack(anchor="w", pady=(0, 2))

        # Number of examples per category
        examples_row = ctk.CTkFrame(
            self._forge_distill_section, fg_color="transparent")
        examples_row.pack(fill="x", padx=10, pady=(4, 4))
        SelectableLabel(
            examples_row, text="Examples per category:",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.distill_num_examples_var = ctk.StringVar(value="50")
        distill_examples_entry = themed_numeric_entry(
            examples_row, mode="int",
            textvariable=self.distill_num_examples_var,
            width=60)
        distill_examples_entry.pack(side="left")
        Tooltip(distill_examples_entry,
                "Number of training examples to generate per\n"
                "selected category. Total will be categories x this.\n"
                "More examples = better quality but slower.")

        # Max response length for generated examples
        distill_len_row = ctk.CTkFrame(
            self._forge_distill_section, fg_color="transparent")
        distill_len_row.pack(fill="x", padx=10, pady=(0, 6))
        SelectableLabel(
            distill_len_row, text="Max response length:",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.distill_max_tokens_var = ctk.StringVar(value="512")
        distill_len_entry = themed_numeric_entry(
            distill_len_row, mode="int",
            textvariable=self.distill_max_tokens_var,
            width=60)
        distill_len_entry.pack(side="left")
        Tooltip(distill_len_entry,
                "Max tokens per teacher response.\n"
                "Longer = richer examples, shorter = faster.")

        # --- External teacher (HTTP) — subprocess wrapper around
        # collect_distill_data.py. Generates a corpus from an off-engine
        # model (Ollama / vLLM / llama.cpp / OpenAI-compatible). On
        # success the FORGE training-data picker auto-fills with the
        # produced .txt path.
        ext_section = ctk.CTkFrame(
            self._forge_distill_section, fg_color="transparent")
        ext_section.pack(fill="x", padx=10, pady=(6, 4))
        SelectableLabel(
            ext_section, text="External teacher (HTTP)",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", pady=(0, 2))

        # Endpoint URL row
        ep_row = ctk.CTkFrame(ext_section, fg_color="transparent")
        ep_row.pack(fill="x", pady=(0, 2))
        SelectableLabel(
            ep_row, text="Endpoint:", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80
        ).pack(side="left")
        self.teacher_endpoint_var = ctk.StringVar(
            value="http://localhost:11434/v1")
        teacher_endpoint_entry = themed_entry(
            ep_row, textvariable=self.teacher_endpoint_var,
            placeholder_text="http://host:port/v1")
        teacher_endpoint_entry.pack(side="left", fill="x", expand=True)
        Tooltip(teacher_endpoint_entry,
                "OpenAI-compatible chat-completions endpoint.\n"
                "Examples: Ollama (http://localhost:11434/v1),\n"
                "vLLM (http://localhost:8000/v1),\n"
                "llama.cpp server (http://localhost:8080/v1).")

        # Model name row
        model_row = ctk.CTkFrame(ext_section, fg_color="transparent")
        model_row.pack(fill="x", pady=(0, 2))
        SelectableLabel(
            model_row, text="Model:", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80
        ).pack(side="left")
        self.teacher_model_var = ctk.StringVar(value="")
        teacher_model_entry = themed_entry(
            model_row, textvariable=self.teacher_model_var,
            placeholder_text="e.g. qwen3:8b, llama3:8b-instruct")
        teacher_model_entry.pack(side="left", fill="x", expand=True)
        Tooltip(teacher_model_entry,
                "Model name as the endpoint expects it.\n"
                "Required — no default.")

        # Mode radio: Magpie (synth) vs Prompts file
        mode_row = ctk.CTkFrame(ext_section, fg_color="transparent")
        mode_row.pack(fill="x", pady=(0, 2))
        SelectableLabel(
            mode_row, text="Mode:", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80
        ).pack(side="left")
        self.teacher_mode_var = ctk.StringVar(value="magpie")
        teacher_mode_magpie = ctk.CTkRadioButton(
            mode_row, text="Magpie (synth)", value="magpie",
            variable=self.teacher_mode_var)
        teacher_mode_magpie.pack(side="left", padx=(0, 8))
        teacher_mode_prompts = ctk.CTkRadioButton(
            mode_row, text="Prompts file", value="prompts",
            variable=self.teacher_mode_var)
        teacher_mode_prompts.pack(side="left")
        Tooltip(teacher_mode_magpie,
                "Magpie: model invents BOTH the instruction AND the\n"
                "answer. No input file needed. Best for fast bootstrap.")
        Tooltip(teacher_mode_prompts,
                "Prompts: read instructions from a .txt or .jsonl file,\n"
                "let the teacher answer each one. Use when you have\n"
                "curated questions. Path required below.")

        # Prompts-file path row (only consulted when mode=Prompts).
        pf_row = ctk.CTkFrame(ext_section, fg_color="transparent")
        pf_row.pack(fill="x", pady=(0, 2))
        SelectableLabel(
            pf_row, text="Prompts:", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80
        ).pack(side="left")
        self.teacher_prompts_path_var = ctk.StringVar(value="")
        teacher_prompts_entry = themed_entry(
            pf_row, textvariable=self.teacher_prompts_path_var,
            placeholder_text="path/to/prompts.txt or .jsonl")
        teacher_prompts_entry.pack(side="left", fill="x", expand=True)
        Tooltip(teacher_prompts_entry,
                "Path to .txt (one prompt per line) or .jsonl file\n"
                "(JSON object per line with a 'prompt' / 'instruction'\n"
                "/ 'question' field). Only used when Mode=Prompts file.")
        teacher_browse_btn = themed_button(
            pf_row, text="...", style="secondary", width=30,
            command=self._browse_teacher_prompts_file)
        teacher_browse_btn.pack(side="left", padx=(2, 0))

        # Magpie N + tag row
        mn_row = ctk.CTkFrame(ext_section, fg_color="transparent")
        mn_row.pack(fill="x", pady=(0, 2))
        SelectableLabel(
            mn_row, text="Magpie N:", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80
        ).pack(side="left")
        self.teacher_magpie_var = ctk.StringVar(value="500")
        teacher_magpie_entry = themed_numeric_entry(
            mn_row, mode="int",
            textvariable=self.teacher_magpie_var, width=70)
        teacher_magpie_entry.pack(side="left")
        Tooltip(teacher_magpie_entry,
                "Number of instruction/answer pairs to synthesize\n"
                "via Magpie empty-prefix method (no input prompts\n"
                "needed). 500 ≈ a few minutes on a fast endpoint.")
        teacher_suggest_btn = themed_button(
            mn_row, text="↻", style="secondary", width=28,
            command=self._suggest_magpie_n_from_tag)
        teacher_suggest_btn.pack(side="left", padx=(2, 0))
        Tooltip(teacher_suggest_btn,
                "Count existing pairs in data/finetune/distill_<tag>.jsonl\n"
                "and suggest Magpie-N = existing + 500. Useful when\n"
                "resuming a corpus you've already partially built.")
        SelectableLabel(
            mn_row, text="  Tag:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(6, 0))
        self.teacher_tag_var = ctk.StringVar(value="external")
        teacher_tag_entry = themed_entry(
            mn_row, textvariable=self.teacher_tag_var, width=120)
        teacher_tag_entry.pack(side="left", padx=(2, 0))
        Tooltip(teacher_tag_entry,
                "Output filename tag — writes\n"
                "data/finetune/distill_<tag>.{jsonl,txt}.\n"
                "Reused tag triggers --resume (skip done prompts).")

        # Max-tokens (mirrors collect_distill_data.py default).
        # Reuses the existing distill_max_tokens_var so the user only
        # configures it once for both in-process and external flows.
        self.teacher_max_tokens_var = self.distill_max_tokens_var

        # Buttons row
        btn_row = ctk.CTkFrame(ext_section, fg_color="transparent")
        btn_row.pack(fill="x", pady=(2, 2))
        self.teacher_start_btn = themed_button(
            btn_row, text="GENERATE EXTERNAL TEACHER CORPUS",
            style="primary",
            command=self._start_external_teacher_corpus)
        self.teacher_start_btn.pack(side="left", fill="x", expand=True)
        self.teacher_stop_btn = themed_button(
            btn_row, text="STOP", style="secondary",
            width=70,
            command=self._stop_external_teacher_corpus)
        self.teacher_stop_btn.pack(side="left", padx=(4, 0))
        self.teacher_stop_btn.configure(state="disabled")

        # === BASIC TRAINING OPTIONS ===
        self._forge_basic_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_basic_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_basic_section, "Training data (required)")
        # D-11b (Pass 156i9): prefer data/finetune/combined_finetune.txt
        # when a fine-tune corpus has been collected, so the FORGE picker
        # surfaces the reasoning data without manual nav.
        # D-11c-DPO (Pass 156q): track the last applied smart default so
        # `_on_training_mode_changed` can swap it for preference-pair
        # modes without clobbering user-customised paths.
        from enigma_engine.gui.scanners import _pick_default_training_file
        _initial_default = _pick_default_training_file(self.training_files)
        self._train_data_smart_default = _initial_default
        self.train_data_var = ctk.StringVar(value=_initial_default)
        data_opts = [
            f"{f['name']} ({f['size_kb'] / 1024:.1f} MB)"
            if f["size_kb"] >= 1024
            else f"{f['name']} ({f['size_kb']} KB)"
            for f in self.training_files]

        # File path entry + Browse button
        data_path_row = ctk.CTkFrame(
            self._forge_basic_section, fg_color="transparent")
        data_path_row.pack(fill="x", padx=10, pady=(0, 4))
        data_path_row.grid_columnconfigure(0, weight=1)
        self._forge_data_entry = themed_entry(
            data_path_row,
            textvariable=self.train_data_var,
            placeholder_text="Path to training file...")
        self._forge_data_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._forge_data_browse_btn = themed_button(
            data_path_row, "Browse", style="secondary",
            width=70,
            font=FONT_SMALL,
            command=self._browse_training_data)
        self._forge_data_browse_btn.grid(row=0, column=1, sticky="e")
        Tooltip(self._forge_data_browse_btn,
                "Browse for a training data file.\n"
                "Supports .txt, .json, .jsonl formats.")

        # Quick dropdown for data/ files
        data_row = ctk.CTkFrame(
            self._forge_basic_section, fg_color="transparent")
        data_row.pack(fill="x", padx=10, pady=(0, 6))
        all_opts = ["(none)"] + data_opts if data_opts else ["(none)"]
        SelectableLabel(
            data_row, text="Quick select:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.train_data_menu = themed_dropdown(
            data_row, all_opts,
            width=220,
            command=self._on_data_selected)
        self.train_data_menu.pack(side="left")
        if data_opts:
            self.train_data_menu.set(data_opts[0])
        else:
            self.train_data_menu.set("(none)")
        Tooltip(self.train_data_menu,
                "Quick select from data/ folder.\n"
                "Or use Browse to pick any file.\n"
                "LoRA auto-selected if model > 7B params.")

        # === AI-GUIDED TRAINING OPTIONS ===
        self._forge_ai_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_ai_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_ai_section,
            "Training topic/goal (required)")
        self.forge_training_topic = themed_entry(
            self._forge_ai_section, width=280,
            placeholder_text="e.g., 'coding assistant' or 'creative writer'")
        self.forge_training_topic.pack(anchor="w", padx=10, pady=(0, 4))
        Tooltip(self.forge_training_topic,
                "What should the AI learn?\n"
                "The TRAINER will generate curriculum based on this.\n"
                "Examples: 'medical Q&A', 'code generation', 'storytelling'")

        self._forge_label(self._forge_ai_section,
            "Supplement data (optional)")
        self.ai_supplement_var = ctk.StringVar(value="")
        # File path entry + Browse button for supplement data
        ai_data_path_row = ctk.CTkFrame(
            self._forge_ai_section, fg_color="transparent")
        ai_data_path_row.pack(fill="x", padx=10, pady=(0, 4))
        ai_data_path_row.grid_columnconfigure(0, weight=1)
        self._forge_ai_data_entry = themed_entry(
            ai_data_path_row,
            textvariable=self.ai_supplement_var,
            placeholder_text="Optional data file...")
        self._forge_ai_data_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._forge_ai_data_browse_btn = themed_button(
            ai_data_path_row, "Browse", style="secondary",
            width=70,
            font=FONT_SMALL,
            command=lambda: self._browse_training_data(
                target_var=self.ai_supplement_var))
        self._forge_ai_data_browse_btn.grid(
            row=0, column=1, sticky="e")

        # Quick dropdown for supplement
        ai_data_opts = ["(none)"] + data_opts if data_opts else ["(none)"]
        ai_dd_row = ctk.CTkFrame(
            self._forge_ai_section, fg_color="transparent")
        ai_dd_row.pack(fill="x", padx=10, pady=(0, 6))
        SelectableLabel(
            ai_dd_row, text="Quick select:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.ai_supplement_menu = themed_dropdown(
            ai_dd_row, ai_data_opts,
            width=220,
            command=self._on_supplement_selected)
        self.ai_supplement_menu.pack(side="left")
        self.ai_supplement_menu.set("(none)")
        Tooltip(self.ai_supplement_menu,
                "Optional: Add your own data to mix with AI-generated curriculum.\n"
                "The trainer will use this as seed material.")

        # === IMAGE TRAINING OPTIONS ===
        self._forge_image_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_image_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_image_section,
            "Image folder (required)")
        self.forge_vision_dir_var = ctk.StringVar(
            value=str(DATA_DIR / "images"))
        vision_dir_row = ctk.CTkFrame(
            self._forge_image_section, fg_color="transparent")
        vision_dir_row.pack(fill="x", padx=10, pady=(0, 6))
        vision_dir_row.grid_columnconfigure(0, weight=1)
        self._forge_vision_dir_entry = themed_entry(
            vision_dir_row, textvariable=self.forge_vision_dir_var)
        self._forge_vision_dir_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._forge_vision_browse_btn = themed_button(
            vision_dir_row, "Browse", style="secondary",
            width=70,
            font=FONT_SMALL,
            command=self._browse_vision_dir)
        self._forge_vision_browse_btn.grid(row=0, column=1, sticky="e")
        Tooltip(self._forge_vision_browse_btn,
            "Pick the folder containing image training data.")
        Tooltip(self._forge_vision_dir_entry,
                "Folder with image+text pairs.\n"
                "Format 1: image.png + image.txt (same name)\n"
                "Format 2: captions.jsonl with image+text fields\n"
                "Video files: Will auto-extract frames")

        self._forge_label(self._forge_image_section, "Vision encoder size")
        self.forge_vision_preset_var = ctk.StringVar(
            value="small - ~4M params, good balance")
        vision_preset_dd = themed_dropdown(
            self._forge_image_section,
            values=[
                "tiny - ~500K params, fast",
                "small - ~4M params, good balance",
                "medium - ~25M params, best quality",
            ],
            variable=self.forge_vision_preset_var,
            width=240)
        vision_preset_dd.pack(anchor="w", padx=10, pady=(0, 6))

        # Code-6b (Pass 156r): unfreeze last N text layers for LLaVA
        # Stage-2 fine-tuning. Default 0 = Stage-1 (projection-only,
        # text frozen). Larger values let the projection adapt the
        # text layers to multimodal inputs at the cost of compute and
        # storage (full fine-tune of those layers).
        self._forge_label(self._forge_image_section,
            "Unfreeze last N text layers (0 = Stage-1, projection only)")
        self.forge_vision_unfreeze_var = ctk.StringVar(value="0")
        unfreeze_entry = themed_entry(
            self._forge_image_section,
            textvariable=self.forge_vision_unfreeze_var,
            width=80,
            placeholder_text="0")
        unfreeze_entry.pack(anchor="w", padx=10, pady=(0, 6))
        Tooltip(unfreeze_entry,
                "LLaVA Stage 2 knob. 0 = projection-only (LLaVA "
                "Stage 1 default).\n"
                "Set to e.g. 4 to also fine-tune the last 4 text "
                "transformer layers.\n"
                "Higher = better multimodal grounding but heavier "
                "training and a larger checkpoint diff.")

        # === AI-GUIDED: TRAINING STAGES ===
        self._forge_stages_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_stages_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_stages_section, "Training stages")
        self.training_stage_var = ctk.StringVar(value="basics")
        stage_frame = ctk.CTkFrame(
            self._forge_stages_section, fg_color="transparent")
        stage_frame.pack(fill="x", padx=10, pady=(0, 6))

        stage_descriptions = {
            "basics": "Fundamental language patterns\nand basic responses",
            "conversation": "Natural dialogue flow\nand contextual responses",
            "commands": "Command recognition\nand structured outputs",
            "web": "Web content understanding\nand information extraction",
        }
        self._stage_buttons = {}
        for stage_name, stage_tip in stage_descriptions.items():
            btn = themed_button(
                stage_frame,
                stage_name.upper(),
                style="primary" if stage_name == "basics"
                    else "secondary",
                width=90, height=30,
                font=FONT_SMALL,
                hover_color=C_ACCENT_DIM,
                text_color=C_GREEN if stage_name == "basics"
                    else C_TEXT,
                command=lambda s=stage_name: self._select_training_stage(s))
            btn.pack(side="left", padx=(0, 4))
            Tooltip(btn, stage_tip)
            self._stage_buttons[stage_name] = btn

        SelectableLabel(
            self._forge_stages_section,
            text="Auto-advances to next stage after completion",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=10, pady=(0, 4))

        # === AI-GUIDED: TRAINING BRIEF ===
        self._forge_brief_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_brief_section.pack(fill="x", padx=0, pady=(8, 0))

        brief_panel = CollapsiblePanel(
            self._forge_brief_section, title="TRAINING BRIEF (OPTIONAL)",
            start_expanded=False)
        brief_panel.pack(fill="x", padx=6, pady=(0, 6))
        brief_inner = brief_panel.content

        SelectableLabel(
            brief_inner,
            text="Fine-tune the curriculum with these fields",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=6, pady=(2, 4))

        # Quick profile fields — structured mad-libs style
        self._brief_field_entries = {}
        for label, placeholder, tip in self._QUICK_PROFILE_FIELDS:
            row = ctk.CTkFrame(brief_inner, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=(3, 0))
            row.grid_columnconfigure(1, weight=1)
            lbl = SelectableLabel(
                row, text=label, font=FONT_TINY,
                text_color=C_TEXT_DIM, anchor="w")
            lbl.grid(row=0, column=0, sticky="w", padx=(0, 4))
            entry = themed_entry(
                row, width=180, placeholder_text=placeholder)
            entry.grid(row=0, column=1, sticky="ew")
            Tooltip(entry, tip)
            self._brief_field_entries[label] = entry

        # Custom brief text area
        SelectableLabel(
            brief_inner, text="Custom instructions",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=6, pady=(8, 2))
        self._brief_custom_text = ctk.CTkTextbox(
            brief_inner, height=80, font=FONT_SMALL,
            fg_color=C_INPUT, text_color=C_TEXT_BRIGHT,
            border_width=1, border_color=C_ACCENT_DIM,
            corner_radius=2, wrap="word")
        self._brief_custom_text.pack(fill="x", padx=6, pady=(0, 4))
        wire_hotkeys(self._brief_custom_text)
        Tooltip(self._brief_custom_text,
                "Freeform instructions for the trainer AI.\n"
                "Describe exactly what you want the student AI to learn.")

        # Load persisted brief values
        self.after(200, self._load_training_brief)

        # === AI-GUIDED: PAIRS/ROUNDS ===
        self._forge_pairs_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_pairs_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_pairs_section,
            "Training examples per stage")
        self.guided_pairs_entry = self._forge_numeric_entry(
            self._forge_pairs_section, "20")
        Tooltip(self.guided_pairs_entry,
                "Number of Q&A pairs to generate per stage.\n"
                "More pairs = better coverage, longer training.")
        # Aliases
        self.dialogue_rounds_entry = self.guided_pairs_entry

        # === HYPERPARAMETERS (all modes) ===
        self._forge_heading(ctrl_scroll, "HYPERPARAMETERS")

        # Training recipe dropdown for quick configuration
        self._forge_label(ctrl_scroll, "Training recipe")
        self._forge_preset_var = ctk.StringVar(value="Balanced - 10 epochs, good quality")
        self._forge_preset_menu = themed_dropdown(
            ctrl_scroll,
            ["Quick - 3 epochs, fast results",
             "Balanced - 10 epochs, good quality",
             "Thorough - 30 epochs, best quality",
             "Custom - set your own values"],
            variable=self._forge_preset_var,
            width=280,
            command=self._on_preset_changed)
        self._forge_preset_menu.pack(anchor="w", padx=10, pady=(0, 6))

        self._forge_label(ctrl_scroll, "Epochs")
        self.ft_epochs_entry = self._forge_numeric_entry(ctrl_scroll, "10")
        Tooltip(self.ft_epochs_entry,
                "How many times the model reads through ALL your data.\n"
                "1 epoch = one full pass. 10 = ten full passes.\n"
                "More = learns deeper, but takes longer.\n"
                "Fine-tuning: 3-30. Pre-training: 1-3.")
        # Aliases for backward compatibility
        self.guided_epochs_entry = self.ft_epochs_entry
        self.epochs_entry = self.ft_epochs_entry

        self._forge_label(ctrl_scroll, "Learning rate")
        self.ft_lr_entry = self._forge_numeric_entry(
            ctrl_scroll, "0.00005", mode="float")
        Tooltip(self.ft_lr_entry,
                "How big each learning step is.\n"
                "Too high = model forgets old knowledge.\n"
                "Too low = barely learns anything.\n"
                "5e-5 = safe default. 1e-5 = gentle. 3e-4 = aggressive.")
        # Aliases
        self.guided_lr_entry = self.ft_lr_entry
        self.lr_entry = self.ft_lr_entry

        # Switch preset to "Custom" when user manually edits fields
        self._preset_programmatic = False
        for entry in (self.ft_epochs_entry, self.ft_lr_entry):
            entry.bind("<KeyRelease>",
                       lambda _e: self._mark_preset_custom())

        # Main train button row
        btn_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(8, 4))

        self.solo_train_btn = themed_button(
            btn_row, "TRAIN", style="primary",
            width=170, height=38,
            font=FONT_SECTION,
            command=self._start_training_by_mode)
        self.solo_train_btn.pack(side="left")
        # Aliases for all mode buttons
        self.guided_train_btn = self.solo_train_btn
        self.dialogue_train_btn = self.solo_train_btn
        self.train_model_btn = self.solo_train_btn
        Tooltip(self.solo_train_btn,
                "Start training with the selected mode")

        self.stop_train_btn = themed_button(
            btn_row, "STOP", style="secondary",
            width=80, height=38,
            font=FONT_SECTION,
            hover_color=C_RED,
            command=self._stop_training,
            state="disabled")
        self.stop_train_btn.pack(side="left", padx=(6, 0))

        # Auto-train — start training after GENERATE DATA or
        # WEB LEARN finishes, using the newly created data file
        self.forge_auto_train_var = ctk.BooleanVar(value=False)
        auto_train_cb = ctk.CTkCheckBox(
            btn_row, text="Auto-train",
            variable=self.forge_auto_train_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        auto_train_cb.pack(side="left", padx=(10, 0))
        Tooltip(auto_train_cb,
                "Automatically start training after\n"
                "GENERATE DATA or WEB LEARN finishes.\n"
                "Uses the newly created data file.")

        # Resume from checkpoint — when enabled, training
        # resumes from the latest checkpoint if one exists.
        self.forge_resume_var = ctk.BooleanVar(value=True)
        resume_cb = ctk.CTkCheckBox(
            btn_row, text="Resume from checkpoint",
            variable=self.forge_resume_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        resume_cb.pack(side="left", padx=(10, 0))
        Tooltip(resume_cb,
                "ON: Click TRAIN to continue from where\n"
                "you left off (loads latest checkpoint).\n"
                "OFF: Start training from scratch.")

        # Save button row (below train)
        save_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        save_row.pack(fill="x", padx=10, pady=(0, 4))

        self.save_ckpt_btn = themed_button(
            save_row, "SAVE CHECKPOINT", style="action",
            width=170, height=34,
            font=FONT_SMALL,
            command=self._save_forge_checkpoint)
        self.save_ckpt_btn.pack(side="left")
        Tooltip(self.save_ckpt_btn,
                "Save STUDENT to a named checkpoint")

        self.load_ckpt_btn = themed_button(
            save_row, "LOAD CHECKPOINT", style="action",
            width=170, height=34,
            font=FONT_SMALL,
            command=self._load_forge_checkpoint)
        self.load_ckpt_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.load_ckpt_btn,
                "Load a saved checkpoint back into STUDENT")

        # Advanced settings (collapsible)
        advanced_panel = CollapsiblePanel(
            ctrl_scroll, title="ADVANCED SETTINGS",
            start_expanded=False)
        advanced_panel.pack(fill="x", padx=6, pady=(8, 2))
        advanced_inner = advanced_panel.content

        # Batch size
        self._forge_label(advanced_inner, "Batch size")
        self.forge_batch_entry = self._forge_numeric_entry(
            advanced_inner, "auto", allow_auto=True)
        self.forge_batch_entry.bind(
            "<KeyRelease>",
            lambda _e: self._mark_preset_custom())
        self.forge_batch_entry.bind(
            "<FocusOut>",
            lambda _e: self._validate_batch_entry())
        Tooltip(self.forge_batch_entry,
                "How many examples the model learns from at once.\n"
                "'auto' = fill available GPU memory.\n"
                "Set to 1 if you get out-of-memory errors.\n"
                "Bigger = faster training, more VRAM needed.")

        # Gradient accumulation
        self._forge_label(advanced_inner, "Gradient accumulation")
        self.forge_accum_entry = self._forge_numeric_entry(
            advanced_inner, "1")
        Tooltip(self.forge_accum_entry,
                "Simulates a bigger batch without using more VRAM.\n"
                "batch_size=1 + accumulation=4 acts like batch_size=4\n"
                "but only uses VRAM for 1 example at a time.\n"
                "Set higher if batch_size is forced to 1 by OOM.")

        # Gradient checkpointing
        ckpt_row = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        ckpt_row.pack(fill="x", padx=10, pady=(0, 6))
        self.forge_grad_ckpt_var = ctk.BooleanVar(value=True)
        self.forge_grad_ckpt_cb = ctk.CTkCheckBox(
            ckpt_row, text="Gradient checkpointing",
            variable=self.forge_grad_ckpt_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        self.forge_grad_ckpt_cb.pack(anchor="w")
        Tooltip(self.forge_grad_ckpt_cb,
                "Trades speed for VRAM savings (~40% less VRAM).\n"
                "ON by default. Disable only if you have VRAM to spare.\n"
                "Training will be ~20% slower but larger models fit.")

        # Rolling best checkpoints
        rolling_row = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        rolling_row.pack(fill="x", padx=10, pady=(0, 6))
        SelectableLabel(
            rolling_row, text="Rolling best K", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.forge_rolling_k_entry = themed_numeric_entry(
            rolling_row, mode="int", width=50)
        self.forge_rolling_k_entry.insert(0, "0")
        self.forge_rolling_k_entry.pack(side="left")
        Tooltip(self.forge_rolling_k_entry,
                "Keep only the K best checkpoints by loss.\n"
                "0 = disabled (keep all). 3 = keep 3 best.\n"
                "Prevents disk bloat during long training.")

        # LoRA settings (show when relevant)
        self._forge_lora_subsection = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        self._forge_lora_subsection.pack(fill="x", padx=0, pady=(4, 0))

        SelectableLabel(
            self._forge_lora_subsection,
            text="LoRA settings (auto-enabled for large models)",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=10, pady=(0, 2))

        lora_row = ctk.CTkFrame(
            self._forge_lora_subsection, fg_color="transparent")
        lora_row.pack(fill="x", padx=10, pady=(0, 4))

        SelectableLabel(
            lora_row, text="Rank", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80, anchor="w"
        ).pack(side="left", padx=(0, 4))
        self.forge_lora_rank_var = ctk.StringVar(value="8")
        lora_rank_entry = themed_numeric_entry(
            lora_row, mode="int",
            textvariable=self.forge_lora_rank_var, width=60)
        lora_rank_entry.pack(side="left", padx=(0, 12))
        Tooltip(lora_rank_entry,
                "LoRA rank (1-128). Lower = fewer params.\n"
                "4-8 for small models, 16-32 for large.\n"
                "Default: 8")

        SelectableLabel(
            lora_row, text="Alpha", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80, anchor="w"
        ).pack(side="left", padx=(0, 4))
        self.forge_lora_alpha_var = ctk.StringVar(value="16")
        lora_alpha_entry = themed_numeric_entry(
            lora_row, mode="int",
            textvariable=self.forge_lora_alpha_var, width=60)
        lora_alpha_entry.pack(side="left")
        Tooltip(lora_alpha_entry,
                "LoRA alpha scaling (1-256).\n"
                "Higher = stronger adapter effect.\n"
                "Common: alpha = 2x rank. Default: 16")

        # Replay buffer settings (RLHF / Self-play)
        SelectableLabel(
            advanced_inner,
            text="Replay buffer (RLHF / Self-play)",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=10, pady=(2, 2))

        replay_row = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        replay_row.pack(fill="x", padx=10, pady=(0, 4))

        SelectableLabel(
            replay_row, text="Capacity", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80, anchor="w"
        ).pack(side="left", padx=(0, 4))
        self.forge_replay_capacity_var = ctk.StringVar(value="256")
        replay_cap_entry = themed_numeric_entry(
            replay_row, mode="int",
            textvariable=self.forge_replay_capacity_var,
            width=60)
        replay_cap_entry.pack(side="left", padx=(0, 12))
        Tooltip(replay_cap_entry,
                "Max stored experiences for replay.\n"
                "0 = disabled. Default: 256")

        SelectableLabel(
            replay_row, text="Ratio", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=80, anchor="w"
        ).pack(side="left", padx=(0, 4))
        self.forge_replay_ratio_var = ctk.StringVar(value="0.25")
        replay_ratio_entry = themed_numeric_entry(
            replay_row, mode="float",
            textvariable=self.forge_replay_ratio_var,
            width=60)
        replay_ratio_entry.pack(side="left")
        Tooltip(replay_ratio_entry,
                "Fraction of minibatch from replay.\n"
                "0.25 = 25% replay experiences. Default: 0.25")

        # Validation split
        val_row = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        val_row.pack(fill="x", padx=10, pady=(2, 4))

        SelectableLabel(
            val_row, text="Val split", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=90, anchor="w"
        ).pack(side="left", padx=(0, 4))
        self.forge_val_split_var = ctk.StringVar(value="0.1")
        self.forge_val_split_entry = themed_numeric_entry(
            val_row, mode="float",
            textvariable=self.forge_val_split_var,
            width=60)
        self.forge_val_split_entry.pack(side="left", padx=(0, 4))
        Tooltip(self.forge_val_split_entry,
                "Fraction of training data held out for validation.\n"
                "0.0 = no validation, 0.1 = 10%. Default: 0.1")

        # --- Tools section (collapsed by default) ---
        tools_panel = CollapsiblePanel(
            ctrl_scroll, title="TOOLS", start_expanded=False)
        tools_panel.pack(fill="x", padx=6, pady=(10, 2))
        tools_inner = tools_panel.content

        # -- Training tools --
        SelectableLabel(
            tools_inner, text="TRAINING",
            font=FONT_TINY, text_color=C_ACCENT_DIM
        ).pack(anchor="w", padx=8, pady=(6, 0))

        tools_row1 = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        tools_row1.pack(fill="x", padx=6, pady=(2, 3))

        self.generate_data_btn = themed_button(
            tools_row1, "GENERATE DATA", style="tool",
            width=140, height=34,
            font=FONT_SMALL,
            command=self._generate_training_data)
        self.generate_data_btn.pack(side="left")
        Tooltip(self.generate_data_btn,
                "TRAINER generates training Q/A pairs")

        self.evaluate_btn = themed_button(
            tools_row1, "EVALUATE", style="tool",
            width=120, height=34,
            font=FONT_SMALL,
            command=self._evaluate_student)
        self.evaluate_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.evaluate_btn,
                "TRAINER tests STUDENT and scores answers")

        self.benchmark_btn = themed_button(
            tools_row1, "BENCHMARK", style="tool",
            width=120, height=34,
            font=FONT_SMALL,
            command=self._coherence_benchmark)
        self.benchmark_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.benchmark_btn,
                "Test model reflection quality (coherence benchmark)")

        # History button
        self._forge_history_btn = themed_button(
            tools_row1, "HISTORY", style="secondary",
            width=90, height=34,
            font=FONT_SMALL,
            command=self._show_training_history)
        self._forge_history_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_history_btn,
                "Show past training runs with loss values")

        # -- Web learning --
        SelectableLabel(
            tools_inner, text="WEB",
            font=FONT_TINY, text_color=C_ACCENT_DIM
        ).pack(anchor="w", padx=8, pady=(6, 0))

        # Web Learn
        web_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        web_row.pack(fill="x", padx=6, pady=(2, 3))

        self.web_learn_topic = themed_entry(
            web_row, width=180, placeholder_text="Topic...")
        self.web_learn_topic.pack(side="left")

        self.web_learn_pages_entry = themed_numeric_entry(
            web_row, mode="int", width=50,
            placeholder_text="3")
        self.web_learn_pages_entry.insert(0, "3")
        self.web_learn_pages_entry.pack(side="left", padx=(6, 0))
        Tooltip(self.web_learn_pages_entry,
                "Max web pages to fetch (1-10)")

        self.web_learn_btn = themed_button(
            web_row, "WEB LEARN", style="tool",
            width=110, height=34,
            font=FONT_SMALL,
            command=self._web_learn)
        self.web_learn_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.web_learn_btn,
                "Search web and generate training data")

        # -- Tokenizer --
        SelectableLabel(
            tools_inner, text="TOKENIZER",
            font=FONT_TINY, text_color=C_ACCENT_DIM
        ).pack(anchor="w", padx=8, pady=(6, 0))

        # Tokenizer training
        tok_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        tok_row.pack(fill="x", padx=6, pady=(2, 3))

        self.vocab_entry = themed_numeric_entry(
            tok_row, mode="int", width=100,
            placeholder_text="Vocab: 32000")
        self.vocab_entry.insert(0, "32000")
        self.vocab_entry.pack(side="left")

        self.train_tok_btn = themed_button(
            tok_row, "TRAIN TOKENIZER", style="tool",
            width=140, height=34,
            font=FONT_SMALL,
            command=self._start_tokenizer_training)
        self.train_tok_btn.pack(side="left", padx=(6, 0))
        utf8_tool_cb = ctk.CTkCheckBox(
            tok_row, text="Byte-level",
            variable=self.pretrain_utf8_bytes_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        utf8_tool_cb.pack(side="left", padx=(6, 0))
        Tooltip(utf8_tool_cb,
                "Use byte-level BPE encoding.\n"
                "Handles any Unicode (emoji, CJK, etc.)")
        Tooltip(self.train_tok_btn,
                "Train a standalone BPE tokenizer on any data file.\n"
                "Saves to models/tokenizer.json.\n"
                "For pre-training, use the checkbox in Pre-Train mode.")

        # Tokenizer analyze row
        tok_analyze_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        tok_analyze_row.pack(fill="x", padx=6, pady=(0, 3))
        self.analyze_tok_btn = themed_button(
            tok_analyze_row, "ANALYZE TOKENIZER", style="tool",
            width=140, height=34,
            font=FONT_SMALL,
            command=self._analyze_tokenizer)
        self.analyze_tok_btn.pack(side="left")
        Tooltip(self.analyze_tok_btn,
                "Analyze the current tokenizer.\n"
                "Shows vocab stats, coverage, and compression.")

        # -- Model export --
        SelectableLabel(
            tools_inner, text="MODEL EXPORT",
            font=FONT_TINY, text_color=C_ACCENT_DIM
        ).pack(anchor="w", padx=8, pady=(6, 0))

        # Quantize model
        quant_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        quant_row.pack(fill="x", padx=6, pady=(2, 3))

        self.quantize_mode_var = ctk.StringVar(
            value="int8 - 8-bit weights")
        self.quantize_mode_dd = themed_dropdown(
            quant_row, width=220,
            values=[
                "dynamic - per-tensor, flexible",
                "int8 - 8-bit weights",
                "int4 - 4-bit, smallest",
            ],
            variable=self.quantize_mode_var)
        self.quantize_mode_dd.pack(side="left")

        self.quantize_btn = themed_button(
            quant_row, "QUANTIZE", style="tool",
            width=120, height=34,
            font=FONT_SMALL,
            command=self._quantize_student)
        self.quantize_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.quantize_btn,
                "Quantize the STUDENT model to reduce size")

        # Export to GGUF
        export_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        export_row.pack(fill="x", padx=6, pady=(6, 3))

        self.export_gguf_mode_var = ctk.StringVar(
            value="Q8_0 - 8-bit, good balance")
        self.export_gguf_mode_dd = themed_dropdown(
            export_row, width=220,
            values=[
                "F16 - float16, best quality",
                "Q8_0 - 8-bit, good balance",
                "Q4_0 - 4-bit, smallest",
            ],
            variable=self.export_gguf_mode_var)
        self.export_gguf_mode_dd.pack(side="left")

        self.export_gguf_btn = themed_button(
            export_row, "EXPORT GGUF", style="tool",
            width=140, height=34,
            font=FONT_SMALL,
            command=self._export_student_gguf)
        self.export_gguf_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.export_gguf_btn,
                "Export STUDENT as a GGUF file for llama.cpp")

        # -- Queue & schedule --
        SelectableLabel(
            tools_inner, text="QUEUE",
            font=FONT_TINY, text_color=C_ACCENT_DIM
        ).pack(anchor="w", padx=8, pady=(6, 0))

        # --- Queue & Schedule (TS-B, TS-C) ---
        queue_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        queue_row.pack(fill="x", padx=6, pady=(2, 3))

        self._forge_add_queue_btn = themed_button(
            queue_row, "ADD TO QUEUE", style="action",
            width=130, height=34,
            font=FONT_SMALL,
            command=self._add_to_training_queue)
        self._forge_add_queue_btn.pack(side="left")
        Tooltip(self._forge_add_queue_btn,
                "Add current settings as a job to the queue.\n"
                "Queue runs jobs sequentially in background.")

        self._forge_show_queue_btn = themed_button(
            queue_row, "QUEUE", style="action",
            width=80, height=34,
            font=FONT_SMALL,
            command=self._show_training_queue)
        self._forge_show_queue_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_show_queue_btn,
                "Show/manage the training queue")

        self._forge_run_queue_btn = themed_button(
            queue_row, "RUN", style="primary",
            width=60, height=34,
            font=FONT_SMALL,
            command=self._run_training_queue)
        self._forge_run_queue_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_run_queue_btn,
                "Start running the training queue")

        # Overnight plan
        plan_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        plan_row.pack(fill="x", padx=6, pady=(3, 3))

        self._forge_save_plan_btn = themed_button(
            plan_row, "SAVE PLAN", style="action",
            width=110, height=34,
            font=FONT_SMALL,
            command=self._save_overnight_plan)
        self._forge_save_plan_btn.pack(side="left")
        Tooltip(self._forge_save_plan_btn,
                "Save the current queue as an overnight plan.\n"
                "Saves progress and resumes on crash.")

        self._forge_load_plan_btn = themed_button(
            plan_row, "LOAD PLAN", style="action",
            width=110, height=34,
            font=FONT_SMALL,
            command=self._load_overnight_plan)
        self._forge_load_plan_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_load_plan_btn,
                "Load a saved overnight training plan.\n"
                "Resumes where it left off.")

        # -- Dataset --
        SelectableLabel(
            tools_inner, text="DATASET",
            font=FONT_TINY, text_color=C_ACCENT_DIM
        ).pack(anchor="w", padx=8, pady=(6, 0))

        # --- Curated Dataset (DA-C) ---
        dataset_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        dataset_row.pack(fill="x", padx=6, pady=(2, 3))

        self._forge_review_dataset_btn = themed_button(
            dataset_row, "REVIEW DATASET", style="action",
            width=140, height=34,
            font=FONT_SMALL,
            command=self._review_curated_dataset)
        self._forge_review_dataset_btn.pack(side="left")
        Tooltip(self._forge_review_dataset_btn,
                "Review/approve/reject entries in the\n"
                "curated master dataset before training.")

        self._forge_approve_all_btn = themed_button(
            dataset_row, "APPROVE ALL", style="action",
            width=110, height=34,
            font=FONT_SMALL,
            command=self._approve_all_dataset)
        self._forge_approve_all_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_approve_all_btn,
                "Approve all pending entries in the\n"
                "curated dataset for training use.")

        # --- Command Policy Generator ---
        cmd_policy_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        cmd_policy_row.pack(fill="x", padx=6, pady=(6, 3))

        self._forge_cmd_policy_btn = themed_button(
            cmd_policy_row, "COMMAND POLICY", style="action",
            width=150, height=34,
            font=FONT_SMALL,
            command=self._generate_command_policy)
        self._forge_cmd_policy_btn.pack(side="left")
        Tooltip(self._forge_cmd_policy_btn,
                "Auto-generate DPO training pairs for all\n"
                "registered engine commands. Reads commands_reference.md\n"
                "and creates chosen/rejected pairs for preference training.")

        # Set initial section visibility for default mode (Basic mode)
        self._on_training_mode_changed("Basic")

        # Add controls panel to pane
        self._forge_pane.add(controls, stretch="always",
                             minsize=300)

        # Right: log output (resizable)
        log_panel = HUDFrame(self._forge_pane, glow_color=C_BORDER)

        log_header_row = ctk.CTkFrame(
            log_panel, fg_color="transparent")
        log_header_row.pack(fill="x", padx=8, pady=(8, 0))
        SectionLabel(log_header_row, "Output Log", color=C_GREEN).pack(
            side="left")

        # Auto-scroll toggle
        self._forge_autoscroll_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            log_header_row, text="Auto-scroll",
            variable=self._forge_autoscroll_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, checkmark_color=C_GREEN,
            width=24, height=24, checkbox_width=16, checkbox_height=16,
        ).pack(side="right", padx=(0, 4))

        # Clear log button
        themed_button(
            log_header_row, text="CLEAR", style="secondary",
            width=55, height=22,
            command=self._clear_forge_log,
        ).pack(side="right", padx=(0, 6))

        # Training progress bar
        progress_frame = ctk.CTkFrame(
            log_panel, fg_color="transparent")
        progress_frame.pack(fill="x", padx=8, pady=(4, 0))
        self._forge_progress_bar = ctk.CTkProgressBar(
            progress_frame, height=6, corner_radius=2,
            fg_color=C_SURFACE, progress_color=C_GREEN_DIM)
        self._forge_progress_bar.pack(side="left", fill="x",
                                      expand=True)
        self._forge_progress_bar.set(0)
        self._forge_progress_label = SelectableLabel(
            progress_frame, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, width=90)
        self._forge_progress_label.pack(side="right", padx=(6, 0))

        # Loss chart panel (EV-D) — collapsible, shows live loss curve
        import tkinter as tk_mod
        self._loss_chart_panel = CollapsiblePanel(
            log_panel, title="LOSS CHART", start_expanded=False)
        self._loss_chart_panel.pack(fill="x", padx=6, pady=(4, 0))
        chart_container = self._loss_chart_panel.content
        self._loss_canvas = tk_mod.Canvas(
            chart_container, bg=C_PANEL, highlightthickness=0,
            height=150)
        self._loss_canvas.pack(fill="x", padx=2, pady=2)
        # Chart info label
        self._loss_chart_info = SelectableLabel(
            chart_container, text="No data yet", font=FONT_TINY,
            text_color=C_TEXT_DIM)
        self._loss_chart_info.pack(anchor="w", padx=4, pady=(0, 2))

        self.train_log = SelectableTextbox(
            log_panel, font=FONT_MONO,
            fg_color=C_PANEL, text_color=C_GREEN,
            border_width=0, corner_radius=2)
        self.train_log.pack(fill="both", expand=True,
                            padx=6, pady=(0, 6))

        self._forge_pane.add(log_panel, stretch="always",
                             minsize=250)

        # Data editor removed — use DOCS page for file editing
        self._editing_data_path = None

        # Set initial button states based on current routes/data
        self.after(100, self._update_forge_button_states)

    def _forge_heading(self, parent, text: str):
        SelectableLabel(
            parent, text=text, font=FONT_SECTION,
            text_color=C_TEXT_BRIGHT
        ).pack(anchor="w", padx=10, pady=(12, 2))
        ctk.CTkFrame(
            parent, height=1, fg_color=C_ACCENT_DIM, corner_radius=0
        ).pack(fill="x", padx=10, pady=(0, 6))

    def _forge_label(self, parent, text: str):
        SelectableLabel(
            parent, text=text, font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=10, pady=(2, 0))

    def _forge_entry(self, parent, default: str) -> ctk.CTkEntry:
        entry = themed_entry(parent)
        entry.insert(0, default)
        entry.pack(anchor="w", padx=10, pady=(0, 6))
        return entry

    def _forge_numeric_entry(
        self, parent, default: str, *,
        mode: str = "int", allow_auto: bool = False,
    ) -> ctk.CTkEntry:
        entry = themed_numeric_entry(
            parent, mode=mode, allow_auto=allow_auto)
        entry.insert(0, default)
        entry.pack(anchor="w", padx=10, pady=(0, 6))
        return entry

    def _mark_preset_custom(self):
        """Switch training recipe to 'Custom' when user manually edits."""
        if getattr(self, "_preset_programmatic", False):
            return
        var = getattr(self, "_forge_preset_var", None)
        if var is not None and not var.get().startswith("Custom"):
            var.set("Custom - set your own values")

    def _validate_batch_entry(self):
        """Flash warning if batch entry has garbage input."""
        widget = getattr(self, "forge_batch_entry", None)
        if widget is None:
            return
        raw = widget.get().strip().lower()
        if not raw or raw == "auto":
            widget.configure(border_color=C_ACCENT_DIM)
            return
        try:
            val = int(raw)
            if val < 1:
                raise ValueError
            widget.configure(border_color=C_ACCENT_DIM)
        except ValueError:
            widget.configure(border_color="#e74c3c")
            self._log(f"[!] Batch size '{raw}' — "
                      f"use a number or 'auto'")

    def _select_training_stage(self, stage: str):
        """Update the selected training stage button highlight."""
        self.training_stage_var.set(stage)
        for name, btn in self._stage_buttons.items():
            if name == stage:
                btn.configure(fg_color=C_GREEN_DIM,
                              text_color=C_GREEN)
            else:
                btn.configure(fg_color=C_SURFACE,
                              text_color=C_TEXT)
