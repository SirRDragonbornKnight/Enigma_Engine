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
    themed_dropdown, themed_entry, themed_scroll,
    wire_hotkeys,
)
from enigma_engine.gui.scanners import DATA_DIR

logger = logging.getLogger(__name__)


class ForgePageMixin:
    """Mixin providing the FORGE page builder for EnigmaGUI.

    Expects the host class to have:
    - _make_page, training_files, _QUICK_PROFILE_FIELDS
    - Various forge callback methods from ForgeMixin
    """

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
            row=0, column=0, rowspan=2, padx=(0, 6))
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
            text_color=C_TEXT_DIM, anchor="e")
        self._forge_trainer_info.grid(
            row=0, column=2, rowspan=2, padx=(4, 0))

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
            row=0, column=0, rowspan=2, padx=(0, 6))
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
            text_color=C_TEXT_DIM, anchor="e")
        self._forge_student_info.grid(
            row=0, column=2, rowspan=2, padx=(4, 0))

        # Param count label — updated after each training session
        self._forge_student_params = SelectableLabel(
            st_inner, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._forge_student_params.grid(
            row=2, column=1, columnspan=2, sticky="w", pady=(2, 0))

        # Kept for backward compatibility
        self._trainer_route_label = self._forge_trainer_name
        self._student_route_label = self._forge_student_name

        # --- Training section ---
        self._forge_heading(ctrl_scroll, "TRAIN")

        # ===== NEW SIMPLIFIED TRAINING MODE SELECTOR =====
        # Instead of dropdown, use card-based selection
        self._forge_label(ctrl_scroll, "Choose training method")

        # Card container
        modes_frame = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        modes_frame.pack(fill="x", padx=10, pady=(0, 8))

        self.training_mode_var = ctk.StringVar(value="Basic")

        # Define the 4 main modes with descriptions
        training_modes = [
            ("Pre-Train", "Language pre-training from scratch on large text data. Builds foundational capabilities"),
            ("Distill", "Teacher generates personality, reasoning, and knowledge data. Student fine-tunes on it"),
            ("Basic", "Train on your own data (text files, JSONL). Auto-selects LoRA for large models"),
            ("AI-Guided", "AI teacher creates curriculum and trains your model. Can work with or without data"),
            ("Image", "Train on images or video. Teach visual understanding. Requires image folder"),
        ]

        # Create radio button cards
        for mode_name, mode_desc in training_modes:
            mode_card = HUDFrame(
                modes_frame, glow_color=C_BORDER)
            mode_card.pack(fill="x", pady=(0, 4))

            card_inner = ctk.CTkFrame(
                mode_card, fg_color="transparent")
            card_inner.pack(fill="x", padx=8, pady=8)
            card_inner.grid_columnconfigure(1, weight=1)

            # Radio button
            radio = ctk.CTkRadioButton(
                card_inner, text="", variable=self.training_mode_var,
                value=mode_name, font=FONT_SMALL,
                fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
                border_color=C_ACCENT_DIM,
                command=self._on_training_mode_selected)
            radio.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 8))

            # Mode title
            SelectableLabel(
                card_inner, text=mode_name.upper(),
                font=FONT_SMALL, text_color=C_TEXT_BRIGHT,
                anchor="w"
            ).grid(row=0, column=1, sticky="w")

            # Mode description (CTkLabel for text wrapping)
            ctk.CTkLabel(
                card_inner, text=mode_desc,
                font=FONT_TINY, text_color=C_TEXT_DIM,
                anchor="w", justify="left",
                wraplength=400
            ).grid(row=1, column=1, sticky="w", pady=(2, 0))

        # Include reasoning (CoT-B) — generates <think> chains in data
        reasoning_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        reasoning_row.pack(fill="x", padx=10, pady=(0, 6))
        self.forge_reasoning_var = ctk.BooleanVar(value=False)
        self._forge_reasoning_cb = ctk.CTkCheckBox(
            reasoning_row, text="Include reasoning chains",
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

        # === KNOWLEDGE PRESERVATION ===
        # The AI should be general-purpose. Training focuses it on a
        # topic without forgetting everything else.
        preserve_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        preserve_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(preserve_section, "Knowledge preservation")

        # Focus balance — how much to focus vs stay general
        focus_row = ctk.CTkFrame(
            preserve_section, fg_color="transparent")
        focus_row.pack(fill="x", padx=10, pady=(0, 4))
        SelectableLabel(
            focus_row, text="General mix:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.forge_general_mix_var = ctk.StringVar(value="20")
        general_mix_entry = themed_entry(
            focus_row, textvariable=self.forge_general_mix_var,
            width=50)
        general_mix_entry.pack(side="left", padx=(0, 4))
        SelectableLabel(
            focus_row, text="%", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left")
        Tooltip(general_mix_entry,
                "How much general knowledge to mix in during\n"
                "focused training. Prevents the AI from\n"
                "forgetting everything except what you trained.\n"
                "0% = only your data, 20% = recommended.\n"
                "Higher = more general, lower = more focused.")

        # General knowledge file
        gen_data_row = ctk.CTkFrame(
            preserve_section, fg_color="transparent")
        gen_data_row.pack(fill="x", padx=10, pady=(0, 6))
        gen_data_row.grid_columnconfigure(0, weight=1)
        self.forge_general_data_var = ctk.StringVar(value="")
        self._forge_general_data_entry = themed_entry(
            gen_data_row,
            textvariable=self.forge_general_data_var,
            placeholder_text="General knowledge file (optional)")
        self._forge_general_data_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._forge_general_browse_btn = ctk.CTkButton(
            gen_data_row, text="Browse", width=70,
            fg_color=C_SURFACE, hover_color=C_PANEL,
            text_color=C_TEXT, font=FONT_SMALL,
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

        self._forge_label(self._forge_pretrain_section, "Model architecture")
        preset_row = ctk.CTkFrame(
            self._forge_pretrain_section, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(0, 4))
        SelectableLabel(
            preset_row, text="Preset:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.pretrain_preset_var = ctk.StringVar()
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, MODEL_DESCRIPTIONS)
        _preset_display = [
            f"{k} - {MODEL_DESCRIPTIONS.get(k, '')}"
            for k in MODEL_PRESETS]
        self.pretrain_preset_var.set(
            _preset_display[list(MODEL_PRESETS.keys()).index("large")])
        self._pretrain_preset_dd = themed_dropdown(
            preset_row, _preset_display,
            variable=self.pretrain_preset_var,
            width=340)
        self._pretrain_preset_dd.pack(side="left", padx=(0, 6))

        self._forge_label(self._forge_pretrain_section,
            "Pre-training data (required)")
        self.pretrain_data_var = ctk.StringVar(value="")
        pt_data_row = ctk.CTkFrame(
            self._forge_pretrain_section, fg_color="transparent")
        pt_data_row.pack(fill="x", padx=10, pady=(0, 4))
        pt_data_row.grid_columnconfigure(0, weight=1)
        self._pretrain_data_entry = themed_entry(
            pt_data_row, textvariable=self.pretrain_data_var,
            placeholder_text="Path to text file or directory...")
        self._pretrain_data_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self._pretrain_browse_btn = ctk.CTkButton(
            pt_data_row, text="Browse", width=70,
            fg_color=C_SURFACE, hover_color=C_PANEL,
            text_color=C_TEXT, font=FONT_SMALL,
            command=lambda: self._browse_training_data(
                target_var=self.pretrain_data_var))
        self._pretrain_browse_btn.grid(row=0, column=1, sticky="e")
        Tooltip(self._pretrain_data_entry,
                "Path to a large text file or directory of text files\n"
                "for pre-training. Supports .txt, .jsonl formats.\n"
                "Recommended starting data: TinyStories (~500M tokens).")

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
        Tooltip(pretrain_vocab_entry,
                "Tokenizer vocabulary size for the new model.\n"
                "32K-50K is standard for pre-training.\n"
                "Larger vocab = better coverage, more params.")
        Tooltip(retrain_tok_cb,
                "Train a new BPE tokenizer on your pre-training data.\n"
                "Recommended for pre-training from scratch.\n"
                "The tokenizer learns which words/subwords appear most.")

        # Model name
        name_row = ctk.CTkFrame(
            self._forge_pretrain_section, fg_color="transparent")
        name_row.pack(fill="x", padx=10, pady=(0, 6))
        SelectableLabel(
            name_row, text="Model name:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.pretrain_model_name_var = ctk.StringVar(
            value="pretrained_model")
        pretrain_name_entry = themed_entry(
            name_row, textvariable=self.pretrain_model_name_var,
            width=200)
        pretrain_name_entry.pack(side="left")
        Tooltip(pretrain_name_entry,
                "Name for the new pre-trained model.\n"
                "Will be saved as models/<name>.pth")

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
        distill_examples_entry = themed_entry(
            examples_row,
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
        distill_len_entry = themed_entry(
            distill_len_row,
            textvariable=self.distill_max_tokens_var,
            width=60)
        distill_len_entry.pack(side="left")
        Tooltip(distill_len_entry,
                "Max tokens per teacher response.\n"
                "Longer = richer examples, shorter = faster.")

        # === BASIC TRAINING OPTIONS ===
        self._forge_basic_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_basic_section.pack(fill="x", padx=0, pady=(8, 0))

        self._forge_label(self._forge_basic_section, "Training data (required)")
        self.train_data_var = ctk.StringVar(
            value=self.training_files[0]["path"]
            if self.training_files else "")
        data_opts = [
            f"{f['name']} ({f['size_kb']} KB)"
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
        self._forge_data_browse_btn = ctk.CTkButton(
            data_path_row, text="Browse", width=70,
            fg_color=C_SURFACE, hover_color=C_PANEL,
            text_color=C_TEXT, font=FONT_SMALL,
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
        self._forge_ai_data_browse_btn = ctk.CTkButton(
            ai_data_path_row, text="Browse", width=70,
            fg_color=C_SURFACE, hover_color=C_PANEL,
            text_color=C_TEXT, font=FONT_SMALL,
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
        self._forge_vision_browse_btn = ctk.CTkButton(
            vision_dir_row, text="Browse", width=70,
            fg_color=C_SURFACE, hover_color=C_PANEL,
            text_color=C_TEXT, font=FONT_SMALL,
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
            btn = ctk.CTkButton(
                stage_frame,
                text=stage_name.upper(),
                width=90, height=30,
                font=FONT_SMALL, corner_radius=2,
                fg_color=C_GREEN_DIM if stage_name == "basics"
                    else C_SURFACE,
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
        self.guided_pairs_entry = self._forge_entry(
            self._forge_pairs_section, "20")
        Tooltip(self.guided_pairs_entry,
                "Number of Q&A pairs to generate per stage.\n"
                "More pairs = better coverage, longer training.")
        # Aliases
        self.dialogue_rounds_entry = self.guided_pairs_entry

        # === HYPERPARAMETERS (all modes) ===
        self._forge_heading(ctrl_scroll, "HYPERPARAMETERS")

        # Preset dropdown for quick configuration
        self._forge_label(ctrl_scroll, "Preset")
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
        self.ft_epochs_entry = self._forge_entry(ctrl_scroll, "10")
        Tooltip(self.ft_epochs_entry,
                "Number of passes through the training data.\n"
                "More epochs = better learning, longer training.\n"
                "Default: 10")
        # Aliases for backward compatibility
        self.guided_epochs_entry = self.ft_epochs_entry
        self.epochs_entry = self.ft_epochs_entry

        self._forge_label(ctrl_scroll, "Learning rate")
        self.ft_lr_entry = self._forge_entry(ctrl_scroll, "0.00005")
        Tooltip(self.ft_lr_entry,
                "How fast the model adapts to training data.\n"
                "Too high = unstable, too low = slow learning.\n"
                "Default: 5e-5 (0.00005)")
        # Aliases
        self.guided_lr_entry = self.ft_lr_entry
        self.lr_entry = self.ft_lr_entry

        # Main train button row
        btn_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(8, 4))

        self.solo_train_btn = ctk.CTkButton(
            btn_row, text="TRAIN", width=170, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a4a2e",
            text_color=C_GREEN,
            command=self._start_training_by_mode)
        self.solo_train_btn.pack(side="left")
        # Aliases for all mode buttons
        self.guided_train_btn = self.solo_train_btn
        self.dialogue_train_btn = self.solo_train_btn
        self.train_model_btn = self.solo_train_btn
        Tooltip(self.solo_train_btn,
                "Start training with the selected mode")

        self.stop_train_btn = ctk.CTkButton(
            btn_row, text="STOP", width=80, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_RED,
            text_color=C_TEXT_DIM, command=self._stop_training,
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

        # Save button row (below train)
        save_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        save_row.pack(fill="x", padx=10, pady=(0, 4))

        self.save_ckpt_btn = ctk.CTkButton(
            save_row, text="SAVE CHECKPOINT", width=170,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._save_forge_checkpoint)
        self.save_ckpt_btn.pack(side="left")
        Tooltip(self.save_ckpt_btn,
                "Save STUDENT to a named checkpoint")

        self.load_ckpt_btn = ctk.CTkButton(
            save_row, text="LOAD CHECKPOINT", width=170,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
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
        self.forge_batch_entry = self._forge_entry(advanced_inner, "4")
        Tooltip(self.forge_batch_entry,
                "Training batch size. Lower = less VRAM.\n"
                "Default: 4")

        # Gradient accumulation
        self._forge_label(advanced_inner, "Gradient accumulation")
        self.forge_accum_entry = self._forge_entry(advanced_inner, "1")
        Tooltip(self.forge_accum_entry,
                "Accumulate gradients over N steps.\n"
                "Effective batch = batch_size × this value.\n"
                "Default: 1 (no accumulation)")

        # Gradient checkpointing
        ckpt_row = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        ckpt_row.pack(fill="x", padx=10, pady=(0, 6))
        self.forge_grad_ckpt_var = ctk.BooleanVar(value=False)
        self.forge_grad_ckpt_cb = ctk.CTkCheckBox(
            ckpt_row, text="Gradient checkpointing",
            variable=self.forge_grad_ckpt_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        self.forge_grad_ckpt_cb.pack(anchor="w")
        Tooltip(self.forge_grad_ckpt_cb,
                "Saves VRAM by recomputing activations.\n"
                "Slower training but fits larger models.")

        # Rolling best checkpoints
        rolling_row = ctk.CTkFrame(
            advanced_inner, fg_color="transparent")
        rolling_row.pack(fill="x", padx=10, pady=(0, 6))
        SelectableLabel(
            rolling_row, text="Rolling best K", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.forge_rolling_k_entry = themed_entry(rolling_row, width=50)
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
        lora_rank_entry = themed_entry(
            lora_row, textvariable=self.forge_lora_rank_var, width=60)
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
        lora_alpha_entry = themed_entry(
            lora_row, textvariable=self.forge_lora_alpha_var, width=60)
        lora_alpha_entry.pack(side="left")
        Tooltip(lora_alpha_entry,
                "LoRA alpha scaling (1-256).\n"
                "Higher = stronger adapter effect.\n"
                "Common: alpha = 2x rank. Default: 16")

        # Audio encoder settings
        self.forge_use_conformer_var = ctk.BooleanVar(value=False)
        self.forge_use_conformer_cb = ctk.CTkCheckBox(
            advanced_inner, text="Use Conformer convolution (audio)",
            variable=self.forge_use_conformer_var,
            font=FONT_TINY, text_color=C_TEXT_DIM,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        self.forge_use_conformer_cb.pack(anchor="w", padx=10, pady=(2, 2))
        Tooltip(self.forge_use_conformer_cb,
                "Enable Conformer convolution in audio encoder.\n"
                "Adds depthwise conv between attention and FFN.\n"
                "Improves audio quality at slight speed cost.")

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
        replay_cap_entry = themed_entry(
            replay_row, textvariable=self.forge_replay_capacity_var,
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
        replay_ratio_entry = themed_entry(
            replay_row, textvariable=self.forge_replay_ratio_var,
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
        val_split_entry = themed_entry(
            val_row, textvariable=self.forge_val_split_var,
            width=60)
        val_split_entry.pack(side="left", padx=(0, 4))
        Tooltip(val_split_entry,
                "Fraction of training data held out for validation.\n"
                "0.0 = no validation, 0.1 = 10%. Default: 0.1")

        # --- Tools section (collapsed by default) ---
        tools_panel = CollapsiblePanel(
            ctrl_scroll, title="TOOLS", start_expanded=False)
        tools_panel.pack(fill="x", padx=6, pady=(10, 2))
        tools_inner = tools_panel.content

        tools_row1 = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        tools_row1.pack(fill="x", padx=6, pady=(6, 3))

        self.generate_data_btn = ctk.CTkButton(
            tools_row1, text="GENERATE DATA", width=140,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._generate_training_data)
        self.generate_data_btn.pack(side="left")
        Tooltip(self.generate_data_btn,
                "TRAINER generates training Q/A pairs")

        self.evaluate_btn = ctk.CTkButton(
            tools_row1, text="EVALUATE", width=120, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._evaluate_student)
        self.evaluate_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.evaluate_btn,
                "TRAINER tests STUDENT and scores answers")

        self.benchmark_btn = ctk.CTkButton(
            tools_row1, text="BENCHMARK", width=120, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._coherence_benchmark)
        self.benchmark_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.benchmark_btn,
                "Test model reflection quality (coherence benchmark)")

        # History button
        self._forge_history_btn = ctk.CTkButton(
            tools_row1, text="HISTORY", width=90, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._show_training_history)
        self._forge_history_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_history_btn,
                "Show past training runs with loss values")

        # Web Learn
        web_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        web_row.pack(fill="x", padx=6, pady=(6, 3))

        self.web_learn_topic = themed_entry(
            web_row, width=180, placeholder_text="Topic...")
        self.web_learn_topic.pack(side="left")

        self.web_learn_pages_entry = themed_entry(
            web_row, width=50,
            placeholder_text="3")
        self.web_learn_pages_entry.insert(0, "3")
        self.web_learn_pages_entry.pack(side="left", padx=(6, 0))
        Tooltip(self.web_learn_pages_entry,
                "Max web pages to fetch (1-10)")

        self.web_learn_btn = ctk.CTkButton(
            web_row, text="WEB LEARN", width=110, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._web_learn)
        self.web_learn_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.web_learn_btn,
                "Search web and generate training data")

        # Tokenizer training
        tok_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        tok_row.pack(fill="x", padx=6, pady=(6, 3))

        self.vocab_entry = themed_entry(
            tok_row, width=100, placeholder_text="Vocab: 8000")
        self.vocab_entry.insert(0, "8000")
        self.vocab_entry.pack(side="left")

        self.train_tok_btn = ctk.CTkButton(
            tok_row, text="TRAIN TOKENIZER", width=140,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._start_tokenizer_training)
        self.train_tok_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.train_tok_btn,
                "Train a BPE tokenizer on selected data")

        # Quantize model
        quant_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        quant_row.pack(fill="x", padx=6, pady=(6, 3))

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

        self.quantize_btn = ctk.CTkButton(
            quant_row, text="QUANTIZE", width=120,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
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

        self.export_gguf_btn = ctk.CTkButton(
            export_row, text="EXPORT GGUF", width=140,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._export_student_gguf)
        self.export_gguf_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.export_gguf_btn,
                "Export STUDENT as a GGUF file for llama.cpp")

        # --- Queue & Schedule (TS-B, TS-C) ---
        queue_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        queue_row.pack(fill="x", padx=6, pady=(6, 3))

        self._forge_add_queue_btn = ctk.CTkButton(
            queue_row, text="ADD TO QUEUE", width=130,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._add_to_training_queue)
        self._forge_add_queue_btn.pack(side="left")
        Tooltip(self._forge_add_queue_btn,
                "Add current settings as a job to the queue.\n"
                "Queue runs jobs sequentially in background.")

        self._forge_show_queue_btn = ctk.CTkButton(
            queue_row, text="QUEUE", width=80,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._show_training_queue)
        self._forge_show_queue_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_show_queue_btn,
                "Show/manage the training queue")

        self._forge_run_queue_btn = ctk.CTkButton(
            queue_row, text="RUN", width=60,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a4a2e",
            text_color=C_GREEN,
            command=self._run_training_queue)
        self._forge_run_queue_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_run_queue_btn,
                "Start running the training queue")

        # Overnight plan
        plan_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        plan_row.pack(fill="x", padx=6, pady=(3, 3))

        self._forge_save_plan_btn = ctk.CTkButton(
            plan_row, text="SAVE PLAN", width=110,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._save_overnight_plan)
        self._forge_save_plan_btn.pack(side="left")
        Tooltip(self._forge_save_plan_btn,
                "Save the current queue as an overnight plan.\n"
                "Saves progress and resumes on crash.")

        self._forge_load_plan_btn = ctk.CTkButton(
            plan_row, text="LOAD PLAN", width=110,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._load_overnight_plan)
        self._forge_load_plan_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_load_plan_btn,
                "Load a saved overnight training plan.\n"
                "Resumes where it left off.")

        # --- Curated Dataset (DA-C) ---
        dataset_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        dataset_row.pack(fill="x", padx=6, pady=(6, 3))

        self._forge_review_dataset_btn = ctk.CTkButton(
            dataset_row, text="REVIEW DATASET", width=140,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._review_curated_dataset)
        self._forge_review_dataset_btn.pack(side="left")
        Tooltip(self._forge_review_dataset_btn,
                "Review/approve/reject entries in the\n"
                "curated master dataset before training.")

        self._forge_approve_all_btn = ctk.CTkButton(
            dataset_row, text="APPROVE ALL", width=110,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._approve_all_dataset)
        self._forge_approve_all_btn.pack(side="left", padx=(6, 0))
        Tooltip(self._forge_approve_all_btn,
                "Approve all pending entries in the\n"
                "curated dataset for training use.")

        # --- Command Policy Generator ---
        cmd_policy_row = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        cmd_policy_row.pack(fill="x", padx=6, pady=(6, 3))

        self._forge_cmd_policy_btn = ctk.CTkButton(
            cmd_policy_row, text="COMMAND POLICY", width=150,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
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
