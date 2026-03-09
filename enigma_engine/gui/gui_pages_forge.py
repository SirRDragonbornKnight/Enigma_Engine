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

        self._forge_label(ctrl_scroll, "Training mode")
        self.training_mode_var = ctk.StringVar(
            value="Self Study")
        self.training_mode_menu = themed_dropdown(
            ctrl_scroll,
            values=["Self Study", "Conversation",
                    "Preference Tuning", "Image Training",
                    "Quick Tune (LoRA)", "Trial & Error",
                    "Adaptive Pipeline", "RLHF", "Self-Play"],
            variable=self.training_mode_var,
            width=200,
            command=self._on_training_mode_changed)
        self.training_mode_menu.pack(
            anchor="w", padx=10, pady=(0, 6))

        # Mode description (updates when mode changes)
        self._training_mode_desc = ctk.CTkLabel(
            ctrl_scroll,
            text="Train your AI directly on a text file.\n"
                 "Needs: STUDENT model + data file.\n"
                 "Best for: Teaching from existing content.",
            font=FONT_TINY, text_color=C_TEXT_DIM,
            justify="left")
        self._training_mode_desc.pack(
            anchor="w", padx=10, pady=(0, 6))

        # Train with AI toggle — use TRAINER model to generate
        # curriculum, train the student, then test what it learned
        ai_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        ai_row.pack(fill="x", padx=10, pady=(0, 6))
        self.train_with_ai_var = ctk.BooleanVar(value=False)
        self.train_with_ai_cb = ctk.CTkCheckBox(
            ai_row, text="Train with AI",
            variable=self.train_with_ai_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._on_train_ai_toggled)
        self.train_with_ai_cb.pack(anchor="w")
        Tooltip(self.train_with_ai_cb,
                "Enable AI-assisted training.\n"
                "A TRAINER model generates curriculum,\n"
                "trains your model, then tests what it learned.\n"
                "Requires: TRAINER route assigned.")

        # Include reasoning (CoT-B) — generates <think> chains in data
        self.forge_reasoning_var = ctk.BooleanVar(value=False)
        self._forge_reasoning_cb = ctk.CTkCheckBox(
            ai_row, text="Include reasoning",
            variable=self.forge_reasoning_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_GREEN_DIM, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2)
        self._forge_reasoning_cb.pack(anchor="w", pady=(2, 0))
        Tooltip(self._forge_reasoning_cb,
                "Generate training data with <think> reasoning\n"
                "chains. Teaches the model to reason step-by-step\n"
                "before answering, not just memorize answers.")

        # --- Data source (shown/hidden per mode) ---
        self._forge_data_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_data_section.pack(
            fill="x", padx=0, pady=0)
        self._forge_data_label = SelectableLabel(
            self._forge_data_section, text="Data source (optional)",
            font=FONT_TINY, text_color=C_TEXT_DIM)
        self._forge_data_label.pack(
            anchor="w", padx=10, pady=(2, 0))
        self.train_data_var = ctk.StringVar(
            value=self.training_files[0]["path"]
            if self.training_files else "")
        data_opts = [
            f"{f['name']} ({f['size_kb']} KB)"
            for f in self.training_files]
        # Always show dropdown — add (none) so data can be deselected
        data_row = ctk.CTkFrame(
            self._forge_data_section, fg_color="transparent")
        data_row.pack(fill="x", padx=10, pady=(0, 6))
        all_opts = ["(none)"] + data_opts
        self.train_data_menu = themed_dropdown(
            data_row, all_opts if all_opts else ["(none)"],
            width=220,
            command=self._on_data_selected)
        self.train_data_menu.pack(side="left")
        # Pre-select first data file if available
        if data_opts:
            self.train_data_menu.set(data_opts[0])
        else:
            self.train_data_menu.set("(none)")

        # --- Training stage (shown for Guided/Dialogue) ---
        self._forge_stages_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_stages_section.pack(
            fill="x", padx=0, pady=0)
        self._forge_label(
            self._forge_stages_section, "Training stage")
        self.training_stage_var = ctk.StringVar(value="basics")

        stage_frame = ctk.CTkFrame(
            self._forge_stages_section, fg_color="transparent")
        stage_frame.pack(fill="x", padx=10, pady=(0, 6))

        stage_descriptions = {
            "basics": "Teach fundamental language patterns,\ngrammar, and basic responses",
            "conversation": "Teach natural dialogue flow\nand contextual responses",
            "commands": "Teach command recognition\nand structured outputs",
            "web": "Teach web content understanding\nand information extraction",
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

        # --- Training Brief (shown for Guided/Dialogue) ---
        self._forge_brief_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_brief_section.pack(
            fill="x", padx=0, pady=0)
        brief_panel = CollapsiblePanel(
            self._forge_brief_section, title="TRAINING BRIEF",
            start_expanded=False)
        brief_panel.pack(fill="x", padx=6, pady=(8, 2))
        brief_inner = brief_panel.content

        # Quick profile fields — structured mad-libs style
        self._brief_field_entries = {}
        for label, placeholder, tip in self._QUICK_PROFILE_FIELDS:
            row = ctk.CTkFrame(
                brief_inner, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=(3, 0))
            row.grid_columnconfigure(1, weight=1)
            lbl = SelectableLabel(
                row, text=label, font=FONT_TINY,
                text_color=C_TEXT_DIM, width=100, anchor="w")
            lbl.grid(row=0, column=0, sticky="w")
            entry = themed_entry(
                row, width=180,
                placeholder_text=placeholder)
            entry.grid(row=0, column=1, sticky="ew", padx=(4, 0))
            Tooltip(entry, tip)
            self._brief_field_entries[label] = entry

        # Custom brief text area — freeform instructions
        SelectableLabel(
            brief_inner, text="Custom instructions",
            font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=6, pady=(8, 2))

        self._brief_custom_text = ctk.CTkTextbox(
            brief_inner, height=80, font=FONT_SMALL,
            fg_color=C_INPUT, text_color=C_TEXT_BRIGHT,
            border_width=1, border_color=C_ACCENT_DIM,
            corner_radius=2, wrap="word")
        self._brief_custom_text.pack(
            fill="x", padx=6, pady=(0, 4))
        # Enable undo/redo
        self._brief_custom_text._textbox.configure(undo=True, maxundo=-1)
        Tooltip(self._brief_custom_text,
                "Freeform instructions for the trainer AI.\n"
                "Describe exactly what you want the student\n"
                "AI to be — anything not covered by the\n"
                "quick fields above.")

        # Save/load brief on change — load persisted values
        self.after(200, self._load_training_brief)

        # Hyperparameter presets — quick / balanced / thorough
        self._forge_label(ctrl_scroll, "Training preset")
        self._forge_preset_menu = themed_dropdown(
            ctrl_scroll,
            ["Custom", "Quick", "Balanced", "Thorough"],
            width=280,
            command=self._on_preset_changed)
        self._forge_preset_menu.pack(
            anchor="w", padx=10, pady=(0, 6))
        Tooltip(self._forge_preset_menu,
                "Quick: 3 epochs, fast results\n"
                "Balanced: 10 epochs, good quality\n"
                "Thorough: 30 epochs, best quality\n"
                "Custom: set your own values below")

        self._forge_label(ctrl_scroll, "Epochs")
        self.ft_epochs_entry = self._forge_entry(ctrl_scroll, "5")
        Tooltip(self.ft_epochs_entry,
                "Number of passes through the training data.\n"
                "More epochs = better learning, longer training.")
        # Aliases so all training modes use the same fields
        self.guided_epochs_entry = self.ft_epochs_entry
        self.epochs_entry = self.ft_epochs_entry

        self._forge_label(ctrl_scroll, "Learning rate")
        self.ft_lr_entry = self._forge_entry(
            ctrl_scroll, "0.00005")
        Tooltip(self.ft_lr_entry,
                "How fast the model adapts to training data.\n"
                "Too high = unstable, too low = slow learning.")
        # Aliases
        self.guided_lr_entry = self.ft_lr_entry
        self.lr_entry = self.ft_lr_entry

        # Batch size — lower values use less VRAM
        self._forge_label(ctrl_scroll, "Batch size")
        self.forge_batch_entry = self._forge_entry(
            ctrl_scroll, "4")
        Tooltip(self.forge_batch_entry,
                "Training batch size. Lower = less VRAM.")

        # Gradient accumulation — simulate larger batches
        self._forge_label(ctrl_scroll, "Grad accumulation")
        self.forge_accum_entry = self._forge_entry(
            ctrl_scroll, "1")
        Tooltip(self.forge_accum_entry,
                "Accumulate gradients over N steps.\n"
                "Effective batch = batch_size × this value.")

        # Gradient checkpointing — trade compute for VRAM
        ckpt_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
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

        # Rolling best checkpoints (CK-C) — keep only K best
        rolling_row = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        rolling_row.pack(fill="x", padx=10, pady=(0, 6))
        SelectableLabel(
            rolling_row, text="Rolling best K", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 6))
        self.forge_rolling_k_entry = themed_entry(
            rolling_row, width=50)
        self.forge_rolling_k_entry.insert(0, "0")
        self.forge_rolling_k_entry.pack(side="left")
        Tooltip(self.forge_rolling_k_entry,
                "Keep only the K best checkpoints by loss.\n"
                "0 = disabled (keep all). 3 = keep 3 best.\n"
                "Prevents disk bloat during long training.")

        # --- Pairs/rounds (shown for Guided/Dialogue) ---
        self._forge_pairs_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_pairs_section.pack(
            fill="x", padx=0, pady=0)
        self._pairs_rounds_label = SelectableLabel(
            self._forge_pairs_section, text="Pairs to generate",
            font=FONT_TINY, text_color=C_TEXT_DIM)
        self._pairs_rounds_label.pack(
            anchor="w", padx=10, pady=(2, 0))
        self.guided_pairs_entry = self._forge_entry(
            self._forge_pairs_section, "20")
        # Aliases
        self.dialogue_rounds_entry = self.guided_pairs_entry

        # --- Vision config (shown for Vision mode) ---
        self._forge_vision_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_vision_section.pack(
            fill="x", padx=0, pady=0)

        self._forge_label(
            self._forge_vision_section, "Image data directory")
        self.forge_vision_dir_var = ctk.StringVar(
            value=str(DATA_DIR / "images"))
        vision_dir_row = ctk.CTkFrame(
            self._forge_vision_section, fg_color="transparent")
        vision_dir_row.pack(fill="x", padx=10, pady=(0, 6))
        vision_dir_row.grid_columnconfigure(0, weight=1)
        self._forge_vision_dir_entry = themed_entry(
            vision_dir_row, textvariable=self.forge_vision_dir_var)
        self._forge_vision_dir_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        Tooltip(self._forge_vision_dir_entry,
                "Folder with image+text pairs.\n"
                "Format 1: image.png + image.txt (same name)\n"
                "Format 2: captions.jsonl with image+text fields")
        self._forge_vision_browse_btn = ctk.CTkButton(
            vision_dir_row, text="Browse", width=70,
            fg_color=C_SURFACE, hover_color=C_PANEL,
            text_color=C_TEXT, font=FONT_SMALL,
            command=self._browse_vision_dir)
        self._forge_vision_browse_btn.grid(
            row=0, column=1, sticky="e")

        self._forge_label(
            self._forge_vision_section, "Encoder size")
        self.forge_vision_preset_var = ctk.StringVar(value="small")
        vision_preset_dd = themed_dropdown(
            self._forge_vision_section,
            values=["tiny", "small", "medium"],
            variable=self.forge_vision_preset_var,
            width=120)
        vision_preset_dd.pack(anchor="w", padx=10, pady=(0, 6))
        Tooltip(vision_preset_dd,
                "Vision encoder size:\n"
                "  tiny — ~500K params (fast, lower quality)\n"
                "  small — ~4M params (default, good balance)\n"
                "  medium — ~25M params (best quality, more VRAM)")

        # --- LoRA config (shown for LoRA mode) ---
        self._forge_lora_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_lora_section.pack(
            fill="x", padx=0, pady=0)

        self._forge_label(self._forge_lora_section, "LoRA rank")
        self.forge_lora_rank_var = ctk.StringVar(value="8")
        lora_rank_entry = themed_entry(
            self._forge_lora_section,
            textvariable=self.forge_lora_rank_var, width=80)
        lora_rank_entry.pack(anchor="w", padx=10, pady=(0, 4))
        Tooltip(lora_rank_entry,
                "LoRA rank (1-128). Lower = fewer trainable params.\n"
                "4-8 for small models, 16-32 for large models.")

        self._forge_label(self._forge_lora_section, "LoRA alpha")
        self.forge_lora_alpha_var = ctk.StringVar(value="16")
        lora_alpha_entry = themed_entry(
            self._forge_lora_section,
            textvariable=self.forge_lora_alpha_var, width=80)
        lora_alpha_entry.pack(anchor="w", padx=10, pady=(0, 6))
        Tooltip(lora_alpha_entry,
                "LoRA alpha scaling (1-256).\n"
                "Higher = stronger adapter effect.\n"
                "Common: alpha = 2x rank.")

        # --- Evolutionary config (shown for Evolutionary mode) ---
        self._forge_evo_section = ctk.CTkFrame(
            ctrl_scroll, fg_color="transparent")
        self._forge_evo_section.pack(
            fill="x", padx=0, pady=0)

        self._forge_label(self._forge_evo_section, "Generations")
        self.forge_evo_gens_var = ctk.StringVar(value="5")
        evo_gens_entry = themed_entry(
            self._forge_evo_section,
            textvariable=self.forge_evo_gens_var, width=80)
        evo_gens_entry.pack(anchor="w", padx=10, pady=(0, 4))
        Tooltip(evo_gens_entry,
                "Number of evolutionary generations (1-100).\n"
                "Each generation: generate, score, keep best, train.")

        self._forge_label(
            self._forge_evo_section, "Candidates per task")
        self.forge_evo_npn_var = ctk.StringVar(value="3")
        evo_npn_entry = themed_entry(
            self._forge_evo_section,
            textvariable=self.forge_evo_npn_var, width=80)
        evo_npn_entry.pack(anchor="w", padx=10, pady=(0, 6))
        Tooltip(evo_npn_entry,
                "Responses generated per task (2-20).\n"
                "Higher = better selection, slower.")

        # --- Focus field (optional, all modes) ---
        self._forge_label(ctrl_scroll, "Focus field (optional)")
        self.forge_focus_field = themed_entry(
            ctrl_scroll, width=280,
            placeholder_text="e.g. medical, coding, cooking...")
        self.forge_focus_field.pack(
            anchor="w", padx=10, pady=(0, 6))
        Tooltip(self.forge_focus_field,
                "Optionally focus training on a specific field.\n"
                "When set, the system prompt tells the trainer AI\n"
                "to prioritize this topic in generated data.\n"
                "Leave empty for general-purpose training.")

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

        tools_row2 = ctk.CTkFrame(
            tools_inner, fg_color="transparent")
        tools_row2.pack(fill="x", padx=6, pady=(3, 3))

        self.save_ckpt_btn = ctk.CTkButton(
            tools_row2, text="SAVE CHECKPOINT", width=140,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._save_forge_checkpoint)
        self.save_ckpt_btn.pack(side="left")
        Tooltip(self.save_ckpt_btn,
                "Save STUDENT to a named checkpoint")

        self.load_ckpt_btn = ctk.CTkButton(
            tools_row2, text="LOAD CHECKPOINT", width=140,
            height=34, font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT,
            command=self._load_forge_checkpoint)
        self.load_ckpt_btn.pack(side="left", padx=(6, 0))
        Tooltip(self.load_ckpt_btn,
                "Load a saved checkpoint back into STUDENT")

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

        self.quantize_mode_var = ctk.StringVar(value="int8")
        self.quantize_mode_dd = themed_dropdown(
            quant_row, width=100,
            values=["dynamic", "int8", "int4"],
            variable=self.quantize_mode_var)
        self.quantize_mode_dd.pack(side="left")
        Tooltip(self.quantize_mode_dd,
                "Quantization mode:\n"
                "  dynamic — per-tensor dynamic range\n"
                "  int8 — 8-bit weights\n"
                "  int4 — 4-bit weights (smallest)")

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

        self.export_gguf_mode_var = ctk.StringVar(value="Q8_0")
        self.export_gguf_mode_dd = themed_dropdown(
            export_row, width=100,
            values=["F16", "Q8_0", "Q4_0"],
            variable=self.export_gguf_mode_var)
        self.export_gguf_mode_dd.pack(side="left")
        Tooltip(self.export_gguf_mode_dd,
                "GGUF quantization type:\n"
                "  F16 — float16 (largest, best quality)\n"
                "  Q8_0 — 8-bit (good balance)\n"
                "  Q4_0 — 4-bit (smallest)")

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

        # Set initial section visibility for default mode
        self._on_training_mode_changed("Self Study")

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
