"""
Model Manager Dialog - Manage, scale, backup, and organize AI models.

This is a large dialog extracted from enhanced_window.py for better maintainability.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QListWidget, QGroupBox,
    QLineEdit, QComboBox, QMessageBox, QTextEdit,
    QApplication, QDialogButtonBox, QCheckBox, QInputDialog,
    QWizard
)
from PyQt5.QtCore import Qt

from ...core.model_registry import ModelRegistry
from ...core.model_scaling import grow_model, shrink_model


class ModelManagerDialog(QDialog):
    """Modern model manager dialog - manage, scale, backup, and organize models."""
    
    def __init__(self, registry: ModelRegistry, current_model: str = None, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.current_model = current_model
        self.selected_model = None
        
        self.setWindowTitle("Model Manager")
        self.setMinimumSize(700, 500)
        self.resize(800, 550)
        
        # Make dialog non-modal so it doesn't block
        self.setModal(False)
        
        self._build_ui()
        self._refresh_list()
        
        # Apply dark style to dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QListWidget {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QListWidget::item:hover {
                background-color: #45475a;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
            QPushButton:pressed {
                background-color: #313244;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #89b4fa;
                subcontrol-origin: margin;
                left: 12px;
            }
            QLabel {
                color: #cdd6f4;
            }
        """)
    
    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel - Model list
        left_panel = QVBoxLayout()
        
        # Header with refresh
        header = QHBoxLayout()
        title = QLabel("Your Models")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        header.addWidget(title)
        header.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedSize(60, 32)
        refresh_btn.setToolTip("Refresh list")
        refresh_btn.clicked.connect(self._refresh_list)
        header.addWidget(refresh_btn)
        left_panel.addLayout(header)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self._on_select_model)
        self.model_list.itemDoubleClicked.connect(self._on_load_model)
        left_panel.addWidget(self.model_list)
        
        # Quick actions under list
        quick_btns = QHBoxLayout()
        
        new_btn = QPushButton("+ New")
        new_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e;")
        new_btn.clicked.connect(self._on_new_model)
        quick_btns.addWidget(new_btn)
        
        load_btn = QPushButton("Load")
        load_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e;")
        load_btn.clicked.connect(self._on_load_model)
        quick_btns.addWidget(load_btn)
        
        left_panel.addLayout(quick_btns)
        
        # HuggingFace Models Section
        hf_group = QGroupBox("HuggingFace Models")
        hf_layout = QVBoxLayout(hf_group)
        hf_layout.setSpacing(8)
        
        # Preset dropdown with model sizes and categories
        self.hf_preset_combo = QComboBox()
        self.hf_preset_combo.addItem("Select a preset model...")
        self.hf_preset_combo.addItem("microsoft/DialoGPT-small (162M) [Small] - Fast chat")
        self.hf_preset_combo.addItem("microsoft/DialoGPT-medium (405M) [Medium] - Conversational")
        self.hf_preset_combo.addItem("Salesforce/codegen-350M-mono (350M) [Small] - Code")
        self.hf_preset_combo.addItem("TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B) [Medium] - Fast chat")
        self.hf_preset_combo.addItem("Qwen/Qwen2-1.5B-Instruct (1.5B) [Medium] - Multilingual")
        self.hf_preset_combo.addItem("stabilityai/stablelm-2-zephyr-1_6b (1.6B) [Medium] - Stable chat")
        self.hf_preset_combo.addItem("google/gemma-2b-it (2B) [Medium] - Google Gemma")
        self.hf_preset_combo.addItem("mistralai/Mistral-7B-Instruct-v0.2 (7B) [Large] GPU")
        self.hf_preset_combo.addItem("HuggingFaceH4/zephyr-7b-beta (7B) [Large] GPU")
        self.hf_preset_combo.addItem("meta-llama/Llama-2-7b-chat-hf (7B) [Large] GPU")
        self.hf_preset_combo.addItem("xai-org/grok-1 (314B) [Huge] Datacenter")
        hf_layout.addWidget(self.hf_preset_combo)
        
        # Custom input
        hf_input_layout = QHBoxLayout()
        self.hf_model_input = QLineEdit()
        self.hf_model_input.setPlaceholderText("Or enter HuggingFace model ID...")
        hf_input_layout.addWidget(self.hf_model_input)
        
        self.hf_add_btn = QPushButton("Add")
        self.hf_add_btn.setStyleSheet("background-color: #fab387; color: #1e1e2e;")
        self.hf_add_btn.clicked.connect(self._on_add_hf_model)
        hf_input_layout.addWidget(self.hf_add_btn)
        hf_layout.addLayout(hf_input_layout)
        
        # Delete HF model button
        self.hf_delete_btn = QPushButton("Delete Selected HF Model")
        self.hf_delete_btn.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
        self.hf_delete_btn.setToolTip("Delete a HuggingFace model from this system")
        self.hf_delete_btn.clicked.connect(self._on_delete_hf_model)
        hf_layout.addWidget(self.hf_delete_btn)
        
        # Tokenizer option
        tokenizer_layout = QHBoxLayout()
        tokenizer_label = QLabel("Tokenizer:")
        tokenizer_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        tokenizer_layout.addWidget(tokenizer_label)
        
        self.hf_tokenizer_combo = QComboBox()
        self.hf_tokenizer_combo.addItem("Model's Own (Recommended)")
        self.hf_tokenizer_combo.addItem("Custom Forge Tokenizer")
        self.hf_tokenizer_combo.setToolTip("Choose which tokenizer to use with HuggingFace models")
        self.hf_tokenizer_combo.setStyleSheet("font-size: 11px;")
        tokenizer_layout.addWidget(self.hf_tokenizer_combo)
        tokenizer_layout.addStretch()
        hf_layout.addLayout(tokenizer_layout)
        
        # Info label
        hf_info = QLabel("Note: Large models need good GPU & HF token for gated models")
        hf_info.setStyleSheet("color: #6c7086; font-size: 10px;")
        hf_info.setWordWrap(True)
        hf_layout.addWidget(hf_info)
        
        left_panel.addWidget(hf_group)
        
        layout.addLayout(left_panel, stretch=1)
        
        # Right panel - Details and actions
        right_panel = QVBoxLayout()
        
        # Model info card
        info_group = QGroupBox("Model Details")
        info_layout = QVBoxLayout(info_group)
        
        self.info_name = QLabel("Select a model")
        self.info_name.setStyleSheet("font-size: 18px; font-weight: bold; color: #f9e2af;")
        info_layout.addWidget(self.info_name)
        
        self.info_details = QLabel("Click a model from the list to see its details")
        self.info_details.setWordWrap(True)
        self.info_details.setStyleSheet("color: #a6adc8; font-size: 12px;")
        info_layout.addWidget(self.info_details)
        
        right_panel.addWidget(info_group)
        
        # Actions grouped
        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        actions_layout.setSpacing(8)
        
        # Row 1 - Safe actions
        self.btn_backup = QPushButton("Backup")
        self.btn_backup.setStyleSheet("background-color: #74c7ec; color: #1e1e2e; font-weight: bold;")
        self.btn_backup.clicked.connect(self._on_backup)
        self.btn_backup.setEnabled(False)
        actions_layout.addWidget(self.btn_backup, 0, 0)
        
        self.btn_clone = QPushButton("Clone")
        self.btn_clone.setStyleSheet("background-color: #cba6f7; color: #1e1e2e; font-weight: bold;")
        self.btn_clone.clicked.connect(self._on_clone)
        self.btn_clone.setEnabled(False)
        actions_layout.addWidget(self.btn_clone, 0, 1)
        
        self.btn_test = QPushButton("Test")
        self.btn_test.setStyleSheet("background-color: #94e2d5; color: #1e1e2e; font-weight: bold;")
        self.btn_test.clicked.connect(self._on_test_model)
        self.btn_test.setEnabled(False)
        self.btn_test.setToolTip("Test model with sample prompts")
        actions_layout.addWidget(self.btn_test, 0, 2)
        
        self.btn_folder = QPushButton("Folder")
        self.btn_folder.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold;")
        self.btn_folder.clicked.connect(self._on_open_folder)
        self.btn_folder.setEnabled(False)
        actions_layout.addWidget(self.btn_folder, 0, 3)
        
        # Row 2 - Scaling
        self.btn_grow = QPushButton("Grow")
        self.btn_grow.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        self.btn_grow.clicked.connect(self._on_grow)
        self.btn_grow.setEnabled(False)
        actions_layout.addWidget(self.btn_grow, 1, 0)
        
        self.btn_shrink = QPushButton("Shrink")
        self.btn_shrink.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold;")
        self.btn_shrink.clicked.connect(self._on_shrink)
        self.btn_shrink.setEnabled(False)
        actions_layout.addWidget(self.btn_shrink, 1, 1)
        
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setStyleSheet("background-color: #94e2d5; color: #1e1e2e; font-weight: bold;")
        self.btn_rename.clicked.connect(self._on_rename)
        self.btn_rename.setEnabled(False)
        actions_layout.addWidget(self.btn_rename, 1, 2)
        
        # Row 3 - Delete buttons
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
        self.btn_delete.clicked.connect(lambda: self._on_delete_action(False))
        self.btn_delete.setEnabled(False)
        actions_layout.addWidget(self.btn_delete, 2, 0)
        
        self.btn_delete_backup = QPushButton("Delete (Keep Backup)")
        self.btn_delete_backup.setStyleSheet("background-color: #fab387; color: #1e1e2e; font-weight: bold;")
        self.btn_delete_backup.clicked.connect(lambda: self._on_delete_action(True))
        self.btn_delete_backup.setEnabled(False)
        actions_layout.addWidget(self.btn_delete_backup, 2, 1, 1, 2)  # Span 2 columns
        
        right_panel.addWidget(actions_group)
        
        # Close button at bottom
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        right_panel.addWidget(close_btn)
        
        layout.addLayout(right_panel, stretch=1)
    
    def _refresh_list(self):
        """Refresh the model list from disk."""
        try:
            self.registry._load_registry()
        except (IOError, OSError, json.JSONDecodeError):
            pass
        
        self.model_list.clear()
        self.selected_model = None
        self._update_buttons_state()
        self.info_name.setText("Select a model")
        self.info_details.setText("Click a model from the list to see its details")
        
        # Sync models to other tabs (like router)
        self._sync_models_everywhere()
        
        models = self.registry.registry.get("models", {})
        for name, info in sorted(models.items()):
            model_path = Path(self.registry.models_dir) / name
            if model_path.exists():
                has_weights = info.get("has_weights", False)
                size = info.get("size", "?")
                source = info.get("source", "forge_ai")
                
                # Different icons for different model sources
                if source == "huggingface":
                    icon = "[HF]"
                elif has_weights:
                    icon = "[OK]"
                else:
                    icon = "[--]"
                    
                self.model_list.addItem(f"{icon} {name} ({size})")
    
    def _update_buttons_state(self):
        """Enable/disable buttons based on selection."""
        has_selection = self.selected_model is not None
        self.btn_backup.setEnabled(has_selection)
        self.btn_clone.setEnabled(has_selection)
        self.btn_test.setEnabled(has_selection)
        self.btn_folder.setEnabled(has_selection)
        self.btn_rename.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)
        self.btn_delete_backup.setEnabled(has_selection)
        
        # Check if this is a HuggingFace model - they can't be resized
        is_huggingface = False
        if has_selection:
            model_info = self.registry.registry.get("models", {}).get(self.selected_model, {})
            is_huggingface = model_info.get("source") == "huggingface"
        
        # Disable grow/shrink for HuggingFace models (they have fixed architecture)
        can_scale = has_selection and not is_huggingface
        self.btn_grow.setEnabled(can_scale)
        self.btn_shrink.setEnabled(can_scale)
        
        # Update button tooltips to explain why they're disabled
        if is_huggingface:
            self.btn_grow.setToolTip("HuggingFace models cannot be resized")
            self.btn_shrink.setToolTip("HuggingFace models cannot be resized")
        else:
            self.btn_grow.setToolTip("Grow model to a larger size")
            self.btn_shrink.setToolTip("Shrink model to a smaller size")
    
    def _sync_models_everywhere(self):
        """Notify all components that model list has changed."""
        try:
            # Refresh router tab dropdowns
            if hasattr(self, 'router_tab') and self.router_tab:
                self.router_tab.refresh_models()
        except Exception:
            pass  # Don't crash if sync fails
    
    def _on_select_model(self, item):
        """Handle model selection."""
        text = item.text()
        # Parse "[OK] name (size)" or "[--] name (size)"
        parts = text.split(" ", 1)
        if len(parts) > 1:
            rest = parts[1]  # "name (size)"
            name = rest.rsplit(" (", 1)[0]
        else:
            name = text
        
        self.selected_model = name
        self._update_buttons_state()
        
        try:
            info = self.registry.get_model_info(name)
            meta = info.get("metadata", {})
            reg_info = info.get("registry", {})
            
            self.info_name.setText(f"{name}")
            
            created = str(meta.get('created', 'Unknown'))[:10]
            last_trained = meta.get('last_trained', 'Never')
            if last_trained and last_trained != 'Never':
                last_trained = str(last_trained)[:10]
            
            epochs = meta.get('total_epochs', 0)
            params = meta.get('estimated_parameters', 0)
            params_str = f"{params:,}" if params else "Unknown"
            checkpoints = len(info.get('checkpoints', []))
            size = reg_info.get('size', '?')
            source = reg_info.get('source', 'forge_ai')
            
            details = f"""
Source: {source.upper()}
Size: {size.upper()}
Parameters: {params_str}
Created: {created}
Last trained: {last_trained}
Total epochs: {epochs}
Checkpoints: {checkpoints}
            """.strip()
            
            # Add note for HuggingFace models
            if source == "huggingface":
                details += "\n\n[HuggingFace models cannot be resized]"
            
            self.info_details.setText(details)
        except Exception as e:
            self.info_details.setText(f"Error loading details:\n{e}")
    
    def _on_load_model(self, item=None):
        """Load the selected model."""
        if not self.selected_model:
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        # Store selected model and close dialog
        self.accept()
    
    def _on_add_hf_model(self):
        """Add a HuggingFace model to the registry and tool router."""
        # Get model ID from preset or input
        preset_text = self.hf_preset_combo.currentText()
        custom_text = self.hf_model_input.text().strip()
        
        model_id = None
        if custom_text:
            model_id = custom_text
        elif preset_text and not preset_text.startswith("Select"):
            # Parse preset: "gpt2 (124M) - Fast, classic" -> "gpt2"
            model_id = preset_text.split(" (")[0].split(" - ")[0].strip()
        
        if not model_id:
            QMessageBox.warning(self, "No Model", "Select a preset or enter a HuggingFace model ID")
            return
        
        # Clean up the model_id
        model_id = model_id.strip()
        
        # Create a local name for the model
        local_name = model_id.replace("/", "_").replace("-", "_").lower()
        
        # Check if already exists
        if local_name in self.registry.registry.get("models", {}):
            # Ask if they want to just assign it to chat
            reply = QMessageBox.question(
                self, "Model Exists",
                f"'{local_name}' already exists in registry.\n\n"
                "Do you want to set it as the active chat AI?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._assign_hf_to_chat(model_id, local_name)
            return
        
        # Add to registry
        try:
            model_path = Path(self.registry.models_dir) / local_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Get tokenizer preference
            use_custom_tokenizer = self.hf_tokenizer_combo.currentIndex() == 1
            
            # Fetch model info (size, params) from HuggingFace
            size_str = "huggingface"
            num_params = 0
            try:
                from ...core.huggingface_loader import get_huggingface_model_info
                info = get_huggingface_model_info(model_id)
                if not info.get("error"):
                    size_str = f"HF-{info['size_str']}"  # e.g., "HF-124M"
                    num_params = info.get("num_parameters", 0)
            except Exception as e:
                print(f"Could not fetch HF model info: {e}")
            
            # Create registry entry
            self.registry.registry.setdefault("models", {})[local_name] = {
                "path": str(model_path),
                "size": size_str,
                "created": datetime.now().isoformat(),
                "has_weights": False,  # Weights are in HF cache, not local
                "source": "huggingface",
                "huggingface_id": model_id,
                "use_custom_tokenizer": use_custom_tokenizer,  # User preference
                "num_parameters": num_params,  # Store actual param count
            }
            self.registry._save_registry()
            
            # Assign to chat tool router
            self._assign_hf_to_chat(model_id, local_name)
            
            self._refresh_list()
            self.hf_model_input.clear()
            self.hf_preset_combo.setCurrentIndex(0)
            
            QMessageBox.information(
                self, "Model Added",
                f"Added HuggingFace model: {model_id}\n\n"
                f"Local name: {local_name}\n"
                f"Estimated size: {size_str}\n\n"
                "It has been set as the active chat AI.\n"
                "The model will download when first used."
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add model: {e}")
    
    def _assign_hf_to_chat(self, model_id: str, local_name: str):
        """Assign a HuggingFace model to the chat tool router."""
        try:
            from ...core.tool_router import get_router
            router = get_router()
            
            # Format as huggingface:model_id
            full_id = f"huggingface:{model_id}"
            
            # Assign with high priority (assign_model already saves)
            router.assign_model("chat", full_id, priority=100)
        except Exception as e:
            print(f"Could not assign to router: {e}")
    
    def _on_delete_hf_model(self):
        """Delete a HuggingFace model from registry and optionally clear cache."""
        # Get all HF models from registry
        hf_models = []
        for name, info in self.registry.registry.get("models", {}).items():
            if info.get("source") == "huggingface":
                hf_id = info.get("huggingface_id", name)
                size = info.get("size", "unknown")
                hf_models.append((name, hf_id, size))
        
        if not hf_models:
            QMessageBox.information(self, "No HF Models", "No HuggingFace models found in registry.")
            return
        
        # Create selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Delete HuggingFace Model")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Select HuggingFace model to delete:")
        label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        layout.addWidget(label)
        
        list_widget = QListWidget()
        list_widget.setStyleSheet("""
            QListWidget { 
                background-color: #313244; 
                color: #cdd6f4; 
                border: 1px solid #45475a;
            }
            QListWidget::item:selected { background-color: #f38ba8; color: #1e1e2e; }
        """)
        for name, hf_id, size in hf_models:
            list_widget.addItem(f"{name} ({hf_id}) [{size}]")
        layout.addWidget(list_widget)
        
        # Option to clear HF cache
        cache_checkbox = QCheckBox("Also clear HuggingFace cache for this model")
        cache_checkbox.setToolTip("This will delete the downloaded model files from HuggingFace cache.\n"
                                   "Re-downloading will be needed if you add this model again.")
        cache_checkbox.setStyleSheet("color: #fab387;")
        layout.addWidget(cache_checkbox)
        
        # Warning
        warning = QLabel("WARNING: This cannot be undone!")
        warning.setStyleSheet("color: #f38ba8; font-weight: bold;")
        layout.addWidget(warning)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        selected = list_widget.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
        
        model_name, hf_id, _ = hf_models[selected]
        clear_cache = cache_checkbox.isChecked()
        
        # Confirm
        reply = QMessageBox.warning(
            self, "Confirm Delete",
            f"Delete HuggingFace model?\n\n"
            f"Registry name: {model_name}\n"
            f"HuggingFace ID: {hf_id}\n"
            f"Clear cache: {'Yes' if clear_cache else 'No'}\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Remove from registry
            if model_name in self.registry.registry.get("models", {}):
                del self.registry.registry["models"][model_name]
                self.registry._save_registry()
            
            # Delete local directory if exists
            model_path = Path(self.registry.models_dir) / model_name
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Clear HuggingFace cache if requested
            cache_msg = ""
            if clear_cache:
                try:
                    from huggingface_hub import scan_cache_dir
                    import os
                    
                    # Get HF cache directory
                    cache_dir = os.environ.get("HF_HOME", 
                                 os.environ.get("HUGGINGFACE_HUB_CACHE",
                                 os.path.expanduser("~/.cache/huggingface/hub")))
                    
                    if os.path.exists(cache_dir):
                        # Look for model in cache
                        cache_info = scan_cache_dir(cache_dir)
                        for repo in cache_info.repos:
                            if repo.repo_id == hf_id:
                                # Delete all revisions of this model
                                delete_strategy = cache_info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
                                delete_strategy.execute()
                                cache_msg = "\n\nHuggingFace cache cleared."
                                break
                        else:
                            cache_msg = "\n\nModel not found in HF cache (may not have been downloaded)."
                except Exception as e:
                    cache_msg = f"\n\nCould not clear HF cache: {e}"
            
            # Clean up tool routing
            self._cleanup_tool_routing(model_name)
            self._cleanup_tool_routing(f"huggingface:{hf_id}")
            
            self._refresh_list()
            
            QMessageBox.information(
                self, "Deleted",
                f"HuggingFace model '{model_name}' has been deleted from registry.{cache_msg}"
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete model: {e}")
    
    def _on_new_model(self):
        """Create a new model via wizard."""
        # Import SetupWizard from enhanced_window (it's still there for now)
        # This will be moved to a separate file later
        try:
            from ..enhanced_window import SetupWizard
        except ImportError:
            QMessageBox.warning(self, "Error", "SetupWizard not available")
            return
        
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(
                    result["name"],
                    size=result["size"],
                    vocab_size=32000
                )
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Created model '{result['name']}'")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_backup(self):
        """Backup the selected model to a zip file."""
        if not self.selected_model:
            return
        
        name = self.selected_model
        model_dir = Path(self.registry.models_dir) / name
        
        # Create backup as a zip file in a backups folder (not as another model)
        backups_dir = Path(self.registry.models_dir) / "_backups"
        backups_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{name}_backup_{timestamp}"
        backup_zip = backups_dir / f"{backup_name}.zip"
        
        try:
            import zipfile
            with zipfile.ZipFile(backup_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_dir)
                        zf.write(file_path, arcname)
            
            QMessageBox.information(
                self, "Backup Complete", 
                f"Backup saved to:\n{backup_zip}\n\n"
                f"To restore, unzip to the models folder."
            )
        except Exception as e:
            QMessageBox.warning(self, "Backup Failed", str(e))
    
    def _on_test_model(self):
        """Test the selected model with sample prompts to verify it works."""
        if not self.selected_model:
            return
        
        name = self.selected_model
        
        # Create test dialog
        test_dialog = QDialog(self)
        test_dialog.setWindowTitle(f"Testing: {name}")
        test_dialog.setMinimumSize(500, 400)
        test_dialog.setStyleSheet("""
            QDialog { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QTextEdit { 
                background-color: #313244; 
                color: #cdd6f4; 
                border: 1px solid #45475a;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        
        layout = QVBoxLayout(test_dialog)
        
        title = QLabel(f"Model Test: {name}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(title)
        
        # Test results area
        results = QTextEdit()
        results.setReadOnly(True)
        layout.addWidget(results)
        
        # Buttons
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(test_dialog.close)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        test_dialog.show()
        QApplication.processEvents()
        
        # Run the test
        results.append("<b>Loading model...</b>")
        QApplication.processEvents()
        
        try:
            # Load the model
            model, config = self.registry.load_model(name)
            results.append(f"<span style='color: #a6e3a1;'>✓ Model loaded successfully</span>")
            results.append(f"<span style='color: #6c7086;'>  Source: {config.get('source', 'local')}</span>")
            QApplication.processEvents()
            
            # Get model info
            is_huggingface = config.get("source") == "huggingface"
            
            if is_huggingface:
                results.append(f"<span style='color: #6c7086;'>  HuggingFace ID: {config.get('huggingface_id', 'unknown')}</span>")
            else:
                results.append(f"<span style='color: #6c7086;'>  Size: {config.get('size', 'unknown')}</span>")
            
            # Test prompts
            test_prompts = [
                "Hello",
                "What is 2 + 2?",
                "How are you?",
            ]
            
            results.append("\n<b>Running test prompts...</b>")
            QApplication.processEvents()
            
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            passed = 0
            failed = 0
            
            for prompt in test_prompts:
                results.append(f"\n<b>Prompt:</b> {prompt}")
                QApplication.processEvents()
                
                try:
                    if is_huggingface:
                        # HuggingFace model
                        response = model.generate(prompt, max_length=50)
                    else:
                        # Local Forge model
                        model.to(device)
                        model.eval()
                        from ...core.tokenizer import load_tokenizer
                        tokenizer = load_tokenizer()
                        
                        tokens = tokenizer.encode(prompt)
                        input_ids = torch.tensor([tokens], device=device)
                        
                        with torch.no_grad():
                            for _ in range(30):
                                logits = model(input_ids)
                                next_token = logits[0, -1, :].argmax().item()
                                input_ids = torch.cat([
                                    input_ids,
                                    torch.tensor([[next_token]], device=device)
                                ], dim=1)
                                if next_token == tokenizer.eos_token_id:
                                    break
                        
                        response = tokenizer.decode(input_ids[0].tolist())
                    
                    # Check response quality
                    response_clean = response.replace(prompt, "").strip()[:100]
                    
                    if len(response_clean) < 2:
                        results.append(f"<span style='color: #f9e2af;'>[!] Response too short: '{response_clean}'</span>")
                        failed += 1
                    elif response_clean.count(response_clean[0]) == len(response_clean):
                        results.append(f"<span style='color: #f38ba8;'>[X] Repetitive output: '{response_clean}'</span>")
                        failed += 1
                    else:
                        results.append(f"<span style='color: #a6e3a1;'>[OK] Response: {response_clean}</span>")
                        passed += 1
                        
                except Exception as e:
                    results.append(f"<span style='color: #f38ba8;'>[X] Error: {e}</span>")
                    failed += 1
                
                QApplication.processEvents()
            
            # Summary
            results.append("\n<b>Test Summary:</b>")
            if failed == 0:
                results.append(f"<span style='color: #a6e3a1;'>All {passed} tests passed! Model looks good.</span>")
            elif passed > failed:
                results.append(f"<span style='color: #f9e2af;'>{passed} passed, {failed} failed. Model may need training.</span>")
            else:
                results.append(f"<span style='color: #f38ba8;'>{passed} passed, {failed} failed. Model needs training or has issues.</span>")
                results.append("<span style='color: #6c7086;'>Tip: Try training with more data or check if weights are corrupted.</span>")
                
        except Exception as e:
            results.append(f"<span style='color: #f38ba8;'>✗ Failed to load model: {e}</span>")
            results.append("<span style='color: #6c7086;'>Check that model files exist and are not corrupted.</span>")
    
    def _on_clone(self):
        """Clone the selected model."""
        if not self.selected_model:
            return
        
        original_name = self.selected_model  # Store before any changes
        
        new_name, ok = QInputDialog.getText(
            self, "Clone Model",
            f"Enter name for clone of '{original_name}':",
            text=f"{original_name}_clone"
        )
        
        if not ok or not new_name.strip():
            return
        
        new_name = new_name.strip().replace(' ', '_').lower()
        
        if new_name in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Name Exists", f"'{new_name}' already exists")
            return
        
        try:
            src = Path(self.registry.models_dir) / original_name
            dst = Path(self.registry.models_dir) / new_name
            shutil.copytree(src, dst)
            
            # Create NEW registry entry with CORRECT path (not copy of old)
            old_info = self.registry.registry["models"][original_name]
            new_info = {
                "path": str(dst),  # NEW path for the clone!
                "size": old_info.get("size", "tiny"),
                "created": datetime.now().isoformat(),
                "has_weights": old_info.get("has_weights", False),
                "data_dir": str(dst / "data"),  # NEW data dir!
                "cloned_from": original_name
            }
            self.registry.registry["models"][new_name] = new_info
            self.registry._save_registry()
            
            self._refresh_list()
            
            # Auto-select the new clone so user can see it's selected
            self.selected_model = new_name
            self._update_buttons_state()
            self.info_name.setText(f"{new_name}")
            self.info_details.setText(f"Clone of: {original_name}\n\nClick to select a different model.")
            
            # Highlight the clone in the list
            for i in range(self.model_list.count()):
                item = self.model_list.item(i)
                if new_name in item.text():
                    self.model_list.setCurrentItem(item)
                    break
            
            QMessageBox.information(self, "Cloned", f"Created clone: {new_name}\n\nThe clone is now selected.")
        except Exception as e:
            QMessageBox.warning(self, "Clone Failed", str(e))
    
    def _on_open_folder(self):
        """Open model folder in file explorer."""
        if not self.selected_model:
            return
        
        from ...config import CONFIG
        folder = Path(CONFIG['models_dir']) / self.selected_model
        
        if not folder.exists():
            QMessageBox.warning(self, "Not Found", "Model folder not found")
            return
        
        from ..tabs.output_helpers import open_folder
        
        try:
            open_folder(folder)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
    
    def _on_grow(self):
        """Grow the model to a larger size."""
        if not self.selected_model:
            return
        
        current_size = self.registry.registry["models"].get(self.selected_model, {}).get("size", "tiny")
        sizes = ["tiny", "small", "medium", "large", "xl"]
        
        try:
            idx = sizes.index(current_size)
            available = sizes[idx + 1:]
        except ValueError:
            available = sizes
        
        if not available:
            QMessageBox.information(self, "Max Size", "Already at maximum size")
            return
        
        size, ok = self._size_dialog("Grow Model", available, f"Current: {current_size}")
        if not ok or not size:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Grow",
            f"Grow '{self.selected_model}' from {current_size} to {size}?\n\nA backup will be created first."
        )
        
        if reply == QMessageBox.Yes:
            self._on_backup()  # Auto backup
            try:
                from ...core.model_scaling import grow_registered_model
                grow_registered_model(self.registry, self.selected_model, self.selected_model, size)
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Model grown to {size}!")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_shrink(self):
        """Shrink the model to a smaller size."""
        if not self.selected_model:
            return
        
        current_size = self.registry.registry["models"].get(self.selected_model, {}).get("size", "large")
        sizes = ["nano", "micro", "tiny", "small", "medium", "large"]
        
        try:
            idx = sizes.index(current_size)
            available = sizes[:idx]
        except ValueError:
            available = sizes[:-1]
        
        if not available:
            QMessageBox.information(self, "Min Size", "Already at minimum size")
            return
        
        size, ok = self._size_dialog("Shrink Model", available, f"Current: {current_size}\nWarning: May lose capacity!")
        if not ok or not size:
            return
        
        reply = QMessageBox.warning(
            self, "Confirm Shrink",
            f"Shrink '{self.selected_model}' to {size}?\n\nWarning: This may reduce model quality.\nA backup will be created first.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._on_backup()
            try:
                model, config = self.registry.load_model(self.selected_model)
                shrunk = shrink_model(model, size, config["vocab_size"])
                self.registry.save_model(self.selected_model, shrunk)
                self.registry.registry["models"][self.selected_model]["size"] = size
                self.registry._save_registry()
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Model shrunk to {size}!")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_rename(self):
        """Rename the selected model."""
        if not self.selected_model:
            return
        
        new_name, ok = QInputDialog.getText(
            self, "Rename Model",
            f"New name for '{self.selected_model}':",
            text=self.selected_model
        )
        
        if not ok or not new_name.strip() or new_name == self.selected_model:
            return
        
        new_name = new_name.strip().replace(' ', '_')
        
        if new_name in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Name Exists", f"'{new_name}' already exists")
            return
        
        try:
            from ...config import CONFIG
            old = Path(CONFIG['models_dir']) / self.selected_model
            new = Path(CONFIG['models_dir']) / new_name
            old.rename(new)
            
            info = self.registry.registry["models"].pop(self.selected_model)
            # Update the path to reflect new name
            info["path"] = str(new)
            self.registry.registry["models"][new_name] = info
            self.registry._save_registry()
            
            self.selected_model = new_name
            self._refresh_list()
            QMessageBox.information(self, "Renamed", f"Model renamed to '{new_name}'")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _on_delete_action(self, create_backup: bool):
        """Handle delete button click."""
        if not self.selected_model:
            QMessageBox.warning(self, "No Selection", "Please select a model first")
            return
        
        model_to_delete = self.selected_model
        
        # Check for existing backups
        models_dir = Path("models")
        backup_found = []
        for backup_dir in models_dir.glob(f"{model_to_delete}_backup*"):
            if backup_dir.is_dir():
                backup_found.append(backup_dir.name)
        
        # Build confirmation message
        if create_backup:
            action_msg = "DELETE WITH BACKUP"
            extra_info = "\n\nA backup will be created before deletion."
        else:
            action_msg = "PERMANENTLY DELETE"
            extra_info = ""
        
        backup_msg = ""
        if backup_found:
            backup_msg = f"\n\nExisting backups:\n" + "\n".join(f"  - {b}" for b in backup_found)
        
        # Single confirmation - no typing required
        reply = QMessageBox.warning(
            self, f"{action_msg}",
            f"Are you sure you want to delete:\n\n"
            f"   {model_to_delete}\n{extra_info}{backup_msg}",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Create backup if requested
            if create_backup:
                self._on_backup()
            
            # Actually delete
            self.registry.delete_model(model_to_delete, confirm=True)
            self.selected_model = None
            self._refresh_list()
            
            # Clean up tool routing
            self._cleanup_tool_routing(model_to_delete)
            
            msg = f"Model '{model_to_delete}' has been deleted."
            if create_backup:
                msg += "\n\nA backup was created in models/_backups/"
            QMessageBox.information(self, "Deleted", msg)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _cleanup_tool_routing(self, deleted_model: str):
        """Remove deleted model from tool_routing.json assignments."""
        try:
            routing_path = Path("information/tool_routing.json")
            if not routing_path.exists():
                return
            
            with open(routing_path, "r") as f:
                routing = json.load(f)
            
            changed = False
            for tool_name, assignments in routing.items():
                if isinstance(assignments, list):
                    # Remove any assignment using this model
                    new_assignments = [
                        a for a in assignments 
                        if not (a.get("model", "").lower() == deleted_model.lower() or
                                deleted_model.lower() in a.get("model", "").lower())
                    ]
                    if len(new_assignments) != len(assignments):
                        routing[tool_name] = new_assignments
                        changed = True
            
            if changed:
                with open(routing_path, "w") as f:
                    json.dump(routing, f, indent=2)
        except Exception:
            pass  # Non-critical, don't block deletion
    
    def _size_dialog(self, title, sizes, message):
        """Show size selection dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setStyleSheet(self.styleSheet())
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(message))
        
        combo = QComboBox()
        combo.addItems(sizes)
        combo.setStyleSheet("padding: 8px; background: #313244; color: #cdd6f4; border-radius: 4px;")
        layout.addWidget(combo)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)
        
        if dialog.exec_() == QDialog.Accepted:
            return combo.currentText(), True
        return None, False
    
    def get_selected_model(self):
        return self.selected_model
    
    def closeEvent(self, event):
        """Handle close - just close, don't block."""
        event.accept()
