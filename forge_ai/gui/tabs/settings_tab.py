"""
Settings Tab - Resource management and application settings.

Allows users to control CPU/RAM usage so the AI doesn't hog resources
while gaming or doing other tasks.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QSlider, QCheckBox,
    QTextEdit, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt

# Qt enum constant for checkbox state
Checked = Qt.CheckState.Checked


def _get_env_key(key_name: str) -> str:
    """Get an environment variable value, return empty string if not set."""
    return os.environ.get(key_name, "")


def _save_api_keys(parent):
    """Save API keys to environment variables and .env file."""
    keys_to_save = {
        "HF_TOKEN": parent.hf_token_input.text().strip(),
        "OPENAI_API_KEY": parent.openai_key_input.text().strip(),
        "REPLICATE_API_TOKEN": parent.replicate_key_input.text().strip(),
        "ELEVENLABS_API_KEY": parent.elevenlabs_key_input.text().strip(),
    }
    
    # Set environment variables for current session
    saved_count = 0
    for key, value in keys_to_save.items():
        if value:
            os.environ[key] = value
            saved_count += 1
    
    # Try to save to .env file for persistence
    try:
        from ...config import CONFIG
        from pathlib import Path
        
        env_file = Path(CONFIG.get("project_root", ".")) / ".env"
        
        # Read existing .env content
        existing_lines = []
        if env_file.exists():
            existing_lines = env_file.read_text().splitlines()
        
        # Update or add keys
        new_lines = []
        keys_found = set()
        for line in existing_lines:
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=")[0].strip()
                if key in keys_to_save:
                    if keys_to_save[key]:  # Only write if value is not empty
                        new_lines.append(f"{key}={keys_to_save[key]}")
                    keys_found.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Add new keys that weren't in file
        for key, value in keys_to_save.items():
            if key not in keys_found and value:
                new_lines.append(f"{key}={value}")
        
        # Write back
        env_file.write_text("\n".join(new_lines) + "\n")
        
        parent.api_status_label.setText(f"Saved {saved_count} key(s) to .env file")
        parent.api_status_label.setStyleSheet("color: #22c55e; font-style: italic;")
    except Exception as e:
        # Still saved to environment, just not persisted
        parent.api_status_label.setText(f"Keys set for this session (couldn't save .env: {e})")
        parent.api_status_label.setStyleSheet("color: #f59e0b; font-style: italic;")


def _toggle_key_visibility(parent):
    """Toggle between showing and hiding API keys."""
    from PyQt5.QtWidgets import QLineEdit
    
    inputs = [
        parent.hf_token_input,
        parent.openai_key_input,
        parent.replicate_key_input,
        parent.elevenlabs_key_input,
    ]
    
    # Check current mode of first input
    if parent.hf_token_input.echoMode() == QLineEdit.Password:
        for inp in inputs:
            inp.setEchoMode(QLineEdit.Normal)
    else:
        for inp in inputs:
            inp.setEchoMode(QLineEdit.Password)


# ===== AVATAR AUTONOMOUS CONTROL =====
def _toggle_avatar_autonomous(parent, state):
    """Toggle avatar autonomous mode."""
    enabled = state == 2  # Qt.Checked
    
    try:
        from forge_ai.avatar import get_avatar
        from forge_ai.avatar.autonomous import get_autonomous_avatar
        
        avatar = get_avatar()
        autonomous = get_autonomous_avatar(avatar)
        
        if enabled:
            avatar.enable()
            autonomous.start()
            parent.avatar_status_label.setText("Avatar: Autonomous (watching screen)")
            parent.avatar_status_label.setStyleSheet("color: #22c55e;")
        else:
            autonomous.stop()
            parent.avatar_status_label.setText("Avatar: Manual mode")
            parent.avatar_status_label.setStyleSheet("color: #888;")
    except Exception as e:
        parent.avatar_status_label.setText(f"Avatar: Error - {e}")
        parent.avatar_status_label.setStyleSheet("color: #ef4444;")


# ===== ROBOT MODE CONTROL =====
def _change_robot_mode(parent):
    """Change robot control mode."""
    mode = parent.robot_mode_combo.currentData()
    
    try:
        from forge_ai.tools.robot_modes import get_mode_controller, RobotMode
        
        controller = get_mode_controller()
        
        mode_map = {
            "disabled": RobotMode.DISABLED,
            "manual": RobotMode.MANUAL,
            "auto": RobotMode.AUTO,
            "safe": RobotMode.SAFE,
        }
        
        robot_mode = mode_map.get(mode, RobotMode.DISABLED)
        success = controller.set_mode(robot_mode)
        
        if success:
            status_map = {
                "disabled": ("Robot: Disabled", "#888"),
                "manual": ("Robot: MANUAL - User control", "#22c55e"),
                "auto": ("Robot: AUTO - AI control enabled", "#3b82f6"),
                "safe": ("Robot: SAFE - Limited AI control", "#f59e0b"),
            }
            text, color = status_map.get(mode, ("Robot: Unknown", "#888"))
            parent.robot_status_label.setText(text)
            parent.robot_status_label.setStyleSheet(f"color: {color};")
        else:
            parent.robot_status_label.setText(f"Robot: Failed to change mode")
            parent.robot_status_label.setStyleSheet("color: #ef4444;")
    except ImportError:
        parent.robot_status_label.setText("Robot: Not configured")
        parent.robot_status_label.setStyleSheet("color: #888;")
    except Exception as e:
        parent.robot_status_label.setText(f"Robot: Error - {e}")
        parent.robot_status_label.setStyleSheet("color: #ef4444;")


def _robot_estop(parent):
    """Emergency stop the robot."""
    try:
        from forge_ai.tools.robot_modes import get_mode_controller
        
        controller = get_mode_controller()
        controller.emergency_stop("User pressed E-STOP button")
        
        parent.robot_status_label.setText("Robot: E-STOP ACTIVE")
        parent.robot_status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        parent.robot_mode_combo.setCurrentIndex(0)  # Set to disabled
        
        QMessageBox.critical(
            parent, "Emergency Stop",
            "Robot has been emergency stopped!\n\n"
            "To resume, set robot to DISABLED mode first, then re-enable."
        )
    except Exception as e:
        QMessageBox.warning(parent, "E-STOP Error", f"Could not E-STOP: {e}")


def _toggle_robot_camera(parent, state):
    """Toggle robot camera feed."""
    enabled = state == 2
    
    try:
        from forge_ai.tools.robot_modes import get_mode_controller, CameraConfig
        
        controller = get_mode_controller()
        
        if enabled:
            controller.setup_camera(CameraConfig(enabled=True, device_id=0))
            controller.start_camera()
            parent.robot_status_label.setText(
                parent.robot_status_label.text() + " (Camera ON)"
            )
        else:
            controller.stop_camera()
    except ImportError:
        parent.robot_camera_check.setChecked(False)
        QMessageBox.information(
            parent, "OpenCV Required",
            "Camera requires OpenCV. Install with:\npip install opencv-python"
        )
    except Exception as e:
        parent.robot_camera_check.setChecked(False)
        QMessageBox.warning(parent, "Camera Error", f"Could not enable camera: {e}")


# ===== GAME AI ROUTING =====
def _toggle_game_detection(parent, state):
    """Toggle automatic game detection."""
    enabled = state == 2
    
    try:
        from forge_ai.tools.game_router import get_game_router
        
        router = get_game_router()
        
        if enabled:
            router.start_detection(interval=5.0)
            router.on_game_detected(lambda game: _on_game_detected(parent, game))
            if hasattr(parent, 'game_combo'):
                parent.game_combo.setEnabled(False)
            if hasattr(parent, 'game_status_label'):
                parent.game_status_label.setText("Watching for games...")
                parent.game_status_label.setStyleSheet("color: #3b82f6;")
        else:
            router.stop_detection()
            if hasattr(parent, 'game_combo'):
                parent.game_combo.setEnabled(True)
            if hasattr(parent, 'game_status_label'):
                parent.game_status_label.setText("Auto-detection disabled")
                parent.game_status_label.setStyleSheet("color: #888;")
    except ImportError:
        if hasattr(parent, 'auto_game_check'):
            parent.auto_game_check.setChecked(False)
        QMessageBox.information(
            parent, "psutil Required",
            "Game detection requires psutil. Install with:\npip install psutil"
        )
    except Exception as e:
        if hasattr(parent, 'auto_game_check'):
            parent.auto_game_check.setChecked(False)
        if hasattr(parent, 'game_status_label'):
            parent.game_status_label.setText(f"Detection error: {e}")
            parent.game_status_label.setStyleSheet("color: #ef4444;")


def _on_game_detected(parent, game_id: str):
    """Called when a game is auto-detected."""
    try:
        from forge_ai.tools.game_router import get_game_router
        router = get_game_router()
        config = router.get_game(game_id)
        
        if config:
            if hasattr(parent, 'game_status_label'):
                parent.game_status_label.setText(f"Detected: {config.name}")
                parent.game_status_label.setStyleSheet("color: #22c55e; font-weight: bold;")
            
            # Update combo without triggering change
            if hasattr(parent, 'game_combo'):
                parent.game_combo.blockSignals(True)
                for i in range(parent.game_combo.count()):
                    if parent.game_combo.itemData(i) == game_id:
                        parent.game_combo.setCurrentIndex(i)
                        break
                parent.game_combo.blockSignals(False)
    except Exception:
        pass


def _change_active_game(parent):
    """Manually change active game."""
    if not hasattr(parent, 'game_combo'):
        return
    
    game_id = parent.game_combo.currentData()
    
    if game_id == "custom":
        # TODO: Show custom game dialog
        QMessageBox.information(
            parent, "Custom Game",
            "Custom game configuration coming soon!\n"
            "For now, add games in forge_ai/tools/game_router.py"
        )
        parent.game_combo.setCurrentIndex(0)
        return
    
    try:
        from forge_ai.tools.game_router import get_game_router
        router = get_game_router()
        
        if game_id == "none":
            router.set_active_game(None)
            if hasattr(parent, 'game_status_label'):
                parent.game_status_label.setText("No game active - using default AI")
                parent.game_status_label.setStyleSheet("color: #888;")
        else:
            router.set_active_game(game_id)
            config = router.get_game(game_id)
            if config and hasattr(parent, 'game_status_label'):
                parent.game_status_label.setText(f"Active: {config.name}")
                parent.game_status_label.setStyleSheet("color: #22c55e;")
    except Exception as e:
        if hasattr(parent, 'game_status_label'):
            parent.game_status_label.setText(f"Error: {e}")
            parent.game_status_label.setStyleSheet("color: #ef4444;")


def _toggle_ai_lock(parent, state):
    """Toggle AI control lock - prevents user from changing settings."""
    is_locked = state == Checked
    
    # If trying to unlock and PIN is set, verify it
    if not is_locked and hasattr(parent, '_ai_lock_pin_set') and parent._ai_lock_pin_set:
        from PyQt5.QtWidgets import QInputDialog
        pin, ok = QInputDialog.getText(
            parent, "Unlock", "Enter PIN to unlock:",
            QLineEdit.Password
        )
        if not ok or pin != parent._ai_lock_pin_set:
            parent.ai_lock_checkbox.setChecked(True)  # Keep locked
            QMessageBox.warning(parent, "Incorrect PIN", "The PIN you entered is incorrect.")
            return
    
    # Save PIN if locking and PIN is set
    if is_locked:
        pin = parent.ai_lock_pin.text().strip()
        if pin:
            parent._ai_lock_pin_set = pin
            parent.ai_lock_pin.clear()
            parent.ai_lock_pin.setPlaceholderText("PIN set")
    
    # Store lock state
    parent._ai_control_locked = is_locked
    
    # Update status
    if is_locked:
        parent.ai_lock_status.setText("LOCKED - Only AI can change settings")
        parent.ai_lock_status.setStyleSheet("color: #ef4444; font-weight: bold;")
    else:
        parent.ai_lock_status.setText("Unlocked")
        parent.ai_lock_status.setStyleSheet("color: #22c55e;")
    
    # Get list of controls to lock/unlock
    lockable_widgets = _get_lockable_widgets(parent)
    
    for widget in lockable_widgets:
        widget.setEnabled(not is_locked)
    
    # Always keep the lock checkbox and PIN field enabled
    parent.ai_lock_checkbox.setEnabled(True)
    parent.ai_lock_pin.setEnabled(not is_locked)


def _get_lockable_widgets(parent):
    """Get list of widgets that should be locked when AI control is enabled."""
    widgets = []
    
    # Settings widgets
    lockable_attrs = [
        # Power mode
        'resource_mode_combo',
        # Theme
        'theme_combo',
        # Autonomous mode
        'autonomous_enabled_check',
        'autonomous_activity_spin',
        # API Keys
        'hf_token_input',
        'openai_key_input', 
        'replicate_key_input',
        'elevenlabs_key_input',
        # Training controls
        'epochs_spin',
        'batch_spin',
        'lr_spin',
        'train_file_combo',
        'btn_train',
        # Chat controls
        'chat_input',
        'send_btn',
        # Personality sliders
        'curiosity_slider',
        'friendliness_slider',
        'creativity_slider',
        'formality_slider',
        'humor_slider',
    ]
    
    for attr in lockable_attrs:
        if hasattr(parent, attr):
            widgets.append(getattr(parent, attr))
    
    return widgets


def _populate_monitors(parent, preserve_selection=False):
    """Populate the monitor dropdown with available displays."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QGuiApplication
    
    # Remember current selection by screen identifier (more robust than index)
    previous_screen_name = None
    previous_index = -1
    if preserve_selection and parent.monitor_combo.count() > 0:
        previous_index = parent.monitor_combo.currentIndex()
        # Also try to remember by screen name in case order changes
        screens_before = QGuiApplication.screens()
        if 0 <= previous_index < len(screens_before):
            previous_screen_name = screens_before[previous_index].name()
    
    parent.monitor_combo.blockSignals(True)  # Prevent triggering move while populating
    parent.monitor_combo.clear()
    screens = QGuiApplication.screens()
    
    for i, screen in enumerate(screens):
        geo = screen.geometry()
        name = screen.name() or f"Display {i + 1}"
        # Shorten long display names
        if len(name) > 20:
            name = name[:17] + "..."
        parent.monitor_combo.addItem(
            f"{i + 1}: {name} ({geo.width()}x{geo.height()})",
            i
        )
    
    # Restore previous selection - try by screen name first, then by index
    restored = False
    if preserve_selection:
        # First try to match by screen name (handles reordering)
        if previous_screen_name:
            for i, screen in enumerate(screens):
                if screen.name() == previous_screen_name:
                    parent.monitor_combo.setCurrentIndex(i)
                    restored = True
                    break
        # Fall back to index if name match failed
        if not restored and 0 <= previous_index < len(screens):
            parent.monitor_combo.setCurrentIndex(previous_index)
            restored = True
    
    if not restored:
        # Select current monitor based on where window actually is
        main_window = parent.window()
        if main_window:
            current_screen = QGuiApplication.screenAt(main_window.geometry().center())
            if current_screen:
                try:
                    idx = screens.index(current_screen)
                    parent.monitor_combo.setCurrentIndex(idx)
                except ValueError:
                    pass
    
    parent.monitor_combo.blockSignals(False)


def _move_to_monitor(parent, monitor_index):
    """Save the selected monitor preference (does NOT move the window)."""
    from PyQt5.QtGui import QGuiApplication
    
    screens = QGuiApplication.screens()
    if monitor_index < 0 or monitor_index >= len(screens):
        return
    
    # Just save the preference - don't move the window
    # The window will open on this monitor next time ForgeAI starts
    main_window = parent.window()
    if main_window and hasattr(main_window, '_gui_settings'):
        main_window._gui_settings["monitor_index"] = monitor_index
        # Save immediately so it persists
        try:
            from pathlib import Path
            import json
            settings_path = Path(__file__).parent.parent.parent.parent / "data" / "gui_settings.json"
            with open(settings_path, 'w') as f:
                json.dump(main_window._gui_settings, f, indent=2)
        except Exception:
            pass
    
    _update_display_info(parent)


def _toggle_cloud_mode(parent, state):
    """Toggle cloud AI mode - routes chat to cloud APIs."""
    enabled = state == Checked
    
    # Enable/disable provider and model dropdowns
    parent.cloud_provider_combo.setEnabled(enabled)
    parent.cloud_model_combo.setEnabled(enabled)
    
    if enabled:
        provider = parent.cloud_provider_combo.currentData()
        
        # Ollama doesn't need an API key - it's FREE and local!
        if provider == 'ollama':
            api_key = None  # Not needed
        else:
            # Check if API key is set for paid providers
            key_map = {
                'openai': 'OPENAI_API_KEY',
                'anthropic': 'ANTHROPIC_API_KEY', 
                'google': 'GOOGLE_API_KEY'
            }
            api_key = os.environ.get(key_map.get(provider, ''), '')
            
            if not api_key:
                parent.cloud_status_label.setText(f"Set {provider.upper()} API key below first!")
                parent.cloud_status_label.setStyleSheet("color: #f59e0b;")
                return
        
        # Try to load the chat_api module
        try:
            main_window = parent.window()
            if hasattr(main_window, 'module_manager'):
                manager = main_window.module_manager
                
                # Unload local inference if loaded
                if manager.is_loaded('inference'):
                    manager.unload('inference')
                
                # Configure and load cloud chat
                config = {
                    'provider': provider,
                    'model': parent.cloud_model_combo.currentData(),
                }
                if api_key:
                    config['api_key'] = api_key
                    
                manager.set_config('chat_api', config)
                
                success, msg = manager.load('chat_api')
                if success:
                    model_name = parent.cloud_model_combo.currentText()
                    if provider == 'ollama':
                        parent.cloud_status_label.setText(f"Ollama active: {model_name}")
                    else:
                        parent.cloud_status_label.setText(f"Cloud AI active: {model_name}")
                    parent.cloud_status_label.setStyleSheet("color: #22c55e;")
                else:
                    if provider == 'ollama':
                        parent.cloud_status_label.setText(
                            f"[X] Ollama not running. Install from https://ollama.ai\n"
                            f"Then run: ollama run {parent.cloud_model_combo.currentData()}"
                        )
                    else:
                        parent.cloud_status_label.setText(f"[X] Failed: {msg}")
                    parent.cloud_status_label.setStyleSheet("color: #ef4444;")
            else:
                parent.cloud_status_label.setText("API mode enabled (apply on restart)")
                parent.cloud_status_label.setStyleSheet("color: #22c55e;")
        except Exception as e:
            parent.cloud_status_label.setText(f"Error: {e}")
            parent.cloud_status_label.setStyleSheet("color: #ef4444;")
    else:
        parent.cloud_status_label.setText("Cloud AI disabled - using local model")
        parent.cloud_status_label.setStyleSheet("color: #888;")
        
        # Try to switch back to local inference
        try:
            main_window = parent.window()
            if hasattr(main_window, 'module_manager'):
                manager = main_window.module_manager
                if manager.is_loaded('chat_api'):
                    manager.unload('chat_api')
        except Exception:
            pass


def _update_cloud_model_options(parent):
    """Update model options when provider changes."""
    provider = parent.cloud_provider_combo.currentData()
    
    parent.cloud_model_combo.clear()
    
    if provider == 'ollama':
        parent.cloud_model_combo.addItem("llama3.2:1b (Fast, FREE)", "llama3.2:1b")
        parent.cloud_model_combo.addItem("llama3.2:3b (Better, FREE)", "llama3.2:3b")
        parent.cloud_model_combo.addItem("mistral:7b (Quality, FREE)", "mistral:7b")
        parent.cloud_model_combo.addItem("phi3:mini (Microsoft, FREE)", "phi3:mini")
        parent.cloud_model_combo.addItem("gemma2:2b (Google, FREE)", "gemma2:2b")
    elif provider == 'openai':
        parent.cloud_model_combo.addItem("gpt-4o (Best)", "gpt-4o")
        parent.cloud_model_combo.addItem("gpt-4-turbo", "gpt-4-turbo")
        parent.cloud_model_combo.addItem("gpt-4", "gpt-4")
        parent.cloud_model_combo.addItem("gpt-3.5-turbo (Cheaper)", "gpt-3.5-turbo")
    elif provider == 'anthropic':
        parent.cloud_model_combo.addItem("Claude Sonnet 4.5 (Best)", "claude-sonnet-4-20250514")
        parent.cloud_model_combo.addItem("Claude Opus 4", "claude-opus-4-20250514")
        parent.cloud_model_combo.addItem("claude-3-opus", "claude-3-opus-20240229")
        parent.cloud_model_combo.addItem("claude-3-sonnet", "claude-3-sonnet-20240229")
        parent.cloud_model_combo.addItem("claude-3-haiku (Fast)", "claude-3-haiku-20240307")
    elif provider == 'google':
        parent.cloud_model_combo.addItem("gemini-pro (Free tier!)", "gemini-pro")
        parent.cloud_model_combo.addItem("gemini-pro-vision", "gemini-pro-vision")


def _toggle_always_on_top(parent, state):
    """Toggle always on top window flag.
    
    When main window is set to always-on-top, Quick Chat should have priority
    (appear above main window) if it's also always-on-top.
    """
    from PyQt5.QtCore import Qt
    
    main_window = parent.window()
    if not main_window:
        return
    
    current_flags = main_window.windowFlags()
    
    if state == Checked:
        main_window.setWindowFlags(current_flags | Qt.WindowStaysOnTopHint)
    else:
        main_window.setWindowFlags(current_flags & ~Qt.WindowStaysOnTopHint)
    
    main_window.show()
    
    # Ensure Quick Chat has priority when visible (raise it above main window)
    if hasattr(main_window, 'mini_chat') and main_window.mini_chat and main_window.mini_chat.isVisible():
        from PyQt5.QtCore import QTimer
        # Brief delay to ensure main window is positioned first, then raise Quick Chat
        QTimer.singleShot(50, lambda: _raise_mini_chat(main_window))


def _raise_mini_chat(main_window):
    """Raise Quick Chat above main window to give it z-order priority."""
    if hasattr(main_window, 'mini_chat') and main_window.mini_chat and main_window.mini_chat.isVisible():
        main_window.mini_chat.raise_()
        main_window.mini_chat.activateWindow()


def _disable_always_on_top(parent):
    """Disable always-on-top for main window."""
    from PyQt5.QtCore import Qt
    
    main_window = parent.window()
    if not main_window:
        return
    
    current_flags = main_window.windowFlags()
    main_window.setWindowFlags(current_flags & ~Qt.WindowStaysOnTopHint)
    main_window.show()
    
    # Update checkbox if it exists
    if hasattr(parent, 'always_on_top_check'):
        parent.always_on_top_check.blockSignals(True)
        parent.always_on_top_check.setChecked(False)
        parent.always_on_top_check.blockSignals(False)


def _reset_window_position(parent):
    """Reset main window to center of primary screen."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QGuiApplication
    
    main_window = parent.window()
    if not main_window:
        return
    
    # Get primary screen
    screen = QGuiApplication.primaryScreen()
    if screen:
        screen_geo = screen.availableGeometry()
        # Center window
        x = screen_geo.x() + (screen_geo.width() - main_window.width()) // 2
        y = screen_geo.y() + (screen_geo.height() - main_window.height()) // 2
        main_window.move(x, y)


def _reset_all_settings(parent):
    """Reset all GUI settings to defaults."""
    import json
    from pathlib import Path
    from PyQt5.QtWidgets import QMessageBox
    from ...config import CONFIG
    
    reply = QMessageBox.question(
        parent, "Reset All Settings",
        "This will reset ALL GUI settings to defaults.\n\n"
        "The application will close. Restart to apply changes.\n\n"
        "Continue?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    
    if reply != QMessageBox.Yes:
        return
    
    try:
        # Delete gui_settings.json (primary location)
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        if settings_path.exists():
            settings_path.unlink()
        
        # Also reset info dir settings (legacy location)
        info_settings = Path(CONFIG.get("info_dir", "information")) / "gui_settings.json"
        if info_settings.exists():
            info_settings.unlink()
        
        QMessageBox.information(
            parent, "Settings Reset",
            "Settings have been reset.\n\nThe application will now close."
        )
        
        # Close the application
        main_window = parent.window()
        if main_window:
            main_window.close()
            
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Could not reset settings: {e}")


def _toggle_mini_chat_on_top(parent, state):
    """Toggle Quick Chat always on top and save setting."""
    import json
    from pathlib import Path
    from ...config import CONFIG
    
    on_top = state == Checked
    
    # Save to gui_settings.json
    try:
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        settings["mini_chat_always_on_top"] = on_top
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Could not save Quick Chat setting: {e}")
    
    # Update the Quick Chat overlay if it exists
    main_window = parent.window()
    if hasattr(main_window, 'mini_chat') and main_window.mini_chat:
        main_window.mini_chat.set_always_on_top(on_top)


def _save_chat_names(parent):
    """Save chat display names to settings."""
    import json
    from pathlib import Path
    from ...config import CONFIG
    
    try:
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        user_name = parent.user_display_name_input.text().strip()
        settings["user_display_name"] = user_name if user_name else "You"
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        # Update main window's user_display_name
        main_window = parent.window()
        if main_window:
            main_window.user_display_name = settings["user_display_name"]
    except Exception as e:
        print(f"Could not save chat names: {e}")


# ===== SYSTEM PROMPT SETTINGS =====
def _save_system_prompt(parent):
    """Save the custom system prompt to settings."""
    import json
    from pathlib import Path
    from ...config import CONFIG
    
    try:
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        preset = parent.system_prompt_preset.currentData()
        custom_prompt = parent.custom_system_prompt.toPlainText().strip()
        
        settings["system_prompt_preset"] = preset
        settings["custom_system_prompt"] = custom_prompt
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        # Update main window's system prompt setting
        main_window = parent.window()
        if main_window:
            main_window._system_prompt_preset = preset
            main_window._custom_system_prompt = custom_prompt
        
        parent.prompt_status_label.setText("System prompt saved!")
        parent.prompt_status_label.setStyleSheet("color: #22c55e; font-style: italic;")
    except Exception as e:
        parent.prompt_status_label.setText(f"Error saving: {e}")
        parent.prompt_status_label.setStyleSheet("color: #ef4444; font-style: italic;")


def _load_system_prompt(parent):
    """Load the system prompt settings."""
    import json
    from pathlib import Path
    from ...config import CONFIG
    
    try:
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            
            preset = settings.get("system_prompt_preset", "simple")
            custom_prompt = settings.get("custom_system_prompt", "")
            
            # Set combo box (without triggering signal)
            parent.system_prompt_preset.blockSignals(True)
            for i in range(parent.system_prompt_preset.count()):
                if parent.system_prompt_preset.itemData(i) == preset:
                    parent.system_prompt_preset.setCurrentIndex(i)
                    break
            parent.system_prompt_preset.blockSignals(False)
            
            # Set custom prompt text only if preset is custom
            if preset == "custom" and custom_prompt:
                parent.custom_system_prompt.setText(custom_prompt)
            
            # Apply the preset to show proper text
            _apply_system_prompt_preset(parent)
            
            # Set on main window
            main_window = parent.window()
            if main_window:
                main_window._system_prompt_preset = preset
                main_window._custom_system_prompt = custom_prompt
    except Exception as e:
        print(f"Could not load system prompt: {e}")
        # Apply default preset on error
        _apply_system_prompt_preset(parent)


def _apply_system_prompt_preset(parent):
    """Apply the selected system prompt preset and display the actual prompt text."""
    preset = parent.system_prompt_preset.currentData()
    
    # Define the actual prompts
    SIMPLE_PROMPT = "You are a helpful AI assistant. Answer questions clearly and conversationally. Be friendly and helpful."
    
    FULL_PROMPT = """You are ForgeAI, an intelligent AI assistant with access to various tools and capabilities.

## Tool Usage
When you need to perform an action (generate media, access files, etc.), use this format:
<tool_call>{"tool": "tool_name", "params": {"param1": "value1"}}</tool_call>

## Available Tools
- generate_image: Create an image from a text description
- generate_video: Generate a video from a description
- generate_code: Generate code for a task
- generate_audio: Generate speech or audio
- generate_3d: Generate a 3D model
- read_file: Read contents of a file
- write_file: Write content to a file (requires permission)
- web_search: Search the web
- screenshot: Take a screenshot
- run_command: Run a system command (requires permission)

## Important
- For system-modifying actions, the user will be asked for permission
- Always explain what you're about to do before using a tool
- Be helpful, accurate, and respect user privacy

Be friendly, concise, and proactive in helping users accomplish their goals."""
    
    if preset == "simple":
        parent.custom_system_prompt.setReadOnly(True)
        parent.custom_system_prompt.setText(SIMPLE_PROMPT)
        parent.custom_system_prompt.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                color: #aaa;
            }
        """)
    elif preset == "full":
        parent.custom_system_prompt.setReadOnly(True)
        parent.custom_system_prompt.setText(FULL_PROMPT)
        parent.custom_system_prompt.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                color: #aaa;
            }
        """)
    else:  # custom
        parent.custom_system_prompt.setReadOnly(False)
        # Don't overwrite existing custom text
        if not parent.custom_system_prompt.toPlainText().strip():
            parent.custom_system_prompt.setPlaceholderText(
                "Enter your custom system prompt here...\n\n"
                "Example: You are a helpful AI assistant. Be friendly and concise."
            )
        parent.custom_system_prompt.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                color: white;
            }
        """)
    
    # Auto-save when preset changes
    _save_system_prompt(parent)


def _reset_system_prompt(parent):
    """Reset system prompt to default."""
    parent.system_prompt_preset.setCurrentIndex(0)  # Simple
    parent.custom_system_prompt.clear()
    _save_system_prompt(parent)
    parent.prompt_status_label.setText("Reset to default (Simple)")
    parent.prompt_status_label.setStyleSheet("color: #3b82f6; font-style: italic;")


def _load_chat_names(parent):
    """Load chat display names from settings."""
    import json
    from pathlib import Path
    from ...config import CONFIG
    
    try:
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            
            user_name = settings.get("user_display_name", "You")
            parent.user_display_name_input.setText(user_name if user_name != "You" else "")
            
            # Set on main window
            main_window = parent.window()
            if main_window:
                main_window.user_display_name = user_name
    except Exception as e:
        print(f"Could not load chat names: {e}")


def _update_display_info(parent):
    """Update the display info label."""
    from PyQt5.QtGui import QGuiApplication
    
    main_window = parent.window()
    if not main_window:
        parent.display_info_label.setText("")
        return
    
    current_screen = QGuiApplication.screenAt(main_window.geometry().center())
    if current_screen:
        geo = current_screen.geometry()
        dpi = current_screen.logicalDotsPerInch()
        refresh = current_screen.refreshRate()
        parent.display_info_label.setText(
            f"Current: {geo.width()}x{geo.height()} @ {refresh:.0f}Hz, {dpi:.0f} DPI"
        )
    else:
        parent.display_info_label.setText("Display info unavailable")


def create_settings_tab(parent):
    """Create the settings/resources tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(15)
    
    # === AI CONTROL LOCK ===
    lock_group = QGroupBox("AI Control Lock")
    lock_layout = QVBoxLayout(lock_group)
    
    lock_desc = QLabel(
        "When locked, only the AI can change settings. "
        "This prevents accidental changes while the AI is working."
    )
    lock_desc.setWordWrap(True)
    lock_layout.addWidget(lock_desc)
    
    lock_row = QHBoxLayout()
    parent.ai_lock_checkbox = QCheckBox("Lock settings for AI control")
    parent.ai_lock_checkbox.stateChanged.connect(
        lambda state: _toggle_ai_lock(parent, state)
    )
    lock_row.addWidget(parent.ai_lock_checkbox)
    
    parent.ai_lock_status = QLabel("")
    parent.ai_lock_status.setStyleSheet("color: #888; font-style: italic;")
    lock_row.addWidget(parent.ai_lock_status)
    lock_row.addStretch()
    lock_layout.addLayout(lock_row)
    
    # Password protection for unlock
    pwd_row = QHBoxLayout()
    pwd_row.addWidget(QLabel("Unlock PIN (optional):"))
    parent.ai_lock_pin = QLineEdit()
    parent.ai_lock_pin.setPlaceholderText("Set a 4-digit PIN")
    parent.ai_lock_pin.setMaxLength(4)
    parent.ai_lock_pin.setMaximumWidth(100)
    parent.ai_lock_pin.setEchoMode(QLineEdit.Password)
    pwd_row.addWidget(parent.ai_lock_pin)
    pwd_row.addStretch()
    lock_layout.addLayout(pwd_row)
    
    layout.addWidget(lock_group)
    
    # === DEVICE INFO ===
    device_group = QGroupBox(" Hardware Detection")
    device_layout = QVBoxLayout(device_group)
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            device_info = f"GPU Available: {gpu_name} ({gpu_mem} MB)"
            device_style = "color: #22c55e; font-weight: bold;"
        else:
            device_info = "No GPU - Using CPU only"
            device_style = "color: #f59e0b; font-weight: bold;"
        
        cpu_count = torch.get_num_threads()
        cpu_info = f"CPU Threads: {cpu_count}"
    except Exception:
        device_info = "Warning: PyTorch not available"
        device_style = "color: #ef4444;"
        cpu_info = ""
    
    device_label = QLabel(device_info)
    device_label.setStyleSheet(device_style)
    device_layout.addWidget(device_label)
    
    if cpu_info:
        cpu_label = QLabel(cpu_info)
        device_layout.addWidget(cpu_label)
    
    layout.addWidget(device_group)
    
    # === POWER MODE ===
    power_group = QGroupBox(" Power Mode")
    power_layout = QVBoxLayout(power_group)
    
    # Mode description
    mode_desc = QLabel(
        "Control how much CPU/GPU the AI uses. Lower settings free up resources for gaming or other apps."
    )
    mode_desc.setWordWrap(True)
    power_layout.addWidget(mode_desc)
    
    # Mode selector
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Mode:"))
    
    parent.resource_mode_combo = QComboBox()
    parent.resource_mode_combo.addItem("Minimal - Best for gaming", "minimal")
    parent.resource_mode_combo.addItem("Gaming - AI in background", "gaming")
    parent.resource_mode_combo.addItem("Balanced - Normal use (default)", "balanced")
    parent.resource_mode_combo.addItem("Performance - Faster AI responses", "performance")
    parent.resource_mode_combo.addItem("Maximum - Use all resources", "max")
    parent.resource_mode_combo.setCurrentIndex(2)  # Default to balanced
    parent.resource_mode_combo.currentIndexChanged.connect(
        lambda idx: _apply_resource_mode(parent)
    )
    mode_row.addWidget(parent.resource_mode_combo)
    mode_row.addStretch()
    power_layout.addLayout(mode_row)
    
    # Mode details
    parent.power_mode_details_label = QLabel(
        "Balanced: Moderate resource usage. Good for normal use."
    )
    parent.power_mode_details_label.setStyleSheet("color: #888; font-style: italic;")
    power_layout.addWidget(parent.power_mode_details_label)
    
    # Custom resource settings
    custom_frame = QGroupBox("Custom Resource Limits")
    custom_layout = QVBoxLayout(custom_frame)
    
    # GPU usage spinbox
    gpu_row = QHBoxLayout()
    gpu_row.addWidget(QLabel("GPU Usage (%):"))
    
    parent.gpu_spinbox = QSpinBox()
    parent.gpu_spinbox.setRange(10, 95)
    parent.gpu_spinbox.setValue(85)  # Match current config default
    parent.gpu_spinbox.setSuffix("%")
    parent.gpu_spinbox.setMinimumWidth(80)
    gpu_row.addWidget(parent.gpu_spinbox)
    gpu_row.addStretch()
    custom_layout.addLayout(gpu_row)
    
    # CPU threads spinbox
    cpu_row = QHBoxLayout()
    cpu_row.addWidget(QLabel("CPU Threads:"))
    
    import os as os_module
    max_threads = os_module.cpu_count() or 8
    parent.cpu_spinbox = QSpinBox()
    parent.cpu_spinbox.setRange(1, max_threads)
    parent.cpu_spinbox.setValue(max(1, max_threads // 2))  # Default to half
    parent.cpu_spinbox.setMinimumWidth(80)
    cpu_row.addWidget(parent.cpu_spinbox)
    
    cpu_info = QLabel(f"(max: {max_threads})")
    cpu_info.setStyleSheet("color: #888;")
    cpu_row.addWidget(cpu_info)
    cpu_row.addStretch()
    custom_layout.addLayout(cpu_row)
    
    # Apply button for custom settings
    apply_row = QHBoxLayout()
    apply_resource_btn = QPushButton("Apply Resource Limits")
    apply_resource_btn.clicked.connect(lambda: _apply_custom_resources(parent))
    apply_row.addWidget(apply_resource_btn)
    apply_row.addStretch()
    custom_layout.addLayout(apply_row)
    
    parent.resource_status_label = QLabel("")
    parent.resource_status_label.setStyleSheet("color: #888; font-style: italic;")
    custom_layout.addWidget(parent.resource_status_label)
    
    power_layout.addWidget(custom_frame)
    
    # Recommended settings info
    rec_frame = QGroupBox("Recommended Settings")
    rec_layout = QVBoxLayout(rec_frame)
    
    rec_info = QLabel(
        "<b>Small models (< 1B params):</b> GPU 50-70%, 2-4 CPU threads<br>"
        "<b>Medium models (1-7B):</b> GPU 70-85%, 4-8 CPU threads<br>"
        "<b>Large models (7B+):</b> GPU 85-95%, 8+ CPU threads<br><br>"
        "<b>Gaming while using AI:</b> Use 'Gaming' mode (GPU 40%, 2 threads)<br>"
        "<b>Best AI performance:</b> Use 'Maximum' mode (GPU 95%, all threads)"
    )
    rec_info.setStyleSheet("color: #bac2de; font-size: 11px; padding: 4px;")
    rec_info.setWordWrap(True)
    rec_layout.addWidget(rec_info)
    
    power_layout.addWidget(rec_frame)
    
    layout.addWidget(power_group)

    # === THEME SELECTOR ===
    theme_group = QGroupBox("Theme")
    theme_layout = QVBoxLayout(theme_group)

    theme_desc = QLabel(
        "Choose a visual theme for the application. "
        "Changes apply immediately."
    )
    theme_desc.setWordWrap(True)
    theme_layout.addWidget(theme_desc)

    theme_row = QHBoxLayout()
    theme_row.addWidget(QLabel("Theme:"))

    parent.theme_combo = QComboBox()
    parent.theme_combo.addItem("Dark (Default)", "dark")
    parent.theme_combo.addItem("Light", "light")
    parent.theme_combo.addItem("Cerulean", "cerulean")
    parent.theme_combo.addItem("High Contrast", "high_contrast")
    parent.theme_combo.addItem("Midnight", "midnight")
    parent.theme_combo.addItem("Forest", "forest")
    parent.theme_combo.addItem("Sunset", "sunset")
    parent.theme_combo.currentIndexChanged.connect(
        lambda idx: _apply_theme(parent)
    )
    theme_row.addWidget(parent.theme_combo)

    theme_row.addStretch()
    theme_layout.addLayout(theme_row)

    parent.theme_description_label = QLabel(
        "Dark theme with soft colors (default)"
    )
    parent.theme_description_label.setStyleSheet("color: #888; font-style: italic;")
    theme_layout.addWidget(parent.theme_description_label)

    layout.addWidget(theme_group)

    # === ZOOM / TEXT SIZE ===
    zoom_group = QGroupBox("Display Zoom")
    zoom_layout = QVBoxLayout(zoom_group)

    zoom_row = QHBoxLayout()
    zoom_row.addWidget(QLabel("Zoom Level:"))

    parent.zoom_spinbox = QSpinBox()
    parent.zoom_spinbox.setRange(80, 200)
    parent.zoom_spinbox.setValue(100)
    parent.zoom_spinbox.setSuffix("%")
    parent.zoom_spinbox.setSingleStep(5)
    parent.zoom_spinbox.setMinimumWidth(80)
    parent.zoom_spinbox.valueChanged.connect(
        lambda val: _apply_zoom(parent, val)
    )
    zoom_row.addWidget(parent.zoom_spinbox)

    reset_zoom_btn = QPushButton("Reset to 100%")
    reset_zoom_btn.clicked.connect(lambda: _reset_zoom(parent))
    zoom_row.addWidget(reset_zoom_btn)
    
    zoom_row.addStretch()
    zoom_layout.addLayout(zoom_row)
    layout.addWidget(zoom_group)

    # === CLOUD AI MODE ===
    cloud_group = QGroupBox("Cloud/API AI Mode")
    cloud_layout = QVBoxLayout(cloud_group)
    
    cloud_desc = QLabel(
        "<b>FREE option:</b> Ollama runs AI locally (no API key needed!)<br>"
        "<b>Paid options:</b> GPT-4, Claude, Gemini (need API keys)<br>"
        "Perfect for Raspberry Pi - use powerful AI without training!"
    )
    cloud_desc.setWordWrap(True)
    cloud_layout.addWidget(cloud_desc)
    
    # Enable cloud mode
    parent.cloud_mode_check = QCheckBox("Enable Cloud/API AI for Chat")
    parent.cloud_mode_check.setToolTip(
        "Route chat to API instead of local trained model.\n"
        "Ollama is FREE! Others need API keys."
    )
    parent.cloud_mode_check.stateChanged.connect(
        lambda state: _toggle_cloud_mode(parent, state)
    )
    cloud_layout.addWidget(parent.cloud_mode_check)
    
    # Provider selection
    provider_row = QHBoxLayout()
    provider_row.addWidget(QLabel("Provider:"))
    parent.cloud_provider_combo = QComboBox()
    parent.cloud_provider_combo.addItem("[FREE] Ollama (Local)", "ollama")
    parent.cloud_provider_combo.addItem("OpenAI (GPT-4)", "openai")
    parent.cloud_provider_combo.addItem("Anthropic (Claude)", "anthropic")
    parent.cloud_provider_combo.addItem("Google (Gemini - Free tier)", "google")
    parent.cloud_provider_combo.setEnabled(False)
    parent.cloud_provider_combo.currentIndexChanged.connect(
        lambda: _update_cloud_model_options(parent)
    )
    provider_row.addWidget(parent.cloud_provider_combo)
    provider_row.addStretch()
    cloud_layout.addLayout(provider_row)
    
    # Model selection
    model_row = QHBoxLayout()
    model_row.addWidget(QLabel("Model:"))
    parent.cloud_model_combo = QComboBox()
    parent.cloud_model_combo.addItem("llama3.2:1b (Fast, FREE)", "llama3.2:1b")
    parent.cloud_model_combo.addItem("llama3.2:3b (Better, FREE)", "llama3.2:3b")
    parent.cloud_model_combo.addItem("mistral:7b (Quality, FREE)", "mistral:7b")
    parent.cloud_model_combo.setEnabled(False)
    model_row.addWidget(parent.cloud_model_combo)
    model_row.addStretch()
    cloud_layout.addLayout(model_row)
    
    # Status
    parent.cloud_status_label = QLabel("For Ollama: Install from https://ollama.ai then run: ollama run llama3.2:1b")
    parent.cloud_status_label.setStyleSheet("color: #888; font-style: italic;")
    parent.cloud_status_label.setWordWrap(True)
    cloud_layout.addWidget(parent.cloud_status_label)
    
    layout.addWidget(cloud_group)

    # === CHAT NAMES ===
    names_group = QGroupBox("Chat Names")
    names_layout = QVBoxLayout(names_group)
    names_layout.setSpacing(6)
    
    names_desc = QLabel("Customize how you and the AI appear in chat.")
    names_desc.setWordWrap(True)
    names_layout.addWidget(names_desc)
    
    # User display name
    user_name_row = QHBoxLayout()
    user_name_row.addWidget(QLabel("Your Name:"))
    parent.user_display_name_input = QLineEdit()
    parent.user_display_name_input.setPlaceholderText("You")
    parent.user_display_name_input.setMaximumWidth(200)
    parent.user_display_name_input.textChanged.connect(
        lambda text: _save_chat_names(parent)
    )
    user_name_row.addWidget(parent.user_display_name_input)
    user_name_row.addStretch()
    names_layout.addLayout(user_name_row)
    
    # Load saved name
    _load_chat_names(parent)
    
    names_note = QLabel("The AI's display name is automatically set to the loaded model name.")
    names_note.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
    names_note.setWordWrap(True)
    names_layout.addWidget(names_note)
    
    layout.addWidget(names_group)

    # === DISPLAY SETTINGS ===
    display_group = QGroupBox("Display Settings")
    display_layout = QVBoxLayout(display_group)
    
    display_desc = QLabel(
        "Configure display options for multi-monitor setups."
    )
    display_desc.setWordWrap(True)
    display_layout.addWidget(display_desc)
    
    # Always on top checkbox
    parent.always_on_top_check = QCheckBox("Always on Top (Main Window)")
    parent.always_on_top_check.setToolTip("Keep the main window above other windows")
    parent.always_on_top_check.stateChanged.connect(
        lambda state: _toggle_always_on_top(parent, state)
    )
    display_layout.addWidget(parent.always_on_top_check)
    
    # Quick Chat always on top checkbox
    parent.mini_chat_on_top_check = QCheckBox("Quick Chat Always on Top")
    parent.mini_chat_on_top_check.setToolTip("Keep the Quick Chat window above other windows (default: on)")
    parent.mini_chat_on_top_check.setChecked(True)  # Default to checked
    parent.mini_chat_on_top_check.stateChanged.connect(
        lambda state: _toggle_mini_chat_on_top(parent, state)
    )
    display_layout.addWidget(parent.mini_chat_on_top_check)
    
    # Monitor selection row
    monitor_row = QHBoxLayout()
    monitor_row.addWidget(QLabel("Display:"))
    
    parent.monitor_combo = QComboBox()
    _populate_monitors(parent)
    parent.monitor_combo.currentIndexChanged.connect(
        lambda idx: _move_to_monitor(parent, idx)
    )
    monitor_row.addWidget(parent.monitor_combo)
    
    refresh_monitors_btn = QPushButton("Refresh")
    refresh_monitors_btn.setMaximumWidth(70)
    refresh_monitors_btn.clicked.connect(lambda: _populate_monitors(parent, preserve_selection=True))
    monitor_row.addWidget(refresh_monitors_btn)
    
    monitor_row.addStretch()
    display_layout.addLayout(monitor_row)
    
    # Current display info
    parent.display_info_label = QLabel("")
    parent.display_info_label.setStyleSheet("color: #888; font-style: italic;")
    _update_display_info(parent)
    display_layout.addWidget(parent.display_info_label)
    
    layout.addWidget(display_group)

    # === AUDIO DEVICE SETTINGS ===
    audio_group = QGroupBox("Audio Devices")
    audio_layout = QVBoxLayout(audio_group)
    
    audio_desc = QLabel(
        "Select input (microphone) and output (speaker) devices for voice features."
    )
    audio_desc.setWordWrap(True)
    audio_layout.addWidget(audio_desc)
    
    # Input device (microphone)
    input_row = QHBoxLayout()
    input_row.addWidget(QLabel("Microphone:"))
    parent.audio_input_combo = QComboBox()
    parent.audio_input_combo.setMinimumWidth(250)
    input_row.addWidget(parent.audio_input_combo)
    
    refresh_audio_btn = QPushButton("↻")
    refresh_audio_btn.setMaximumWidth(30)
    refresh_audio_btn.setToolTip("Refresh device list")
    refresh_audio_btn.clicked.connect(lambda: _refresh_audio_devices(parent))
    input_row.addWidget(refresh_audio_btn)
    input_row.addStretch()
    audio_layout.addLayout(input_row)
    
    # Output device (speaker)
    output_row = QHBoxLayout()
    output_row.addWidget(QLabel("Speaker:"))
    parent.audio_output_combo = QComboBox()
    parent.audio_output_combo.setMinimumWidth(250)
    output_row.addWidget(parent.audio_output_combo)
    output_row.addStretch()
    audio_layout.addLayout(output_row)
    
    # Mic test section
    mic_test_row = QHBoxLayout()
    parent.mic_test_btn = QPushButton("Test Microphone")
    parent.mic_test_btn.clicked.connect(lambda: _test_microphone(parent))
    mic_test_row.addWidget(parent.mic_test_btn)
    
    parent.mic_level_bar = QSlider(Qt.Orientation.Horizontal)
    parent.mic_level_bar.setRange(0, 100)
    parent.mic_level_bar.setValue(0)
    parent.mic_level_bar.setEnabled(False)
    parent.mic_level_bar.setStyleSheet("""
        QSlider::groove:horizontal { background: #333; height: 10px; border-radius: 5px; }
        QSlider::handle:horizontal { width: 0px; }
        QSlider::sub-page:horizontal { background: #22c55e; border-radius: 5px; }
    """)
    mic_test_row.addWidget(parent.mic_level_bar)
    
    parent.mic_status_label = QLabel("Not tested")
    parent.mic_status_label.setStyleSheet("color: #888;")
    mic_test_row.addWidget(parent.mic_status_label)
    mic_test_row.addStretch()
    audio_layout.addLayout(mic_test_row)
    
    # Populate audio devices
    _refresh_audio_devices(parent)
    
    layout.addWidget(audio_group)

    # === AVATAR CONTROL ===
    avatar_group = QGroupBox("Avatar Control")
    avatar_layout = QVBoxLayout(avatar_group)
    
    avatar_desc = QLabel(
        "Avatar can react to screen content and behave autonomously."
    )
    avatar_desc.setWordWrap(True)
    avatar_layout.addWidget(avatar_desc)
    
    # Avatar auto expressions
    parent.auto_avatar_check = QCheckBox("Auto Expressions (from AI text)")
    parent.auto_avatar_check.setChecked(True)
    parent.auto_avatar_check.stateChanged.connect(
        lambda state: setattr(parent, 'auto_avatar_enabled', state == 2)
    )
    parent.auto_avatar_enabled = True
    avatar_layout.addWidget(parent.auto_avatar_check)
    
    # Avatar autonomous mode
    parent.avatar_autonomous_check = QCheckBox("Autonomous Mode (react to screen)")
    parent.avatar_autonomous_check.setChecked(False)
    parent.avatar_autonomous_check.stateChanged.connect(
        lambda state: _toggle_avatar_autonomous(parent, state)
    )
    avatar_layout.addWidget(parent.avatar_autonomous_check)
    
    # Avatar behavior settings
    avatar_behavior_row = QHBoxLayout()
    avatar_behavior_row.addWidget(QLabel("Activity:"))
    parent.avatar_activity_slider = QSlider(Qt.Orientation.Horizontal)
    parent.avatar_activity_slider.setRange(1, 10)
    parent.avatar_activity_slider.setValue(5)
    parent.avatar_activity_slider.setMaximumWidth(100)
    parent.avatar_activity_slider.setToolTip("How active the avatar is (1=calm, 10=energetic)")
    avatar_behavior_row.addWidget(parent.avatar_activity_slider)
    avatar_behavior_row.addWidget(QLabel("Slow"))
    avatar_behavior_row.addStretch()
    avatar_behavior_row.addWidget(QLabel("Fast"))
    avatar_layout.addLayout(avatar_behavior_row)
    
    parent.avatar_status_label = QLabel("Avatar: Idle")
    parent.avatar_status_label.setStyleSheet("color: #888; font-style: italic;")
    avatar_layout.addWidget(parent.avatar_status_label)
    
    layout.addWidget(avatar_group)
    
    # === ROBOT CONTROL ===
    robot_group = QGroupBox("Robot Control")
    robot_layout = QVBoxLayout(robot_group)
    
    robot_desc = QLabel(
        "Control robot with safety modes. AUTO = AI controls, MANUAL = you control."
    )
    robot_desc.setWordWrap(True)
    robot_layout.addWidget(robot_desc)
    
    # Robot mode selector
    robot_mode_row = QHBoxLayout()
    robot_mode_row.addWidget(QLabel("Mode:"))
    parent.robot_mode_combo = QComboBox()
    parent.robot_mode_combo.addItem("DISABLED", "disabled")
    parent.robot_mode_combo.addItem("MANUAL (User)", "manual")
    parent.robot_mode_combo.addItem("AUTO (AI)", "auto")
    parent.robot_mode_combo.addItem("SAFE (Limited)", "safe")
    parent.robot_mode_combo.currentIndexChanged.connect(
        lambda: _change_robot_mode(parent)
    )
    robot_mode_row.addWidget(parent.robot_mode_combo)
    robot_mode_row.addStretch()
    robot_layout.addLayout(robot_mode_row)
    
    # E-STOP button
    parent.estop_btn = QPushButton("EMERGENCY STOP")
    parent.estop_btn.setStyleSheet("""
        QPushButton {
            background-color: #dc2626;
            color: white;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #b91c1c;
        }
    """)
    parent.estop_btn.clicked.connect(lambda: _robot_estop(parent))
    robot_layout.addWidget(parent.estop_btn)
    
    # Camera toggle
    parent.robot_camera_check = QCheckBox("Enable Camera Feed")
    parent.robot_camera_check.stateChanged.connect(
        lambda state: _toggle_robot_camera(parent, state)
    )
    robot_layout.addWidget(parent.robot_camera_check)
    
    # Robot status
    parent.robot_status_label = QLabel("Robot: Disabled")
    parent.robot_status_label.setStyleSheet("color: #888; font-style: italic;")
    robot_layout.addWidget(parent.robot_status_label)
    
    layout.addWidget(robot_group)
    
    # NOTE: Game AI Routing moved to Game tab for better organization

    # === AUTONOMOUS MODE ===
    autonomous_group = QGroupBox("Autonomous Mode")
    autonomous_layout = QVBoxLayout(autonomous_group)
    
    autonomous_desc = QLabel(
        "Allow AI to act on its own - explore curiosities, learn from the web, "
        "and evolve personality when you're not chatting. "
        "Can be turned off at any time."
    )
    autonomous_desc.setWordWrap(True)
    autonomous_layout.addWidget(autonomous_desc)
    
    parent.autonomous_enabled_check = QCheckBox("Enable Autonomous Mode")
    parent.autonomous_enabled_check.stateChanged.connect(
        lambda state: _toggle_autonomous(parent, state)
    )
    autonomous_layout.addWidget(parent.autonomous_enabled_check)
    
    # Autonomous settings
    autonomous_settings = QHBoxLayout()
    autonomous_settings.addWidget(QLabel("Activity Level:"))
    
    parent.autonomous_activity_spin = QSpinBox()
    parent.autonomous_activity_spin.setRange(1, 20)
    parent.autonomous_activity_spin.setValue(12)
    parent.autonomous_activity_spin.setSuffix(" actions/hour")
    parent.autonomous_activity_spin.setToolTip("How many autonomous actions per hour")
    parent.autonomous_activity_spin.setEnabled(False)  # Disabled until autonomous mode enabled
    autonomous_settings.addWidget(parent.autonomous_activity_spin)
    autonomous_settings.addStretch()
    autonomous_layout.addLayout(autonomous_settings)
    
    layout.addWidget(autonomous_group)
    
    # === PERSONALITY SETTINGS (COMPACT) ===
    personality_group = QGroupBox("AI Personality")
    personality_layout = QVBoxLayout(personality_group)
    personality_layout.setSpacing(6)
    
    # Preset row
    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel("Preset:"))
    parent.settings_personality_combo = QComboBox()
    parent.settings_personality_combo.addItem("Balanced (Default)", "balanced")
    parent.settings_personality_combo.addItem("Professional", "professional")
    parent.settings_personality_combo.addItem("Friendly", "friendly")
    parent.settings_personality_combo.addItem("Creative", "creative")
    parent.settings_personality_combo.addItem("Analytical", "analytical")
    parent.settings_personality_combo.addItem("Teacher", "teacher")
    parent.settings_personality_combo.currentIndexChanged.connect(
        lambda: _apply_personality_preset(parent)
    )
    preset_row.addWidget(parent.settings_personality_combo)
    preset_row.addStretch()
    personality_layout.addLayout(preset_row)
    
    # Quick trait sliders (compact - most important ones)
    trait_row1 = QHBoxLayout()
    trait_row1.addWidget(QLabel("Formal"))
    parent.formality_slider = QSlider(Qt.Orientation.Horizontal)
    parent.formality_slider.setRange(0, 100)
    parent.formality_slider.setValue(50)
    parent.formality_slider.setMaximumWidth(150)
    parent.formality_slider.valueChanged.connect(lambda v: _update_personality_trait(parent, 'formality', v))
    trait_row1.addWidget(parent.formality_slider)
    trait_row1.addWidget(QLabel("Casual"))
    trait_row1.addSpacing(20)
    trait_row1.addWidget(QLabel("Brief"))
    parent.verbosity_slider = QSlider(Qt.Orientation.Horizontal)
    parent.verbosity_slider.setRange(0, 100)
    parent.verbosity_slider.setValue(50)
    parent.verbosity_slider.setMaximumWidth(150)
    parent.verbosity_slider.valueChanged.connect(lambda v: _update_personality_trait(parent, 'verbosity', v))
    trait_row1.addWidget(parent.verbosity_slider)
    trait_row1.addWidget(QLabel("Detailed"))
    trait_row1.addStretch()
    personality_layout.addLayout(trait_row1)
    
    trait_row2 = QHBoxLayout()
    trait_row2.addWidget(QLabel("Serious"))
    parent.humor_slider = QSlider(Qt.Orientation.Horizontal)
    parent.humor_slider.setRange(0, 100)
    parent.humor_slider.setValue(50)
    parent.humor_slider.setMaximumWidth(150)
    parent.humor_slider.valueChanged.connect(lambda v: _update_personality_trait(parent, 'humor_level', v))
    trait_row2.addWidget(parent.humor_slider)
    trait_row2.addWidget(QLabel("Humorous"))
    trait_row2.addSpacing(20)
    trait_row2.addWidget(QLabel("Factual"))
    parent.creativity_slider = QSlider(Qt.Orientation.Horizontal)
    parent.creativity_slider.setRange(0, 100)
    parent.creativity_slider.setValue(50)
    parent.creativity_slider.setMaximumWidth(150)
    parent.creativity_slider.valueChanged.connect(lambda v: _update_personality_trait(parent, 'creativity', v))
    trait_row2.addWidget(parent.creativity_slider)
    trait_row2.addWidget(QLabel("Creative"))
    trait_row2.addStretch()
    personality_layout.addLayout(trait_row2)
    
    # Evolution toggle
    evolution_row = QHBoxLayout()
    parent.settings_evolution_check = QCheckBox("Allow personality evolution from conversations")
    parent.settings_evolution_check.setChecked(True)
    parent.settings_evolution_check.stateChanged.connect(
        lambda state: _toggle_personality_evolution(parent, state)
    )
    evolution_row.addWidget(parent.settings_evolution_check)
    evolution_row.addStretch()
    personality_layout.addLayout(evolution_row)
    
    layout.addWidget(personality_group)
    
    # === API KEYS ===
    api_group = QGroupBox("API Keys")
    api_layout = QVBoxLayout(api_group)
    
    api_desc = QLabel(
        "Configure API keys for cloud services. Keys are stored in environment variables. "
        "Leave blank to use local models only."
    )
    api_desc.setWordWrap(True)
    api_layout.addWidget(api_desc)
    
    # HuggingFace Token
    hf_row = QHBoxLayout()
    hf_row.addWidget(QLabel("HuggingFace Token:"))
    parent.hf_token_input = QLineEdit()
    parent.hf_token_input.setPlaceholderText("hf_... (for gated models like Llama)")
    parent.hf_token_input.setEchoMode(QLineEdit.Password)
    parent.hf_token_input.setText(_get_env_key("HF_TOKEN"))
    hf_row.addWidget(parent.hf_token_input)
    api_layout.addLayout(hf_row)
    
    # OpenAI API Key
    openai_row = QHBoxLayout()
    openai_row.addWidget(QLabel("OpenAI API Key:"))
    parent.openai_key_input = QLineEdit()
    parent.openai_key_input.setPlaceholderText("sk-... (for DALL-E, GPT-4)")
    parent.openai_key_input.setEchoMode(QLineEdit.Password)
    parent.openai_key_input.setText(_get_env_key("OPENAI_API_KEY"))
    openai_row.addWidget(parent.openai_key_input)
    api_layout.addLayout(openai_row)
    
    # Replicate Token
    replicate_row = QHBoxLayout()
    replicate_row.addWidget(QLabel("Replicate Token:"))
    parent.replicate_key_input = QLineEdit()
    parent.replicate_key_input.setPlaceholderText("r8_... (for cloud video/audio/3D)")
    parent.replicate_key_input.setEchoMode(QLineEdit.Password)
    parent.replicate_key_input.setText(_get_env_key("REPLICATE_API_TOKEN"))
    replicate_row.addWidget(parent.replicate_key_input)
    api_layout.addLayout(replicate_row)
    
    # ElevenLabs Key
    eleven_row = QHBoxLayout()
    eleven_row.addWidget(QLabel("ElevenLabs Key:"))
    parent.elevenlabs_key_input = QLineEdit()
    parent.elevenlabs_key_input.setPlaceholderText("(for cloud TTS)")
    parent.elevenlabs_key_input.setEchoMode(QLineEdit.Password)
    parent.elevenlabs_key_input.setText(_get_env_key("ELEVENLABS_API_KEY"))
    eleven_row.addWidget(parent.elevenlabs_key_input)
    api_layout.addLayout(eleven_row)
    
    # Save keys button
    api_buttons = QHBoxLayout()
    save_keys_btn = QPushButton("Save API Keys")
    save_keys_btn.clicked.connect(lambda: _save_api_keys(parent))
    api_buttons.addWidget(save_keys_btn)
    
    show_keys_btn = QPushButton("Show/Hide")
    show_keys_btn.clicked.connect(lambda: _toggle_key_visibility(parent))
    api_buttons.addWidget(show_keys_btn)
    
    api_buttons.addStretch()
    api_layout.addLayout(api_buttons)
    
    parent.api_status_label = QLabel("")
    parent.api_status_label.setStyleSheet("color: #888; font-style: italic;")
    api_layout.addWidget(parent.api_status_label)
    
    layout.addWidget(api_group)
    
    # === SYSTEM PROMPT ===
    prompt_group = QGroupBox("System Prompt")
    prompt_layout = QVBoxLayout(prompt_group)
    
    prompt_desc = QLabel(
        "Customize the system prompt that tells the AI how to behave. "
        "This affects both the main Chat and Quick Chat."
    )
    prompt_desc.setWordWrap(True)
    prompt_layout.addWidget(prompt_desc)
    
    # Preset selector
    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel("Preset:"))
    parent.system_prompt_preset = QComboBox()
    parent.system_prompt_preset.addItem("Simple (recommended for small models)", "simple")
    parent.system_prompt_preset.addItem("Full (with tools, for larger models)", "full")
    parent.system_prompt_preset.addItem("Custom", "custom")
    parent.system_prompt_preset.currentIndexChanged.connect(
        lambda: _apply_system_prompt_preset(parent)
    )
    preset_row.addWidget(parent.system_prompt_preset)
    preset_row.addStretch()
    prompt_layout.addLayout(preset_row)
    
    # Custom prompt text area
    parent.custom_system_prompt = QTextEdit()
    parent.custom_system_prompt.setPlaceholderText(
        "Enter your custom system prompt here...\n\n"
        "Example: You are a helpful AI assistant. Be friendly and concise."
    )
    parent.custom_system_prompt.setMaximumHeight(120)
    parent.custom_system_prompt.setStyleSheet("""
        QTextEdit {
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 8px;
            font-family: monospace;
        }
    """)
    prompt_layout.addWidget(parent.custom_system_prompt)
    
    # Save button
    prompt_buttons = QHBoxLayout()
    save_prompt_btn = QPushButton("Save System Prompt")
    save_prompt_btn.clicked.connect(lambda: _save_system_prompt(parent))
    prompt_buttons.addWidget(save_prompt_btn)
    
    reset_prompt_btn = QPushButton("Reset to Default")
    reset_prompt_btn.clicked.connect(lambda: _reset_system_prompt(parent))
    prompt_buttons.addWidget(reset_prompt_btn)
    prompt_buttons.addStretch()
    prompt_layout.addLayout(prompt_buttons)
    
    parent.prompt_status_label = QLabel("")
    parent.prompt_status_label.setStyleSheet("color: #888; font-style: italic;")
    prompt_layout.addWidget(parent.prompt_status_label)
    
    layout.addWidget(prompt_group)
    
    # Load saved system prompt
    _load_system_prompt(parent)
    
    # === AI CONNECTION STATUS ===
    connection_group = QGroupBox("AI Connection Status")
    connection_layout = QVBoxLayout(connection_group)
    
    conn_desc = QLabel("Shows what AI is currently active and which components are loaded.")
    conn_desc.setWordWrap(True)
    connection_layout.addWidget(conn_desc)
    
    # Active AI Model display
    parent.active_ai_label = QLabel("Active AI: Not configured")
    parent.active_ai_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px; background: #1e1e2e; border-radius: 4px;")
    connection_layout.addWidget(parent.active_ai_label)
    
    # Connection indicators
    parent.connection_indicators = QWidget()
    conn_grid = QHBoxLayout(parent.connection_indicators)
    conn_grid.setSpacing(20)
    
    # Create indicator labels
    parent.model_status = _create_status_indicator("Model", "disconnected")
    parent.tokenizer_status = _create_status_indicator("Tokenizer", "disconnected")
    parent.inference_status = _create_status_indicator("Inference", "disconnected")
    parent.memory_status = _create_status_indicator("Memory", "disconnected")
    
    conn_grid.addWidget(parent.model_status)
    conn_grid.addWidget(parent.tokenizer_status)
    conn_grid.addWidget(parent.inference_status)
    conn_grid.addWidget(parent.memory_status)
    conn_grid.addStretch()
    
    connection_layout.addWidget(parent.connection_indicators)
    
    # Refresh connection button
    conn_refresh_btn = QPushButton("Check Connections")
    conn_refresh_btn.clicked.connect(lambda: _refresh_connections(parent))
    connection_layout.addWidget(conn_refresh_btn)
    
    layout.addWidget(connection_group)
    
    # === CACHE MANAGEMENT ===
    cache_group = QGroupBox("Downloaded Models (Cache)")
    cache_layout = QVBoxLayout(cache_group)
    
    cache_desc = QLabel(
        "HuggingFace models are downloaded to your cache folder. "
        "Delete unused models to free up disk space."
    )
    cache_desc.setWordWrap(True)
    cache_layout.addWidget(cache_desc)
    
    # Cache info
    parent.cache_size_label = QLabel("Calculating cache size...")
    parent.cache_size_label.setStyleSheet("font-weight: bold;")
    cache_layout.addWidget(parent.cache_size_label)
    
    parent.cache_path_label = QLabel("")
    parent.cache_path_label.setStyleSheet("color: #888; font-size: 10px;")
    parent.cache_path_label.setWordWrap(True)
    cache_layout.addWidget(parent.cache_path_label)
    
    # Cache buttons
    cache_buttons = QHBoxLayout()
    
    open_cache_btn = QPushButton("Open Cache Folder")
    open_cache_btn.clicked.connect(lambda: _open_cache_folder(parent))
    cache_buttons.addWidget(open_cache_btn)
    
    refresh_cache_btn = QPushButton("Refresh Size")
    refresh_cache_btn.clicked.connect(lambda: _refresh_cache_info(parent))
    cache_buttons.addWidget(refresh_cache_btn)
    
    clear_cache_btn = QPushButton("Clear All Cache")
    clear_cache_btn.setStyleSheet("background: #7f1d1d; color: white;")
    clear_cache_btn.clicked.connect(lambda: _clear_cache(parent))
    cache_buttons.addWidget(clear_cache_btn)
    
    cache_buttons.addStretch()
    cache_layout.addLayout(cache_buttons)
    
    layout.addWidget(cache_group)
    
    # === CURRENT STATUS ===
    status_group = QGroupBox("Current Status")
    status_layout = QVBoxLayout(status_group)
    
    parent.power_status = QTextEdit()
    parent.power_status.setReadOnly(True)
    parent.power_status.setMaximumHeight(150)
    parent.power_status.setStyleSheet("font-family: Consolas, monospace;")
    status_layout.addWidget(parent.power_status)
    
    refresh_btn = QPushButton("Refresh Status")
    refresh_btn.clicked.connect(lambda: _refresh_power_status(parent))
    status_layout.addWidget(refresh_btn)
    
    layout.addWidget(status_group)
    
    # === RESET SETTINGS ===
    reset_group = QGroupBox("Reset Settings")
    reset_layout = QVBoxLayout(reset_group)
    
    reset_desc = QLabel(
        "Reset GUI settings to defaults. Use this if you're having issues "
        "or want to restore original window positions and preferences."
    )
    reset_desc.setWordWrap(True)
    reset_layout.addWidget(reset_desc)
    
    reset_buttons = QHBoxLayout()
    
    reset_window_btn = QPushButton("Reset Window Position")
    reset_window_btn.setToolTip("Center window on primary screen")
    reset_window_btn.setStyleSheet("""
        QPushButton {
            background: #3b82f6;
            color: white;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
        }
        QPushButton:hover {
            background: #2563eb;
        }
    """)
    reset_window_btn.clicked.connect(lambda: _reset_window_position(parent))
    reset_buttons.addWidget(reset_window_btn)
    
    reset_ontop_btn = QPushButton("Disable Always-On-Top")
    reset_ontop_btn.setToolTip("Turn off always-on-top for main window")
    reset_ontop_btn.setStyleSheet("""
        QPushButton {
            background: #8b5cf6;
            color: white;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
        }
        QPushButton:hover {
            background: #7c3aed;
        }
    """)
    reset_ontop_btn.clicked.connect(lambda: _disable_always_on_top(parent))
    reset_buttons.addWidget(reset_ontop_btn)
    
    reset_all_btn = QPushButton("Reset All Settings")
    reset_all_btn.setStyleSheet("""
        QPushButton {
            background: #dc2626;
            color: white;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
        }
        QPushButton:hover {
            background: #b91c1c;
        }
    """)
    reset_all_btn.setToolTip("Reset all settings to defaults (requires restart)")
    reset_all_btn.clicked.connect(lambda: _reset_all_settings(parent))
    reset_buttons.addWidget(reset_all_btn)
    
    reset_buttons.addStretch()
    reset_layout.addLayout(reset_buttons)
    
    layout.addWidget(reset_group)
    
    layout.addStretch()
    
    # Initial status refresh
    _refresh_power_status(parent)
    _refresh_cache_info(parent)
    _refresh_connections(parent)
    
    # Load saved mini chat on top setting
    _load_mini_chat_on_top_setting(parent)
    
    return tab


def _load_mini_chat_on_top_setting(parent):
    """Load the saved mini chat on top setting."""
    try:
        import json
        from pathlib import Path
        from ...config import CONFIG
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            on_top = settings.get("mini_chat_always_on_top", True)  # Default to True
            parent.mini_chat_on_top_check.setChecked(on_top)
    except Exception:
        pass  # Keep default (checked)


def _create_status_indicator(name: str, status: str) -> QLabel:
    """Create a status indicator widget."""
    colors = {
        "connected": "#22c55e",  # Green
        "disconnected": "#ef4444",  # Red
        "loading": "#f59e0b",  # Yellow
    }
    symbols = {
        "connected": "[ON]",
        "disconnected": "[OFF]",
        "loading": "[...]",
    }
    
    color = colors.get(status, colors["disconnected"])
    symbol = symbols.get(status, symbols["disconnected"])
    
    label = QLabel(f"{symbol} {name}")
    label.setStyleSheet(f"color: {color}; font-weight: bold;")
    label.setProperty("status", status)
    label.setProperty("name", name)
    return label


def _update_status_indicator(label: QLabel, status: str):
    """Update a status indicator."""
    colors = {
        "connected": "#22c55e",
        "disconnected": "#ef4444", 
        "loading": "#f59e0b",
    }
    symbols = {
        "connected": "[ON]",
        "disconnected": "[OFF]",
        "loading": "[...]",
    }
    
    name = label.property("name")
    color = colors.get(status, colors["disconnected"])
    symbol = symbols.get(status, symbols["disconnected"])
    
    label.setText(f"{symbol} {name}")
    label.setStyleSheet(f"color: {color}; font-weight: bold;")
    label.setProperty("status", status)


def _refresh_connections(parent):
    """Check and update AI connection status."""
    try:
        # First check if we have a model assigned via ToolRouter
        active_ai_text = "Active AI: Not configured"
        active_ai_style = "font-weight: bold; font-size: 12px; padding: 5px; background: #1e1e2e; border-radius: 4px; color: #888;"
        
        try:
            from ...core.tool_router import get_router
            router = get_router()
            
            # Use the public get_active_ai() method
            active_ai = router.get_active_ai()
            
            if active_ai:
                model_type = active_ai["model_type"]
                model_name = active_ai["model_name"]
                is_loaded = active_ai["loaded"]
                
                # Parse model type for display
                if model_type == "huggingface":
                    active_ai_text = f"Active AI: [HF] {model_name}"
                    active_ai_style = "font-weight: bold; font-size: 12px; padding: 5px; background: #1e1e2e; border-radius: 4px; color: #f59e0b;"
                    
                    if is_loaded:
                        _update_status_indicator(parent.model_status, "connected")
                        _update_status_indicator(parent.tokenizer_status, "connected")
                        _update_status_indicator(parent.inference_status, "connected")
                    else:
                        _update_status_indicator(parent.model_status, "loading")
                        _update_status_indicator(parent.tokenizer_status, "loading")
                        _update_status_indicator(parent.inference_status, "disconnected")
                        active_ai_text += " (not loaded yet)"
                        
                elif model_type == "forge_ai":
                    active_ai_text = f"Active AI: Forge - {model_name}"
                    active_ai_style = "font-weight: bold; font-size: 12px; padding: 5px; background: #1e1e2e; border-radius: 4px; color: #22c55e;"
                    
                    if is_loaded:
                        _update_status_indicator(parent.model_status, "connected")
                        _update_status_indicator(parent.tokenizer_status, "connected")
                        _update_status_indicator(parent.inference_status, "connected")
                    else:
                        _update_status_indicator(parent.model_status, "loading")
                        _update_status_indicator(parent.tokenizer_status, "loading")
                        _update_status_indicator(parent.inference_status, "disconnected")
                        
                elif model_type == "api":
                    active_ai_text = f"Active AI: [API] {model_name.upper()} API"
                    active_ai_style = "font-weight: bold; font-size: 12px; padding: 5px; background: #1e1e2e; border-radius: 4px; color: #3b82f6;"
                    _update_status_indicator(parent.model_status, "connected")
                    _update_status_indicator(parent.tokenizer_status, "connected")
                    _update_status_indicator(parent.inference_status, "connected")
                    
                elif model_type == "local":
                    active_ai_text = f"Active AI: [Local] {model_name}"
                    active_ai_style = "font-weight: bold; font-size: 12px; padding: 5px; background: #1e1e2e; border-radius: 4px; color: #a855f7;"
                    if is_loaded:
                        _update_status_indicator(parent.model_status, "connected")
                        _update_status_indicator(parent.tokenizer_status, "connected")
                        _update_status_indicator(parent.inference_status, "connected")
                    else:
                        _update_status_indicator(parent.model_status, "loading")
                        _update_status_indicator(parent.tokenizer_status, "loading")
                        _update_status_indicator(parent.inference_status, "disconnected")
                else:
                    active_ai_text = f"Active AI: {active_ai['model_id']}"
                    
        except ImportError:
            pass  # ToolRouter not available, fall back to module check
        except Exception as e:
            print(f"Router check error: {e}")
        
        # Update active AI label
        parent.active_ai_label.setText(active_ai_text)
        parent.active_ai_label.setStyleSheet(active_ai_style)
        
        # Check for module manager (fallback/additional checks)
        main_window = parent.window()
        module_manager = getattr(main_window, 'module_manager', None) if main_window else None
        
        if module_manager:
            loaded = module_manager.list_loaded()
            
            # Only update if not already set by router check
            if parent.model_status.property("status") == "disconnected":
                _update_status_indicator(
                    parent.model_status, 
                    "connected" if "model" in loaded else "disconnected"
                )
            if parent.tokenizer_status.property("status") == "disconnected":
                _update_status_indicator(
                    parent.tokenizer_status,
                    "connected" if "tokenizer" in loaded else "disconnected"
                )
            if parent.inference_status.property("status") == "disconnected":
                _update_status_indicator(
                    parent.inference_status,
                    "connected" if "inference" in loaded else "disconnected"
                )
            _update_status_indicator(
                parent.memory_status,
                "connected" if "memory" in loaded else "disconnected"
            )
        else:
            # Try direct checks for memory
            try:
                from ...memory import ConversationManager
                _update_status_indicator(parent.memory_status, "connected")
            except Exception:
                _update_status_indicator(parent.memory_status, "disconnected")
            
    except Exception as e:
        print(f"Connection check error: {e}")


def _get_cache_path():
    """Get the HuggingFace cache path."""
    import os
    from pathlib import Path
    
    # Check environment variable first
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    
    # Default paths
    home = Path.home()
    
    # Windows
    if os.name == 'nt':
        return home / ".cache" / "huggingface"
    # Unix
    return home / ".cache" / "huggingface"


def _get_folder_size(path):
    """Calculate total size of a folder in bytes."""
    from pathlib import Path
    total = 0
    try:
        for item in Path(path).rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def _format_size(size_bytes):
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _refresh_cache_info(parent):
    """Refresh cache size information."""
    try:
        cache_path = _get_cache_path()
        parent.cache_path_label.setText(f"Location: {cache_path}")
        
        if cache_path.exists():
            size = _get_folder_size(cache_path)
            parent.cache_size_label.setText(f"Cache Size: {_format_size(size)}")
            
            if size > 10 * 1024 * 1024 * 1024:  # > 10GB
                parent.cache_size_label.setStyleSheet("font-weight: bold; color: #f59e0b;")
            elif size > 1 * 1024 * 1024 * 1024:  # > 1GB
                parent.cache_size_label.setStyleSheet("font-weight: bold; color: #22c55e;")
            else:
                parent.cache_size_label.setStyleSheet("font-weight: bold;")
        else:
            parent.cache_size_label.setText("Cache folder not found (no models downloaded yet)")
    except Exception as e:
        parent.cache_size_label.setText(f"Error: {e}")


def _open_cache_folder(parent):
    """Open the cache folder in file explorer."""
    cache_path = _get_cache_path()
    
    if not cache_path.exists():
        QMessageBox.information(
            parent, "No Cache",
            "The cache folder doesn't exist yet.\n"
            "It will be created when you first use a HuggingFace model."
        )
        return
    
    from .output_helpers import open_folder
    open_folder(cache_path)


def _clear_cache(parent):
    """Clear all HuggingFace cache."""
    import shutil
    
    cache_path = _get_cache_path()
    
    if not cache_path.exists():
        QMessageBox.information(parent, "No Cache", "Cache folder is already empty.")
        return
    
    # Get size first
    size = _get_folder_size(cache_path)
    
    reply = QMessageBox.warning(
        parent,
        "Clear Cache?",
        f"This will delete ALL downloaded models ({_format_size(size)}).\n\n"
        f"Location: {cache_path}\n\n"
        "Models will need to be re-downloaded when you use them again.\n\n"
        "Are you sure?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        try:
            # Delete hub folder (where models are stored)
            hub_path = cache_path / "hub"
            if hub_path.exists():
                shutil.rmtree(hub_path)
            
            _refresh_cache_info(parent)
            QMessageBox.information(
                parent, "Cache Cleared",
                f"Successfully freed {_format_size(size)} of disk space."
            )
        except Exception as e:
            QMessageBox.critical(parent, "Error", f"Failed to clear cache: {e}")


def _load_saved_settings(parent):
    """Load saved settings from CONFIG into the UI."""
    from ...config import CONFIG
    
    # Load resource mode
    saved_mode = CONFIG.get("resource_mode", "balanced")
    mode_map = {"minimal": 0, "gaming": 1, "balanced": 2, "performance": 3, "max": 4}
    if saved_mode in mode_map:
        parent.resource_mode_combo.setCurrentIndex(mode_map[saved_mode])


def _apply_theme(parent):
    """Apply selected theme to the application."""
    theme_id = parent.theme_combo.currentData()

    # Theme descriptions
    descriptions = {
        "dark": "Dark theme with soft colors (default)",
        "light": "Light theme for bright environments",
        "high_contrast": "High contrast for accessibility",
        "midnight": "Deep blue midnight theme",
        "forest": "Nature-inspired green theme",
        "sunset": "Warm sunset colors"
    }
    parent.theme_description_label.setText(descriptions.get(theme_id, ""))

    try:
        from ..theme_system import ThemeManager

        # Get or create theme manager
        if not hasattr(parent, 'theme_manager'):
            parent.theme_manager = ThemeManager()

        # Apply theme
        if parent.theme_manager.set_theme(theme_id):
            stylesheet = parent.theme_manager.get_current_stylesheet()
            # Apply to main window
            main_window = parent.window()
            if main_window:
                main_window.setStyleSheet(stylesheet)
    except ImportError:
        QMessageBox.warning(
            parent, "Theme Error",
            "Theme system module (theme_system) not found.\n\n"
            "Please ensure the forge_ai.gui.theme_system module is properly installed."
        )
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to apply theme: {e}")


def _apply_resource_mode(parent):
    """Apply selected resource mode."""
    mode = parent.resource_mode_combo.currentData()
    
    # Update description
    descriptions = {
        "minimal": "Minimal: Uses 1 CPU thread, low priority. Best while gaming!",
        "gaming": "Gaming: AI runs in background, prioritizes gaming performance.",
        "balanced": "Balanced: Uses moderate resources. Good for normal use.",
        "performance": "Performance: Uses more resources for faster AI responses.",
        "max": "Maximum: Uses all available resources. May slow other apps."
    }
    parent.power_mode_details_label.setText(descriptions.get(mode, ""))
    
    # Update spinboxes to match mode presets
    mode_presets = {
        "minimal": (20, 1),
        "gaming": (30, 2),
        "balanced": (50, 4),
        "performance": (70, 6),
        "max": (90, 8)
    }
    if mode in mode_presets:
        gpu, cpu = mode_presets[mode]
        if hasattr(parent, 'gpu_spinbox'):
            parent.gpu_spinbox.setValue(gpu)
        if hasattr(parent, 'cpu_spinbox'):
            parent.cpu_spinbox.setValue(min(cpu, parent.cpu_spinbox.maximum()))


def _apply_custom_resources(parent):
    """Apply custom GPU/CPU resource limits."""
    try:
        from ...config import CONFIG
        
        gpu_percent = parent.gpu_spinbox.value()
        cpu_threads = parent.cpu_spinbox.value()
        
        # Update CONFIG
        CONFIG.set("gpu_memory_fraction", gpu_percent / 100.0)
        CONFIG.set("max_cpu_threads", cpu_threads)
        
        # Also update PyTorch settings if possible
        try:
            import torch
            torch.set_num_threads(cpu_threads)
        except:
            pass
        
        # Save settings
        try:
            from pathlib import Path
            import json
            settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
            
            settings = {}
            if settings_path.exists():
                settings = json.loads(settings_path.read_text())
            
            settings["gpu_memory_fraction"] = gpu_percent / 100.0
            settings["max_cpu_threads"] = cpu_threads
            
            settings_path.write_text(json.dumps(settings, indent=2))
        except Exception as e:
            print(f"Could not save resource settings: {e}")
        
        parent.resource_status_label.setText(
            f"Applied: GPU {gpu_percent}%, CPU {cpu_threads} threads"
        )
        parent.resource_status_label.setStyleSheet("color: #22c55e; font-style: italic;")
        
        QMessageBox.information(
            parent, 
            "Resource Limits Applied",
            f"Custom resource limits set:\n\n"
            f"GPU Memory: {gpu_percent}%\n"
            f"CPU Threads: {cpu_threads}\n\n"
            "Note: Some changes may require restart to fully apply."
        )
    except Exception as e:
        parent.resource_status_label.setText(f"Error: {e}")
        parent.resource_status_label.setStyleSheet("color: #ef4444; font-style: italic;")
        QMessageBox.warning(parent, "Error", f"Failed to apply resource limits: {e}")


def _update_gpu_label(parent, value):
    """Update GPU percentage label - only if gpu_label exists."""
    if hasattr(parent, 'gpu_label'):
        parent.gpu_label.setText(f"{value}%")


def _update_cpu_threads(parent, value):
    """Handle CPU thread change."""
    pass  # Applied when Apply button is clicked


def _update_priority(parent, state):
    """Handle priority checkbox change."""
    pass  # Applied when Apply button is clicked


def _apply_all_settings(parent):
    """Apply all resource settings."""
    try:
        from ...core.power_mode import get_power_manager, PowerLevel
        
        mode = parent.resource_mode_combo.currentData()
        power_mgr = get_power_manager()
        
        # Convert string to PowerLevel enum
        level = PowerLevel(mode)
        power_mgr.set_level(level)
        
        # Update description - match the mode values from resource_mode_combo
        descriptions = {
            "minimal": "Minimal: Uses 1 CPU thread, low priority. Best while gaming!",
            "gaming": "Gaming: AI runs in background, prioritizes gaming performance.",
            "balanced": "Balanced: Moderate resource usage. Good for normal use.",
            "performance": "Performance: Uses more resources for faster AI responses.",
            "max": "Maximum: Uses all available resources. May slow other apps."
        }
        parent.power_mode_details_label.setText(descriptions.get(mode, ""))
        
        # Refresh status
        _refresh_power_status(parent)
        
        QMessageBox.information(parent, "Power Mode Changed", 
            f"Power mode set to: {mode.upper()}\n\n"
            f"Batch size: {power_mgr.settings.max_batch_size}\n"
            f"Max tokens: {power_mgr.settings.max_tokens}\n"
            f"GPU: {'Enabled' if power_mgr.settings.use_gpu else 'Disabled'}"
        )
    except ImportError:
        QMessageBox.warning(parent, "Error", "Power mode manager not available")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to apply power mode: {e}")


def _toggle_autonomous(parent, state):
    """Toggle autonomous mode on/off."""
    try:
        from ...core.autonomous import AutonomousManager
        
        # Get current model name
        model_name = getattr(parent, 'current_model_name', 'forge_ai')
        
        # Check if using HuggingFace model
        main_window = parent.window()
        is_hf = False
        if main_window and hasattr(main_window, 'engine'):
            is_hf = getattr(main_window.engine, '_is_huggingface', False)
        
        autonomous = AutonomousManager.get(model_name)
        
        if state == Checked:
            # Warn about HF model limitations
            if is_hf:
                QMessageBox.warning(parent, "Limited Functionality",
                    "Autonomous mode has limited functionality with HuggingFace models.\n\n"
                    "What works:\n"
                    "• Web browsing and learning\n"
                    "• Personality evolution\n"
                    "• Curiosity exploration\n\n"
                    "What doesn't work:\n"
                    "• Model self-improvement (HF models are read-only)\n"
                    "• Response practice (requires training)\n\n"
                    "For full autonomous features, use a local Forge model."
                )
            
            # Set activity level
            max_actions = parent.autonomous_activity_spin.value()
            autonomous.max_actions_per_hour = max_actions
            
            # Start autonomous mode
            autonomous.start()
            parent.autonomous_activity_spin.setEnabled(True)
            
            QMessageBox.information(parent, "Autonomous Mode", 
                "Autonomous mode enabled!\n\n"
                "AI will explore topics, learn, and evolve on its own.\n"
                "You can disable this at any time."
            )
        else:
            # Stop autonomous mode
            autonomous.stop()
            parent.autonomous_activity_spin.setEnabled(False)
            
    except ImportError:
        QMessageBox.warning(parent, "Error", "Autonomous mode not available")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to toggle autonomous mode: {e}")


def _refresh_power_status(parent):
    """Refresh power status display."""
    try:
        from ...core.power_mode import get_power_manager
        import torch
        
        power_mgr = get_power_manager()
        
        status_text = f"""Power Mode: {power_mgr.level.value.upper()}

Settings:
  Max Batch Size: {power_mgr.settings.max_batch_size}
  Max Tokens: {power_mgr.settings.max_tokens}
  GPU Enabled: {'Yes' if power_mgr.settings.use_gpu else 'No'}
  Thread Count: {power_mgr.settings.thread_count if power_mgr.settings.thread_count > 0 else 'Auto'}
  Response Delay: {power_mgr.settings.response_delay}s
  Paused: {'Yes' if power_mgr.is_paused else 'No'}

System:
  PyTorch Threads: {torch.get_num_threads()}"""
        
        if torch.cuda.is_available():
            status_text += f"""
  GPU Available: Yes
  GPU Name: {torch.cuda.get_device_name(0)}"""
        else:
            status_text += """
  GPU Available: No"""
        
        parent.power_status.setPlainText(status_text)
        
    except ImportError:
        parent.power_status.setPlainText(
            "Power mode manager not available.\n"
            "Make sure forge_ai.core.power_mode module exists."
        )
    except Exception as e:
        parent.power_status.setPlainText(f"Error getting status: {e}")


def _apply_zoom(parent, value: int):
    """Apply zoom level to the application."""
    try:
        from PyQt5.QtWidgets import QApplication, QWidget
        from PyQt5.QtGui import QFont
        from typing import cast
        
        app = QApplication.instance()
        if app is None:
            return
            
        app = cast(QApplication, app)
        
        # Calculate font size based on zoom (base size is ~10pt at 100%)
        base_size = 10
        new_size = max(6, int(base_size * value / 100))
        
        # Create and apply new font
        font = QFont()
        font.setPointSize(new_size)
        app.setFont(font)
        
        # Force all widgets to update with new font
        main_window = parent.window()
        if main_window:
            # Recursively update all child widgets
            for widget in main_window.findChildren(QWidget):
                widget.setFont(font)
                widget.update()
            
            # Force a repaint
            main_window.update()
            main_window.repaint()
            
        # Store zoom value for persistence
        parent._current_zoom = value
                    
    except Exception as e:
        print(f"Zoom error: {e}")


def _reset_zoom(parent):
    """Reset zoom to 100%."""
    parent.zoom_spinbox.setValue(100)


def _apply_personality_preset(parent):
    """Apply personality preset from settings."""
    preset = parent.settings_personality_combo.currentData()
    if not preset:
        return
    
    # Define preset values
    presets = {
        "balanced": {"formality": 50, "verbosity": 50, "humor_level": 50, "creativity": 50},
        "professional": {"formality": 80, "verbosity": 60, "humor_level": 20, "creativity": 40},
        "friendly": {"formality": 30, "verbosity": 50, "humor_level": 60, "creativity": 50},
        "creative": {"formality": 40, "verbosity": 60, "humor_level": 50, "creativity": 85},
        "analytical": {"formality": 70, "verbosity": 70, "humor_level": 15, "creativity": 30},
        "teacher": {"formality": 50, "verbosity": 75, "humor_level": 40, "creativity": 60},
    }
    
    values = presets.get(preset, presets["balanced"])
    
    # Update sliders
    parent.formality_slider.blockSignals(True)
    parent.formality_slider.setValue(values["formality"])
    parent.formality_slider.blockSignals(False)
    
    parent.verbosity_slider.blockSignals(True)
    parent.verbosity_slider.setValue(values["verbosity"])
    parent.verbosity_slider.blockSignals(False)
    
    parent.humor_slider.blockSignals(True)
    parent.humor_slider.setValue(values["humor_level"])
    parent.humor_slider.blockSignals(False)
    
    parent.creativity_slider.blockSignals(True)
    parent.creativity_slider.setValue(values["creativity"])
    parent.creativity_slider.blockSignals(False)
    
    # Apply to personality system if available
    try:
        from ...core.personality import AIPersonality
        model_name = getattr(parent, 'current_model_name', 'forge_ai')
        personality = AIPersonality(model_name)
        personality.set_preset(preset)
        personality.save()
    except Exception:
        pass  # Personality system may not be available


def _update_personality_trait(parent, trait_key: str, value: int):
    """Update a single personality trait."""
    try:
        from ...core.personality import AIPersonality
        model_name = getattr(parent, 'current_model_name', 'forge_ai')
        personality = AIPersonality(model_name)
        personality.load()
        
        # Set trait value (convert from 0-100 to 0.0-1.0)
        trait_value = value / 100.0
        if hasattr(personality.traits, trait_key):
            setattr(personality.traits, trait_key, trait_value)
            personality.save()
    except Exception:
        pass  # Personality system may not be available


def _toggle_personality_evolution(parent, state):
    """Toggle personality evolution on/off."""
    try:
        from ...core.personality import AIPersonality
        model_name = getattr(parent, 'current_model_name', 'forge_ai')
        personality = AIPersonality(model_name)
        personality.load()
        personality.allow_evolution = (state == Checked)
        personality.save()
    except Exception:
        pass  # Personality system may not be available


# =============================================================================
# Audio Device Functions
# =============================================================================

def _get_audio_devices():
    """Get list of audio input and output devices."""
    input_devices = [("Default", -1)]
    output_devices = [("Default", -1)]
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = info.get('name', f'Device {i}')
            
            # Truncate long names
            if len(name) > 40:
                name = name[:37] + "..."
            
            if info.get('maxInputChannels', 0) > 0:
                input_devices.append((name, i))
            if info.get('maxOutputChannels', 0) > 0:
                output_devices.append((name, i))
        
        p.terminate()
    except (ImportError, OSError, TypeError) as e:
        # pyaudio not installed, or audio system issue (e.g., portaudio ctypes callback)
        print(f"[Audio] Could not enumerate devices: {type(e).__name__}")
    except Exception as e:
        print(f"Error getting audio devices: {e}")
    
    return input_devices, output_devices


def _refresh_audio_devices(parent):
    """Refresh the audio device dropdowns."""
    input_devices, output_devices = _get_audio_devices()
    
    # Save current selections
    current_input = parent.audio_input_combo.currentData() if parent.audio_input_combo.count() > 0 else -1
    current_output = parent.audio_output_combo.currentData() if parent.audio_output_combo.count() > 0 else -1
    
    # Update input devices
    parent.audio_input_combo.clear()
    for name, idx in input_devices:
        parent.audio_input_combo.addItem(name, idx)
    
    # Restore selection or default
    for i in range(parent.audio_input_combo.count()):
        if parent.audio_input_combo.itemData(i) == current_input:
            parent.audio_input_combo.setCurrentIndex(i)
            break
    
    # Update output devices
    parent.audio_output_combo.clear()
    for name, idx in output_devices:
        parent.audio_output_combo.addItem(name, idx)
    
    # Restore selection or default
    for i in range(parent.audio_output_combo.count()):
        if parent.audio_output_combo.itemData(i) == current_output:
            parent.audio_output_combo.setCurrentIndex(i)
            break


def _test_microphone(parent):
    """Test the selected microphone and show audio levels."""
    import threading
    
    # If already testing, stop the test
    if hasattr(parent, '_mic_test_running') and parent._mic_test_running:
        parent._mic_test_stop_requested = True
        parent.mic_test_btn.setText("Stopping...")
        return
    
    device_index = parent.audio_input_combo.currentData()
    if device_index is None:
        device_index = -1
    
    parent._mic_test_running = True
    parent._mic_test_stop_requested = False
    parent.mic_test_btn.setEnabled(True)
    parent.mic_test_btn.setText("Stop Test")
    parent.mic_status_label.setText("Listening... (click Stop to end)")
    parent.mic_status_label.setStyleSheet("color: #f59e0b;")
    
    def run_test():
        try:
            import pyaudio
            import struct
            import math
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            DURATION = 3  # seconds
            
            p = pyaudio.PyAudio()
            
            # Open stream
            stream_kwargs = {
                'format': FORMAT,
                'channels': CHANNELS,
                'rate': RATE,
                'input': True,
                'frames_per_buffer': CHUNK,
            }
            if device_index >= 0:
                stream_kwargs['input_device_index'] = device_index
            
            stream = p.open(**stream_kwargs)
            
            max_level = 0
            frames_read = 0
            total_frames = int(RATE / CHUNK * DURATION)
            
            while frames_read < total_frames and not getattr(parent, '_mic_test_stop_requested', False):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # Calculate RMS level
                    count = len(data) // 2
                    shorts = struct.unpack(f'{count}h', data)
                    sum_squares = sum(s * s for s in shorts)
                    rms = math.sqrt(sum_squares / count) if count > 0 else 0
                    
                    # Convert to percentage (0-100)
                    level = min(100, int(rms / 32768 * 500))
                    max_level = max(max_level, level)
                    
                    # Update UI from main thread
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        parent.mic_level_bar, "setValue",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(int, level)
                    )
                    
                    frames_read += 1
                except Exception:
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Update final status
            def update_ui():
                parent._mic_test_running = False
                parent._mic_test_stop_requested = False
                parent.mic_test_btn.setEnabled(True)
                parent.mic_test_btn.setText("Test Microphone")
                if getattr(parent, '_mic_test_stop_requested', False):
                    parent.mic_status_label.setText("Test stopped")
                    parent.mic_status_label.setStyleSheet("color: #888;")
                elif max_level > 10:
                    parent.mic_status_label.setText(f"Working (peak: {max_level}%)")
                    parent.mic_status_label.setStyleSheet("color: #22c55e;")
                else:
                    parent.mic_status_label.setText("Low/no signal detected")
                    parent.mic_status_label.setStyleSheet("color: #f59e0b;")
            
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, update_ui)
            
        except (ImportError, OSError, TypeError) as e:
            def show_error():
                parent._mic_test_running = False
                parent._mic_test_stop_requested = False
                parent.mic_test_btn.setEnabled(True)
                parent.mic_test_btn.setText("Test Microphone")
                if isinstance(e, ImportError):
                    parent.mic_status_label.setText("Install: pip install pyaudio")
                else:
                    parent.mic_status_label.setText("Audio system unavailable")
                parent.mic_status_label.setStyleSheet("color: #ef4444;")
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, show_error)
            
        except Exception as e:
            def show_error():
                parent._mic_test_running = False
                parent._mic_test_stop_requested = False
                parent.mic_test_btn.setEnabled(True)
                parent.mic_test_btn.setText("Test Microphone")
                parent.mic_status_label.setText(f"Error: {str(e)[:30]}")
                parent.mic_status_label.setStyleSheet("color: #ef4444;")
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, show_error)
    
    thread = threading.Thread(target=run_test, daemon=True)
    thread.start()

