"""
Internationalization (i18n) for Enigma AI Engine

Multi-language support for the UI.

Features:
- Translation loading
- Locale detection
- Plural forms
- RTL support
- String interpolation

Usage:
    from enigma_engine.gui.i18n import I18n, get_i18n
    
    i18n = get_i18n()
    i18n.load_locale("es")
    
    # Translate
    text = i18n.t("hello")  # "Hola"
    
    # With parameters
    text = i18n.t("greeting", name="User")  # "Hola, User"
"""

import json
import locale
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# RTL (right-to-left) languages
RTL_LANGUAGES = {'ar', 'he', 'fa', 'ur', 'yi'}


@dataclass
class LocaleInfo:
    """Locale information."""
    code: str  # e.g., "en", "es", "zh-CN"
    name: str  # e.g., "English"
    native_name: str  # e.g., "Español"
    direction: str = "ltr"  # "ltr" or "rtl"
    
    # Formatting
    date_format: str = "YYYY-MM-DD"
    time_format: str = "HH:MM:SS"
    decimal_separator: str = "."
    thousands_separator: str = ","


# Built-in locale info
LOCALE_INFO: Dict[str, LocaleInfo] = {
    "en": LocaleInfo("en", "English", "English"),
    "es": LocaleInfo("es", "Spanish", "Español"),
    "fr": LocaleInfo("fr", "French", "Français"),
    "de": LocaleInfo("de", "German", "Deutsch"),
    "it": LocaleInfo("it", "Italian", "Italiano"),
    "pt": LocaleInfo("pt", "Portuguese", "Português"),
    "ru": LocaleInfo("ru", "Russian", "Русский"),
    "zh": LocaleInfo("zh", "Chinese", "中文"),
    "zh-CN": LocaleInfo("zh-CN", "Chinese (Simplified)", "简体中文"),
    "zh-TW": LocaleInfo("zh-TW", "Chinese (Traditional)", "繁體中文"),
    "ja": LocaleInfo("ja", "Japanese", "日本語"),
    "ko": LocaleInfo("ko", "Korean", "한국어"),
    "ar": LocaleInfo("ar", "Arabic", "العربية", direction="rtl"),
    "he": LocaleInfo("he", "Hebrew", "עברית", direction="rtl"),
    "hi": LocaleInfo("hi", "Hindi", "हिन्दी"),
    "th": LocaleInfo("th", "Thai", "ไทย"),
    "vi": LocaleInfo("vi", "Vietnamese", "Tiếng Việt"),
    "nl": LocaleInfo("nl", "Dutch", "Nederlands"),
    "pl": LocaleInfo("pl", "Polish", "Polski"),
    "tr": LocaleInfo("tr", "Turkish", "Türkçe"),
}


# Base English translations
BASE_TRANSLATIONS = {
    # General
    "app_name": "Enigma AI Engine",
    "ok": "OK",
    "cancel": "Cancel",
    "save": "Save",
    "load": "Load",
    "delete": "Delete",
    "close": "Close",
    "yes": "Yes",
    "no": "No",
    "error": "Error",
    "warning": "Warning",
    "info": "Info",
    "success": "Success",
    
    # Chat
    "send": "Send",
    "message_placeholder": "Type a message...",
    "thinking": "Thinking...",
    "generating": "Generating...",
    "copy": "Copy",
    "regenerate": "Regenerate",
    
    # Models
    "model": "Model",
    "models": "Models",
    "load_model": "Load Model",
    "unload_model": "Unload Model",
    "model_loaded": "Model loaded: {name}",
    "model_unloaded": "Model unloaded",
    "no_model": "No model loaded",
    
    # Training
    "train": "Train",
    "training": "Training",
    "epoch": "Epoch",
    "loss": "Loss",
    "training_complete": "Training complete",
    "training_failed": "Training failed",
    
    # Settings
    "settings": "Settings",
    "language": "Language",
    "theme": "Theme",
    "font_size": "Font Size",
    "save_settings": "Save Settings",
    
    # Tabs
    "chat": "Chat",
    "build_ai": "Build AI",
    "memory": "Memory",
    "tools": "Tools",
    "modules": "Modules",
    "avatar": "Avatar",
    "voice": "Voice",
    
    # Files
    "open_file": "Open File",
    "save_file": "Save File",
    "file_saved": "File saved: {path}",
    "file_not_found": "File not found: {path}",
    
    # Plurals (with count)
    "item": "{count} item",
    "items": "{count} items",
    "message": "{count} message",
    "messages": "{count} messages",
    "token": "{count} token",
    "tokens": "{count} tokens",
}


class I18n:
    """Internationalization manager."""
    
    def __init__(self, locale_dir: Optional[str] = None):
        """
        Initialize i18n.
        
        Args:
            locale_dir: Directory containing locale files
        """
        self._locale_dir = Path(locale_dir) if locale_dir else Path("locales")
        self._current_locale = "en"
        self._translations: Dict[str, str] = dict(BASE_TRANSLATIONS)
        self._fallback_translations = dict(BASE_TRANSLATIONS)
        
        # Callbacks for locale change
        self._callbacks: List[Callable[[str], None]] = []
        
        # Try to detect system locale
        self._detect_system_locale()
        
        logger.info(f"I18n initialized with locale: {self._current_locale}")
    
    def _detect_system_locale(self):
        """Detect system locale."""
        try:
            # Get system locale
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                # Extract language code
                lang = system_locale.split('_')[0]
                if lang in LOCALE_INFO:
                    self._current_locale = lang
        except:
            pass
    
    @property
    def locale(self) -> str:
        """Get current locale code."""
        return self._current_locale
    
    @property
    def locale_info(self) -> LocaleInfo:
        """Get current locale info."""
        return LOCALE_INFO.get(self._current_locale, LOCALE_INFO['en'])
    
    @property
    def is_rtl(self) -> bool:
        """Check if current locale is RTL."""
        return self._current_locale[:2] in RTL_LANGUAGES
    
    def load_locale(self, locale_code: str) -> bool:
        """
        Load a locale.
        
        Args:
            locale_code: Locale code (e.g., "es", "zh-CN")
            
        Returns:
            True if loaded successfully
        """
        # Try exact match first
        if not self._load_locale_file(locale_code):
            # Try base language
            base_lang = locale_code.split('-')[0]
            if not self._load_locale_file(base_lang):
                logger.warning(f"Locale not found: {locale_code}")
                return False
        
        self._current_locale = locale_code
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(locale_code)
            except Exception as e:
                logger.error(f"Locale change callback error: {e}")
        
        logger.info(f"Loaded locale: {locale_code}")
        return True
    
    def _load_locale_file(self, locale_code: str) -> bool:
        """Load translations from file."""
        locale_file = self._locale_dir / f"{locale_code}.json"
        
        if not locale_file.exists():
            return False
        
        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            # Merge with base translations (fallback)
            self._translations = dict(self._fallback_translations)
            self._translations.update(translations)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading locale file: {e}")
            return False
    
    def t(self, key: str, **kwargs) -> str:
        """
        Translate a key.
        
        Args:
            key: Translation key
            **kwargs: Interpolation parameters
            
        Returns:
            Translated string
        """
        text = self._translations.get(key, self._fallback_translations.get(key, key))
        
        # Interpolate parameters
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
        
        return text
    
    def tp(self, singular_key: str, plural_key: str, count: int, **kwargs) -> str:
        """
        Translate with pluralization.
        
        Args:
            singular_key: Key for singular form
            plural_key: Key for plural form
            count: Count for pluralization
            **kwargs: Additional parameters
            
        Returns:
            Translated string
        """
        key = singular_key if count == 1 else plural_key
        return self.t(key, count=count, **kwargs)
    
    def get_available_locales(self) -> List[str]:
        """Get list of available locale codes."""
        locales = ['en']  # Always available
        
        if self._locale_dir.exists():
            for f in self._locale_dir.glob("*.json"):
                locales.append(f.stem)
        
        return sorted(set(locales))
    
    def get_locale_display_name(self, locale_code: str) -> str:
        """Get display name for locale."""
        info = LOCALE_INFO.get(locale_code)
        if info:
            return f"{info.native_name} ({info.name})"
        return locale_code
    
    def on_locale_change(self, callback: Callable[[str], None]):
        """Register callback for locale changes."""
        self._callbacks.append(callback)
    
    def create_locale_file(self, locale_code: str, translations: Dict[str, str]):
        """
        Create a new locale file.
        
        Args:
            locale_code: Locale code
            translations: Dictionary of translations
        """
        self._locale_dir.mkdir(parents=True, exist_ok=True)
        
        locale_file = self._locale_dir / f"{locale_code}.json"
        
        with open(locale_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created locale file: {locale_file}")
    
    def export_template(self, output_path: str):
        """
        Export translation template.
        
        Args:
            output_path: Path to save template
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(BASE_TRANSLATIONS, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported translation template to {output_path}")


class TranslationHelper:
    """Helper for generating translations."""
    
    @staticmethod
    def generate_stubs(
        base_translations: Dict[str, str],
        target_locale: str
    ) -> Dict[str, str]:
        """
        Generate translation stubs with markers.
        
        Args:
            base_translations: Base English translations
            target_locale: Target locale code
            
        Returns:
            Dictionary with stub translations
        """
        stubs = {}
        for key, value in base_translations.items():
            stubs[key] = f"[{target_locale}] {value}"
        return stubs
    
    @staticmethod
    def validate_translations(
        translations: Dict[str, str],
        base_translations: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Validate translations against base.
        
        Returns:
            Dict with 'missing', 'extra', and 'invalid' keys
        """
        result = {
            'missing': [],
            'extra': [],
            'invalid': []
        }
        
        base_keys = set(base_translations.keys())
        trans_keys = set(translations.keys())
        
        result['missing'] = list(base_keys - trans_keys)
        result['extra'] = list(trans_keys - base_keys)
        
        # Check for invalid placeholders
        for key in base_keys & trans_keys:
            base_placeholders = set(
                p for p in base_translations[key].split('{')
                if '}' in p
            )
            trans_placeholders = set(
                p for p in translations[key].split('{')
                if '}' in p
            )
            
            if base_placeholders != trans_placeholders:
                result['invalid'].append(key)
        
        return result


# Global instance
_i18n: Optional[I18n] = None


def get_i18n(locale_dir: Optional[str] = None) -> I18n:
    """Get or create global i18n instance."""
    global _i18n
    if _i18n is None:
        _i18n = I18n(locale_dir)
    return _i18n


# Convenience function
def t(key: str, **kwargs) -> str:
    """Translate a key using global i18n."""
    return get_i18n().t(key, **kwargs)
