"""
Internationalization (i18n) and Localization (L10n) System

Provides:
- UI translations for multiple languages
- RTL layout support
- Locale-aware formatting (dates, numbers, currencies)
- Dynamic language switching
"""

import json
import locale
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


class TextDirection(Enum):
    """Text direction for languages"""
    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left


@dataclass
class LanguageInfo:
    """Information about a supported language"""
    code: str  # ISO 639-1 code (e.g., "en", "ja", "ar")
    name: str  # English name
    native_name: str  # Name in native language
    direction: TextDirection = TextDirection.LTR
    locale_code: str = ""  # Full locale (e.g., "en_US", "ja_JP")
    font_family: Optional[str] = None  # Preferred font for this language


# Supported languages
LANGUAGES: dict[str, LanguageInfo] = {
    "en": LanguageInfo("en", "English", "English", TextDirection.LTR, "en_US"),
    "es": LanguageInfo("es", "Spanish", "Español", TextDirection.LTR, "es_ES"),
    "fr": LanguageInfo("fr", "French", "Français", TextDirection.LTR, "fr_FR"),
    "de": LanguageInfo("de", "German", "Deutsch", TextDirection.LTR, "de_DE"),
    "it": LanguageInfo("it", "Italian", "Italiano", TextDirection.LTR, "it_IT"),
    "pt": LanguageInfo("pt", "Portuguese", "Português", TextDirection.LTR, "pt_BR"),
    "ru": LanguageInfo("ru", "Russian", "Русский", TextDirection.LTR, "ru_RU"),
    "zh": LanguageInfo("zh", "Chinese (Simplified)", "简体中文", TextDirection.LTR, "zh_CN", "Noto Sans SC"),
    "zh_TW": LanguageInfo("zh_TW", "Chinese (Traditional)", "繁體中文", TextDirection.LTR, "zh_TW", "Noto Sans TC"),
    "ja": LanguageInfo("ja", "Japanese", "日本語", TextDirection.LTR, "ja_JP", "Noto Sans JP"),
    "ko": LanguageInfo("ko", "Korean", "한국어", TextDirection.LTR, "ko_KR", "Noto Sans KR"),
    "ar": LanguageInfo("ar", "Arabic", "العربية", TextDirection.RTL, "ar_SA", "Noto Sans Arabic"),
    "he": LanguageInfo("he", "Hebrew", "עברית", TextDirection.RTL, "he_IL", "Noto Sans Hebrew"),
    "fa": LanguageInfo("fa", "Persian", "فارسی", TextDirection.RTL, "fa_IR", "Noto Sans Arabic"),
    "hi": LanguageInfo("hi", "Hindi", "हिन्दी", TextDirection.LTR, "hi_IN", "Noto Sans Devanagari"),
    "th": LanguageInfo("th", "Thai", "ไทย", TextDirection.LTR, "th_TH", "Noto Sans Thai"),
    "vi": LanguageInfo("vi", "Vietnamese", "Tiếng Việt", TextDirection.LTR, "vi_VN"),
    "tr": LanguageInfo("tr", "Turkish", "Türkçe", TextDirection.LTR, "tr_TR"),
    "pl": LanguageInfo("pl", "Polish", "Polski", TextDirection.LTR, "pl_PL"),
    "nl": LanguageInfo("nl", "Dutch", "Nederlands", TextDirection.LTR, "nl_NL"),
    "uk": LanguageInfo("uk", "Ukrainian", "Українська", TextDirection.LTR, "uk_UA"),
}


# Default English translations (base strings)
DEFAULT_TRANSLATIONS = {
    # Main Window
    "app.title": "Enigma AI Engine",
    "app.subtitle": "AI Assistant",
    
    # Menu
    "menu.file": "File",
    "menu.file.new": "New Chat",
    "menu.file.open": "Open",
    "menu.file.save": "Save",
    "menu.file.save_as": "Save As...",
    "menu.file.export": "Export",
    "menu.file.import": "Import",
    "menu.file.settings": "Settings",
    "menu.file.exit": "Exit",
    
    "menu.edit": "Edit",
    "menu.edit.undo": "Undo",
    "menu.edit.redo": "Redo",
    "menu.edit.cut": "Cut",
    "menu.edit.copy": "Copy",
    "menu.edit.paste": "Paste",
    "menu.edit.select_all": "Select All",
    "menu.edit.clear": "Clear",
    
    "menu.view": "View",
    "menu.view.zoom_in": "Zoom In",
    "menu.view.zoom_out": "Zoom Out",
    "menu.view.fullscreen": "Fullscreen",
    "menu.view.sidebar": "Show Sidebar",
    "menu.view.dark_mode": "Dark Mode",
    
    "menu.tools": "Tools",
    "menu.tools.modules": "Module Manager",
    "menu.tools.models": "Model Manager",
    "menu.tools.plugins": "Plugins",
    
    "menu.help": "Help",
    "menu.help.about": "About",
    "menu.help.docs": "Documentation",
    "menu.help.shortcuts": "Keyboard Shortcuts",
    "menu.help.check_updates": "Check for Updates",
    
    # Chat
    "chat.input_placeholder": "Type a message...",
    "chat.send": "Send",
    "chat.stop": "Stop",
    "chat.regenerate": "Regenerate",
    "chat.copy": "Copy",
    "chat.edit": "Edit",
    "chat.delete": "Delete",
    "chat.branch": "Branch",
    "chat.thinking": "Thinking...",
    "chat.typing": "AI is typing...",
    "chat.new_chat": "New Chat",
    "chat.clear_history": "Clear History",
    "chat.export_chat": "Export Chat",
    
    # Settings
    "settings.title": "Settings",
    "settings.general": "General",
    "settings.appearance": "Appearance",
    "settings.language": "Language",
    "settings.theme": "Theme",
    "settings.font_size": "Font Size",
    "settings.voice": "Voice",
    "settings.privacy": "Privacy",
    "settings.advanced": "Advanced",
    "settings.save": "Save",
    "settings.cancel": "Cancel",
    "settings.reset": "Reset to Defaults",
    
    # Modules
    "modules.title": "Module Manager",
    "modules.search": "Search modules...",
    "modules.enabled": "Enabled",
    "modules.disabled": "Disabled",
    "modules.install": "Install",
    "modules.uninstall": "Uninstall",
    "modules.update": "Update",
    "modules.configure": "Configure",
    "modules.dependencies": "Dependencies",
    "modules.conflicts": "Conflicts",
    
    # Models
    "models.title": "Model Manager",
    "models.local": "Local Models",
    "models.download": "Download",
    "models.delete": "Delete",
    "models.size": "Size",
    "models.parameters": "Parameters",
    "models.quantization": "Quantization",
    "models.select": "Select Model",
    
    # Generation Tabs
    "tab.chat": "Chat",
    "tab.image": "Image",
    "tab.code": "Code",
    "tab.video": "Video",
    "tab.audio": "Audio",
    "tab.3d": "3D",
    "tab.embeddings": "Embeddings",
    "tab.training": "Training",
    
    # Image Generation
    "image.prompt": "Image prompt...",
    "image.negative_prompt": "Negative prompt...",
    "image.generate": "Generate",
    "image.width": "Width",
    "image.height": "Height",
    "image.steps": "Steps",
    "image.guidance": "Guidance Scale",
    "image.seed": "Seed",
    "image.save": "Save Image",
    
    # Code Generation
    "code.prompt": "Describe the code you want...",
    "code.generate": "Generate Code",
    "code.language": "Language",
    "code.copy": "Copy Code",
    "code.run": "Run",
    "code.explain": "Explain",
    
    # Audio/Voice
    "voice.speak": "Speak",
    "voice.listen": "Listen",
    "voice.mute": "Mute",
    "voice.unmute": "Unmute",
    "voice.voice_input": "Voice Input",
    "voice.text_to_speech": "Text to Speech",
    
    # Training
    "training.start": "Start Training",
    "training.stop": "Stop Training",
    "training.pause": "Pause",
    "training.resume": "Resume",
    "training.epochs": "Epochs",
    "training.batch_size": "Batch Size",
    "training.learning_rate": "Learning Rate",
    "training.progress": "Progress",
    "training.loss": "Loss",
    
    # Common Actions
    "action.ok": "OK",
    "action.cancel": "Cancel",
    "action.yes": "Yes",
    "action.no": "No",
    "action.apply": "Apply",
    "action.close": "Close",
    "action.refresh": "Refresh",
    "action.search": "Search",
    "action.filter": "Filter",
    "action.sort": "Sort",
    "action.loading": "Loading...",
    "action.error": "Error",
    "action.success": "Success",
    "action.warning": "Warning",
    
    # Status Messages
    "status.connected": "Connected",
    "status.disconnected": "Disconnected",
    "status.connecting": "Connecting...",
    "status.ready": "Ready",
    "status.busy": "Busy",
    "status.offline": "Offline",
    "status.online": "Online",
    
    # Errors
    "error.generic": "An error occurred",
    "error.network": "Network error",
    "error.timeout": "Request timed out",
    "error.auth": "Authentication failed",
    "error.permission": "Permission denied",
    "error.not_found": "Not found",
    "error.invalid_input": "Invalid input",
    
    # Time
    "time.just_now": "Just now",
    "time.minutes_ago": "{n} minutes ago",
    "time.hours_ago": "{n} hours ago",
    "time.days_ago": "{n} days ago",
    "time.yesterday": "Yesterday",
    "time.today": "Today",
}


# Additional translations for other languages
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": DEFAULT_TRANSLATIONS,
    
    "es": {
        "app.title": "Enigma AI Engine",
        "app.subtitle": "Asistente de IA",
        "menu.file": "Archivo",
        "menu.file.new": "Nueva conversación",
        "menu.file.open": "Abrir",
        "menu.file.save": "Guardar",
        "menu.file.settings": "Configuración",
        "menu.file.exit": "Salir",
        "menu.edit": "Editar",
        "menu.view": "Ver",
        "menu.tools": "Herramientas",
        "menu.help": "Ayuda",
        "chat.input_placeholder": "Escribe un mensaje...",
        "chat.send": "Enviar",
        "chat.stop": "Detener",
        "chat.thinking": "Pensando...",
        "settings.title": "Configuración",
        "settings.language": "Idioma",
        "action.ok": "Aceptar",
        "action.cancel": "Cancelar",
        "action.loading": "Cargando...",
        "error.generic": "Ha ocurrido un error",
    },
    
    "fr": {
        "app.subtitle": "Assistant IA",
        "menu.file": "Fichier",
        "menu.file.new": "Nouvelle conversation",
        "menu.file.open": "Ouvrir",
        "menu.file.save": "Enregistrer",
        "menu.file.settings": "Paramètres",
        "menu.file.exit": "Quitter",
        "menu.edit": "Édition",
        "menu.view": "Affichage",
        "menu.tools": "Outils",
        "menu.help": "Aide",
        "chat.input_placeholder": "Tapez un message...",
        "chat.send": "Envoyer",
        "chat.stop": "Arrêter",
        "chat.thinking": "Réflexion...",
        "settings.title": "Paramètres",
        "settings.language": "Langue",
        "action.ok": "OK",
        "action.cancel": "Annuler",
        "action.loading": "Chargement...",
        "error.generic": "Une erreur s'est produite",
    },
    
    "de": {
        "app.subtitle": "KI-Assistent",
        "menu.file": "Datei",
        "menu.file.new": "Neuer Chat",
        "menu.file.open": "Öffnen",
        "menu.file.save": "Speichern",
        "menu.file.settings": "Einstellungen",
        "menu.file.exit": "Beenden",
        "menu.edit": "Bearbeiten",
        "menu.view": "Ansicht",
        "menu.tools": "Werkzeuge",
        "menu.help": "Hilfe",
        "chat.input_placeholder": "Nachricht eingeben...",
        "chat.send": "Senden",
        "chat.stop": "Stoppen",
        "chat.thinking": "Überlege...",
        "settings.title": "Einstellungen",
        "settings.language": "Sprache",
        "action.ok": "OK",
        "action.cancel": "Abbrechen",
        "action.loading": "Laden...",
        "error.generic": "Ein Fehler ist aufgetreten",
    },
    
    "ja": {
        "app.subtitle": "AIアシスタント",
        "menu.file": "ファイル",
        "menu.file.new": "新しいチャット",
        "menu.file.open": "開く",
        "menu.file.save": "保存",
        "menu.file.settings": "設定",
        "menu.file.exit": "終了",
        "menu.edit": "編集",
        "menu.view": "表示",
        "menu.tools": "ツール",
        "menu.help": "ヘルプ",
        "chat.input_placeholder": "メッセージを入力...",
        "chat.send": "送信",
        "chat.stop": "停止",
        "chat.thinking": "考え中...",
        "settings.title": "設定",
        "settings.language": "言語",
        "action.ok": "OK",
        "action.cancel": "キャンセル",
        "action.loading": "読み込み中...",
        "error.generic": "エラーが発生しました",
    },
    
    "zh": {
        "app.subtitle": "AI助手",
        "menu.file": "文件",
        "menu.file.new": "新对话",
        "menu.file.open": "打开",
        "menu.file.save": "保存",
        "menu.file.settings": "设置",
        "menu.file.exit": "退出",
        "menu.edit": "编辑",
        "menu.view": "视图",
        "menu.tools": "工具",
        "menu.help": "帮助",
        "chat.input_placeholder": "输入消息...",
        "chat.send": "发送",
        "chat.stop": "停止",
        "chat.thinking": "思考中...",
        "settings.title": "设置",
        "settings.language": "语言",
        "action.ok": "确定",
        "action.cancel": "取消",
        "action.loading": "加载中...",
        "error.generic": "发生错误",
    },
    
    "ko": {
        "app.subtitle": "AI 어시스턴트",
        "menu.file": "파일",
        "menu.file.new": "새 대화",
        "menu.file.open": "열기",
        "menu.file.save": "저장",
        "menu.file.settings": "설정",
        "menu.file.exit": "종료",
        "menu.edit": "편집",
        "menu.view": "보기",
        "menu.tools": "도구",
        "menu.help": "도움말",
        "chat.input_placeholder": "메시지를 입력하세요...",
        "chat.send": "보내기",
        "chat.stop": "중지",
        "chat.thinking": "생각 중...",
        "settings.title": "설정",
        "settings.language": "언어",
        "action.ok": "확인",
        "action.cancel": "취소",
        "action.loading": "로딩 중...",
        "error.generic": "오류가 발생했습니다",
    },
    
    "ar": {
        "app.subtitle": "مساعد الذكاء الاصطناعي",
        "menu.file": "ملف",
        "menu.file.new": "محادثة جديدة",
        "menu.file.open": "فتح",
        "menu.file.save": "حفظ",
        "menu.file.settings": "الإعدادات",
        "menu.file.exit": "خروج",
        "menu.edit": "تحرير",
        "menu.view": "عرض",
        "menu.tools": "أدوات",
        "menu.help": "مساعدة",
        "chat.input_placeholder": "اكتب رسالة...",
        "chat.send": "إرسال",
        "chat.stop": "إيقاف",
        "chat.thinking": "جاري التفكير...",
        "settings.title": "الإعدادات",
        "settings.language": "اللغة",
        "action.ok": "موافق",
        "action.cancel": "إلغاء",
        "action.loading": "جاري التحميل...",
        "error.generic": "حدث خطأ",
    },
    
    "ru": {
        "app.subtitle": "ИИ-ассистент",
        "menu.file": "Файл",
        "menu.file.new": "Новый чат",
        "menu.file.open": "Открыть",
        "menu.file.save": "Сохранить",
        "menu.file.settings": "Настройки",
        "menu.file.exit": "Выход",
        "menu.edit": "Редактировать",
        "menu.view": "Вид",
        "menu.tools": "Инструменты",
        "menu.help": "Справка",
        "chat.input_placeholder": "Введите сообщение...",
        "chat.send": "Отправить",
        "chat.stop": "Остановить",
        "chat.thinking": "Думаю...",
        "settings.title": "Настройки",
        "settings.language": "Язык",
        "action.ok": "ОК",
        "action.cancel": "Отмена",
        "action.loading": "Загрузка...",
        "error.generic": "Произошла ошибка",
    },
    
    "pt": {
        "app.subtitle": "Assistente de IA",
        "menu.file": "Arquivo",
        "menu.file.new": "Nova conversa",
        "menu.file.open": "Abrir",
        "menu.file.save": "Salvar",
        "menu.file.settings": "Configurações",
        "menu.file.exit": "Sair",
        "chat.input_placeholder": "Digite uma mensagem...",
        "chat.send": "Enviar",
        "chat.thinking": "Pensando...",
        "settings.language": "Idioma",
        "action.ok": "OK",
        "action.cancel": "Cancelar",
        "action.loading": "Carregando...",
    },
    
    "he": {
        "app.subtitle": "עוזר AI",
        "menu.file": "קובץ",
        "menu.file.new": "שיחה חדשה",
        "menu.file.settings": "הגדרות",
        "menu.file.exit": "יציאה",
        "chat.input_placeholder": "הקלד הודעה...",
        "chat.send": "שלח",
        "chat.thinking": "חושב...",
        "settings.language": "שפה",
        "action.ok": "אישור",
        "action.cancel": "ביטול",
    },
}


class TranslationManager:
    """
    Translation manager for internationalization
    
    Singleton that manages translations and locale-aware formatting.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._current_language = "en"
        self._translations = TRANSLATIONS.copy()
        self._custom_translations: dict[str, dict[str, str]] = {}
        self._callbacks: list[Callable[[str], None]] = []
        self._translations_dir: Optional[Path] = None
        
        # Try to detect system language
        self._detect_system_language()
    
    def _detect_system_language(self) -> None:
        """Detect system language from locale"""
        try:
            system_locale = locale.getdefaultlocale()[0] or "en_US"
            lang_code = system_locale.split('_')[0]
            
            if lang_code in LANGUAGES:
                self._current_language = lang_code
                logger.info(f"Detected system language: {lang_code}")
        except Exception:
            pass  # Intentionally silent
    
    def set_translations_directory(self, path: Union[str, Path]) -> None:
        """Set directory for loading additional translation files"""
        self._translations_dir = Path(path)
        self._load_translation_files()
    
    def _load_translation_files(self) -> None:
        """Load translation files from directory"""
        if not self._translations_dir or not self._translations_dir.exists():
            return
        
        for json_file in self._translations_dir.glob("*.json"):
            try:
                lang_code = json_file.stem
                with open(json_file, encoding='utf-8') as f:
                    translations = json.load(f)
                
                if lang_code not in self._translations:
                    self._translations[lang_code] = {}
                
                self._translations[lang_code].update(translations)
                logger.info(f"Loaded translations from {json_file}")
                
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
    
    def save_translations(self, lang_code: str, path: Union[str, Path]) -> None:
        """Save translations to JSON file"""
        translations = self._translations.get(lang_code, {})
        translations.update(self._custom_translations.get(lang_code, {}))
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)
    
    @property
    def current_language(self) -> str:
        """Get current language code"""
        return self._current_language
    
    @property
    def current_language_info(self) -> LanguageInfo:
        """Get current language info"""
        return LANGUAGES.get(self._current_language, LANGUAGES["en"])
    
    @property
    def text_direction(self) -> TextDirection:
        """Get current text direction"""
        return self.current_language_info.direction
    
    @property
    def is_rtl(self) -> bool:
        """Check if current language is RTL"""
        return self.text_direction == TextDirection.RTL
    
    def set_language(self, lang_code: str) -> bool:
        """Set current language"""
        if lang_code not in LANGUAGES:
            logger.warning(f"Unsupported language: {lang_code}")
            return False
        
        old_lang = self._current_language
        self._current_language = lang_code
        
        # Set locale for formatting
        try:
            lang_info = LANGUAGES[lang_code]
            locale.setlocale(locale.LC_ALL, lang_info.locale_code)
        except locale.Error:
            # Fallback to just the language code
            try:
                locale.setlocale(locale.LC_ALL, lang_code)
            except locale.Error:
                pass  # Intentionally silent
        
        # Notify callbacks
        if old_lang != lang_code:
            for callback in self._callbacks:
                try:
                    callback(lang_code)
                except Exception as e:
                    logger.error(f"Language change callback error: {e}")
        
        logger.info(f"Language changed to: {lang_code}")
        return True
    
    def add_language_change_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for language changes"""
        self._callbacks.append(callback)
    
    def remove_language_change_callback(self, callback: Callable[[str], None]) -> None:
        """Remove language change callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_supported_languages(self) -> dict[str, LanguageInfo]:
        """Get all supported languages"""
        return LANGUAGES.copy()
    
    def t(self, key: str, **kwargs) -> str:
        """
        Translate a key to the current language
        
        Args:
            key: Translation key (e.g., "chat.send")
            **kwargs: Formatting arguments
            
        Returns:
            Translated string
        """
        # Try current language
        translations = self._translations.get(self._current_language, {})
        custom = self._custom_translations.get(self._current_language, {})
        
        text = custom.get(key) or translations.get(key)
        
        # Fallback to English
        if text is None:
            text = DEFAULT_TRANSLATIONS.get(key, key)
        
        # Format with arguments
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass  # Intentionally silent
        
        return text
    
    def add_translation(self, lang_code: str, key: str, value: str) -> None:
        """Add a custom translation"""
        if lang_code not in self._custom_translations:
            self._custom_translations[lang_code] = {}
        self._custom_translations[lang_code][key] = value
    
    def add_translations(self, lang_code: str, translations: dict[str, str]) -> None:
        """Add multiple custom translations"""
        if lang_code not in self._custom_translations:
            self._custom_translations[lang_code] = {}
        self._custom_translations[lang_code].update(translations)


class LocaleFormatter:
    """Locale-aware formatting for dates, numbers, and currencies"""
    
    def __init__(self, translation_manager: TranslationManager = None):
        self.tm = translation_manager or get_translation_manager()
    
    def format_number(self, value: Union[int, float], 
                     decimal_places: int = None) -> str:
        """Format number according to locale"""
        try:
            if decimal_places is not None:
                return locale.format_string(f"%.{decimal_places}f", value, grouping=True)
            return locale.format_string("%g", value, grouping=True)
        except Exception:
            return str(value)
    
    def format_currency(self, value: float, currency: str = None) -> str:
        """Format currency according to locale"""
        try:
            return locale.currency(value, symbol=True, grouping=True)
        except Exception:
            return f"${value:.2f}"
    
    def format_percent(self, value: float, decimal_places: int = 1) -> str:
        """Format percentage according to locale"""
        try:
            return locale.format_string(f"%.{decimal_places}f%%", value * 100)
        except Exception:
            return f"{value * 100:.{decimal_places}f}%"
    
    def format_date(self, dt: datetime, style: str = "medium") -> str:
        """
        Format date according to locale
        
        Args:
            dt: datetime object
            style: "short", "medium", "long", or "full"
        """
        formats = {
            "short": "%x",
            "medium": "%b %d, %Y",
            "long": "%B %d, %Y",
            "full": "%A, %B %d, %Y"
        }
        
        fmt = formats.get(style, formats["medium"])
        
        try:
            return dt.strftime(fmt)
        except Exception:
            return str(dt.date())
    
    def format_time(self, dt: datetime, style: str = "medium") -> str:
        """
        Format time according to locale
        
        Args:
            dt: datetime object
            style: "short", "medium", or "long"
        """
        formats = {
            "short": "%H:%M",
            "medium": "%X",
            "long": "%X %Z"
        }
        
        fmt = formats.get(style, formats["medium"])
        
        try:
            return dt.strftime(fmt)
        except Exception:
            return str(dt.time())
    
    def format_datetime(self, dt: datetime, 
                       date_style: str = "medium",
                       time_style: str = "short") -> str:
        """Format datetime according to locale"""
        date_str = self.format_date(dt, date_style)
        time_str = self.format_time(dt, time_style)
        return f"{date_str} {time_str}"
    
    def format_relative_time(self, dt: datetime) -> str:
        """Format time relative to now (e.g., "5 minutes ago")"""
        now = datetime.now()
        diff = now - dt
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return self.tm.t("time.just_now")
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return self.tm.t("time.minutes_ago", n=minutes)
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return self.tm.t("time.hours_ago", n=hours)
        elif seconds < 172800:
            return self.tm.t("time.yesterday")
        else:
            days = int(seconds / 86400)
            return self.tm.t("time.days_ago", n=days)
    
    def format_file_size(self, bytes_size: int) -> str:
        """Format file size with locale-aware numbers"""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(bytes_size)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        return f"{self.format_number(size, 2)} {units[unit_index]}"


class RTLLayoutHelper:
    """Helper for Right-to-Left layout adjustments"""
    
    def __init__(self, translation_manager: TranslationManager = None):
        self.tm = translation_manager or get_translation_manager()
    
    @property
    def is_rtl(self) -> bool:
        """Check if current language is RTL"""
        return self.tm.is_rtl
    
    def get_qt_layout_direction(self):
        """Get Qt layout direction constant"""
        try:
            from PyQt5.QtCore import Qt
            return Qt.RightToLeft if self.is_rtl else Qt.LeftToRight
        except ImportError:
            return None
    
    def get_css_direction(self) -> str:
        """Get CSS direction value"""
        return "rtl" if self.is_rtl else "ltr"
    
    def get_text_align(self, default: str = "left") -> str:
        """Get text alignment (flipped for RTL)"""
        if not self.is_rtl:
            return default
        
        flip_map = {"left": "right", "right": "left"}
        return flip_map.get(default, default)
    
    def flip_margin(self, left: int, right: int) -> tuple:
        """Flip left/right margins for RTL"""
        if self.is_rtl:
            return (right, left)
        return (left, right)
    
    def get_stylesheet_direction(self) -> str:
        """Get stylesheet rules for direction"""
        direction = self.get_css_direction()
        text_align = self.get_text_align()
        
        return f"""
            * {{
                direction: {direction};
                text-align: {text_align};
            }}
            QLineEdit, QTextEdit, QPlainTextEdit {{
                text-align: {text_align};
            }}
            QMenuBar {{
                direction: {direction};
            }}
            QToolBar {{
                direction: {direction};
            }}
        """
    
    def apply_to_qt_widget(self, widget) -> None:
        """Apply RTL settings to a Qt widget"""
        try:
            from PyQt5.QtCore import Qt
            
            direction = Qt.RightToLeft if self.is_rtl else Qt.LeftToRight
            widget.setLayoutDirection(direction)
            
        except ImportError:
            pass  # Intentionally silent
    
    def apply_to_qt_application(self, app) -> None:
        """Apply RTL settings to entire Qt application"""
        try:
            from PyQt5.QtCore import Qt
            
            direction = Qt.RightToLeft if self.is_rtl else Qt.LeftToRight
            app.setLayoutDirection(direction)
            
            # Add direction-specific stylesheet
            current_style = app.styleSheet()
            direction_style = self.get_stylesheet_direction()
            app.setStyleSheet(current_style + direction_style)
            
        except ImportError:
            pass  # Intentionally silent


# Singleton accessor
_translation_manager: Optional[TranslationManager] = None


def get_translation_manager() -> TranslationManager:
    """Get the singleton translation manager"""
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def get_locale_formatter() -> LocaleFormatter:
    """Get a locale formatter instance"""
    return LocaleFormatter(get_translation_manager())


def get_rtl_helper() -> RTLLayoutHelper:
    """Get an RTL layout helper instance"""
    return RTLLayoutHelper(get_translation_manager())


# Convenience function for translations
def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current language
    
    Usage:
        from enigma_engine.i18n import t
        label = t("chat.send")
        message = t("time.minutes_ago", n=5)
    """
    return get_translation_manager().t(key, **kwargs)


# Convenience function for setting language
def set_language(lang_code: str) -> bool:
    """Set the current language"""
    return get_translation_manager().set_language(lang_code)


# Export for easy imports
__all__ = [
    'TranslationManager',
    'LocaleFormatter', 
    'RTLLayoutHelper',
    'LanguageInfo',
    'TextDirection',
    'LANGUAGES',
    'get_translation_manager',
    'get_locale_formatter',
    'get_rtl_helper',
    't',
    'set_language',
]
