"""
ForgeAI Internationalization (i18n) Module

Provides translation and localization support for multiple languages.
"""

from .translations import (
    LANGUAGES,
    LanguageInfo,
    LocaleFormatter,
    RTLLayoutHelper,
    TextDirection,
    TranslationManager,
    get_locale_formatter,
    get_rtl_helper,
    get_translation_manager,
    set_language,
    t,
)

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
