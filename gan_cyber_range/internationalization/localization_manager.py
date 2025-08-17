"""
Comprehensive localization management for global cyber range deployment.

This module provides advanced localization capabilities including dynamic content
translation, cultural adaptation, and compliance with international standards.
"""

import logging
import json
import yaml
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import re
import locale
import gettext
import babel
from babel import Locale, dates, numbers
from babel.messages import Catalog
from babel.messages.extract import extract_from_dir
import threading

logger = logging.getLogger(__name__)


class LocaleRegion(Enum):
    """Major locale regions"""
    NORTH_AMERICA = "NA"
    EUROPE = "EU"
    ASIA_PACIFIC = "APAC"
    LATIN_AMERICA = "LATAM"
    MIDDLE_EAST_AFRICA = "MEA"


class TextDirection(Enum):
    """Text direction for different languages"""
    LEFT_TO_RIGHT = "ltr"
    RIGHT_TO_LEFT = "rtl"
    TOP_TO_BOTTOM = "ttb"


@dataclass
class LocaleConfig:
    """Configuration for a specific locale"""
    locale_code: str  # e.g., 'en-US', 'fr-FR', 'zh-CN'
    language_code: str  # e.g., 'en', 'fr', 'zh'
    country_code: str  # e.g., 'US', 'FR', 'CN'
    display_name: str
    native_name: str
    region: LocaleRegion
    text_direction: TextDirection = TextDirection.LEFT_TO_RIGHT
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = "#,##0.##"
    currency_code: str = "USD"
    timezone: str = "UTC"
    enabled: bool = True
    completion_percentage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    fallback_locale: Optional[str] = None
    cultural_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationEntry:
    """A single translation entry"""
    key: str
    original_text: str
    translated_text: str
    context: Optional[str] = None
    translation_date: datetime = field(default_factory=datetime.now)
    translator: Optional[str] = None
    approved: bool = False
    needs_review: bool = False
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationStats:
    """Statistics about localization progress"""
    total_strings: int
    translated_strings: int
    approved_strings: int
    pending_review: int
    completion_percentage: float
    last_update: datetime
    active_translators: int
    languages_supported: int


class LocalizationManager:
    """Advanced localization management system"""
    
    def __init__(
        self,
        base_path: Path = None,
        default_locale: str = "en-US",
        fallback_locale: str = "en"
    ):
        self.base_path = base_path or Path("locales")
        self.default_locale = default_locale
        self.fallback_locale = fallback_locale
        
        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "templates").mkdir(exist_ok=True)
        (self.base_path / "translations").mkdir(exist_ok=True)
        
        # Locale configurations
        self.locale_configs: Dict[str, LocaleConfig] = {}
        self.current_locale = default_locale
        
        # Translation storage
        self.translations: Dict[str, Dict[str, TranslationEntry]] = {}
        self.message_catalogs: Dict[str, Catalog] = {}
        
        # Gettext support
        self.gettext_domains: Dict[str, gettext.GNUTranslations] = {}
        
        # Thread-local storage for locale context
        self._thread_local = threading.local()
        
        # Caching
        self.translation_cache: Dict[str, str] = {}
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1 hour
        
        # Event handlers
        self.translation_handlers: List[callable] = []
        self.locale_change_handlers: List[callable] = []
        
        # Initialize supported locales
        self._initialize_supported_locales()
        self._load_translations()
        
        logger.info(f"LocalizationManager initialized with base path: {self.base_path}")
    
    def register_locale(self, config: LocaleConfig) -> None:
        """Register a new locale configuration"""
        
        self.locale_configs[config.locale_code] = config
        
        # Create locale directory structure
        locale_path = self.base_path / "translations" / config.locale_code
        locale_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty translation dictionary
        if config.locale_code not in self.translations:
            self.translations[config.locale_code] = {}
        
        logger.info(f"Registered locale: {config.locale_code} ({config.display_name})")
    
    def set_current_locale(self, locale_code: str) -> bool:
        """Set the current locale for the thread"""
        
        if locale_code not in self.locale_configs:
            logger.warning(f"Locale {locale_code} not registered")
            return False
        
        self.current_locale = locale_code
        self._thread_local.current_locale = locale_code
        
        # Trigger locale change handlers
        for handler in self.locale_change_handlers:
            try:
                handler(locale_code)
            except Exception as e:
                logger.error(f"Error in locale change handler: {e}")
        
        logger.info(f"Current locale set to: {locale_code}")
        return True
    
    def get_current_locale(self) -> str:
        """Get the current locale for the thread"""
        
        return getattr(self._thread_local, 'current_locale', self.current_locale)
    
    def translate(
        self,
        key: str,
        default: Optional[str] = None,
        locale: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """Translate a text key to the current or specified locale"""
        
        target_locale = locale or self.get_current_locale()
        
        # Check cache first
        cache_key = f"{target_locale}:{key}:{context}"
        if self.cache_enabled and cache_key in self.translation_cache:
            translated = self.translation_cache[cache_key]
        else:
            translated = self._get_translation(key, target_locale, context, default)
            
            # Cache the result
            if self.cache_enabled:
                self.translation_cache[cache_key] = translated
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                translated = translated.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"String formatting failed for key '{key}': {e}")
        
        # Update usage statistics
        self._update_translation_usage(key, target_locale)
        
        return translated
    
    def add_translation(
        self,
        key: str,
        text: str,
        locale: str,
        original_text: Optional[str] = None,
        context: Optional[str] = None,
        translator: Optional[str] = None
    ) -> None:
        """Add a new translation"""
        
        if locale not in self.locale_configs:
            logger.warning(f"Locale {locale} not registered")
            return
        
        if locale not in self.translations:
            self.translations[locale] = {}
        
        entry = TranslationEntry(
            key=key,
            original_text=original_text or key,
            translated_text=text,
            context=context,
            translator=translator
        )
        
        self.translations[locale][key] = entry
        
        # Clear cache for this key
        self._clear_cache_for_key(key, locale)
        
        # Trigger translation handlers
        for handler in self.translation_handlers:
            try:
                handler('added', key, locale, text)
            except Exception as e:
                logger.error(f"Error in translation handler: {e}")
        
        logger.info(f"Added translation for key '{key}' in locale '{locale}'")
    
    def update_translation(
        self,
        key: str,
        text: str,
        locale: str,
        translator: Optional[str] = None
    ) -> bool:
        """Update an existing translation"""
        
        if locale not in self.translations or key not in self.translations[locale]:
            logger.warning(f"Translation key '{key}' not found for locale '{locale}'")
            return False
        
        entry = self.translations[locale][key]
        entry.translated_text = text
        entry.translation_date = datetime.now()
        entry.approved = False  # Reset approval status
        entry.needs_review = True
        
        if translator:
            entry.translator = translator
        
        # Clear cache
        self._clear_cache_for_key(key, locale)
        
        # Trigger handlers
        for handler in self.translation_handlers:
            try:
                handler('updated', key, locale, text)
            except Exception as e:
                logger.error(f"Error in translation handler: {e}")
        
        logger.info(f"Updated translation for key '{key}' in locale '{locale}'")
        return True
    
    def approve_translation(self, key: str, locale: str, approver: str) -> bool:
        """Approve a translation"""
        
        if locale not in self.translations or key not in self.translations[locale]:
            return False
        
        entry = self.translations[locale][key]
        entry.approved = True
        entry.needs_review = False
        entry.metadata['approved_by'] = approver
        entry.metadata['approval_date'] = datetime.now().isoformat()
        
        logger.info(f"Approved translation for key '{key}' in locale '{locale}'")
        return True
    
    def get_untranslated_keys(self, locale: str) -> List[str]:
        """Get list of keys that need translation for a locale"""
        
        if locale not in self.locale_configs:
            return []
        
        # Get all keys from the default locale
        default_keys = set()
        if self.default_locale in self.translations:
            default_keys = set(self.translations[self.default_locale].keys())
        
        # Get translated keys for target locale
        translated_keys = set()
        if locale in self.translations:
            translated_keys = set(self.translations[locale].keys())
        
        # Return missing keys
        return list(default_keys - translated_keys)
    
    def get_pending_review_keys(self, locale: str) -> List[str]:
        """Get list of translation keys pending review"""
        
        if locale not in self.translations:
            return []
        
        return [
            key for key, entry in self.translations[locale].items()
            if entry.needs_review and not entry.approved
        ]
    
    def export_translations(
        self,
        locale: str,
        format_type: str = "json",
        include_metadata: bool = False
    ) -> str:
        """Export translations in specified format"""
        
        if locale not in self.translations:
            return ""
        
        translations = self.translations[locale]
        
        if format_type == "json":
            if include_metadata:
                export_data = {
                    key: {
                        'text': entry.translated_text,
                        'context': entry.context,
                        'approved': entry.approved,
                        'last_updated': entry.translation_date.isoformat()
                    }
                    for key, entry in translations.items()
                }
            else:
                export_data = {
                    key: entry.translated_text
                    for key, entry in translations.items()
                }
            
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        
        elif format_type == "yaml":
            if include_metadata:
                export_data = {
                    key: {
                        'text': entry.translated_text,
                        'context': entry.context,
                        'approved': entry.approved
                    }
                    for key, entry in translations.items()
                }
            else:
                export_data = {
                    key: entry.translated_text
                    for key, entry in translations.items()
                }
            
            return yaml.dump(export_data, default_flow_style=False, allow_unicode=True)
        
        elif format_type == "po":
            return self._export_po_format(locale)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def import_translations(
        self,
        locale: str,
        data: str,
        format_type: str = "json",
        translator: Optional[str] = None
    ) -> int:
        """Import translations from data string"""
        
        imported_count = 0
        
        try:
            if format_type == "json":
                translation_data = json.loads(data)
            elif format_type == "yaml":
                translation_data = yaml.safe_load(data)
            else:
                raise ValueError(f"Unsupported import format: {format_type}")
            
            for key, value in translation_data.items():
                if isinstance(value, dict):
                    # Format with metadata
                    text = value.get('text', '')
                    context = value.get('context')
                else:
                    # Simple key-value format
                    text = str(value)
                    context = None
                
                if text:
                    self.add_translation(key, text, locale, context=context, translator=translator)
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} translations for locale '{locale}'")
            
        except Exception as e:
            logger.error(f"Error importing translations: {e}")
        
        return imported_count
    
    def get_localization_stats(self, locale: Optional[str] = None) -> Union[LocalizationStats, Dict[str, LocalizationStats]]:
        """Get localization statistics"""
        
        if locale:
            return self._calculate_locale_stats(locale)
        else:
            # Return stats for all locales
            return {
                locale_code: self._calculate_locale_stats(locale_code)
                for locale_code in self.locale_configs.keys()
            }
    
    def format_date(
        self,
        date_obj: datetime,
        format_type: str = "medium",
        locale: Optional[str] = None
    ) -> str:
        """Format date according to locale conventions"""
        
        target_locale = locale or self.get_current_locale()
        babel_locale = Locale.parse(target_locale.replace('-', '_'))
        
        return dates.format_date(date_obj, format=format_type, locale=babel_locale)
    
    def format_datetime(
        self,
        datetime_obj: datetime,
        format_type: str = "medium",
        locale: Optional[str] = None
    ) -> str:
        """Format datetime according to locale conventions"""
        
        target_locale = locale or self.get_current_locale()
        babel_locale = Locale.parse(target_locale.replace('-', '_'))
        
        return dates.format_datetime(datetime_obj, format=format_type, locale=babel_locale)
    
    def format_number(
        self,
        number: Union[int, float],
        locale: Optional[str] = None
    ) -> str:
        """Format number according to locale conventions"""
        
        target_locale = locale or self.get_current_locale()
        babel_locale = Locale.parse(target_locale.replace('-', '_'))
        
        return numbers.format_number(number, locale=babel_locale)
    
    def format_currency(
        self,
        amount: Union[int, float],
        currency: Optional[str] = None,
        locale: Optional[str] = None
    ) -> str:
        """Format currency according to locale conventions"""
        
        target_locale = locale or self.get_current_locale()
        babel_locale = Locale.parse(target_locale.replace('-', '_'))
        
        # Get currency from locale config if not specified
        if not currency and target_locale in self.locale_configs:
            currency = self.locale_configs[target_locale].currency_code
        
        currency = currency or "USD"
        
        return numbers.format_currency(amount, currency, locale=babel_locale)
    
    def get_text_direction(self, locale: Optional[str] = None) -> TextDirection:
        """Get text direction for locale"""
        
        target_locale = locale or self.get_current_locale()
        
        if target_locale in self.locale_configs:
            return self.locale_configs[target_locale].text_direction
        
        return TextDirection.LEFT_TO_RIGHT
    
    def pluralize(
        self,
        count: int,
        singular_key: str,
        plural_key: Optional[str] = None,
        locale: Optional[str] = None,
        **kwargs
    ) -> str:
        """Handle pluralization for different locales"""
        
        target_locale = locale or self.get_current_locale()
        
        # Simple English pluralization
        if target_locale.startswith('en'):
            key = singular_key if count == 1 else (plural_key or f"{singular_key}_plural")
        else:
            # For other languages, would need more sophisticated pluralization rules
            key = singular_key if count == 1 else (plural_key or f"{singular_key}_plural")
        
        return self.translate(key, locale=target_locale, count=count, **kwargs)
    
    def save_translations(self, locale: Optional[str] = None) -> None:
        """Save translations to files"""
        
        locales_to_save = [locale] if locale else self.translations.keys()
        
        for locale_code in locales_to_save:
            if locale_code not in self.translations:
                continue
            
            try:
                # Save as JSON
                json_path = self.base_path / "translations" / locale_code / "messages.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {key: entry.translated_text for key, entry in self.translations[locale_code].items()},
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
                
                # Update locale config
                if locale_code in self.locale_configs:
                    config = self.locale_configs[locale_code]
                    config.last_updated = datetime.now()
                    config.completion_percentage = self._calculate_completion_percentage(locale_code)
                
                logger.info(f"Saved translations for locale: {locale_code}")
                
            except Exception as e:
                logger.error(f"Error saving translations for {locale_code}: {e}")
    
    def _initialize_supported_locales(self) -> None:
        """Initialize supported locale configurations"""
        
        default_locales = [
            LocaleConfig(
                locale_code="en-US",
                language_code="en",
                country_code="US",
                display_name="English (United States)",
                native_name="English (United States)",
                region=LocaleRegion.NORTH_AMERICA,
                currency_code="USD",
                timezone="America/New_York"
            ),
            LocaleConfig(
                locale_code="es-ES",
                language_code="es",
                country_code="ES",
                display_name="Spanish (Spain)",
                native_name="Español (España)",
                region=LocaleRegion.EUROPE,
                currency_code="EUR",
                timezone="Europe/Madrid"
            ),
            LocaleConfig(
                locale_code="fr-FR",
                language_code="fr",
                country_code="FR",
                display_name="French (France)",
                native_name="Français (France)",
                region=LocaleRegion.EUROPE,
                currency_code="EUR",
                timezone="Europe/Paris"
            ),
            LocaleConfig(
                locale_code="de-DE",
                language_code="de",
                country_code="DE",
                display_name="German (Germany)",
                native_name="Deutsch (Deutschland)",
                region=LocaleRegion.EUROPE,
                currency_code="EUR",
                timezone="Europe/Berlin"
            ),
            LocaleConfig(
                locale_code="ja-JP",
                language_code="ja",
                country_code="JP",
                display_name="Japanese (Japan)",
                native_name="日本語 (日本)",
                region=LocaleRegion.ASIA_PACIFIC,
                currency_code="JPY",
                timezone="Asia/Tokyo"
            ),
            LocaleConfig(
                locale_code="zh-CN",
                language_code="zh",
                country_code="CN",
                display_name="Chinese (Simplified)",
                native_name="中文 (简体)",
                region=LocaleRegion.ASIA_PACIFIC,
                currency_code="CNY",
                timezone="Asia/Shanghai"
            ),
            LocaleConfig(
                locale_code="ar-SA",
                language_code="ar",
                country_code="SA",
                display_name="Arabic (Saudi Arabia)",
                native_name="العربية (المملكة العربية السعودية)",
                region=LocaleRegion.MIDDLE_EAST_AFRICA,
                text_direction=TextDirection.RIGHT_TO_LEFT,
                currency_code="SAR",
                timezone="Asia/Riyadh"
            )
        ]
        
        for locale_config in default_locales:
            self.register_locale(locale_config)
    
    def _load_translations(self) -> None:
        """Load existing translations from files"""
        
        translations_dir = self.base_path / "translations"
        
        if not translations_dir.exists():
            return
        
        for locale_dir in translations_dir.iterdir():
            if not locale_dir.is_dir():
                continue
            
            locale_code = locale_dir.name
            
            # Load JSON translations
            json_file = locale_dir / "messages.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        translation_data = json.load(f)
                    
                    if locale_code not in self.translations:
                        self.translations[locale_code] = {}
                    
                    for key, text in translation_data.items():
                        entry = TranslationEntry(
                            key=key,
                            original_text=key,
                            translated_text=text,
                            approved=True  # Assume file-based translations are approved
                        )
                        self.translations[locale_code][key] = entry
                    
                    logger.info(f"Loaded {len(translation_data)} translations for {locale_code}")
                    
                except Exception as e:
                    logger.error(f"Error loading translations for {locale_code}: {e}")
    
    def _get_translation(
        self,
        key: str,
        locale: str,
        context: Optional[str],
        default: Optional[str]
    ) -> str:
        """Get translation with fallback logic"""
        
        # Try exact locale
        if locale in self.translations and key in self.translations[locale]:
            return self.translations[locale][key].translated_text
        
        # Try language code only (e.g., 'en' for 'en-US')
        language_code = locale.split('-')[0]
        for loc_code, translations in self.translations.items():
            if loc_code.startswith(language_code) and key in translations:
                return translations[key].translated_text
        
        # Try fallback locale
        if (self.fallback_locale in self.translations and 
            key in self.translations[self.fallback_locale]):
            return self.translations[self.fallback_locale][key].translated_text
        
        # Return default or key
        return default or key
    
    def _update_translation_usage(self, key: str, locale: str) -> None:
        """Update usage statistics for a translation"""
        
        if locale in self.translations and key in self.translations[locale]:
            self.translations[locale][key].usage_count += 1
    
    def _clear_cache_for_key(self, key: str, locale: str) -> None:
        """Clear cache entries for a specific key and locale"""
        
        keys_to_remove = [
            cache_key for cache_key in self.translation_cache.keys()
            if cache_key.startswith(f"{locale}:{key}:")
        ]
        
        for cache_key in keys_to_remove:
            del self.translation_cache[cache_key]
    
    def _calculate_locale_stats(self, locale: str) -> LocalizationStats:
        """Calculate statistics for a specific locale"""
        
        # Get total strings from default locale
        total_strings = 0
        if self.default_locale in self.translations:
            total_strings = len(self.translations[self.default_locale])
        
        # Get translated strings for target locale
        translated_strings = 0
        approved_strings = 0
        pending_review = 0
        
        if locale in self.translations:
            translated_strings = len(self.translations[locale])
            approved_strings = len([
                entry for entry in self.translations[locale].values()
                if entry.approved
            ])
            pending_review = len([
                entry for entry in self.translations[locale].values()
                if entry.needs_review
            ])
        
        completion_percentage = (translated_strings / total_strings * 100) if total_strings > 0 else 0
        
        # Get unique translators
        translators = set()
        if locale in self.translations:
            for entry in self.translations[locale].values():
                if entry.translator:
                    translators.add(entry.translator)
        
        return LocalizationStats(
            total_strings=total_strings,
            translated_strings=translated_strings,
            approved_strings=approved_strings,
            pending_review=pending_review,
            completion_percentage=completion_percentage,
            last_update=datetime.now(),
            active_translators=len(translators),
            languages_supported=len(self.locale_configs)
        )
    
    def _calculate_completion_percentage(self, locale: str) -> float:
        """Calculate completion percentage for a locale"""
        
        total_strings = 0
        if self.default_locale in self.translations:
            total_strings = len(self.translations[self.default_locale])
        
        translated_strings = 0
        if locale in self.translations:
            translated_strings = len(self.translations[locale])
        
        return (translated_strings / total_strings * 100) if total_strings > 0 else 0
    
    def _export_po_format(self, locale: str) -> str:
        """Export translations in PO (Portable Object) format"""
        
        if locale not in self.translations:
            return ""
        
        lines = [
            '# Translation file for cyber range',
            f'# Locale: {locale}',
            f'# Generated: {datetime.now().isoformat()}',
            '',
            'msgid ""',
            'msgstr ""',
            f'"Language: {locale}\\n"',
            f'"Content-Type: text/plain; charset=UTF-8\\n"',
            ''
        ]
        
        for key, entry in self.translations[locale].items():
            if entry.context:
                lines.append(f'msgctxt "{entry.context}"')
            
            lines.append(f'msgid "{entry.original_text}"')
            lines.append(f'msgstr "{entry.translated_text}"')
            lines.append('')
        
        return '\n'.join(lines)


# Context manager for temporary locale switching
class locale_context:
    """Context manager for temporary locale switching"""
    
    def __init__(self, localization_manager: LocalizationManager, locale: str):
        self.localization_manager = localization_manager
        self.new_locale = locale
        self.original_locale = None
    
    def __enter__(self):
        self.original_locale = self.localization_manager.get_current_locale()
        self.localization_manager.set_current_locale(self.new_locale)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_locale:
            self.localization_manager.set_current_locale(self.original_locale)


# Decorator for marking translatable strings
def translatable(key: str, context: Optional[str] = None):
    """Decorator to mark functions with translatable strings"""
    
    def decorator(func):
        func._translation_key = key
        func._translation_context = context
        return func
    
    return decorator