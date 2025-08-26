"""
Comprehensive Internationalization and Localization System

Multi-language support with regional compliance and cultural adaptations
for global cybersecurity training deployment.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from ..utils.robust_error_handler import robust, ErrorSeverity

logger = logging.getLogger(__name__)


class SupportedLocale(Enum):
    """Supported locales with regional variants"""
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB" 
    SPANISH_ES = "es-ES"
    SPANISH_MX = "es-MX"
    FRENCH_FR = "fr-FR"
    GERMAN_DE = "de-DE"
    JAPANESE_JP = "ja-JP"
    CHINESE_CN = "zh-CN"
    CHINESE_TW = "zh-TW"
    KOREAN_KR = "ko-KR"
    PORTUGUESE_BR = "pt-BR"
    RUSSIAN_RU = "ru-RU"
    ITALIAN_IT = "it-IT"
    DUTCH_NL = "nl-NL"
    ARABIC_SA = "ar-SA"
    HINDI_IN = "hi-IN"


class RegionalCompliance(Enum):
    """Regional compliance frameworks"""
    GDPR = "gdpr"          # European Union
    CCPA = "ccpa"          # California, USA
    PDPA = "pdpa"          # Singapore, Thailand
    LGPD = "lgpd"          # Brazil
    PIPEDA = "pipeda"      # Canada
    APPI = "appi"          # Japan
    KVKK = "kvkk"          # Turkey
    DPA = "dpa"            # UK
    POPI = "popi"          # South Africa


@dataclass
class LocalizedContent:
    """Localized content structure"""
    locale: str
    text: str
    context: Optional[str] = None
    plural_forms: Optional[Dict[str, str]] = None
    cultural_notes: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceRequirement:
    """Regional compliance requirement"""
    framework: RegionalCompliance
    requirement_id: str
    description: str
    mandatory: bool = True
    applicable_locales: List[str] = field(default_factory=list)
    implementation_notes: str = ""


class ComprehensiveI18nManager:
    """Advanced internationalization manager"""
    
    def __init__(self):
        self.current_locale = SupportedLocale.ENGLISH_US.value
        self.fallback_locale = SupportedLocale.ENGLISH_US.value
        
        # Translation storage
        self.translations: Dict[str, Dict[str, LocalizedContent]] = {}
        self.pluralization_rules: Dict[str, Dict[str, Any]] = {}
        
        # Regional data
        self.regional_formats: Dict[str, Dict[str, str]] = {}
        self.compliance_requirements: Dict[str, List[ComplianceRequirement]] = {}
        
        # Cultural adaptations
        self.cultural_settings: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default data
        self._initialize_locales()
        self._initialize_compliance_frameworks()
        self._initialize_cultural_settings()
        
        logger.info("Comprehensive I18n manager initialized")
    
    @robust(severity=ErrorSeverity.LOW)
    def set_locale(self, locale: Union[str, SupportedLocale]) -> bool:
        """Set current locale"""
        if isinstance(locale, SupportedLocale):
            locale = locale.value
        
        if locale not in [loc.value for loc in SupportedLocale]:
            logger.warning(f"Unsupported locale: {locale}")
            return False
        
        self.current_locale = locale
        logger.info(f"Locale set to: {locale}")
        return True
    
    @robust(severity=ErrorSeverity.LOW)
    def translate(
        self, 
        key: str, 
        locale: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """Translate text key to specified or current locale"""
        target_locale = locale or self.current_locale
        
        # Get translation
        translation = self._get_translation(key, target_locale, context)
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                return translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting error for key '{key}': {e}")
                return translation
        
        return translation
    
    def _get_translation(self, key: str, locale: str, context: Optional[str] = None) -> str:
        """Get translation with fallback logic"""
        
        # Try exact locale match
        if locale in self.translations and key in self.translations[locale]:
            content = self.translations[locale][key]
            if not context or content.context == context:
                return content.text
        
        # Try language family fallback (e.g., es-MX -> es-ES)
        language_code = locale.split('-')[0]
        for loc in self.translations.keys():
            if loc.startswith(language_code + '-') and key in self.translations[loc]:
                content = self.translations[loc][key]
                if not context or content.context == context:
                    logger.debug(f"Using language family fallback: {loc} for {locale}")
                    return content.text
        
        # Try fallback locale
        if self.fallback_locale in self.translations and key in self.translations[self.fallback_locale]:
            content = self.translations[self.fallback_locale][key]
            logger.debug(f"Using fallback locale for key: {key}")
            return content.text
        
        # Return key as last resort
        logger.warning(f"No translation found for key: {key}")
        return f"[{key}]"
    
    @robust(severity=ErrorSeverity.LOW)
    def add_translation(
        self, 
        key: str, 
        locale: str, 
        text: str,
        context: Optional[str] = None,
        plural_forms: Optional[Dict[str, str]] = None,
        cultural_notes: Optional[str] = None
    ):
        """Add translation for a key"""
        if locale not in self.translations:
            self.translations[locale] = {}
        
        self.translations[locale][key] = LocalizedContent(
            locale=locale,
            text=text,
            context=context,
            plural_forms=plural_forms,
            cultural_notes=cultural_notes
        )
        
        logger.debug(f"Added translation: {key} -> {locale}")
    
    @robust(severity=ErrorSeverity.LOW)
    def pluralize(
        self, 
        key: str, 
        count: int, 
        locale: Optional[str] = None
    ) -> str:
        """Get pluralized translation"""
        target_locale = locale or self.current_locale
        
        # Get plural form
        plural_form = self._get_plural_form(count, target_locale)
        
        # Get translation with plural context
        if target_locale in self.translations and key in self.translations[target_locale]:
            content = self.translations[target_locale][key]
            if content.plural_forms and plural_form in content.plural_forms:
                return content.plural_forms[plural_form].format(count=count)
        
        # Fallback to regular translation
        return self.translate(key, target_locale, count=count)
    
    def _get_plural_form(self, count: int, locale: str) -> str:
        """Get plural form for count and locale"""
        language = locale.split('-')[0]
        
        # Simplified pluralization rules
        if language in ['en', 'de', 'nl', 'it', 'pt']:
            return 'one' if count == 1 else 'other'
        elif language in ['fr', 'es']:
            return 'one' if count == 0 or count == 1 else 'other'
        elif language in ['ru']:
            if count % 10 == 1 and count % 100 != 11:
                return 'one'
            elif 2 <= count % 10 <= 4 and not (12 <= count % 100 <= 14):
                return 'few'
            else:
                return 'many'
        elif language in ['ja', 'ko', 'zh']:
            return 'other'  # No pluralization
        else:
            return 'other'  # Default fallback
    
    @robust(severity=ErrorSeverity.MEDIUM)
    def format_datetime(
        self, 
        dt: datetime, 
        format_type: str = "full",
        locale: Optional[str] = None
    ) -> str:
        """Format datetime according to locale"""
        target_locale = locale or self.current_locale
        
        formats = self.regional_formats.get(target_locale, {})
        
        if format_type == "full":
            fmt = formats.get("datetime_full", "%Y-%m-%d %H:%M:%S")
        elif format_type == "date":
            fmt = formats.get("date_format", "%Y-%m-%d")
        elif format_type == "time":
            fmt = formats.get("time_format", "%H:%M:%S")
        else:
            fmt = formats.get(format_type, "%Y-%m-%d %H:%M:%S")
        
        try:
            return dt.strftime(fmt)
        except Exception as e:
            logger.warning(f"DateTime formatting error: {e}")
            return dt.isoformat()
    
    @robust(severity=ErrorSeverity.LOW)
    def format_number(
        self, 
        number: Union[int, float], 
        locale: Optional[str] = None
    ) -> str:
        """Format number according to locale"""
        target_locale = locale or self.current_locale
        
        formats = self.regional_formats.get(target_locale, {})
        
        if isinstance(number, float):
            decimal_sep = formats.get("decimal_separator", ".")
            thousand_sep = formats.get("thousand_separator", ",")
            
            # Simple formatting
            if number >= 1000:
                int_part = int(number)
                decimal_part = number - int_part
                
                # Format integer part with thousands separator
                int_str = f"{int_part:,}".replace(",", thousand_sep)
                
                if decimal_part > 0:
                    return f"{int_str}{decimal_sep}{decimal_part:.2f}".split('.')[1]
                else:
                    return int_str
            else:
                return str(number).replace(".", decimal_sep)
        else:
            thousand_sep = formats.get("thousand_separator", ",")
            return f"{number:,}".replace(",", thousand_sep)
    
    def get_cultural_setting(
        self, 
        setting_key: str, 
        locale: Optional[str] = None
    ) -> Any:
        """Get cultural setting for locale"""
        target_locale = locale or self.current_locale
        
        settings = self.cultural_settings.get(target_locale, {})
        return settings.get(setting_key)
    
    def get_compliance_requirements(
        self, 
        locale: Optional[str] = None
    ) -> List[ComplianceRequirement]:
        """Get compliance requirements for locale"""
        target_locale = locale or self.current_locale
        
        return self.compliance_requirements.get(target_locale, [])
    
    def is_rtl_locale(self, locale: Optional[str] = None) -> bool:
        """Check if locale uses right-to-left text direction"""
        target_locale = locale or self.current_locale
        language = target_locale.split('-')[0]
        
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return language in rtl_languages
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales with metadata"""
        return [
            {
                "code": locale.value,
                "name": self._get_locale_display_name(locale.value),
                "native_name": self._get_native_name(locale.value),
                "rtl": self.is_rtl_locale(locale.value)
            }
            for locale in SupportedLocale
        ]
    
    def _get_locale_display_name(self, locale: str) -> str:
        """Get display name for locale"""
        display_names = {
            "en-US": "English (United States)",
            "en-GB": "English (United Kingdom)",
            "es-ES": "Spanish (Spain)",
            "es-MX": "Spanish (Mexico)",
            "fr-FR": "French (France)",
            "de-DE": "German (Germany)",
            "ja-JP": "Japanese (Japan)",
            "zh-CN": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "ko-KR": "Korean (South Korea)",
            "pt-BR": "Portuguese (Brazil)",
            "ru-RU": "Russian (Russia)",
            "it-IT": "Italian (Italy)",
            "nl-NL": "Dutch (Netherlands)",
            "ar-SA": "Arabic (Saudi Arabia)",
            "hi-IN": "Hindi (India)"
        }
        return display_names.get(locale, locale)
    
    def _get_native_name(self, locale: str) -> str:
        """Get native name for locale"""
        native_names = {
            "en-US": "English",
            "en-GB": "English",
            "es-ES": "Español",
            "es-MX": "Español",
            "fr-FR": "Français",
            "de-DE": "Deutsch",
            "ja-JP": "日本語",
            "zh-CN": "中文(简体)",
            "zh-TW": "中文(繁體)",
            "ko-KR": "한국어",
            "pt-BR": "Português",
            "ru-RU": "Русский",
            "it-IT": "Italiano",
            "nl-NL": "Nederlands",
            "ar-SA": "العربية",
            "hi-IN": "हिन्दी"
        }
        return native_names.get(locale, locale)
    
    def _initialize_locales(self):
        """Initialize default locale data"""
        
        # Core UI translations
        core_translations = {
            "app.title": {
                "en-US": "GAN Cyber Range - Defensive Security Training",
                "es-ES": "Rango Cibernético GAN - Entrenamiento de Seguridad Defensiva",
                "fr-FR": "Portée Cyber GAN - Formation Sécuritaire Défensive",
                "de-DE": "GAN Cyber Range - Defensive Sicherheitsschulung",
                "ja-JP": "GAN サイバーレンジ - 防御的セキュリティトレーニング",
                "zh-CN": "GAN 网络靶场 - 防御性安全培训"
            },
            "attack.generated": {
                "en-US": "Generated {count} attacks",
                "es-ES": "Generados {count} ataques",
                "fr-FR": "Généré {count} attaques",
                "de-DE": "{count} Angriffe generiert",
                "ja-JP": "{count}個の攻撃を生成しました",
                "zh-CN": "生成了{count}次攻击"
            },
            "security.ethical_violation": {
                "en-US": "This request violates our ethical guidelines",
                "es-ES": "Esta solicitud viola nuestras pautas éticas",
                "fr-FR": "Cette demande viole nos directives éthiques",
                "de-DE": "Diese Anfrage verstößt gegen unsere ethischen Richtlinien",
                "ja-JP": "このリクエストは倫理ガイドラインに違反しています",
                "zh-CN": "此请求违反了我们的道德准则"
            },
            "training.defensive_focus": {
                "en-US": "This system is designed exclusively for defensive cybersecurity training",
                "es-ES": "Este sistema está diseñado exclusivamente para entrenamiento defensivo de ciberseguridad",
                "fr-FR": "Ce système est conçu exclusivement pour la formation défensive en cybersécurité",
                "de-DE": "Dieses System ist ausschließlich für defensive Cybersecurity-Schulungen konzipiert",
                "ja-JP": "このシステムは防御的サイバーセキュリティトレーニング専用に設計されています",
                "zh-CN": "该系统专为防御性网络安全培训而设计"
            }
        }
        
        # Add translations to storage
        for key, translations in core_translations.items():
            for locale, text in translations.items():
                self.add_translation(key, locale, text)
        
        # Initialize regional formats
        self.regional_formats = {
            "en-US": {
                "date_format": "%m/%d/%Y",
                "time_format": "%I:%M:%S %p",
                "datetime_full": "%m/%d/%Y %I:%M:%S %p",
                "decimal_separator": ".",
                "thousand_separator": ","
            },
            "en-GB": {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M:%S",
                "datetime_full": "%d/%m/%Y %H:%M:%S",
                "decimal_separator": ".",
                "thousand_separator": ","
            },
            "de-DE": {
                "date_format": "%d.%m.%Y",
                "time_format": "%H:%M:%S",
                "datetime_full": "%d.%m.%Y %H:%M:%S",
                "decimal_separator": ",",
                "thousand_separator": "."
            },
            "fr-FR": {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M:%S",
                "datetime_full": "%d/%m/%Y %H:%M:%S",
                "decimal_separator": ",",
                "thousand_separator": " "
            },
            "ja-JP": {
                "date_format": "%Y年%m月%d日",
                "time_format": "%H:%M:%S",
                "datetime_full": "%Y年%m月%d日 %H:%M:%S",
                "decimal_separator": ".",
                "thousand_separator": ","
            },
            "zh-CN": {
                "date_format": "%Y年%m月%d日",
                "time_format": "%H:%M:%S",
                "datetime_full": "%Y年%m月%d日 %H:%M:%S",
                "decimal_separator": ".",
                "thousand_separator": ","
            }
        }
    
    def _initialize_compliance_frameworks(self):
        """Initialize compliance framework requirements"""
        
        # GDPR requirements
        gdpr_requirements = [
            ComplianceRequirement(
                framework=RegionalCompliance.GDPR,
                requirement_id="gdpr_consent",
                description="Explicit user consent required for data processing",
                applicable_locales=["en-GB", "de-DE", "fr-FR", "es-ES", "it-IT", "nl-NL"]
            ),
            ComplianceRequirement(
                framework=RegionalCompliance.GDPR,
                requirement_id="gdpr_data_minimization",
                description="Process only data necessary for the specified purpose",
                applicable_locales=["en-GB", "de-DE", "fr-FR", "es-ES", "it-IT", "nl-NL"]
            ),
            ComplianceRequirement(
                framework=RegionalCompliance.GDPR,
                requirement_id="gdpr_right_to_erasure",
                description="Users have the right to request data deletion",
                applicable_locales=["en-GB", "de-DE", "fr-FR", "es-ES", "it-IT", "nl-NL"]
            )
        ]
        
        # Add GDPR requirements to applicable locales
        for req in gdpr_requirements:
            for locale in req.applicable_locales:
                if locale not in self.compliance_requirements:
                    self.compliance_requirements[locale] = []
                self.compliance_requirements[locale].append(req)
        
        # CCPA requirements
        ccpa_requirements = [
            ComplianceRequirement(
                framework=RegionalCompliance.CCPA,
                requirement_id="ccpa_disclosure",
                description="Disclose categories of personal information collected",
                applicable_locales=["en-US"]
            ),
            ComplianceRequirement(
                framework=RegionalCompliance.CCPA,
                requirement_id="ccpa_opt_out",
                description="Provide option to opt-out of data sale",
                applicable_locales=["en-US"]
            )
        ]
        
        for req in ccpa_requirements:
            for locale in req.applicable_locales:
                if locale not in self.compliance_requirements:
                    self.compliance_requirements[locale] = []
                self.compliance_requirements[locale].append(req)
    
    def _initialize_cultural_settings(self):
        """Initialize cultural adaptation settings"""
        
        self.cultural_settings = {
            "en-US": {
                "attack_examples": ["corporate_network", "financial_system", "healthcare_database"],
                "security_priorities": ["privacy", "availability", "integrity"],
                "training_style": "direct",
                "formal_tone": False
            },
            "de-DE": {
                "attack_examples": ["industrial_control", "automotive_system", "banking_infrastructure"],
                "security_priorities": ["privacy", "compliance", "integrity"],
                "training_style": "systematic",
                "formal_tone": True
            },
            "ja-JP": {
                "attack_examples": ["manufacturing_system", "public_infrastructure", "financial_network"],
                "security_priorities": ["reliability", "consensus", "continuity"],
                "training_style": "collaborative",
                "formal_tone": True
            },
            "zh-CN": {
                "attack_examples": ["smart_city", "e_commerce", "social_platform"],
                "security_priorities": ["stability", "scalability", "compliance"],
                "training_style": "practical",
                "formal_tone": True
            }
        }
    
    @robust(severity=ErrorSeverity.LOW)
    def export_translations(self, output_path: str) -> bool:
        """Export translations to JSON file"""
        try:
            export_data = {
                "translations": {},
                "regional_formats": self.regional_formats,
                "compliance_requirements": {},
                "cultural_settings": self.cultural_settings
            }
            
            # Convert translations
            for locale, translations in self.translations.items():
                export_data["translations"][locale] = {}
                for key, content in translations.items():
                    export_data["translations"][locale][key] = {
                        "text": content.text,
                        "context": content.context,
                        "plural_forms": content.plural_forms,
                        "cultural_notes": content.cultural_notes
                    }
            
            # Convert compliance requirements
            for locale, requirements in self.compliance_requirements.items():
                export_data["compliance_requirements"][locale] = [
                    {
                        "framework": req.framework.value,
                        "requirement_id": req.requirement_id,
                        "description": req.description,
                        "mandatory": req.mandatory,
                        "implementation_notes": req.implementation_notes
                    }
                    for req in requirements
                ]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Translations exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def get_localization_status(self) -> Dict[str, Any]:
        """Get comprehensive localization status"""
        total_keys = len(set(
            key for translations in self.translations.values() 
            for key in translations.keys()
        ))
        
        locale_coverage = {}
        for locale_enum in SupportedLocale:
            locale = locale_enum.value
            translated_keys = len(self.translations.get(locale, {}))
            coverage = (translated_keys / max(total_keys, 1)) * 100
            
            locale_coverage[locale] = {
                "coverage_percent": coverage,
                "translated_keys": translated_keys,
                "missing_keys": max(0, total_keys - translated_keys),
                "compliance_requirements": len(self.compliance_requirements.get(locale, [])),
                "cultural_adaptations": len(self.cultural_settings.get(locale, {}))
            }
        
        return {
            "total_translation_keys": total_keys,
            "supported_locales": len(SupportedLocale),
            "locale_coverage": locale_coverage,
            "current_locale": self.current_locale,
            "rtl_support": any(
                self.is_rtl_locale(locale.value) for locale in SupportedLocale
            )
        }


# Global i18n manager
i18n_manager = ComprehensiveI18nManager()

# Convenience functions
def translate(key: str, **kwargs) -> str:
    """Global translation function"""
    return i18n_manager.translate(key, **kwargs)

def set_locale(locale: Union[str, SupportedLocale]) -> bool:
    """Global locale setting"""
    return i18n_manager.set_locale(locale)

def get_current_locale() -> str:
    """Get current locale"""
    return i18n_manager.current_locale