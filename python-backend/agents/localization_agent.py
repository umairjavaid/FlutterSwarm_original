"""
Localization Agent for Flutter Development
Handles internationalization, locale management, and translation workflows
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from core.agent_types import AgentType
import json
import re

class LocalizationAgent(BaseAgent):
    def __init__(self, agent_id: str = "localization_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.LOCALIZATION,
            name="Localization Agent",
            description="Specialized agent for Flutter internationalization and localization"
        )
        
        self.capabilities = [
            "arb_file_generation",
            "locale_setup",
            "translation_management",
            "pluralization_rules",
            "date_time_formatting",
            "number_formatting",
            "rtl_support",
            "dynamic_locale_switching",
            "translation_validation",
            "context_translations",
            "gender_specific_translations",
            "translation_extraction"
        ]
        
        self.supported_locales = [
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ]
        
        self.common_translations = {
            "buttons": ["save", "cancel", "delete", "edit", "submit", "ok"],
            "navigation": ["home", "settings", "profile", "logout", "back"],
            "messages": ["loading", "error", "success", "warning", "info"],
            "forms": ["email", "password", "username", "confirm", "required"]
        }

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process localization-related requests"""
        try:
            request_type = request.get('type', '')
            
            if request_type == 'setup_localization':
                return await self._setup_localization(request)
            elif request_type == 'generate_arb_files':
                return await self._generate_arb_files(request)
            elif request_type == 'add_translations':
                return await self._add_translations(request)
            elif request_type == 'setup_rtl_support':
                return await self._setup_rtl_support(request)
            elif request_type == 'create_locale_switching':
                return await self._create_locale_switching(request)
            elif request_type == 'validate_translations':
                return await self._validate_translations(request)
            elif request_type == 'extract_translatable_strings':
                return await self._extract_translatable_strings(request)
            else:
                return await self._analyze_localization_needs(request)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Localization processing failed: {str(e)}",
                'agent_id': self.agent_id
            }

    async def _setup_localization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Flutter localization infrastructure"""
        locales = request.get('locales', ['en', 'es'])
        app_name = request.get('app_name', 'MyApp')
        
        # Generate pubspec.yaml dependencies
        pubspec_config = self._generate_pubspec_config()
        
        # Generate l10n.yaml configuration
        l10n_config = self._generate_l10n_config(locales)
        
        # Generate main app configuration
        main_app_config = self._generate_main_app_config(app_name, locales)
        
        # Generate initial ARB files
        arb_files = self._generate_initial_arb_files(locales)
        
        return {
            'success': True,
            'pubspec_config': pubspec_config,
            'l10n_config': l10n_config,
            'main_app_config': main_app_config,
            'arb_files': arb_files,
            'setup_steps': self._get_setup_steps(),
            'agent_id': self.agent_id
        }

    async def _generate_arb_files(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ARB (Application Resource Bundle) files"""
        translations = request.get('translations', {})
        locales = request.get('locales', ['en'])
        base_locale = request.get('base_locale', 'en')
        
        arb_files = {}
        
        for locale in locales:
            arb_content = self._create_arb_content(
                translations.get(locale, {}), 
                locale, 
                base_locale
            )
            arb_files[f'app_{locale}.arb'] = arb_content
        
        return {
            'success': True,
            'arb_files': arb_files,
            'generation_summary': self._get_generation_summary(arb_files),
            'usage_instructions': self._get_arb_usage_instructions(),
            'agent_id': self.agent_id
        }

    async def _add_translations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add new translations to existing ARB files"""
        new_translations = request.get('translations', {})
        existing_arb = request.get('existing_arb', {})
        pluralization = request.get('pluralization', False)
        
        updated_arb = {}
        translation_keys = []
        
        for locale, translations in new_translations.items():
            arb_file = f'app_{locale}.arb'
            current_arb = existing_arb.get(arb_file, {})
            
            # Add new translations
            for key, value in translations.items():
                if pluralization and isinstance(value, dict):
                    # Handle plural forms
                    current_arb[key] = self._create_plural_translation(value, locale)
                    current_arb[f'@{key}'] = self._create_plural_metadata(value)
                else:
                    current_arb[key] = value
                    current_arb[f'@{key}'] = {"description": f"Translation for {key}"}
                
                translation_keys.append(key)
            
            updated_arb[arb_file] = current_arb
        
        return {
            'success': True,
            'updated_arb': updated_arb,
            'added_keys': translation_keys,
            'pluralization_used': pluralization,
            'validation_results': self._validate_translation_keys(translation_keys),
            'agent_id': self.agent_id
        }

    async def _setup_rtl_support(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Right-to-Left language support"""
        rtl_locales = request.get('rtl_locales', ['ar', 'he', 'fa'])
        app_name = request.get('app_name', 'MyApp')
        
        # Generate RTL-aware MaterialApp configuration
        rtl_app_config = self._generate_rtl_app_config(app_name, rtl_locales)
        
        # Generate RTL-aware widget examples
        rtl_widgets = self._generate_rtl_widgets()
        
        # Generate text direction utilities
        text_direction_utils = self._generate_text_direction_utils()
        
        return {
            'success': True,
            'rtl_app_config': rtl_app_config,
            'rtl_widgets': rtl_widgets,
            'text_direction_utils': text_direction_utils,
            'rtl_best_practices': self._get_rtl_best_practices(),
            'testing_guidelines': self._get_rtl_testing_guidelines(),
            'agent_id': self.agent_id
        }

    async def _create_locale_switching(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic locale switching functionality"""
        locales = request.get('locales', ['en', 'es'])
        storage_type = request.get('storage_type', 'shared_preferences')
        
        # Generate locale provider/state management
        locale_provider = self._generate_locale_provider(locales, storage_type)
        
        # Generate locale switching UI
        locale_switcher_ui = self._generate_locale_switcher_ui(locales)
        
        # Generate locale persistence logic
        locale_persistence = self._generate_locale_persistence(storage_type)
        
        return {
            'success': True,
            'locale_provider': locale_provider,
            'locale_switcher_ui': locale_switcher_ui,
            'locale_persistence': locale_persistence,
            'integration_steps': self._get_locale_switching_steps(),
            'dependencies': self._get_locale_switching_dependencies(storage_type),
            'agent_id': self.agent_id
        }

    async def _validate_translations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate translation completeness and consistency"""
        arb_files = request.get('arb_files', {})
        base_locale = request.get('base_locale', 'en')
        
        validation_results = {
            'missing_translations': {},
            'extra_translations': {},
            'placeholder_mismatches': {},
            'consistency_issues': []
        }
        
        base_arb = arb_files.get(f'app_{base_locale}.arb', {})
        base_keys = set(k for k in base_arb.keys() if not k.startswith('@'))
        
        for arb_file, content in arb_files.items():
            if arb_file == f'app_{base_locale}.arb':
                continue
                
            locale = arb_file.replace('app_', '').replace('.arb', '')
            translation_keys = set(k for k in content.keys() if not k.startswith('@'))
            
            # Find missing translations
            missing = base_keys - translation_keys
            if missing:
                validation_results['missing_translations'][locale] = list(missing)
            
            # Find extra translations
            extra = translation_keys - base_keys
            if extra:
                validation_results['extra_translations'][locale] = list(extra)
            
            # Validate placeholders
            placeholder_issues = self._validate_placeholders(base_arb, content, locale)
            if placeholder_issues:
                validation_results['placeholder_mismatches'][locale] = placeholder_issues
        
        return {
            'success': True,
            'validation_results': validation_results,
            'completion_percentage': self._calculate_completion_percentage(validation_results, arb_files),
            'recommendations': self._get_validation_recommendations(validation_results),
            'agent_id': self.agent_id
        }

    async def _extract_translatable_strings(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract translatable strings from Dart code"""
        dart_code = request.get('code', '')
        file_path = request.get('file_path', '')
        
        # Extract string literals
        string_patterns = [
            r"Text\s*\(\s*['\"]([^'\"]+)['\"]",  # Text widgets
            r"title:\s*['\"]([^'\"]+)['\"]",      # Title properties
            r"label:\s*['\"]([^'\"]+)['\"]",      # Label properties
            r"hintText:\s*['\"]([^'\"]+)['\"]",   # Hint text
        ]
        
        extracted_strings = []
        suggestions = []
        
        for pattern in string_patterns:
            matches = re.finditer(pattern, dart_code)
            for match in matches:
                string_value = match.group(1)
                line_number = dart_code[:match.start()].count('\n') + 1
                
                # Generate key suggestion
                key_suggestion = self._generate_translation_key(string_value)
                
                extracted_strings.append({
                    'original': string_value,
                    'suggested_key': key_suggestion,
                    'line_number': line_number,
                    'context': match.group(0)
                })
        
        # Generate replacement code
        replacement_code = self._generate_replacement_code(dart_code, extracted_strings)
        
        return {
            'success': True,
            'extracted_strings': extracted_strings,
            'replacement_code': replacement_code,
            'arb_entries': self._generate_arb_entries(extracted_strings),
            'extraction_summary': f"Found {len(extracted_strings)} translatable strings",
            'agent_id': self.agent_id
        }

    def _generate_pubspec_config(self) -> str:
        """Generate pubspec.yaml localization configuration"""
        return '''
dependencies:
  flutter:
    sdk: flutter
  flutter_localizations:
    sdk: flutter
  intl: any

dev_dependencies:
  flutter_gen: ^5.3.2

flutter_gen:
  output: lib/gen/
  line_length: 80
  integrations:
    flutter_localization:
      enabled: true
      class_name: S
'''

    def _generate_l10n_config(self, locales: List[str]) -> str:
        """Generate l10n.yaml configuration"""
        return f'''
arb-dir: lib/l10n
template-arb-file: app_en.arb
output-localization-file: app_localizations.dart
output-class: AppLocalizations
output-dir: lib/gen/l10n
preferred-supported-locales: [{', '.join([f'"{loc}"' for loc in locales])}]
'''

    def _generate_main_app_config(self, app_name: str, locales: List[str]) -> str:
        """Generate main app localization configuration"""
        return f'''
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'gen/l10n/app_localizations.dart';

class {app_name} extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    return MaterialApp(
      title: '{app_name}',
      localizationsDelegates: [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: [
        {', '.join([f"Locale('{loc}')" for loc in locales])},
      ],
      locale: Locale('en'), // Default locale
      home: MyHomePage(),
    );
  }}
}}

// Usage in widgets:
class LocalizedWidget extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    final l10n = AppLocalizations.of(context)!;
    
    return Column(
      children: [
        Text(l10n.welcome),
        Text(l10n.goodbye),
        ElevatedButton(
          onPressed: () {{}},
          child: Text(l10n.save),
        ),
      ],
    );
  }}
}}
'''

    def _generate_initial_arb_files(self, locales: List[str]) -> Dict[str, str]:
        """Generate initial ARB files with common translations"""
        arb_files = {}
        
        base_translations = {
            "welcome": "Welcome",
            "goodbye": "Goodbye", 
            "save": "Save",
            "cancel": "Cancel",
            "delete": "Delete",
            "edit": "Edit",
            "loading": "Loading...",
            "error": "Error",
            "success": "Success"
        }
        
        for locale in locales:
            arb_content = {"@@locale": locale}
            
            for key, value in base_translations.items():
                # For base locale (usually English)
                if locale == 'en':
                    arb_content[key] = value
                    arb_content[f"@{key}"] = {"description": f"Translation for {key}"}
                else:
                    # Placeholder for other locales
                    arb_content[key] = f"TODO: Translate '{value}' to {locale}"
                    arb_content[f"@{key}"] = {"description": f"Translation for {key}"}
            
            arb_files[f'app_{locale}.arb'] = json.dumps(arb_content, indent=2)
        
        return arb_files

    def _generate_locale_provider(self, locales: List[str], storage_type: str) -> str:
        """Generate locale provider for state management"""
        return f'''
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LocaleProvider extends ChangeNotifier {{
  Locale _locale = Locale('en');
  
  Locale get locale => _locale;
  
  static const String _localeKey = 'selected_locale';
  
  LocaleProvider() {{
    _loadLocale();
  }}
  
  void setLocale(Locale locale) async {{
    if (!supportedLocales.contains(locale)) return;
    
    _locale = locale;
    notifyListeners();
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_localeKey, locale.languageCode);
  }}
  
  void _loadLocale() async {{
    final prefs = await SharedPreferences.getInstance();
    final localeCode = prefs.getString(_localeKey) ?? 'en';
    _locale = Locale(localeCode);
    notifyListeners();
  }}
  
  static List<Locale> supportedLocales = [
    {', '.join([f"Locale('{loc}')" for loc in locales])},
  ];
}}
'''

    def _generate_locale_switcher_ui(self, locales: List[str]) -> str:
        """Generate UI for locale switching"""
        locale_options = ', '.join([f"'{loc}': '{self._get_locale_display_name(loc)}'" for loc in locales])
        
        return f'''
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'locale_provider.dart';

class LocaleSwitcher extends StatelessWidget {{
  final Map<String, String> localeNames = {{
    {locale_options}
  }};
  
  @override
  Widget build(BuildContext context) {{
    final localeProvider = Provider.of<LocaleProvider>(context);
    
    return DropdownButton<Locale>(
      value: localeProvider.locale,
      icon: Icon(Icons.language),
      items: LocaleProvider.supportedLocales.map((Locale locale) {{
        return DropdownMenuItem<Locale>(
          value: locale,
          child: Row(
            children: [
              Text(localeNames[locale.languageCode] ?? locale.languageCode),
              SizedBox(width: 8),
              Text('(${locale.languageCode.toUpperCase()})', 
                   style: TextStyle(color: Colors.grey)),
            ],
          ),
        );
      }}).toList(),
      onChanged: (Locale? newLocale) {{
        if (newLocale != null) {{
          localeProvider.setLocale(newLocale);
        }}
      }},
    );
  }}
}}

// Settings page integration
class LanguageSettingsPage extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: AppBar(title: Text('Language Settings')),
      body: ListView(
        children: [
          ListTile(
            leading: Icon(Icons.language),
            title: Text('Language'),
            subtitle: Text('Select your preferred language'),
            trailing: LocaleSwitcher(),
          ),
        ],
      ),
    );
  }}
}}
'''

    def _get_locale_display_name(self, locale_code: str) -> str:
        """Get display name for locale"""
        locale_names = {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português',
            'ru': 'Русский',
            'zh': '中文',
            'ja': '日本語',
            'ko': '한국어',
            'ar': 'العربية',
            'hi': 'हिन्दी',
            'tr': 'Türkçe'
        }
        return locale_names.get(locale_code, locale_code.capitalize())

    def _generate_translation_key(self, string_value: str) -> str:
        """Generate a translation key from string value"""
        # Convert to snake_case and remove special characters
        key = re.sub(r'[^a-zA-Z0-9\s]', '', string_value.lower())
        key = re.sub(r'\s+', '_', key.strip())
        return key[:50]  # Limit key length

    async def _analyze_localization_needs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze localization requirements"""
        app_description = request.get('description', '')
        target_markets = request.get('target_markets', [])
        
        recommended_locales = self._recommend_locales(target_markets)
        complexity_analysis = self._analyze_l10n_complexity(app_description)
        
        return {
            'success': True,
            'recommended_locales': recommended_locales,
            'complexity_analysis': complexity_analysis,
            'implementation_roadmap': self._create_implementation_roadmap(complexity_analysis),
            'best_practices': self._get_localization_best_practices(),
            'agent_id': self.agent_id
        }

    def _get_localization_best_practices(self) -> List[str]:
        """Get localization best practices"""
        return [
            "Always use ARB files for translation management",
            "Implement proper pluralization rules for each locale",
            "Test RTL languages thoroughly",
            "Use context-aware translations where needed",
            "Implement proper date and number formatting",
            "Consider cultural differences in UI design",
            "Validate translations with native speakers",
            "Use translation management tools for large projects",
            "Implement fallback mechanisms for missing translations",
            "Test locale switching functionality extensively"
        ]
