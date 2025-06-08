"""
Accessibility Agent for Flutter Development
Handles accessibility implementation, WCAG compliance, and inclusive design
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from core.agent_types import AgentType

class AccessibilityAgent(BaseAgent):
    def __init__(self, agent_id: str = "accessibility_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ACCESSIBILITY,
            name="Accessibility Agent", 
            description="Specialized agent for Flutter accessibility implementation and WCAG compliance"
        )
        
        self.capabilities = [
            "semantic_labels",
            "screen_reader_support", 
            "keyboard_navigation",
            "focus_management",
            "color_contrast_analysis",
            "accessibility_testing",
            "voice_over_support",
            "talkback_support",
            "wcag_compliance",
            "accessible_widgets",
            "gesture_alternatives",
            "accessibility_announcements"
        ]
        
        self.wcag_levels = ['A', 'AA', 'AAA']
        self.accessibility_features = [
            'semantic_labels', 'focus_order', 'keyboard_navigation',
            'screen_reader', 'high_contrast', 'large_text', 'voice_control'
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process accessibility-related requests"""
        try:
            request_type = request.get('type', '')
            
            if request_type == 'audit_accessibility':
                return await self._audit_accessibility(request)
            elif request_type == 'add_semantic_labels':
                return await self._add_semantic_labels(request)
            elif request_type == 'implement_keyboard_navigation':
                return await self._implement_keyboard_navigation(request)
            elif request_type == 'setup_focus_management':
                return await self._setup_focus_management(request)
            elif request_type == 'check_color_contrast':
                return await self._check_color_contrast(request)
            elif request_type == 'create_accessible_widget':
                return await self._create_accessible_widget(request)
            elif request_type == 'setup_accessibility_testing':
                return await self._setup_accessibility_testing(request)
            else:
                return await self._analyze_accessibility_requirements(request)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Accessibility processing failed: {str(e)}",
                'agent_id': self.agent_id
            }

    async def _audit_accessibility(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive accessibility audit"""
        widget_code = request.get('code', '')
        target_level = request.get('wcag_level', 'AA')
        
        audit_results = {
            'semantic_issues': [],
            'contrast_issues': [],
            'navigation_issues': [],
            'focus_issues': [],
            'screen_reader_issues': []
        }
        
        # Analyze semantic labels
        semantic_analysis = self._analyze_semantic_labels(widget_code)
        audit_results['semantic_issues'] = semantic_analysis['issues']
        
        # Analyze focus management
        focus_analysis = self._analyze_focus_management(widget_code)
        audit_results['focus_issues'] = focus_analysis['issues']
        
        # Analyze keyboard navigation
        keyboard_analysis = self._analyze_keyboard_navigation(widget_code)
        audit_results['navigation_issues'] = keyboard_analysis['issues']
        
        # Generate recommendations
        recommendations = self._generate_accessibility_recommendations(audit_results, target_level)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(audit_results)
        
        return {
            'success': True,
            'audit_results': audit_results,
            'recommendations': recommendations,
            'compliance_score': compliance_score,
            'wcag_level': target_level,
            'priority_fixes': self._prioritize_fixes(audit_results),
            'agent_id': self.agent_id
        }

    async def _add_semantic_labels(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add semantic labels to widgets"""
        widget_code = request.get('code', '')
        widget_type = request.get('widget_type', 'general')
        
        # Generate semantic labels based on widget type
        semantic_enhancements = self._generate_semantic_enhancements(widget_code, widget_type)
        
        # Create enhanced code with semantic labels
        enhanced_code = self._apply_semantic_enhancements(widget_code, semantic_enhancements)
        
        return {
            'success': True,
            'enhanced_code': enhanced_code,
            'semantic_enhancements': semantic_enhancements,
            'testing_instructions': self._get_semantic_testing_instructions(),
            'best_practices': self._get_semantic_best_practices(),
            'agent_id': self.agent_id
        }

    async def _implement_keyboard_navigation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement keyboard navigation support"""
        widget_code = request.get('code', '')
        navigation_type = request.get('navigation_type', 'tab_order')
        
        # Generate keyboard navigation implementation
        keyboard_code = self._generate_keyboard_navigation(widget_code, navigation_type)
        
        # Generate focus handling utilities
        focus_utilities = self._generate_focus_utilities()
        
        return {
            'success': True,
            'keyboard_navigation_code': keyboard_code,
            'focus_utilities': focus_utilities,
            'keyboard_shortcuts': self._get_keyboard_shortcuts(),
            'testing_guide': self._get_keyboard_testing_guide(),
            'agent_id': self.agent_id
        }

    async def _setup_focus_management(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Setup focus management system"""
        screen_type = request.get('screen_type', 'form')
        focus_strategy = request.get('focus_strategy', 'sequential')
        
        # Generate focus management implementation
        focus_manager_code = self._generate_focus_manager(screen_type, focus_strategy)
        
        # Generate focus node management
        focus_node_code = self._generate_focus_node_management()
        
        # Generate focus restoration logic
        focus_restoration = self._generate_focus_restoration()
        
        return {
            'success': True,
            'focus_manager_code': focus_manager_code,
            'focus_node_code': focus_node_code,
            'focus_restoration': focus_restoration,
            'focus_strategies': self._get_focus_strategies(),
            'implementation_tips': self._get_focus_implementation_tips(),
            'agent_id': self.agent_id
        }

    async def _check_color_contrast(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check color contrast for accessibility compliance"""
        colors = request.get('colors', {})
        target_level = request.get('wcag_level', 'AA')
        
        contrast_results = []
        
        for color_pair_name, color_data in colors.items():
            foreground = color_data.get('foreground', '#000000')
            background = color_data.get('background', '#FFFFFF')
            
            contrast_ratio = self._calculate_contrast_ratio(foreground, background)
            compliance = self._check_contrast_compliance(contrast_ratio, target_level)
            
            contrast_results.append({
                'name': color_pair_name,
                'foreground': foreground,
                'background': background,
                'contrast_ratio': contrast_ratio,
                'compliant': compliance['compliant'],
                'required_ratio': compliance['required_ratio'],
                'suggestions': self._get_color_suggestions(foreground, background, compliance)
            })
        
        return {
            'success': True,
            'contrast_results': contrast_results,
            'overall_compliance': all(result['compliant'] for result in contrast_results),
            'color_palette_suggestions': self._generate_accessible_color_palette(),
            'testing_tools': self._get_contrast_testing_tools(),
            'agent_id': self.agent_id
        }

    async def _create_accessible_widget(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create fully accessible widget implementation"""
        widget_type = request.get('widget_type', 'button')
        functionality = request.get('functionality', {})
        
        # Generate accessible widget code
        accessible_code = self._generate_accessible_widget(widget_type, functionality)
        
        # Generate usage examples
        usage_examples = self._generate_accessible_usage_examples(widget_type)
        
        # Generate testing code
        testing_code = self._generate_accessibility_tests(widget_type)
        
        return {
            'success': True,
            'accessible_widget_code': accessible_code,
            'usage_examples': usage_examples,
            'testing_code': testing_code,
            'accessibility_features': self._get_widget_accessibility_features(widget_type),
            'customization_options': self._get_accessibility_customization_options(),
            'agent_id': self.agent_id
        }

    async def _setup_accessibility_testing(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Setup accessibility testing framework"""
        testing_scope = request.get('scope', 'widget')
        automation_level = request.get('automation_level', 'basic')
        
        # Generate testing utilities
        testing_utilities = self._generate_accessibility_testing_utilities()
        
        # Generate test cases
        test_cases = self._generate_accessibility_test_cases(testing_scope)
        
        # Generate CI/CD integration
        ci_integration = self._generate_accessibility_ci_integration()
        
        return {
            'success': True,
            'testing_utilities': testing_utilities,
            'test_cases': test_cases,
            'ci_integration': ci_integration,
            'testing_checklist': self._get_accessibility_testing_checklist(),
            'manual_testing_guide': self._get_manual_testing_guide(),
            'agent_id': self.agent_id
        }

    def _generate_accessible_widget(self, widget_type: str, functionality: Dict[str, Any]) -> str:
        """Generate accessible widget implementation"""
        if widget_type == 'button':
            return self._generate_accessible_button(functionality)
        elif widget_type == 'form':
            return self._generate_accessible_form(functionality)
        elif widget_type == 'list':
            return self._generate_accessible_list(functionality)
        elif widget_type == 'dialog':
            return self._generate_accessible_dialog(functionality)
        else:
            return self._generate_generic_accessible_widget(widget_type, functionality)

    def _generate_accessible_button(self, functionality: Dict[str, Any]) -> str:
        """Generate accessible button widget"""
        button_text = functionality.get('text', 'Button')
        action_description = functionality.get('action_description', 'Performs an action')
        
        return f'''
class AccessibleButton extends StatelessWidget {{
  final String text;
  final VoidCallback? onPressed;
  final String? semanticLabel;
  final String? tooltip;
  final bool enabled;
  
  const AccessibleButton({{
    Key? key,
    required this.text,
    this.onPressed,
    this.semanticLabel,
    this.tooltip,
    this.enabled = true,
  }}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {{
    return Semantics(
      label: semanticLabel ?? text,
      hint: '{action_description}',
      button: true,
      enabled: enabled,
      child: Tooltip(
        message: tooltip ?? text,
        child: ElevatedButton(
          onPressed: enabled ? onPressed : null,
          style: ElevatedButton.styleFrom(
            // Ensure minimum touch target size (44x44 points)
            minimumSize: Size(44, 44),
            // Ensure sufficient contrast
            foregroundColor: Theme.of(context).colorScheme.onPrimary,
            backgroundColor: Theme.of(context).colorScheme.primary,
          ),
          child: Text(
            text,
            style: TextStyle(
              fontSize: 16, // Minimum font size for readability
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }}
}}

// Usage example:
AccessibleButton(
  text: '{button_text}',
  semanticLabel: 'Button to {action_description.lower()}',
  tooltip: '{action_description}',
  onPressed: () {{
    // Announce action completion to screen readers
    SemanticsService.announce(
      'Action completed successfully',
      TextDirection.ltr,
    );
  }},
)
'''

    def _generate_accessible_form(self, functionality: Dict[str, Any]) -> str:
        """Generate accessible form widget"""
        fields = functionality.get('fields', ['email', 'password'])
        
        form_fields = []
        for field in fields:
            form_fields.append(f'''
        Semantics(
          label: '{field.capitalize()} field',
          hint: 'Enter your {field}',
          textField: true,
          child: TextFormField(
            decoration: InputDecoration(
              labelText: '{field.capitalize()}',
              hintText: 'Enter your {field}',
              border: OutlineInputBorder(),
              // Ensure sufficient contrast
              fillColor: Theme.of(context).colorScheme.surface,
              filled: true,
            ),
            validator: (value) {{
              if (value == null || value.isEmpty) {{
                return '{field.capitalize()} is required';
              }}
              return null;
            }},
            // Announce errors to screen readers
            onChanged: (value) {{
              // Validate and announce errors if needed
            }},
          ),
        ),
        SizedBox(height: 16),''')
        
        return f'''
class AccessibleForm extends StatefulWidget {{
  @override
  _AccessibleFormState createState() => _AccessibleFormState();
}}

class _AccessibleFormState extends State<AccessibleForm> {{
  final _formKey = GlobalKey<FormState>();
  final List<FocusNode> _focusNodes = [];
  
  @override
  void initState() {{
    super.initState();
    // Create focus nodes for each field
    for (int i = 0; i < {len(fields)}; i++) {{
      _focusNodes.add(FocusNode());
    }}
  }}
  
  @override
  void dispose() {{
    // Dispose focus nodes
    for (final node in _focusNodes) {{
      node.dispose();
    }}
    super.dispose();
  }}
  
  @override
  Widget build(BuildContext context) {{
    return Semantics(
      label: 'Registration form',
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Form title
            Semantics(
              header: true,
              child: Text(
                'Create Account',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
            ),
            SizedBox(height: 24),
            
            {''.join(form_fields)}
            
            // Submit button
            AccessibleButton(
              text: 'Create Account',
              semanticLabel: 'Create account button',
              tooltip: 'Submit form to create your account',
              onPressed: () {{
                if (_formKey.currentState!.validate()) {{
                  SemanticsService.announce(
                    'Account created successfully',
                    TextDirection.ltr,
                  );
                }}
              }},
            ),
          ],
        ),
      ),
    );
  }}
}}
'''

    def _generate_keyboard_navigation(self, widget_code: str, navigation_type: str) -> str:
        """Generate keyboard navigation implementation"""
        return '''
class KeyboardNavigationWidget extends StatefulWidget {
  @override
  _KeyboardNavigationWidgetState createState() => _KeyboardNavigationWidgetState();
}

class _KeyboardNavigationWidgetState extends State<KeyboardNavigationWidget> {
  final List<FocusNode> _focusNodes = [];
  int _currentFocusIndex = 0;
  
  @override
  void initState() {
    super.initState();
    // Initialize focus nodes
    for (int i = 0; i < 5; i++) {
      _focusNodes.add(FocusNode());
    }
  }
  
  @override
  void dispose() {
    for (final node in _focusNodes) {
      node.dispose();
    }
    super.dispose();
  }
  
  void _handleKeyEvent(RawKeyEvent event) {
    if (event is RawKeyDownEvent) {
      if (event.logicalKey == LogicalKeyboardKey.tab) {
        _moveFocus(event.isShiftPressed ? -1 : 1);
      } else if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
        _moveFocus(1);
      } else if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
        _moveFocus(-1);
      } else if (event.logicalKey == LogicalKeyboardKey.enter ||
                 event.logicalKey == LogicalKeyboardKey.space) {
        _activateCurrentItem();
      }
    }
  }
  
  void _moveFocus(int direction) {
    setState(() {
      _currentFocusIndex = (_currentFocusIndex + direction) % _focusNodes.length;
      if (_currentFocusIndex < 0) {
        _currentFocusIndex = _focusNodes.length - 1;
      }
      _focusNodes[_currentFocusIndex].requestFocus();
    });
    
    // Announce focus change to screen readers
    SemanticsService.announce(
      'Focused item ${_currentFocusIndex + 1} of ${_focusNodes.length}',
      TextDirection.ltr,
    );
  }
  
  void _activateCurrentItem() {
    // Handle activation of current focused item
    SemanticsService.announce(
      'Item activated',
      TextDirection.ltr,
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return RawKeyboardListener(
      focusNode: FocusNode(),
      onKey: _handleKeyEvent,
      child: Column(
        children: List.generate(_focusNodes.length, (index) {
          return Focus(
            focusNode: _focusNodes[index],
            child: Builder(
              builder: (context) {
                final hasFocus = Focus.of(context).hasFocus;
                return Container(
                  decoration: BoxDecoration(
                    border: hasFocus 
                        ? Border.all(color: Theme.of(context).focusColor, width: 2)
                        : null,
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: ListTile(
                    title: Text('Item ${index + 1}'),
                    onTap: () => _activateCurrentItem(),
                  ),
                );
              },
            ),
          );
        }),
      ),
    );
  }
}
'''

    def _calculate_contrast_ratio(self, foreground: str, background: str) -> float:
        """Calculate contrast ratio between two colors"""
        # Convert hex to RGB
        def hex_to_rgb(hex_color: str) -> tuple:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def get_relative_luminance(rgb: tuple) -> float:
            """Calculate relative luminance of RGB color"""
            def normalize_rgb(c):
                c = c / 255.0
                return c / 12.92 if c <= 0.03928 else pow((c + 0.055) / 1.055, 2.4)
            
            r, g, b = [normalize_rgb(c) for c in rgb]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Calculate luminance for both colors
        fg_rgb = hex_to_rgb(foreground)
        bg_rgb = hex_to_rgb(background)
        
        fg_luminance = get_relative_luminance(fg_rgb)
        bg_luminance = get_relative_luminance(bg_rgb)
        
        # Calculate contrast ratio
        lighter = max(fg_luminance, bg_luminance)
        darker = min(fg_luminance, bg_luminance)
        
        return (lighter + 0.05) / (darker + 0.05)

    def _check_contrast_compliance(self, ratio: float, level: str) -> Dict[str, Any]:
        """Check if contrast ratio meets WCAG requirements"""
        requirements = {
            'A': {'normal': 3.0, 'large': 3.0},
            'AA': {'normal': 4.5, 'large': 3.0},
            'AAA': {'normal': 7.0, 'large': 4.5}
        }
        
        required_normal = requirements[level]['normal']
        required_large = requirements[level]['large']
        
        return {
            'compliant': ratio >= required_normal,
            'compliant_large_text': ratio >= required_large,
            'required_ratio': required_normal,
            'required_ratio_large': required_large,
            'actual_ratio': ratio
        }

    def _get_accessibility_testing_checklist(self) -> List[str]:
        """Get comprehensive accessibility testing checklist"""
        return [
            "Test with screen readers (VoiceOver on iOS, TalkBack on Android)",
            "Verify keyboard navigation works properly",
            "Check color contrast meets WCAG AA standards (4.5:1 for normal text)",
            "Ensure touch targets are at least 44x44 points",
            "Test with high contrast mode enabled",
            "Verify text scaling works up to 200%",
            "Check focus indicators are visible and clear",
            "Test voice control functionality",
            "Verify semantic labels are meaningful",
            "Check form validation messages are announced",
            "Test modal dialogs trap focus properly",
            "Verify error messages are associated with form fields",
            "Check loading states are announced to screen readers",
            "Test with reduced motion preferences",
            "Verify content reflows properly at different zoom levels"
        ]

    async def _analyze_accessibility_requirements(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accessibility requirements for the application"""
        app_type = request.get('app_type', 'general')
        target_audience = request.get('target_audience', [])
        compliance_level = request.get('compliance_level', 'AA')
        
        requirements = self._generate_accessibility_requirements(app_type, target_audience, compliance_level)
        implementation_plan = self._create_accessibility_implementation_plan(requirements)
        
        return {
            'success': True,
            'requirements': requirements,
            'implementation_plan': implementation_plan,
            'compliance_checklist': self._get_compliance_checklist(compliance_level),
            'testing_strategy': self._get_testing_strategy(app_type),
            'resources': self._get_accessibility_resources(),
            'agent_id': self.agent_id
        }

    def _get_accessibility_resources(self) -> Dict[str, List[str]]:
        """Get accessibility resources and tools"""
        return {
            'testing_tools': [
                'Accessibility Scanner (Android)',
                'Accessibility Inspector (iOS)',
                'Flutter Inspector',
                'Contrast ratio analyzers',
                'Screen reader testing'
            ],
            'guidelines': [
                'WCAG 2.1 Guidelines',
                'Flutter Accessibility Guide',
                'Material Design Accessibility',
                'iOS Human Interface Guidelines - Accessibility',
                'Android Accessibility Guidelines'
            ],
            'libraries': [
                'flutter/semantics',
                'flutter/services (SemanticsService)',
                'flutter_accessibility_service',
                'accessible_text_field'
            ]
        }
