"""
Navigation Agent - Manages Flutter app navigation and routing.
Handles route configuration, deep linking, and navigation patterns.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.agent_types import (
    AgentType, AgentMessage, MessageType, TaskDefinition, TaskStatus,
    AgentResponse, Priority, ProjectContext, WorkflowState
)
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class NavigationAgent(BaseAgent):
    """
    Specialized agent for Flutter navigation and routing.
    Handles route configuration, deep linking, nested navigation, and navigation patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.NAVIGATION_ROUTING, config)
        
        # Navigation frameworks
        self.navigation_frameworks = {
            "go_router": "go_router",
            "auto_route": "auto_route",
            "flutter_navigator": "flutter_navigator",
            "beamer": "beamer"
        }
        
        # Configuration
        self.preferred_framework = config.get("navigation_framework", "go_router")
        self.enable_deep_links = config.get("enable_deep_links", True)
        self.enable_guards = config.get("enable_guards", True)
        self.enable_nested_navigation = config.get("enable_nested_navigation", True)
        
    def _define_capabilities(self) -> List[str]:
        """Define navigation agent capabilities."""
        return [
            "route_configuration",
            "go_router_setup",
            "auto_route_implementation",
            "deep_link_handling",
            "navigation_guard_implementation",
            "nested_navigation_setup",
            "tab_navigation_configuration",
            "navigation_animation_setup",
            "route_parameter_handling",
            "navigation_state_management",
            "bottom_navigation_setup",
            "drawer_navigation_setup",
            "modal_navigation_handling",
            "custom_route_transitions"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process navigation tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'navigation_setup')
            
            if task_type == "navigation_setup":
                return await self._handle_navigation_setup(state)
            elif task_type == "route_generation":
                return await self._handle_route_generation(state)
            elif task_type == "deep_link_setup":
                return await self._handle_deep_link_setup(state)
            elif task_type == "navigation_guards":
                return await self._handle_navigation_guards(state)
            elif task_type == "tab_navigation":
                return await self._handle_tab_navigation(state)
            elif task_type == "nested_navigation":
                return await self._handle_nested_navigation(state)
            else:
                return await self._handle_default_navigation_setup(state)
                
        except Exception as e:
            logger.error(f"Navigation agent error: {e}")
            return self._create_error_response(f"Navigation setup failed: {str(e)}")
    
    async def _handle_navigation_setup(self, state: WorkflowState) -> AgentResponse:
        """Set up navigation framework and routing."""
        try:
            project_context = state.project_context
            routes = self.current_task.parameters.get("routes", [])
            framework = self.current_task.parameters.get("framework", self.preferred_framework)
            
            generated_files = []
            
            if framework == "go_router":
                navigation_setup = await self._generate_go_router_setup(routes, project_context)
            elif framework == "auto_route":
                navigation_setup = await self._generate_auto_route_setup(routes, project_context)
            else:
                navigation_setup = await self._generate_flutter_navigator_setup(routes, project_context)
            
            generated_files.extend(navigation_setup)
            
            # Generate route definitions
            route_definitions = await self._generate_route_definitions(routes)
            generated_files.append({
                "file_name": "AppRoutes",
                "file_path": "lib/core/app_routes.dart",
                "content": route_definitions
            })
            
            # Generate navigation service
            navigation_service = await self._generate_navigation_service(framework)
            generated_files.append({
                "file_name": "NavigationService",
                "file_path": "lib/services/navigation_service.dart",
                "content": navigation_service
            })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Navigation setup completed using {framework}",
                data={
                    "generated_files": generated_files,
                    "framework": framework,
                    "deep_links_enabled": self.enable_deep_links,
                    "guards_enabled": self.enable_guards
                },
                metadata={
                    "file_count": len(generated_files),
                    "route_count": len(routes)
                }
            )
            
        except Exception as e:
            logger.error(f"Navigation setup failed: {e}")
            return self._create_error_response(f"Navigation setup failed: {str(e)}")
    
    async def _generate_go_router_setup(self, routes: List[Dict[str, Any]], project_context: ProjectContext) -> List[Dict[str, Any]]:
        """Generate GoRouter configuration."""
        router_config = f"""// GoRouter Configuration
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

{self._generate_imports_for_routes(routes)}

class AppRouter {{
  static final GoRouter _router = GoRouter(
    routes: [
{self._generate_go_routes(routes)}
    ],
    initialLocation: '/',
    errorBuilder: (context, state) => ErrorPage(error: state.error),
    redirect: (context, state) {{
      return _handleRedirect(context, state);
    }},
  );

  static GoRouter get router => _router;

  static String? _handleRedirect(BuildContext context, GoRouterState state) {{
    // Implement authentication and other redirect logic
    return null;
  }}
}}

class ErrorPage extends StatelessWidget {{
  final Exception? error;
  
  const ErrorPage({{Key? key, this.error}}) : super(key: key);

  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: AppBar(title: const Text('Error')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text('An error occurred: ${{error?.toString() ?? "Unknown error"}}'),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => context.go('/'),
              child: const Text('Go Home'),
            ),
          ],
        ),
      ),
    );
  }}
}}
"""
        
        return [{
            "file_name": "AppRouter",
            "file_path": "lib/core/app_router.dart",
            "content": router_config
        }]
    
    def _generate_go_routes(self, routes: List[Dict[str, Any]]) -> str:
        """Generate GoRoute configurations."""
        route_configs = []
        
        for route in routes:
            path = route.get("path", "/")
            name = route.get("name", "")
            widget = route.get("widget", "HomePage")
            children = route.get("children", [])
            
            route_config = f"""      GoRoute(
        path: '{path}',
        name: '{name}',
        builder: (context, state) => const {widget}(),"""
            
            if children:
                route_config += f"""
        routes: [
{self._generate_go_routes(children)}
        ],"""
            
            route_config += "\n      ),"
            route_configs.append(route_config)
        
        return "\n".join(route_configs)
    
    async def _generate_auto_route_setup(self, routes: List[Dict[str, Any]], project_context: ProjectContext) -> List[Dict[str, Any]]:
        """Generate AutoRoute configuration."""
        app_router = f"""// AutoRoute Configuration
import 'package:auto_route/auto_route.dart';
import 'package:flutter/material.dart';

{self._generate_imports_for_routes(routes)}

part 'app_router.gr.dart';

@AutoRouterConfig()
class AppRouter extends _$AppRouter {{
  @override
  List<AutoRoute> get routes => [
{self._generate_auto_routes(routes)}
  ];
}}
"""
        
        return [{
            "file_name": "AppRouter",
            "file_path": "lib/core/app_router.dart", 
            "content": app_router
        }]
    
    def _generate_auto_routes(self, routes: List[Dict[str, Any]]) -> str:
        """Generate AutoRoute configurations."""
        route_configs = []
        
        for route in routes:
            path = route.get("path", "/")
            widget = route.get("widget", "HomePage")
            initial = route.get("initial", False)
            
            route_config = f"""    AutoRoute(
      page: {widget}Route.page,
      path: '{path}',
      initial: {str(initial).lower()},
    ),"""
            
            route_configs.append(route_config)
        
        return "\n".join(route_configs)
    
    async def _generate_flutter_navigator_setup(self, routes: List[Dict[str, Any]], project_context: ProjectContext) -> List[Dict[str, Any]]:
        """Generate Flutter Navigator 2.0 setup."""
        route_delegate = f"""// Route Delegate
import 'package:flutter/material.dart';

class AppRouteDelegate extends RouterDelegate<AppRouteConfiguration>
    with ChangeNotifier, PopNavigatorRouterDelegateMixin<AppRouteConfiguration> {{
  
  @override
  final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();
  
  AppRouteConfiguration _configuration = AppRouteConfiguration.home();
  
  AppRouteConfiguration get configuration => _configuration;
  
  @override
  AppRouteConfiguration get currentConfiguration => _configuration;
  
  @override
  Widget build(BuildContext context) {{
    return Navigator(
      key: navigatorKey,
      pages: _buildPages(),
      onPopPage: _onPopPage,
    );
  }}
  
  List<Page> _buildPages() {{
    final List<Page> pages = [];
    
    // Build pages based on current configuration
    switch (_configuration.routeName) {{
{self._generate_navigator_pages(routes)}
      default:
        pages.add(const MaterialPage(child: HomePage()));
    }}
    
    return pages;
  }}
  
  bool _onPopPage(Route<dynamic> route, dynamic result) {{
    if (!route.didPop(result)) return false;
    
    // Handle back navigation
    _configuration = AppRouteConfiguration.home();
    notifyListeners();
    
    return true;
  }}
  
  @override
  Future<void> setNewRoutePath(AppRouteConfiguration configuration) async {{
    _configuration = configuration;
    notifyListeners();
  }}
  
  void navigateTo(String routeName, [Map<String, dynamic>? arguments]) {{
    _configuration = AppRouteConfiguration(routeName, arguments);
    notifyListeners();
  }}
}}

class AppRouteConfiguration {{
  final String routeName;
  final Map<String, dynamic>? arguments;
  
  AppRouteConfiguration(this.routeName, [this.arguments]);
  
  AppRouteConfiguration.home() : this('/');
  AppRouteConfiguration.profile() : this('/profile');
  // Add more named constructors as needed
}}
"""
        
        return [{
            "file_name": "AppRouteDelegate",
            "file_path": "lib/core/app_route_delegate.dart",
            "content": route_delegate
        }]
    
    def _generate_navigator_pages(self, routes: List[Dict[str, Any]]) -> str:
        """Generate Navigator pages for Flutter Navigator 2.0."""
        page_configs = []
        
        for route in routes:
            path = route.get("path", "/")
            widget = route.get("widget", "HomePage")
            
            page_config = f"""      case '{path}':
        pages.add(MaterialPage(child: const {widget}()));
        break;"""
        
        page_configs.append(page_config)
    
        return "\n".join(page_configs)
    
    def _generate_imports_for_routes(self, routes: List[Dict[str, Any]]) -> str:
        """Generate imports for route widgets."""
        imports = set()
        
        for route in routes:
            widget = route.get("widget", "")
            if widget:
                # Convert widget name to file path
                file_name = self._widget_to_file_name(widget)
                imports.add(f"import '../pages/{file_name}.dart';")
        
        return "\n".join(sorted(imports))
    
    def _widget_to_file_name(self, widget_name: str) -> str:
        """Convert widget name to file name."""
        # Convert CamelCase to snake_case
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', widget_name).lower()
    
    async def _generate_route_definitions(self, routes: List[Dict[str, Any]]) -> str:
        """Generate route definitions and constants."""
        return f"""// Route Definitions
class AppRoutes {{
  // Route names
{self._generate_route_constants(routes)}

  // Route paths
{self._generate_path_constants(routes)}

  // All routes map
  static final Map<String, String> routes = {{
{self._generate_routes_map(routes)}
  }};

  // Route validation
  static bool isValidRoute(String route) {{
    return routes.containsKey(route);
  }}

  // Get path by name
  static String? getPath(String name) {{
    return routes[name];
  }}

  // Build route with parameters
  static String buildRoute(String route, [Map<String, String>? params]) {{
    if (params == null || params.isEmpty) return route;
    
    String result = route;
    params.forEach((key, value) {{
      result = result.replaceAll(':$key', value);
    }});
    
    return result;
  }}
}}
"""
    
    def _generate_route_constants(self, routes: List[Dict[str, Any]]) -> str:
        """Generate route name constants."""
        constants = []
        
        for route in routes:
            name = route.get("name", "")
            if name:
                constant_name = name.upper().replace(" ", "_")
                constants.append(f"  static const String {constant_name} = '{name}';")
        
        return "\n".join(constants)
    
    def _generate_path_constants(self, routes: List[Dict[str, Any]]) -> str:
        """Generate route path constants."""
        constants = []
        
        for route in routes:
            name = route.get("name", "")
            path = route.get("path", "/")
            if name:
                constant_name = f"{name.upper().replace(' ', '_')}_PATH"
                constants.append(f"  static const String {constant_name} = '{path}';")
        
        return "\n".join(constants)
    
    def _generate_routes_map(self, routes: List[Dict[str, Any]]) -> str:
        """Generate routes map."""
        mappings = []
        
        for route in routes:
            name = route.get("name", "")
            path = route.get("path", "/")
            if name:
                mappings.append(f"    '{name}': '{path}',")
        
        return "\n".join(mappings)
    
    async def _generate_navigation_service(self, framework: str) -> str:
        """Generate navigation service."""
        if framework == "go_router":
            return self._generate_go_router_service()
        elif framework == "auto_route":
            return self._generate_auto_route_service()
        else:
            return self._generate_flutter_navigator_service()
    
    def _generate_go_router_service(self) -> str:
        """Generate navigation service for GoRouter."""
        return """// Navigation Service for GoRouter
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class NavigationService {
  static final NavigationService _instance = NavigationService._internal();
  factory NavigationService() => _instance;
  NavigationService._internal();

  // Navigation methods
  void push(BuildContext context, String route, {Object? extra}) {
    context.push(route, extra: extra);
  }

  void pushReplacement(BuildContext context, String route, {Object? extra}) {
    context.pushReplacement(route, extra: extra);
  }

  void go(BuildContext context, String route, {Object? extra}) {
    context.go(route, extra: extra);
  }

  void pop(BuildContext context, [Object? result]) {
    context.pop(result);
  }

  void popUntil(BuildContext context, String route) {
    while (GoRouter.of(context).location != route) {
      if (!context.canPop()) break;
      context.pop();
    }
  }

  void pushAndClearStack(BuildContext context, String route, {Object? extra}) {
    context.go(route, extra: extra);
  }

  // Route checking
  bool canPop(BuildContext context) {
    return context.canPop();
  }

  String getCurrentRoute(BuildContext context) {
    return GoRouter.of(context).location;
  }

  // Deep link handling
  void handleDeepLink(BuildContext context, String link) {
    try {
      final uri = Uri.parse(link);
      context.go(uri.path, extra: uri.queryParameters);
    } catch (e) {
      // Handle invalid deep link
      context.go('/');
    }
  }

  // Navigation with parameters
  void pushWithParams(BuildContext context, String route, Map<String, String> params) {
    String finalRoute = route;
    params.forEach((key, value) {
      finalRoute = finalRoute.replaceAll(':$key', value);
    });
    context.push(finalRoute);
  }
}
"""
    
    def _generate_auto_route_service(self) -> str:
        """Generate navigation service for AutoRoute."""
        return """// Navigation Service for AutoRoute
import 'package:auto_route/auto_route.dart';
import 'package:flutter/material.dart';

class NavigationService {
  static final NavigationService _instance = NavigationService._internal();
  factory NavigationService() => _instance;
  NavigationService._internal();

  // Navigation methods
  Future<T?> push<T extends Object?>(BuildContext context, PageRouteInfo route) {
    return context.router.push<T>(route);
  }

  Future<void> pushAndClearStack(BuildContext context, PageRouteInfo route) {
    return context.router.pushAndClearStack(route);
  }

  Future<T?> pushReplacement<T extends Object?>(BuildContext context, PageRouteInfo route) {
    return context.router.pushReplacement<T>(route);
  }

  void pop<T extends Object?>(BuildContext context, [T? result]) {
    context.router.pop<T>(result);
  }

  void popUntil(BuildContext context, bool Function(Route) predicate) {
    context.router.popUntil(predicate);
  }

  void popUntilRoot(BuildContext context) {
    context.router.popUntilRoot();
  }

  // Route information
  String getCurrentRoute(BuildContext context) {
    return context.router.current.name;
  }

  bool canPop(BuildContext context) {
    return context.router.canPop();
  }

  // Nested navigation
  void pushToTab(BuildContext context, PageRouteInfo route, int tabIndex) {
    // Implementation for tab-specific navigation
    context.router.push(route);
  }
}
"""
    
    def _generate_flutter_navigator_service(self) -> str:
        """Generate navigation service for Flutter Navigator."""
        return """// Navigation Service for Flutter Navigator
import 'package:flutter/material.dart';

class NavigationService {
  static final NavigationService _instance = NavigationService._internal();
  factory NavigationService() => _instance;
  NavigationService._internal();

  final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

  BuildContext? get context => navigatorKey.currentContext;

  // Navigation methods
  Future<T?> push<T>(String routeName, {Object? arguments}) {
    return navigatorKey.currentState!.pushNamed<T>(routeName, arguments: arguments);
  }

  Future<T?> pushReplacement<T>(String routeName, {Object? arguments}) {
    return navigatorKey.currentState!.pushReplacementNamed<T>(routeName, arguments: arguments);
  }

  Future<T?> pushAndClearStack<T>(String routeName, {Object? arguments}) {
    return navigatorKey.currentState!.pushNamedAndRemoveUntil<T>(
      routeName, 
      (route) => false,
      arguments: arguments,
    );
  }

  void pop<T>([T? result]) {
    navigatorKey.currentState!.pop<T>(result);
  }

  void popUntil(String routeName) {
    navigatorKey.currentState!.popUntil(ModalRoute.withName(routeName));
  }

  bool canPop() {
    return navigatorKey.currentState!.canPop();
  }

  // Modal navigation
  Future<T?> showModal<T>(Widget child) {
    return showDialog<T>(
      context: context!,
      builder: (context) => child,
    );
  }

  Future<T?> showBottomSheet<T>(Widget child) {
    return showModalBottomSheet<T>(
      context: context!,
      builder: (context) => child,
    );
  }
}
"""
    
    async def _handle_deep_link_setup(self, state: WorkflowState) -> AgentResponse:
        """Set up deep linking configuration."""
        try:
            deep_link_config = self._generate_deep_link_config()
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content="Deep linking setup completed",
                data={
                    "deep_link_config": deep_link_config,
                    "android_intent_filter": self._generate_android_intent_filter(),
                    "ios_url_scheme": self._generate_ios_url_scheme()
                }
            )
            
        except Exception as e:
            logger.error(f"Deep link setup failed: {e}")
            return self._create_error_response(f"Deep link setup failed: {str(e)}")
    
    def _generate_deep_link_config(self) -> str:
        """Generate deep link configuration."""
        return """// Deep Link Configuration
class DeepLinkConfig {
  static const String scheme = 'myapp';
  static const String host = 'example.com';
  
  // Deep link patterns
  static final Map<String, String> patterns = {
    r'^/profile/(\d+)$': '/profile',
    r'^/product/([a-zA-Z0-9]+)$': '/product',
    r'^/settings': '/settings',
  };
  
  static String? matchRoute(String path) {
    for (final pattern in patterns.keys) {
      if (RegExp(pattern).hasMatch(path)) {
        return patterns[pattern];
      }
    }
    return null;
  }
  
  static Map<String, String> extractParams(String path, String pattern) {
    final regex = RegExp(pattern);
    final match = regex.firstMatch(path);
    
    if (match == null) return {};
    
    final params = <String, String>{};
    for (int i = 1; i <= match.groupCount; i++) {
      params['param$i'] = match.group(i) ?? '';
    }
    
    return params;
  }
}
"""
    
    def _generate_android_intent_filter(self) -> str:
        """Generate Android intent filter configuration."""
        return """<!-- Add to android/app/src/main/AndroidManifest.xml -->
<activity
    android:name=".MainActivity"
    android:launchMode="singleTop"
    android:theme="@style/LaunchTheme">
    <!-- Existing intent filter -->
    <intent-filter android:autoVerify="true">
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
    </intent-filter>
    
    <!-- Deep link intent filter -->
    <intent-filter android:autoVerify="true">
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
        <category android:name="android.intent.category.BROWSABLE" />
        <data android:scheme="myapp" />
    </intent-filter>
    
    <!-- HTTP/HTTPS deep links -->
    <intent-filter android:autoVerify="true">
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
        <category android:name="android.intent.category.BROWSABLE" />
        <data android:scheme="https"
              android:host="example.com" />
    </intent-filter>
</activity>
"""
    
    def _generate_ios_url_scheme(self) -> str:
        """Generate iOS URL scheme configuration."""
        return """<!-- Add to ios/Runner/Info.plist -->
<key>CFBundleURLTypes</key>
<array>
    <dict>
        <key>CFBundleURLName</key>
        <string>example.com</string>
        <key>CFBundleURLSchemes</key>
        <array>
            <string>myapp</string>
        </array>
    </dict>
    <dict>
        <key>CFBundleURLName</key>
        <string>example.com</string>
        <key>CFBundleURLSchemes</key>
        <array>
            <string>https</string>
        </array>
    </dict>
</array>
"""
    
    async def _handle_default_navigation_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle default navigation setup workflow."""
        try:
            # Default workflow: setup navigation with routing and deep links
            navigation_result = await self._handle_navigation_setup(state)
            
            if navigation_result.status == TaskStatus.FAILED:
                return navigation_result
            
            # Add deep linking if enabled
            if self.enable_deep_links:
                deep_link_result = await self._handle_deep_link_setup(state)
                navigation_result.data["deep_link_setup"] = deep_link_result.data
            
            return navigation_result
            
        except Exception as e:
            logger.error(f"Default navigation setup failed: {e}")
            return self._create_error_response(f"Navigation setup workflow failed: {str(e)}")
