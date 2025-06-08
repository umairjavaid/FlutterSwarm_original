"""
API Integration Agent - Handles network communication and API integration.
Manages HTTP clients, authentication, and API service generation.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse

from core.agent_types import (
    AgentType, AgentMessage, MessageType, TaskDefinition, TaskStatus,
    AgentResponse, Priority, ProjectContext, WorkflowState
)
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class APIIntegrationAgent(BaseAgent):
    """
    Specialized agent for API integration and network communication.
    Handles HTTP clients, REST/GraphQL APIs, authentication, and service generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.API_INTEGRATION, config)
        
        # API configuration
        self.http_clients = {
            "dio": "dio",
            "http": "http",
            "chopper": "chopper",
            "retrofit": "retrofit"
        }
        
        # Authentication methods
        self.auth_methods = {
            "jwt": "JSON Web Token",
            "oauth2": "OAuth 2.0",
            "api_key": "API Key",
            "basic": "Basic Authentication",
            "bearer": "Bearer Token"
        }
        
        # Configuration
        self.default_timeout = config.get("api_timeout", 30)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.generate_models = config.get("generate_models", True)
        self.preferred_client = config.get("preferred_client", "dio")
        
    def _define_capabilities(self) -> List[str]:
        """Define API integration agent capabilities."""
        return [
            "http_client_setup",
            "dio_configuration",
            "rest_api_service_generation",
            "graphql_client_setup",
            "websocket_implementation",
            "authentication_flow_handling",
            "api_error_handling",
            "request_retry_logic",
            "network_connectivity_handling",
            "api_model_generation",
            "json_serialization_setup",
            "api_response_parsing",
            "multipart_file_upload",
            "api_caching_implementation"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process API integration tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'api_service_generation')
            
            if task_type == "api_service_generation":
                return await self._handle_api_service_generation(state)
            elif task_type == "http_client_setup":
                return await self._handle_http_client_setup(state)
            elif task_type == "authentication_setup":
                return await self._handle_authentication_setup(state)
            elif task_type == "api_model_generation":
                return await self._handle_api_model_generation(state)
            elif task_type == "websocket_setup":
                return await self._handle_websocket_setup(state)
            elif task_type == "graphql_setup":
                return await self._handle_graphql_setup(state)
            else:
                return await self._handle_default_api_integration(state)
                
        except Exception as e:
            logger.error(f"API integration agent error: {e}")
            return self._create_error_response(f"API integration failed: {str(e)}")
    
    async def _handle_api_service_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate API service classes and methods."""
        try:
            api_spec = self.current_task.parameters.get("api_spec", {})
            base_url = api_spec.get("base_url", "https://api.example.com")
            endpoints = api_spec.get("endpoints", [])
            
            generated_services = []
            
            # Generate main API service
            main_service = await self._generate_main_api_service(base_url, endpoints)
            generated_services.append({
                "service_name": "ApiService",
                "file_path": "lib/services/api_service.dart",
                "content": main_service
            })
            
            # Generate specific endpoint services
            for endpoint_group in self._group_endpoints_by_resource(endpoints):
                resource_name = endpoint_group["resource"]
                service_content = await self._generate_resource_service(resource_name, endpoint_group["endpoints"])
                
                generated_services.append({
                    "service_name": f"{resource_name.capitalize()}Service",
                    "file_path": f"lib/services/{resource_name.lower()}_service.dart",
                    "content": service_content
                })
            
            # Generate API client configuration
            client_config = await self._generate_api_client_config(api_spec)
            generated_services.append({
                "service_name": "ApiClient",
                "file_path": "lib/core/api_client.dart",
                "content": client_config
            })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_services)} API service files",
                data={
                    "generated_services": generated_services,
                    "http_client": self.preferred_client,
                    "base_url": base_url
                },
                metadata={
                    "service_count": len(generated_services),
                    "endpoint_count": len(endpoints)
                }
            )
            
        except Exception as e:
            logger.error(f"API service generation failed: {e}")
            return self._create_error_response(f"API service generation failed: {str(e)}")
    
    async def _generate_main_api_service(self, base_url: str, endpoints: List[Dict[str, Any]]) -> str:
        """Generate the main API service class."""
        return f"""// Main API Service
import 'package:dio/dio.dart';
import '../core/api_client.dart';
import '../models/api_response.dart';

class ApiService {{
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  final ApiClient _client = ApiClient();

  // Base URL
  static const String baseUrl = '{base_url}';

  // HTTP client getter
  Dio get client => _client.dio;

  // Generic GET request
  Future<ApiResponse<T>> get<T>(
    String endpoint, {{
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(Map<String, dynamic>)? fromJson,
  }}) async {{
    try {{
      final response = await _client.dio.get(
        endpoint,
        queryParameters: queryParameters,
        options: options,
      );

      if (fromJson != null && response.data != null) {{
        return ApiResponse<T>.success(fromJson(response.data));
      }}

      return ApiResponse<T>.success(response.data as T);
    }} on DioException catch (e) {{
      return ApiResponse<T>.error(_handleDioError(e));
    }} catch (e) {{
      return ApiResponse<T>.error('Unexpected error: ${{e.toString()}}');
    }}
  }}

  // Generic POST request
  Future<ApiResponse<T>> post<T>(
    String endpoint, {{
    dynamic data,
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(Map<String, dynamic>)? fromJson,
  }}) async {{
    try {{
      final response = await _client.dio.post(
        endpoint,
        data: data,
        queryParameters: queryParameters,
        options: options,
      );

      if (fromJson != null && response.data != null) {{
        return ApiResponse<T>.success(fromJson(response.data));
      }}

      return ApiResponse<T>.success(response.data as T);
    }} on DioException catch (e) {{
      return ApiResponse<T>.error(_handleDioError(e));
    }} catch (e) {{
      return ApiResponse<T>.error('Unexpected error: ${{e.toString()}}');
    }}
  }}

  // Generic PUT request
  Future<ApiResponse<T>> put<T>(
    String endpoint, {{
    dynamic data,
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(Map<String, dynamic>)? fromJson,
  }}) async {{
    try {{
      final response = await _client.dio.put(
        endpoint,
        data: data,
        queryParameters: queryParameters,
        options: options,
      );

      if (fromJson != null && response.data != null) {{
        return ApiResponse<T>.success(fromJson(response.data));
      }}

      return ApiResponse<T>.success(response.data as T);
    }} on DioException catch (e) {{
      return ApiResponse<T>.error(_handleDioError(e));
    }} catch (e) {{
      return ApiResponse<T>.error('Unexpected error: ${{e.toString()}}');
    }}
  }}

  // Generic DELETE request
  Future<ApiResponse<T>> delete<T>(
    String endpoint, {{
    dynamic data,
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(Map<String, dynamic>)? fromJson,
  }}) async {{
    try {{
      final response = await _client.dio.delete(
        endpoint,
        data: data,
        queryParameters: queryParameters,
        options: options,
      );

      if (fromJson != null && response.data != null) {{
        return ApiResponse<T>.success(fromJson(response.data));
      }}

      return ApiResponse<T>.success(response.data as T);
    }} on DioException catch (e) {{
      return ApiResponse<T>.error(_handleDioError(e));
    }} catch (e) {{
      return ApiResponse<T>.error('Unexpected error: ${{e.toString()}}');
    }}
  }}

  // File upload
  Future<ApiResponse<T>> uploadFile<T>(
    String endpoint,
    String filePath, {{
    String fieldName = 'file',
    Map<String, dynamic>? additionalData,
    ProgressCallback? onSendProgress,
    T Function(Map<String, dynamic>)? fromJson,
  }}) async {{
    try {{
      final formData = FormData.fromMap({{
        fieldName: await MultipartFile.fromFile(filePath),
        ...?additionalData,
      }});

      final response = await _client.dio.post(
        endpoint,
        data: formData,
        options: Options(
          headers: {{'Content-Type': 'multipart/form-data'}},
        ),
        onSendProgress: onSendProgress,
      );

      if (fromJson != null && response.data != null) {{
        return ApiResponse<T>.success(fromJson(response.data));
      }}

      return ApiResponse<T>.success(response.data as T);
    }} on DioException catch (e) {{
      return ApiResponse<T>.error(_handleDioError(e));
    }} catch (e) {{
      return ApiResponse<T>.error('Unexpected error: ${{e.toString()}}');
    }}
  }}

  // Error handling
  String _handleDioError(DioException error) {{
    switch (error.type) {{
      case DioExceptionType.connectionTimeout:
        return 'Connection timeout';
      case DioExceptionType.sendTimeout:
        return 'Send timeout';
      case DioExceptionType.receiveTimeout:
        return 'Receive timeout';
      case DioExceptionType.badResponse:
        return 'Server error: ${{error.response?.statusCode}}';
      case DioExceptionType.cancel:
        return 'Request cancelled';
      case DioExceptionType.connectionError:
        return 'Connection error';
      default:
        return 'Network error: ${{error.message}}';
    }}
  }}
}}
"""
    
    async def _generate_resource_service(self, resource_name: str, endpoints: List[Dict[str, Any]]) -> str:
        """Generate a service for a specific resource."""
        class_name = f"{resource_name.capitalize()}Service"
        model_name = resource_name.capitalize()
        
        service_content = f"""// {class_name}
import '../core/api_service.dart';
import '../models/{resource_name.lower()}.dart';
import '../models/api_response.dart';

class {class_name} {{
  static final {class_name} _instance = {class_name}._internal();
  factory {class_name}() => _instance;
  {class_name}._internal();

  final ApiService _apiService = ApiService();

"""
        
        # Generate methods for each endpoint
        for endpoint in endpoints:
            method_name = endpoint.get("name", "").lower()
            http_method = endpoint.get("method", "GET").upper()
            path = endpoint.get("path", "/")
            
            if http_method == "GET":
                if "{id}" in path:
                    service_content += f"""
  // Get {resource_name} by ID
  Future<ApiResponse<{model_name}>> get{model_name}(String id) async {{
    return await _apiService.get<{model_name}>(
      '{path}'.replaceAll('{{id}}', id),
      fromJson: (json) => {model_name}.fromJson(json),
    );
  }}

"""
                else:
                    service_content += f"""
  // Get all {resource_name}s
  Future<ApiResponse<List<{model_name}>>> getAll{model_name}s() async {{
    final response = await _apiService.get<Map<String, dynamic>>('{path}');
    
    if (response.isSuccess && response.data != null) {{
      final List<dynamic> data = response.data!['data'] ?? response.data!;
      final List<{model_name}> items = data
          .map((json) => {model_name}.fromJson(json))
          .toList();
      return ApiResponse<List<{model_name}>>.success(items);
    }}
    
    return ApiResponse<List<{model_name}>>.error(response.error ?? 'Unknown error');
  }}

"""
            elif http_method == "POST":
                service_content += f"""
  // Create new {resource_name}
  Future<ApiResponse<{model_name}>> create{model_name}({model_name} {resource_name.lower()}) async {{
    return await _apiService.post<{model_name}>(
      '{path}',
      data: {resource_name.lower()}.toJson(),
      fromJson: (json) => {model_name}.fromJson(json),
    );
  }}

"""
            elif http_method == "PUT":
                service_content += f"""
  // Update {resource_name}
  Future<ApiResponse<{model_name}>> update{model_name}(String id, {model_name} {resource_name.lower()}) async {{
    return await _apiService.put<{model_name}>(
      '{path}'.replaceAll('{{id}}', id),
      data: {resource_name.lower()}.toJson(),
      fromJson: (json) => {model_name}.fromJson(json),
    );
  }}

"""
            elif http_method == "DELETE":
                service_content += f"""
  // Delete {resource_name}
  Future<ApiResponse<bool>> delete{model_name}(String id) async {{
    final response = await _apiService.delete<Map<String, dynamic>>(
      '{path}'.replaceAll('{{id}}', id),
    );
    
    return ApiResponse<bool>.success(response.isSuccess);
  }}

"""
        
        service_content += "}\n"
        return service_content
    
    async def _generate_api_client_config(self, api_spec: Dict[str, Any]) -> str:
        """Generate API client configuration."""
        base_url = api_spec.get("base_url", "https://api.example.com")
        auth_type = api_spec.get("auth_type", "bearer")
        
        return f"""// API Client Configuration
import 'package:dio/dio.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

class ApiClient {{
  static final ApiClient _instance = ApiClient._internal();
  factory ApiClient() => _instance;
  ApiClient._internal();

  late final Dio _dio;

  Dio get dio => _dio;

  void initialize() {{
    _dio = Dio(BaseOptions(
      baseUrl: '{base_url}',
      connectTimeout: const Duration(seconds: {self.default_timeout}),
      receiveTimeout: const Duration(seconds: {self.default_timeout}),
      headers: {{
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      }},
    ));

    // Add interceptors
    _dio.interceptors.addAll([
      _AuthInterceptor(),
      _RetryInterceptor(),
      PrettyDioLogger(
        requestHeader: true,
        requestBody: true,
        responseBody: true,
        responseHeader: false,
        error: true,
        compact: true,
        maxWidth: 90,
      ),
    ]);
  }}
}}

// Authentication Interceptor
class _AuthInterceptor extends Interceptor {{
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {{
    // Add authentication token
    final token = _getAuthToken();
    if (token != null) {{
      options.headers['Authorization'] = 'Bearer $token';
    }}
    
    handler.next(options);
  }}

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {{
    if (err.response?.statusCode == 401) {{
      // Handle token refresh or redirect to login
      _handleUnauthorized();
    }}
    
    handler.next(err);
  }}

  String? _getAuthToken() {{
    // Implement token retrieval logic
    // This could come from secure storage, shared preferences, etc.
    return null;
  }}

  void _handleUnauthorized() {{
    // Implement unauthorized handling
    // Clear token, redirect to login, etc.
  }}
}}

// Retry Interceptor
class _RetryInterceptor extends Interceptor {{
  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {{
    if (_shouldRetry(err) && (err.requestOptions.extra['retryCount'] ?? 0) < {self.retry_attempts}) {{
      err.requestOptions.extra['retryCount'] = (err.requestOptions.extra['retryCount'] ?? 0) + 1;
      
      // Wait before retry
      await Future.delayed(Duration(seconds: err.requestOptions.extra['retryCount'] * 2));
      
      try {{
        final response = await _dio.fetch(err.requestOptions);
        handler.resolve(response);
      }} catch (e) {{
        handler.next(err);
      }}
    }} else {{
      handler.next(err);
    }}
  }}

  bool _shouldRetry(DioException err) {{
    return err.type == DioExceptionType.connectionTimeout ||
           err.type == DioExceptionType.receiveTimeout ||
           err.type == DioExceptionType.connectionError ||
           (err.response?.statusCode != null && err.response!.statusCode! >= 500);
  }}
}}
"""
    
    def _group_endpoints_by_resource(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group endpoints by resource type."""
        resources = {}
        
        for endpoint in endpoints:
            path = endpoint.get("path", "")
            # Extract resource name from path (e.g., /users/{id} -> users)
            resource = path.split("/")[1] if len(path.split("/")) > 1 else "default"
            
            if resource not in resources:
                resources[resource] = {
                    "resource": resource,
                    "endpoints": []
                }
            
            resources[resource]["endpoints"].append(endpoint)
        
        return list(resources.values())
    
    async def _handle_http_client_setup(self, state: WorkflowState) -> AgentResponse:
        """Set up HTTP client configuration."""
        try:
            client_type = self.current_task.parameters.get("client_type", self.preferred_client)
            
            if client_type == "dio":
                client_setup = await self._generate_dio_setup()
            else:
                client_setup = await self._generate_http_setup()
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"HTTP client setup completed using {client_type}",
                data={
                    "client_type": client_type,
                    "setup_code": client_setup,
                    "configuration_file": f"lib/core/{client_type}_client.dart"
                }
            )
            
        except Exception as e:
            logger.error(f"HTTP client setup failed: {e}")
            return self._create_error_response(f"HTTP client setup failed: {str(e)}")
    
    async def _generate_dio_setup(self) -> str:
        """Generate Dio HTTP client setup."""
        return """// Dio HTTP Client Setup
import 'package:dio/dio.dart';

class DioClient {
  static final DioClient _instance = DioClient._internal();
  factory DioClient() => _instance;
  DioClient._internal();

  late final Dio _dio;

  Dio get dio => _dio;

  void initialize({
    String? baseUrl,
    Duration? connectTimeout,
    Duration? receiveTimeout,
    Map<String, dynamic>? headers,
  }) {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl ?? '',
      connectTimeout: connectTimeout ?? const Duration(seconds: 30),
      receiveTimeout: receiveTimeout ?? const Duration(seconds: 30),
      headers: headers ?? {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    ));

    // Add logging interceptor in debug mode
    if (kDebugMode) {
      _dio.interceptors.add(PrettyDioLogger());
    }
  }
}
"""
    
    async def _handle_default_api_integration(self, state: WorkflowState) -> AgentResponse:
        """Handle default API integration workflow."""
        try:
            # Default workflow: setup client and generate basic services
            client_result = await self._handle_http_client_setup(state)
            
            if client_result.status == TaskStatus.FAILED:
                return client_result
            
            # Generate API services
            service_result = await self._handle_api_service_generation(state)
            
            return service_result
            
        except Exception as e:
            logger.error(f"Default API integration failed: {e}")
            return self._create_error_response(f"API integration workflow failed: {str(e)}")
