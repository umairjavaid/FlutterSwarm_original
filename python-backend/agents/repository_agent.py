"""
Repository Agent - Implements repository pattern for data access.
Coordinates between local storage and remote APIs as single source of truth.
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


class RepositoryAgent(BaseAgent):
    """
    Specialized agent for implementing repository pattern.
    Coordinates between local storage and remote APIs as single source of truth.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.REPOSITORY, config)
        
        # Repository configuration
        self.cache_strategies = {
            "cache_first": "Cache First",
            "network_first": "Network First", 
            "cache_only": "Cache Only",
            "network_only": "Network Only",
            "stale_while_revalidate": "Stale While Revalidate"
        }
        
        # Configuration
        self.default_cache_strategy = config.get("cache_strategy", "cache_first")
        self.sync_on_network = config.get("sync_on_network", True)
        self.offline_support = config.get("offline_support", True)
        self.generate_interfaces = config.get("generate_interfaces", True)
        
    def _define_capabilities(self) -> List[str]:
        """Define repository agent capabilities."""
        return [
            "repository_pattern_implementation",
            "data_source_coordination",
            "cache_first_strategies",
            "network_first_strategies",
            "offline_data_handling",
            "data_synchronization",
            "repository_interface_creation",
            "data_consistency_management",
            "crud_operation_implementation",
            "batch_operation_handling",
            "data_conflict_resolution",
            "repository_testing_setup"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process repository pattern tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'repository_generation')
            
            if task_type == "repository_generation":
                return await self._handle_repository_generation(state)
            elif task_type == "repository_interface_creation":
                return await self._handle_repository_interface_creation(state)
            elif task_type == "data_source_coordination":
                return await self._handle_data_source_coordination(state)
            elif task_type == "sync_strategy_implementation":
                return await self._handle_sync_strategy_implementation(state)
            else:
                return await self._handle_default_repository_implementation(state)
                
        except Exception as e:
            logger.error(f"Repository agent error: {e}")
            return self._create_error_response(f"Repository implementation failed: {str(e)}")
    
    async def _handle_repository_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate repository classes for data entities."""
        try:
            project_context = state.project_context
            entities = self.current_task.parameters.get("entities", [])
            
            generated_repositories = []
            
            # Generate base repository
            base_repo = await self._generate_base_repository()
            generated_repositories.append({
                "repository_name": "BaseRepository",
                "file_path": "lib/repositories/base_repository.dart",
                "content": base_repo
            })
            
            # Generate specific repositories for each entity
            for entity in entities:
                entity_name = entity.get("name", "")
                if entity_name:
                    repo_content = await self._generate_entity_repository(entity, project_context)
                    generated_repositories.append({
                        "repository_name": f"{entity_name}Repository",
                        "file_path": f"lib/repositories/{entity_name.lower()}_repository.dart",
                        "content": repo_content
                    })
            
            # Generate repository provider/injector
            provider_content = await self._generate_repository_provider(entities)
            generated_repositories.append({
                "repository_name": "RepositoryProvider",
                "file_path": "lib/repositories/repository_provider.dart",
                "content": provider_content
            })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_repositories)} repository files",
                data={
                    "generated_repositories": generated_repositories,
                    "cache_strategy": self.default_cache_strategy,
                    "offline_support": self.offline_support
                },
                metadata={
                    "repository_count": len(generated_repositories),
                    "entities_processed": len(entities)
                }
            )
            
        except Exception as e:
            logger.error(f"Repository generation failed: {e}")
            return self._create_error_response(f"Repository generation failed: {str(e)}")
    
    async def _generate_base_repository(self) -> str:
        """Generate base repository class with common functionality."""
        return f"""// Base Repository
import 'dart:async';
import 'package:connectivity_plus/connectivity_plus.dart';

abstract class BaseRepository<T, ID> {{
  // Abstract methods to be implemented by concrete repositories
  Future<T?> getById(ID id);
  Future<List<T>> getAll();
  Future<T> create(T entity);
  Future<T> update(T entity);
  Future<bool> delete(ID id);
  
  // Optional batch operations
  Future<List<T>> createBatch(List<T> entities) async {{
    final List<T> results = [];
    for (final entity in entities) {{
      results.add(await create(entity));
    }}
    return results;
  }}
  
  Future<List<T>> updateBatch(List<T> entities) async {{
    final List<T> results = [];
    for (final entity in entities) {{
      results.add(await update(entity));
    }}
    return results;
  }}
  
  Future<bool> deleteBatch(List<ID> ids) async {{
    for (final id in ids) {{
      final success = await delete(id);
      if (!success) return false;
    }}
    return true;
  }}
  
  // Network connectivity check
  Future<bool> get isOnline async {{
    final connectivityResult = await Connectivity().checkConnectivity();
    return connectivityResult != ConnectivityResult.none;
  }}
  
  // Cache invalidation
  Future<void> invalidateCache() async {{
    // Implement cache invalidation logic
  }}
  
  // Sync with remote
  Future<void> syncWithRemote() async {{
    if (await isOnline) {{
      await _performSync();
    }}
  }}
  
  Future<void> _performSync() async {{
    // Implement synchronization logic
  }}
}}

// Repository result wrapper
class RepositoryResult<T> {{
  final T? data;
  final String? error;
  final bool isFromCache;
  final DateTime timestamp;
  
  const RepositoryResult.success(
    this.data, {{
    this.isFromCache = false,
  }}) : error = null, timestamp = DateTime.now();
  
  const RepositoryResult.error(
    this.error, {{
    this.isFromCache = false,
  }}) : data = null, timestamp = DateTime.now();
  
  const RepositoryResult.cached(
    this.data, {{
    required this.timestamp,
  }}) : error = null, isFromCache = true;
  
  bool get isSuccess => error == null;
  bool get isError => error != null;
}}

// Cache strategy enum
enum CacheStrategy {{
  cacheFirst,
  networkFirst,
  cacheOnly,
  networkOnly,
  staleWhileRevalidate,
}}

// Repository configuration
class RepositoryConfig {{
  final CacheStrategy cacheStrategy;
  final Duration cacheTimeout;
  final bool enableOfflineMode;
  final bool autoSync;
  
  const RepositoryConfig({{
    this.cacheStrategy = CacheStrategy.{self.default_cache_strategy.split('_')[0]}{''.join(word.capitalize() for word in self.default_cache_strategy.split('_')[1:])},
    this.cacheTimeout = const Duration(minutes: 30),
    this.enableOfflineMode = {str(self.offline_support).lower()},
    this.autoSync = {str(self.sync_on_network).lower()},
  }});
}}
"""
    
    async def _generate_entity_repository(self, entity: Dict[str, Any], project_context: ProjectContext) -> str:
        """Generate repository for a specific entity."""
        entity_name = entity.get("name", "Entity")
        entity_id_type = entity.get("id_type", "String")
        
        return f"""// {entity_name} Repository
import 'dart:async';
import 'package:rxdart/rxdart.dart';

import '../models/{entity_name.lower()}.dart';
import '../services/{entity_name.lower()}_service.dart';
import '../core/local_storage.dart';
import 'base_repository.dart';

class {entity_name}Repository extends BaseRepository<{entity_name}, {entity_id_type}> {{
  static final {entity_name}Repository _instance = {entity_name}Repository._internal();
  factory {entity_name}Repository() => _instance;
  {entity_name}Repository._internal();

  // Dependencies
  final {entity_name}Service _apiService = {entity_name}Service();
  final LocalStorage _localStorage = LocalStorage();
  
  // Configuration
  final RepositoryConfig _config = const RepositoryConfig();
  
  // Cache and streams
  final BehaviorSubject<List<{entity_name}>> _{entity_name.lower()}sController = 
      BehaviorSubject<List<{entity_name}>>();
  
  Stream<List<{entity_name}>> get {entity_name.lower()}s => _{entity_name.lower()}sController.stream;
  
  // Cache keys
  static const String _cacheKey = '{entity_name.lower()}s';
  static const String _timestampKey = '{entity_name.lower()}s_timestamp';

  @override
  Future<{entity_name}?> getById({entity_id_type} id) async {{
    try {{
      switch (_config.cacheStrategy) {{
        case CacheStrategy.cacheFirst:
          return await _getCacheFirst(id);
        case CacheStrategy.networkFirst:
          return await _getNetworkFirst(id);
        case CacheStrategy.cacheOnly:
          return await _getCacheOnly(id);
        case CacheStrategy.networkOnly:
          return await _getNetworkOnly(id);
        case CacheStrategy.staleWhileRevalidate:
          return await _getStaleWhileRevalidate(id);
      }}
    }} catch (e) {{
      logger.error('Error getting {entity_name.lower()} by id: $e');
      return null;
    }}
  }}

  @override
  Future<List<{entity_name}>> getAll() async {{
    try {{
      switch (_config.cacheStrategy) {{
        case CacheStrategy.cacheFirst:
          return await _getAllCacheFirst();
        case CacheStrategy.networkFirst:
          return await _getAllNetworkFirst();
        case CacheStrategy.cacheOnly:
          return await _getAllCacheOnly();
        case CacheStrategy.networkOnly:
          return await _getAllNetworkOnly();
        case CacheStrategy.staleWhileRevalidate:
          return await _getAllStaleWhileRevalidate();
      }}
    }} catch (e) {{
      logger.error('Error getting all {entity_name.lower()}s: $e');
      return [];
    }}
  }}

  @override
  Future<{entity_name}> create({entity_name} entity) async {{
    try {{
      // Always try network first for create operations
      if (await isOnline) {{
        final response = await _apiService.create{entity_name}(entity);
        if (response.isSuccess && response.data != null) {{
          // Cache the created entity
          await _cacheEntity(response.data!);
          _updateStream();
          return response.data!;
        }}
        throw Exception(response.error);
      }} else {{
        // Offline: queue for sync
        await _queueForSync('create', entity);
        return entity;
      }}
    }} catch (e) {{
      logger.error('Error creating {entity_name.lower()}: $e');
      rethrow;
    }}
  }}

  @override
  Future<{entity_name}> update({entity_name} entity) async {{
    try {{
      if (await isOnline) {{
        final response = await _apiService.update{entity_name}(entity.id, entity);
        if (response.isSuccess && response.data != null) {{
          await _cacheEntity(response.data!);
          _updateStream();
          return response.data!;
        }}
        throw Exception(response.error);
      }} else {{
        // Offline: queue for sync
        await _queueForSync('update', entity);
        await _cacheEntity(entity);
        _updateStream();
        return entity;
      }}
    }} catch (e) {{
      logger.error('Error updating {entity_name.lower()}: $e');
      rethrow;
    }}
  }}

  @override
  Future<bool> delete({entity_id_type} id) async {{
    try {{
      if (await isOnline) {{
        final response = await _apiService.delete{entity_name}(id);
        if (response.isSuccess) {{
          await _removeFromCache(id);
          _updateStream();
          return true;
        }}
        return false;
      }} else {{
        // Offline: queue for sync
        await _queueForSync('delete', {{'id': id}});
        await _removeFromCache(id);
        _updateStream();
        return true;
      }}
    }} catch (e) {{
      logger.error('Error deleting {entity_name.lower()}: $e');
      return false;
    }}
  }}

  // Cache-first strategy
  Future<{entity_name}?> _getCacheFirst({entity_id_type} id) async {{
    // Try cache first
    final cached = await _getCachedEntity(id);
    if (cached != null && !_isCacheExpired()) {{
      return cached;
    }}

    // Fallback to network
    if (await isOnline) {{
      return await _getNetworkOnly(id);
    }}

    // Return stale cache if offline
    return cached;
  }}

  // Network-first strategy  
  Future<{entity_name}?> _getNetworkFirst({entity_id_type} id) async {{
    if (await isOnline) {{
      try {{
        return await _getNetworkOnly(id);
      }} catch (e) {{
        // Fallback to cache on network error
        return await _getCachedEntity(id);
      }}
    }}
    
    return await _getCachedEntity(id);
  }}

  // Cache-only strategy
  Future<{entity_name}?> _getCacheOnly({entity_id_type} id) async {{
    return await _getCachedEntity(id);
  }}

  // Network-only strategy
  Future<{entity_name}?> _getNetworkOnly({entity_id_type} id) async {{
    final response = await _apiService.get{entity_name}(id);
    if (response.isSuccess && response.data != null) {{
      await _cacheEntity(response.data!);
      return response.data!;
    }}
    throw Exception(response.error);
  }}

  // Stale-while-revalidate strategy
  Future<{entity_name}?> _getStaleWhileRevalidate({entity_id_type} id) async {{
    // Return cache immediately if available
    final cached = await _getCachedEntity(id);
    
    // Revalidate in background if online
    if (await isOnline) {{
      _getNetworkOnly(id).then((_) => _updateStream()).catchError((e) {{
        logger.warning('Background revalidation failed: $e');
      }});
    }}
    
    return cached;
  }}

  // Get all methods for different strategies
  Future<List<{entity_name}>> _getAllCacheFirst() async {{
    final cached = await _getCachedEntities();
    if (cached.isNotEmpty && !_isCacheExpired()) {{
      return cached;
    }}

    if (await isOnline) {{
      return await _getAllNetworkOnly();
    }}

    return cached;
  }}

  Future<List<{entity_name}>> _getAllNetworkFirst() async {{
    if (await isOnline) {{
      try {{
        return await _getAllNetworkOnly();
      }} catch (e) {{
        return await _getCachedEntities();
      }}
    }}
    
    return await _getCachedEntities();
  }}

  Future<List<{entity_name}>> _getAllCacheOnly() async {{
    return await _getCachedEntities();
  }}

  Future<List<{entity_name}>> _getAllNetworkOnly() async {{
    final response = await _apiService.getAll{entity_name}s();
    if (response.isSuccess && response.data != null) {{
      await _cacheEntities(response.data!);
      _updateStream();
      return response.data!;
    }}
    throw Exception(response.error);
  }}

  Future<List<{entity_name}>> _getAllStaleWhileRevalidate() async {{
    final cached = await _getCachedEntities();
    
    if (await isOnline) {{
      _getAllNetworkOnly().then((_) => _updateStream()).catchError((e) {{
        logger.warning('Background revalidation failed: $e');
      }});
    }}
    
    return cached;
  }}

  // Cache management
  Future<{entity_name}?> _getCachedEntity({entity_id_type} id) async {{
    final entities = await _getCachedEntities();
    try {{
      return entities.firstWhere((e) => e.id == id);
    }} catch (e) {{
      return null;
    }}
  }}

  Future<List<{entity_name}>> _getCachedEntities() async {{
    final jsonList = await _localStorage.getList(_cacheKey);
    return jsonList.map((json) => {entity_name}.fromJson(json)).toList();
  }}

  Future<void> _cacheEntity({entity_name} entity) async {{
    final entities = await _getCachedEntities();
    final index = entities.indexWhere((e) => e.id == entity.id);
    
    if (index >= 0) {{
      entities[index] = entity;
    }} else {{
      entities.add(entity);
    }}
    
    await _cacheEntities(entities);
  }}

  Future<void> _cacheEntities(List<{entity_name}> entities) async {{
    final jsonList = entities.map((e) => e.toJson()).toList();
    await _localStorage.saveList(_cacheKey, jsonList);
    await _localStorage.saveInt(_timestampKey, DateTime.now().millisecondsSinceEpoch);
  }}

  Future<void> _removeFromCache({entity_id_type} id) async {{
    final entities = await _getCachedEntities();
    entities.removeWhere((e) => e.id == id);
    await _cacheEntities(entities);
  }}

  bool _isCacheExpired() {{
    final timestamp = _localStorage.getInt(_timestampKey) ?? 0;
    final cacheTime = DateTime.fromMillisecondsSinceEpoch(timestamp);
    return DateTime.now().difference(cacheTime) > _config.cacheTimeout;
  }}

  // Sync management
  Future<void> _queueForSync(String operation, dynamic data) async {{
    // Implement sync queue logic
    final syncQueue = await _localStorage.getList('sync_queue') ?? [];
    syncQueue.add({{
      'entity_type': '{entity_name.lower()}',
      'operation': operation,
      'data': data is Map ? data : (data as {entity_name}).toJson(),
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    }});
    await _localStorage.saveList('sync_queue', syncQueue);
  }}

  @override
  Future<void> syncWithRemote() async {{
    if (!await isOnline) return;

    final syncQueue = await _localStorage.getList('sync_queue') ?? [];
    final {entity_name.lower()}Items = syncQueue
        .where((item) => item['entity_type'] == '{entity_name.lower()}')
        .toList();

    for (final item in {entity_name.lower()}Items) {{
      try {{
        await _processSyncItem(item);
        syncQueue.remove(item);
      }} catch (e) {{
        logger.error('Sync failed for item: $item, error: $e');
      }}
    }}

    await _localStorage.saveList('sync_queue', syncQueue);
  }}

  Future<void> _processSyncItem(Map<String, dynamic> item) async {{
    final operation = item['operation'];
    final data = item['data'];

    switch (operation) {{
      case 'create':
        final entity = {entity_name}.fromJson(data);
        await _apiService.create{entity_name}(entity);
        break;
      case 'update':
        final entity = {entity_name}.fromJson(data);
        await _apiService.update{entity_name}(entity.id, entity);
        break;
      case 'delete':
        await _apiService.delete{entity_name}(data['id']);
        break;
    }}
  }}

  void _updateStream() {{
    _getCachedEntities().then((entities) {{
      _{entity_name.lower()}sController.add(entities);
    }});
  }}

  void dispose() {{
    _{entity_name.lower()}sController.close();
  }}
}}
"""
    
    async def _generate_repository_provider(self, entities: List[Dict[str, Any]]) -> str:
        """Generate repository provider for dependency injection."""
        imports = []
        repositories = []
        
        for entity in entities:
            entity_name = entity.get("name", "")
            if entity_name:
                imports.append(f"import '{entity_name.lower()}_repository.dart';")
                repositories.append(f"  {entity_name}Repository get {entity_name.lower()}Repository => {entity_name}Repository();")
        
        imports_str = "\n".join(imports)
        repositories_str = "\n".join(repositories)
        
        return f"""// Repository Provider
{imports_str}

class RepositoryProvider {{
  static final RepositoryProvider _instance = RepositoryProvider._internal();
  factory RepositoryProvider() => _instance;
  RepositoryProvider._internal();

{repositories_str}

  // Initialize all repositories
  Future<void> initialize() async {{
    // Perform any initialization logic
    await _initializeConnectivity();
    await _startSyncScheduler();
  }}

  Future<void> _initializeConnectivity() async {{
    // Setup connectivity monitoring
  }}

  Future<void> _startSyncScheduler() async {{
    // Setup periodic sync for offline changes
    Timer.periodic(const Duration(minutes: 5), (_) async {{
{chr(10).join(f"      await {entity.get('name', '').lower()}Repository.syncWithRemote();" for entity in entities if entity.get('name'))}
    }});
  }}

  // Clear all caches
  Future<void> clearAllCaches() async {{
{chr(10).join(f"    await {entity.get('name', '').lower()}Repository.invalidateCache();" for entity in entities if entity.get('name'))}
  }}

  // Force sync all repositories
  Future<void> syncAll() async {{
{chr(10).join(f"    await {entity.get('name', '').lower()}Repository.syncWithRemote();" for entity in entities if entity.get('name'))}
  }}
}}
"""
    
    async def _handle_default_repository_implementation(self, state: WorkflowState) -> AgentResponse:
        """Handle default repository implementation workflow."""
        try:
            # Default workflow: generate repositories with caching and sync
            generation_result = await self._handle_repository_generation(state)
            
            if generation_result.status == TaskStatus.FAILED:
                return generation_result
            
            # Generate repository interfaces if enabled
            if self.generate_interfaces:
                interface_result = await self._handle_repository_interface_creation(state)
                
                # Merge results
                generation_result.data["generated_interfaces"] = interface_result.data.get("generated_interfaces", [])
            
            return generation_result
            
        except Exception as e:
            logger.error(f"Default repository implementation failed: {e}")
            return self._create_error_response(f"Repository implementation workflow failed: {str(e)}")
