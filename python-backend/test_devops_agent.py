#!/usr/bin/env python3
"""
Test script for the DevOps Agent.
Validates the functionality of deployment automation and CI/CD pipeline setup.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from agents.devops_agent import DevOpsAgent
from core.agent_types import TaskDefinition, AgentType, Priority


async def test_devops_agent():
    """Test the DevOps Agent functionality."""
    print("🚀 Testing DevOps Agent...")
    
    # Create test configuration
    config = {
        'deployment_targets': ['android', 'ios', 'web'],
        'ci_cd_provider': 'github_actions',
        'infrastructure_provider': 'firebase',
        'environments': {
            'development': {'domain': 'dev.example.com', 'api_base_url': 'https://dev-api.example.com'},
            'staging': {'domain': 'staging.example.com', 'api_base_url': 'https://staging-api.example.com'},
            'production': {'domain': 'example.com', 'api_base_url': 'https://api.example.com'}
        }
    }
    
    # Initialize DevOps Agent
    devops_agent = DevOpsAgent(config)
    print(f"✅ DevOps Agent initialized: {devops_agent.agent_id}")
    
    # Create temporary project directory
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir) / "test_flutter_project"
        project_path.mkdir()
        
        # Create basic Flutter project structure
        (project_path / "lib").mkdir()
        (project_path / "test").mkdir()
        (project_path / "pubspec.yaml").write_text("""
name: test_app
description: A test Flutter application
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.16.0"

dependencies:
  flutter:
    sdk: flutter

dev_dependencies:
  flutter_test:
    sdk: flutter
""")
        
        # Test 1: Setup CI/CD Pipeline
        print("\n📋 Test 1: Setting up CI/CD Pipeline...")
        task1 = TaskDefinition(
            task_id="test_cicd_setup",
            task_type="setup_cicd",
            description="Setup CI/CD pipeline for test project",
            priority=Priority.HIGH,
            context={
                'project_path': str(project_path),
                'provider': 'github_actions',
                'platforms': ['android', 'web']
            }
        )
        
        response1 = await devops_agent.process_task(task1)
        print(f"   Result: {response1.status.value}")
        print(f"   Message: {response1.result.get('message', 'No message')}")
        print(f"   Files created: {len(response1.result.get('files_created', []))}")
        
        # Test 2: Setup Deployment Configuration
        print("\n🔧 Test 2: Setting up Deployment Configuration...")
        task2 = TaskDefinition(
            task_id="test_deployment_config",
            task_type="setup_deployment",
            description="Setup deployment configurations",
            priority=Priority.HIGH,
            context={
                'project_path': str(project_path),
                'platforms': ['android', 'web'],
                'environment': 'production'
            }
        )
        
        response2 = await devops_agent.process_task(task2)
        print(f"   Result: {response2.status.value}")
        print(f"   Message: {response2.result.get('message', 'No message')}")
        print(f"   Platforms configured: {len(response2.result.get('platforms', []))}")
        
        # Test 3: Setup Infrastructure
        print("\n🏗️ Test 3: Setting up Infrastructure...")
        task3 = TaskDefinition(
            task_id="test_infrastructure_setup",
            task_type="setup_infrastructure",
            description="Setup cloud infrastructure",
            priority=Priority.HIGH,
            context={
                'provider': 'firebase',
                'project_name': 'test-flutter-app',
                'services': ['hosting', 'firestore', 'auth']
            }
        )
        
        response3 = await devops_agent.process_task(task3)
        print(f"   Result: {response3.status.value}")
        print(f"   Message: {response3.result.get('message', 'No message')}")
        print(f"   Provider: {response3.result.get('provider', 'Unknown')}")
        print(f"   Services: {response3.result.get('services', [])}")
        
        # Test 4: Configure Environments
        print("\n🌍 Test 4: Configuring Environments...")
        task4 = TaskDefinition(
            task_id="test_environment_config",
            task_type="environment_config",
            description="Configure deployment environments",
            priority=Priority.MEDIUM,
            context={
                'project_path': str(project_path),
                'environments': ['development', 'staging', 'production']
            }
        )
        
        response4 = await devops_agent.process_task(task4)
        print(f"   Result: {response4.status.value}")
        print(f"   Message: {response4.result.get('message', 'No message')}")
        print(f"   Environments: {response4.result.get('environments', [])}")
        
        # Test 5: Setup Monitoring
        print("\n📊 Test 5: Setting up Monitoring...")
        task5 = TaskDefinition(
            task_id="test_monitoring_setup",
            task_type="setup_monitoring",
            description="Setup monitoring and alerting",
            priority=Priority.MEDIUM,
            context={
                'platforms': ['web', 'android'],
                'services': ['performance', 'errors', 'usage'],
                'alert_channels': ['email', 'slack']
            }
        )
        
        response5 = await devops_agent.process_task(task5)
        print(f"   Result: {response5.status.value}")
        print(f"   Message: {response5.result.get('message', 'No message')}")
        print(f"   Monitoring services: {response5.result.get('monitoring_services', [])}")
        
        # Test 6: Deploy Application (simulation)
        print("\n🚀 Test 6: Deploying Application...")
        task6 = TaskDefinition(
            task_id="test_app_deployment",
            task_type="deploy_app",
            description="Deploy application to platforms",
            priority=Priority.HIGH,
            context={
                'project_path': str(project_path),
                'platforms': ['web'],
                'environment': 'staging',
                'version': '1.0.0'
            }
        )
        
        response6 = await devops_agent.process_task(task6)
        print(f"   Result: {response6.status.value}")
        print(f"   Message: {response6.result.get('message', 'No message')}")
        print(f"   Successful deployments: {response6.result.get('successful_count', 0)}/{response6.result.get('total_count', 0)}")
        
        # Test 7: Check Deployment Status
        print("\n📈 Test 7: Checking Deployment Status...")
        task7 = TaskDefinition(
            task_id="test_deployment_status",
            task_type="deployment_status",
            description="Check deployment status",
            priority=Priority.LOW,
            context={
                'deployment_id': 'test-deployment-123',
                'platforms': ['web', 'android']
            }
        )
        
        response7 = await devops_agent.process_task(task7)
        print(f"   Result: {response7.status.value}")
        print(f"   Message: {response7.result.get('message', 'No message')}")
        print(f"   Platforms checked: {response7.metadata.get('platforms_checked', 0)}")
        
        # Test 8: Generic DevOps Task
        print("\n🛠️ Test 8: Generic DevOps Task (Docker Setup)...")
        task8 = TaskDefinition(
            task_id="test_docker_setup",
            task_type="docker_setup",
            description="Setup Docker environment",
            priority=Priority.MEDIUM,
            context={
                'project_path': str(project_path)
            }
        )
        
        response8 = await devops_agent.process_task(task8)
        print(f"   Result: {response8.status.value}")
        print(f"   Message: {response8.result.get('message', 'No message')}")
        
        # Test 9: Get Agent Capabilities
        print("\n🎯 Test 9: Agent Capabilities...")
        capabilities = devops_agent.get_capabilities()
        print(f"   Capability categories: {len(capabilities)}")
        for category, tasks in capabilities.items():
            print(f"   - {category}: {len(tasks)} tasks")
        
        # Verify created files
        print("\n📁 Created Files Verification:")
        deployment_dir = project_path / "deployment"
        if deployment_dir.exists():
            print(f"   ✅ Deployment directory created")
            for file_path in deployment_dir.rglob("*"):
                if file_path.is_file():
                    print(f"      - {file_path.relative_to(project_path)}")
        
        github_workflows = project_path / ".github" / "workflows"
        if github_workflows.exists():
            print(f"   ✅ GitHub workflows directory created")
            for file_path in github_workflows.rglob("*"):
                if file_path.is_file():
                    print(f"      - {file_path.relative_to(project_path)}")
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   ✅ All 9 tests completed")
        print(f"   🎯 DevOps Agent ID: {devops_agent.agent_id}")
        print(f"   🌐 Deployment Targets: {devops_agent.deployment_targets}")
        print(f"   🔄 CI/CD Provider: {devops_agent.ci_cd_provider}")
        print(f"   ☁️ Infrastructure Provider: {devops_agent.infrastructure_provider}")


async def test_devops_templates():
    """Test DevOps template generation."""
    print("\n🔧 Testing DevOps Templates...")
    
    config = {
        'deployment_targets': ['android', 'ios', 'web'],
        'ci_cd_provider': 'github_actions',
        'infrastructure_provider': 'firebase'
    }
    
    devops_agent = DevOpsAgent(config)
    
    # Test pipeline templates
    print("   📋 Testing pipeline templates...")
    github_template = devops_agent._get_github_actions_template()
    gitlab_template = devops_agent._get_gitlab_ci_template()
    azure_template = devops_agent._get_azure_devops_template()
    jenkins_template = devops_agent._get_jenkins_template()
    
    print(f"      ✅ GitHub Actions template: {len(github_template)} keys")
    print(f"      ✅ GitLab CI template: {len(gitlab_template)} keys")
    print(f"      ✅ Azure DevOps template: {len(azure_template)} keys")
    print(f"      ✅ Jenkins template: {len(jenkins_template)} keys")
    
    # Test deployment configs
    print("   🚀 Testing deployment configurations...")
    android_config = devops_agent._get_android_deployment_config()
    ios_config = devops_agent._get_ios_deployment_config()
    web_config = devops_agent._get_web_deployment_config()
    desktop_config = devops_agent._get_desktop_deployment_config()
    
    print(f"      ✅ Android config: {len(android_config)} settings")
    print(f"      ✅ iOS config: {len(ios_config)} settings")
    print(f"      ✅ Web config: {len(web_config)} settings")
    print(f"      ✅ Desktop config: {len(desktop_config)} settings")


async def main():
    """Main test function."""
    print("🔥 DevOps Agent Test Suite")
    print("=" * 50)
    
    try:
        await test_devops_agent()
        await test_devops_templates()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
