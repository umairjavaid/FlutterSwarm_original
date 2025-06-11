"""
Security Agent for FlutterSwarm Multi-Agent System.

This agent specializes in security analysis, vulnerability assessment,
and security best practices for Flutter applications.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, TaskResult
from ..models.task_models import TaskContext
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..config import get_logger

logger = get_logger("security_agent")


class SecurityAgent(BaseAgent):
    """
    Specialized agent for Flutter application security analysis and hardening.
    
    This agent handles:
    - Security vulnerability assessment and scanning
    - Code security analysis and best practices
    - Data protection and privacy compliance
    - Authentication and authorization implementation
    - Secure communication and API security
    - Security testing and penetration testing
    - Compliance and regulatory requirements
    - Security incident response and forensics
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        # Override config for security-specific settings
        security_config = AgentConfig(
            agent_id=config.agent_id or f"security_agent_{str(uuid.uuid4())[:8]}",
            agent_type="security",
            capabilities=[
                AgentCapability.SECURITY_ANALYSIS,
                AgentCapability.CODE_GENERATION,
                AgentCapability.VULNERABILITY_SCANNING
            ],
            max_concurrent_tasks=3,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.2,  # Lower temperature for consistent security analysis
            max_tokens=7000,
            timeout=800,
            metadata=config.metadata
        )
        
        super().__init__(security_config, llm_client, memory_manager, event_bus)
        
        # Security-specific state
        self.security_domains = [
            "authentication", "authorization", "data_protection", 
            "communication_security", "input_validation", "session_management",
            "cryptography", "platform_security", "api_security", "compliance"
        ]
        
        self.vulnerability_categories = {
            "owasp_mobile": [
                "M1-Improper_Platform_Usage",
                "M2-Insecure_Data_Storage", 
                "M3-Insecure_Communication",
                "M4-Insecure_Authentication",
                "M5-Insufficient_Cryptography",
                "M6-Insecure_Authorization",
                "M7-Client_Code_Quality",
                "M8-Code_Tampering",
                "M9-Reverse_Engineering",
                "M10-Extraneous_Functionality"
            ],
            "flutter_specific": [
                "dart_security", "native_bridge_security", "widget_security",
                "state_management_security", "package_security"
            ]
        }
        
        self.security_frameworks = {
            "authentication": ["firebase_auth", "oauth2", "openid_connect", "biometric_auth"],
            "encryption": ["encrypt", "cryptography", "pointycastle"],
            "secure_storage": ["flutter_secure_storage", "keychain", "keystore"],
            "network_security": ["dio_interceptors", "certificate_pinning", "https_enforcement"],
            "compliance": ["gdpr", "ccpa", "hipaa", "pci_dss"]
        }
        
        logger.info(f"Security Agent {self.agent_id} initialized")

    async def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the security agent."""
        return """
You are the Security Agent in the FlutterSwarm multi-agent system, specializing in Flutter application security analysis and hardening.

CORE EXPERTISE:
- Flutter and Dart security best practices and vulnerabilities
- Mobile application security (OWASP Mobile Top 10)
- Cross-platform security considerations (iOS, Android, Web, Desktop)
- Authentication and authorization implementation

Always provide actionable security recommendations with implementation details and compliance guidance.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of security-specific capabilities."""
        return [
            "vulnerability_assessment",
            "security_code_review",
            "threat_modeling",
            "authentication_implementation",
            "authorization_design",
            "data_encryption",
            "secure_communication",
            "privacy_compliance",
            "security_testing",
            "penetration_testing",
            "incident_response",
            "compliance_auditing"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute security-specific processing logic.
        """
        try:
            task_type = task_context.task_type.value
            
            if task_type == "vulnerability_assessment":
                return await self._assess_vulnerabilities(task_context, llm_analysis)
            elif task_type == "security_review":
                return await self._conduct_security_review(task_context, llm_analysis)
            elif task_type == "authentication_setup":
                return await self._implement_authentication(task_context, llm_analysis)
            elif task_type == "data_protection":
                return await self._implement_data_protection(task_context, llm_analysis)
            elif task_type == "compliance_check":
                return await self._check_compliance(task_context, llm_analysis)
            elif task_type == "security_testing":
                return await self._create_security_tests(task_context, llm_analysis)
            else:
                # Generic security processing
                return await self._process_security_request(task_context, llm_analysis)
                
        except Exception as e:
            logger.error(f"Security processing failed: {e}")
            return {
                "error": str(e),
                "security_report": {},
                "recommendations": []
            }

    async def _assess_vulnerabilities(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct comprehensive vulnerability assessment."""
        logger.info(f"Assessing vulnerabilities for task: {task_context.task_id}")
        
        vulnerability_prompt = self._create_vulnerability_assessment_prompt(task_context, llm_analysis)
        
        vulnerability_report = await self.execute_llm_task(
            user_prompt=vulnerability_prompt,
            context={
                "task": task_context.to_dict(),
                "vulnerability_categories": self.vulnerability_categories,
                "security_frameworks": self.security_frameworks
            },
            structured_output=True
        )
        
        # Store vulnerability assessment
        await self.memory_manager.store_memory(
            content=f"Vulnerability assessment: {json.dumps(vulnerability_report)}",
            metadata={
                "type": "vulnerability_assessment",
                "severity_counts": vulnerability_report.get('severity_summary', {}),
                "categories": vulnerability_report.get('categories', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.9,
            long_term=True
        )
        
        return {
            "vulnerability_report": vulnerability_report,
            "critical_vulnerabilities": vulnerability_report.get("critical", []),
            "high_vulnerabilities": vulnerability_report.get("high", []),
            "medium_vulnerabilities": vulnerability_report.get("medium", []),
            "low_vulnerabilities": vulnerability_report.get("low", []),
            "remediation_plan": vulnerability_report.get("remediation", {}),
            "security_score": vulnerability_report.get("security_score", 0)
        }

    async def _conduct_security_review(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct comprehensive security code review."""
        logger.info(f"Conducting security review for task: {task_context.task_id}")
        
        security_review_prompt = self._create_security_review_prompt(task_context, llm_analysis)
        
        security_review = await self.execute_llm_task(
            user_prompt=security_review_prompt,
            context={
                "task": task_context.to_dict(),
                "security_patterns": self._get_security_patterns(),
                "best_practices": self._get_security_best_practices()
            },
            structured_output=True
        )
        
        return {
            "security_review": security_review,
            "code_issues": security_review.get("issues", []),
            "security_violations": security_review.get("violations", []),
            "improvement_recommendations": security_review.get("improvements", []),
            "compliance_status": security_review.get("compliance", {}),
            "security_metrics": security_review.get("metrics", {})
        }

    async def _implement_authentication(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement secure authentication system."""
        logger.info(f"Implementing authentication for task: {task_context.task_id}")
        
        auth_prompt = self._create_authentication_prompt(task_context, llm_analysis)
        
        auth_implementation = await self.execute_llm_task(
            user_prompt=auth_prompt,
            context={
                "task": task_context.to_dict(),
                "auth_frameworks": self.security_frameworks["authentication"],
                "auth_patterns": self._get_authentication_patterns()
            },
            structured_output=True
        )
        
        return {
            "auth_implementation": auth_implementation,
            "auth_services": auth_implementation.get("services", {}),
            "security_config": auth_implementation.get("config", {}),
            "integration_code": auth_implementation.get("code", {}),
            "testing_suite": auth_implementation.get("tests", {}),
            "security_documentation": auth_implementation.get("documentation", "")
        }

    async def _implement_data_protection(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement comprehensive data protection measures."""
        logger.info(f"Implementing data protection for task: {task_context.task_id}")
        
        data_protection_prompt = self._create_data_protection_prompt(task_context, llm_analysis)
        
        data_protection = await self.execute_llm_task(
            user_prompt=data_protection_prompt,
            context={
                "task": task_context.to_dict(),
                "encryption_tools": self.security_frameworks["encryption"],
                "storage_patterns": self._get_secure_storage_patterns()
            },
            structured_output=True
        )
        
        return {
            "data_protection": data_protection,
            "encryption_implementation": data_protection.get("encryption", {}),
            "secure_storage": data_protection.get("storage", {}),
            "data_handling_policies": data_protection.get("policies", {}),
            "privacy_controls": data_protection.get("privacy", {}),
            "compliance_measures": data_protection.get("compliance", {})
        }

    # Prompt creation methods
    def _create_vulnerability_assessment_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for vulnerability assessment."""
        return f"""
Conduct a comprehensive security vulnerability assessment for the following Flutter application:

APPLICATION ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

ASSESSMENT SCOPE:
{task_context.description}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please provide detailed vulnerability assessment including:

1. THREAT MODELING:
   - Identify potential attack vectors and threat actors
   - Analyze data flow and trust boundaries
   - Assess platform-specific security risks
   - Evaluate third-party dependency risks

2. OWASP MOBILE TOP 10 ANALYSIS:
   - M1: Improper Platform Usage assessment
   - M2: Insecure Data Storage evaluation
   - M3: Insecure Communication analysis
   - M4: Insecure Authentication review
   - M5: Insufficient Cryptography check
   - M6: Insecure Authorization assessment
   - M7: Client Code Quality analysis
   - M8: Code Tampering vulnerability
   - M9: Reverse Engineering protection
   - M10: Extraneous Functionality audit

3. FLUTTER-SPECIFIC VULNERABILITIES:
   - Dart code security issues
   - Platform channel security gaps
   - Widget security vulnerabilities
   - State management security flaws
   - Package and dependency vulnerabilities

4. SECURITY CONTROLS ASSESSMENT:
   - Authentication mechanism evaluation
   - Authorization control effectiveness
   - Data encryption implementation
   - Input validation and sanitization
   - Session management security

5. COMPLIANCE ANALYSIS:
   - GDPR compliance assessment
   - Platform store security requirements
   - Industry-specific compliance gaps
   - Privacy policy implementation

6. REMEDIATION PRIORITIES:
   - Critical vulnerability fixes
   - High-priority security improvements
   - Medium and low-priority enhancements
   - Long-term security roadmap

Provide detailed findings with severity ratings, impact analysis, and specific remediation guidance.
"""

    def _create_security_review_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for security code review."""
        return f"""
Conduct comprehensive security code review for the following Flutter application:

CODE ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

REVIEW SCOPE:
{task_context.description}

Please perform thorough security code review including:

1. AUTHENTICATION SECURITY:
   - Authentication implementation analysis
   - Token handling and storage security
   - Session management implementation
   - Multi-factor authentication setup

2. AUTHORIZATION CONTROLS:
   - Access control implementation
   - Role-based access control (RBAC)
   - Attribute-based access control (ABAC)
   - API endpoint protection

3. DATA SECURITY:
   - Sensitive data identification and handling
   - Encryption implementation and key management
   - Data storage security (local and remote)
   - Data transmission security

4. INPUT VALIDATION:
   - User input validation and sanitization
   - SQL injection prevention
   - Cross-site scripting (XSS) protection
   - Command injection prevention

5. API SECURITY:
   - API authentication and authorization
   - Rate limiting and throttling
   - Input validation for API endpoints
   - Error handling and information disclosure

6. PLATFORM SECURITY:
   - Platform-specific security implementations
   - Native code security (iOS/Android)
   - Web security considerations
   - Desktop application security

Provide specific code issues, security violations, and detailed remediation recommendations.
"""

    def _create_authentication_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for authentication implementation."""
        return f"""
Implement secure authentication system for the following Flutter application:

AUTHENTICATION REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

IMPLEMENTATION SCOPE:
{task_context.description}

Please design and implement comprehensive authentication including:

1. AUTHENTICATION STRATEGY:
   - Authentication method selection (OAuth, JWT, biometric)
   - Multi-factor authentication implementation
   - Social login integration
   - Enterprise authentication (SSO, LDAP)

2. SECURE IMPLEMENTATION:
   - Token-based authentication
   - Secure token storage and management
   - Session management and lifecycle
   - Logout and token revocation

3. SECURITY FEATURES:
   - Password policy enforcement
   - Account lockout mechanisms
   - Brute force protection
   - Device fingerprinting

4. PLATFORM INTEGRATION:
   - iOS Keychain integration
   - Android Keystore integration
   - Biometric authentication (Face ID, Touch ID)
   - Platform-specific security features

5. ERROR HANDLING:
   - Secure error messages
   - Authentication failure handling
   - Network error resilience
   - Offline authentication support

6. TESTING AND VALIDATION:
   - Authentication test cases
   - Security validation tests
   - Penetration testing scenarios
   - Compliance verification

Provide complete authentication implementation with security best practices.
"""

    def _create_data_protection_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for data protection implementation."""
        return f"""
Implement comprehensive data protection for the following Flutter application:

DATA PROTECTION REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

PROTECTION SCOPE:
{task_context.description}

Please design data protection strategy including:

1. DATA CLASSIFICATION:
   - Sensitive data identification
   - Data classification levels
   - Personal data inventory
   - Compliance data mapping

2. ENCRYPTION IMPLEMENTATION:
   - Data encryption at rest
   - Data encryption in transit
   - Key management and rotation
   - Cryptographic algorithm selection

3. SECURE STORAGE:
   - Local secure storage implementation
   - Cloud storage security
   - Database encryption
   - File system protection

4. PRIVACY CONTROLS:
   - Data minimization principles
   - Consent management
   - Data retention policies
   - Right to deletion implementation

5. ACCESS CONTROLS:
   - Data access authorization
   - Audit logging and monitoring
   - Data loss prevention (DLP)
   - Insider threat protection

6. COMPLIANCE IMPLEMENTATION:
   - GDPR compliance measures
   - CCPA compliance implementation
   - Industry-specific requirements
   - Cross-border data transfer protection

Provide complete data protection implementation with compliance documentation.
"""

    # Helper methods for security patterns and practices
    def _get_security_patterns(self) -> Dict[str, Any]:
        """Get security design patterns and practices."""
        return {
            "authentication_patterns": [
                "token_based_auth", "oauth2_flow", "jwt_implementation", 
                "biometric_auth", "multi_factor_auth"
            ],
            "authorization_patterns": [
                "rbac", "abac", "policy_based_auth", 
                "resource_based_auth", "claim_based_auth"
            ],
            "data_protection_patterns": [
                "encryption_at_rest", "encryption_in_transit", 
                "key_management", "data_masking", "tokenization"
            ],
            "secure_communication": [
                "certificate_pinning", "mutual_tls", 
                "api_gateway", "rate_limiting"
            ]
        }

    def _get_security_best_practices(self) -> Dict[str, Any]:
        """Get security best practices and guidelines."""
        return {
            "secure_coding": [
                "input_validation", "output_encoding", 
                "error_handling", "logging_security"
            ],
            "mobile_security": [
                "secure_storage", "root_detection", 
                "anti_tampering", "obfuscation"
            ],
            "privacy_by_design": [
                "data_minimization", "consent_management", 
                "transparency", "user_control"
            ],
            "compliance": [
                "gdpr_requirements", "ccpa_requirements", 
                "industry_standards", "audit_requirements"
            ]
        }

    def _get_authentication_patterns(self) -> Dict[str, Any]:
        """Get authentication implementation patterns."""
        return {
            "oauth2": "OAuth 2.0 authorization framework",
            "jwt": "JSON Web Token implementation",
            "biometric": "Biometric authentication integration",
            "mfa": "Multi-factor authentication",
            "sso": "Single sign-on implementation"
        }

    def _get_secure_storage_patterns(self) -> Dict[str, Any]:
        """Get secure storage implementation patterns."""
        return {
            "keychain": "iOS Keychain Services",
            "keystore": "Android Keystore",
            "secure_storage": "Flutter Secure Storage",
            "encrypted_prefs": "Encrypted SharedPreferences",
            "database_encryption": "SQLCipher implementation"
        }
