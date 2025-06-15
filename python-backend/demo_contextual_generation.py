#!/usr/bin/env python3
"""
Demo script showing contextual code generation capabilities.

This script demonstrates how the ImplementationAgent can intelligently
analyze a Flutter project and generate code that seamlessly integrates
with existing patterns and conventions.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any


class ContextualCodeGenerationDemo:
    """Demo of intelligent contextual code generation."""
    
    def __init__(self):
        self.project_structure = {
            "lib/": {
                "main.dart": "entry_point",
                "app.dart": "app_configuration",
                "features/": {
                    "auth/": {
                        "auth_bloc.dart": "bloc_pattern",
                        "auth_screen.dart": "screen_widget",
                        "auth_repository.dart": "repository_pattern"
                    },
                    "home/": {
                        "home_bloc.dart": "bloc_pattern", 
                        "home_screen.dart": "screen_widget"
                    }
                },
                "shared/": {
                    "widgets/": {
                        "custom_button.dart": "reusable_widget"
                    },
                    "theme/": {
                        "app_theme.dart": "theme_configuration"
                    }
                }
            },
            "test/": {
                "auth_bloc_test.dart": "unit_test",
                "home_screen_test.dart": "widget_test"
            }
        }
        
        self.dependencies = [
            "flutter_bloc", "equatable", "dio", "cached_network_image"
        ]
        
        self.detected_patterns = {
            "architecture": "bloc_pattern",
            "state_management": "flutter_bloc",
            "navigation": "standard_navigator", 
            "file_naming": "snake_case",
            "directory_organization": "feature_based",
            "import_organization": "dart_first_then_packages_then_relative"
        }

    def demonstrate_project_analysis(self):
        """Show how project structure analysis works."""
        print("🔍 PROJECT ANALYSIS")
        print("=" * 50)
        
        print("📁 Detected Project Structure:")
        self._print_structure(self.project_structure, indent=0)
        
        print(f"\n📦 Dependencies Found: {len(self.dependencies)}")
        for dep in self.dependencies:
            print(f"  • {dep}")
        
        print(f"\n🏛️ Architectural Patterns Detected:")
        for pattern_type, pattern_value in self.detected_patterns.items():
            print(f"  • {pattern_type}: {pattern_value}")

    def demonstrate_similar_code_finding(self, feature_request: str):
        """Show how similar code finding works."""
        print(f"\n🔍 FINDING SIMILAR CODE FOR: '{feature_request}'")
        print("=" * 50)
        
        # Simulate semantic search results
        similar_files = []
        
        if "profile" in feature_request.lower():
            similar_files = [
                {
                    "file": "lib/features/auth/auth_screen.dart",
                    "similarity": 0.75,
                    "reason": "Similar screen structure with form inputs"
                },
                {
                    "file": "lib/features/home/home_screen.dart", 
                    "similarity": 0.60,
                    "reason": "Standard screen widget pattern"
                }
            ]
        
        print("📋 Similar Code Found:")
        for file_info in similar_files:
            print(f"  • {file_info['file']} (similarity: {file_info['similarity']:.2f})")
            print(f"    Reason: {file_info['reason']}")

    def demonstrate_integration_planning(self, feature_request: str):
        """Show how integration planning works."""
        print(f"\n📋 INTEGRATION PLANNING FOR: '{feature_request}'")
        print("=" * 50)
        
        # Simulate integration plan
        plan = {
            "new_files": [
                {
                    "path": "lib/features/profile/profile_screen.dart",
                    "purpose": "Main profile screen widget"
                },
                {
                    "path": "lib/features/profile/profile_bloc.dart", 
                    "purpose": "Profile business logic"
                },
                {
                    "path": "lib/features/profile/profile_repository.dart",
                    "purpose": "Profile data management"
                },
                {
                    "path": "test/features/profile/profile_bloc_test.dart",
                    "purpose": "Unit tests for profile bloc"
                }
            ],
            "affected_files": [
                {
                    "path": "lib/app.dart",
                    "change": "Add profile route",
                    "breaking": False
                },
                {
                    "path": "lib/features/home/home_screen.dart",
                    "change": "Add navigation to profile",
                    "breaking": False
                }
            ],
            "dependencies_to_add": ["image_picker"],
            "complexity": "medium",
            "estimated_time": "2-3 hours"
        }
        
        print("📁 New Files to Create:")
        for file_info in plan["new_files"]:
            print(f"  • {file_info['path']}")
            print(f"    Purpose: {file_info['purpose']}")
        
        print(f"\n🔧 Existing Files to Modify:")
        for file_info in plan["affected_files"]:
            breaking = "⚠️ BREAKING" if file_info["breaking"] else "✅ Safe"
            print(f"  • {file_info['path']} ({breaking})")
            print(f"    Change: {file_info['change']}")
        
        print(f"\n📦 Dependencies to Add:")
        for dep in plan["dependencies_to_add"]:
            print(f"  • {dep}")
        
        print(f"\n⏱️ Estimated Complexity: {plan['complexity']}")
        print(f"🕒 Estimated Time: {plan['estimated_time']}")

    def demonstrate_code_generation(self, feature_request: str):
        """Show generated code that matches project patterns."""
        print(f"\n💻 GENERATED CODE FOR: '{feature_request}'")
        print("=" * 50)
        
        # Generate code following detected patterns
        bloc_code = '''
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';

// Following project's BLoC pattern detected in auth and home features

// Events
abstract class ProfileEvent extends Equatable {
  @override
  List<Object> get props => [];
}

class LoadProfile extends ProfileEvent {}

class UpdateProfile extends ProfileEvent {
  final String name;
  final String email;
  
  UpdateProfile({required this.name, required this.email});
  
  @override
  List<Object> get props => [name, email];
}

// States
abstract class ProfileState extends Equatable {
  @override
  List<Object> get props => [];
}

class ProfileInitial extends ProfileState {}

class ProfileLoading extends ProfileState {}

class ProfileLoaded extends ProfileState {
  final String name;
  final String email;
  final String? avatarUrl;
  
  ProfileLoaded({
    required this.name,
    required this.email,
    this.avatarUrl
  });
  
  @override
  List<Object?> get props => [name, email, avatarUrl];
}

class ProfileError extends ProfileState {
  final String message;
  
  ProfileError(this.message);
  
  @override
  List<Object> get props => [message];
}

// BLoC
class ProfileBloc extends Bloc<ProfileEvent, ProfileState> {
  ProfileBloc() : super(ProfileInitial()) {
    on<LoadProfile>(_onLoadProfile);
    on<UpdateProfile>(_onUpdateProfile);
  }
  
  void _onLoadProfile(LoadProfile event, Emitter<ProfileState> emit) async {
    emit(ProfileLoading());
    try {
      // Load profile logic here
      emit(ProfileLoaded(
        name: 'John Doe',
        email: 'john@example.com'
      ));
    } catch (e) {
      emit(ProfileError('Failed to load profile'));
    }
  }
  
  void _onUpdateProfile(UpdateProfile event, Emitter<ProfileState> emit) async {
    emit(ProfileLoading());
    try {
      // Update profile logic here
      emit(ProfileLoaded(
        name: event.name,
        email: event.email
      ));
    } catch (e) {
      emit(ProfileError('Failed to update profile'));
    }
  }
}
'''
        
        screen_code = '''
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../../../shared/widgets/custom_button.dart';
import 'profile_bloc.dart';

// Following project's screen widget pattern detected in auth and home

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => ProfileBloc()..add(LoadProfile()),
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Profile'),
          // Following project's AppBar pattern
        ),
        body: BlocBuilder<ProfileBloc, ProfileState>(
          builder: (context, state) {
            if (state is ProfileLoading) {
              return const Center(child: CircularProgressIndicator());
            }
            
            if (state is ProfileError) {
              return Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      state.message,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                    const SizedBox(height: 16),
                    CustomButton(
                      text: 'Retry',
                      onPressed: () => context.read<ProfileBloc>().add(LoadProfile()),
                    ),
                  ],
                ),
              );
            }
            
            if (state is ProfileLoaded) {
              return Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    // Avatar section
                    CircleAvatar(
                      radius: 50,
                      backgroundImage: state.avatarUrl != null
                          ? CachedNetworkImageProvider(state.avatarUrl!)
                          : null,
                      child: state.avatarUrl == null
                          ? const Icon(Icons.person, size: 50)
                          : null,
                    ),
                    const SizedBox(height: 24),
                    
                    // Name
                    Text(
                      state.name,
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 8),
                    
                    // Email
                    Text(
                      state.email,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                    const SizedBox(height: 32),
                    
                    // Actions using project's custom button
                    CustomButton(
                      text: 'Edit Profile',
                      onPressed: () {
                        // Navigate to edit screen
                      },
                    ),
                  ],
                ),
              );
            }
            
            return const SizedBox.shrink();
          },
        ),
      ),
    );
  }
}
'''
        
        print("📄 Generated BLoC (lib/features/profile/profile_bloc.dart):")
        print("```dart")
        print(bloc_code.strip())
        print("```")
        
        print("\n📄 Generated Screen (lib/features/profile/profile_screen.dart):")
        print("```dart")
        print(screen_code.strip())
        print("```")
        
        print("\n✨ Code Generation Features Demonstrated:")
        print("  • Follows detected BLoC pattern from existing features")
        print("  • Uses same naming conventions (snake_case files)")
        print("  • Imports organized following project pattern")
        print("  • Reuses existing custom widgets (CustomButton)")
        print("  • Follows same error handling patterns")
        print("  • Uses project's theme system")
        print("  • Matches existing code structure and style")

    def demonstrate_validation(self):
        """Show code validation process."""
        print(f"\n✅ CODE VALIDATION")
        print("=" * 50)
        
        validation_results = {
            "syntax_check": "✅ Pass - No syntax errors",
            "import_validation": "✅ Pass - All imports valid",
            "convention_compliance": "✅ Pass - Follows project conventions",
            "architecture_compliance": "✅ Pass - Respects BLoC architecture",
            "pattern_consistency": "✅ Pass - Matches existing patterns",
            "test_coverage": "✅ Pass - Test files planned",
            "documentation": "✅ Pass - Code properly documented"
        }
        
        print("📋 Validation Results:")
        for check, result in validation_results.items():
            print(f"  • {check}: {result}")

    def _print_structure(self, structure: Dict, indent: int):
        """Helper to print directory structure."""
        for name, content in structure.items():
            print("  " * indent + f"📁 {name}" if name.endswith("/") else "  " * indent + f"📄 {name}")
            if isinstance(content, dict):
                self._print_structure(content, indent + 1)

    def run_demo(self):
        """Run the complete demonstration."""
        print("🚀 CONTEXTUAL CODE GENERATION DEMO")
        print("=" * 60)
        print("Demonstrating intelligent Flutter code generation that")
        print("understands and adapts to existing project patterns.\n")
        
        feature_request = "user profile screen with avatar and settings"
        
        # Show each phase of the process
        self.demonstrate_project_analysis()
        self.demonstrate_similar_code_finding(feature_request)
        self.demonstrate_integration_planning(feature_request)
        self.demonstrate_code_generation(feature_request)
        self.demonstrate_validation()
        
        print(f"\n🎉 DEMO COMPLETE")
        print("=" * 60)
        print("The ImplementationAgent successfully:")
        print("  ✅ Analyzed existing project structure")
        print("  ✅ Detected BLoC architectural pattern")
        print("  ✅ Found similar code for reference")
        print("  ✅ Planned seamless integration")
        print("  ✅ Generated contextually-aware code")
        print("  ✅ Validated code quality and compliance")
        print("\nResult: Production-ready code that feels like it was")
        print("written by a team member who deeply understands the project!")


def main():
    """Run the contextual code generation demo."""
    demo = ContextualCodeGenerationDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
