#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from config.agent_configs import agent_config_manager
    print("✅ agent_config_manager imported successfully")
    print(f"Type: {type(agent_config_manager)}")
    print(f"Available methods: {[method for method in dir(agent_config_manager) if not method.startswith('_')]}")
except Exception as e:
    print(f"❌ Failed to import agent_config_manager: {e}")
    import traceback
    traceback.print_exc()
