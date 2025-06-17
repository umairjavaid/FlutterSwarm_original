#!/usr/bin/env python3
"""
Very simple test to isolate LLM client issues.
"""

import sys
import os
import asyncio
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.DEBUG)

async def test_simple_llm():
    """Simple LLM test."""
    print("🧪 Testing Simple LLM Client")
    
    try:
        from src.core.llm_client import LLMClient, LLMRequest
        
        # Create LLM client
        client = LLMClient()
        print(f"✅ LLM Client created, default provider: {client.default_provider}")
        
        # Create a simple request
        request = LLMRequest(
            prompt="Create a personal finance tracker app JSON with main_dart",
            model="local_fallback",
            temperature=0.7,
            max_tokens=4000
        )
        
        print("🔄 Sending request to LLM...")
        response = await client.generate(request.prompt)
        
        print(f"✅ Response received, length: {len(response)} characters")
        print("📋 First 500 characters:")
        print(response[:500])
        
        # Check if it's JSON
        import json
        try:
            parsed = json.loads(response)
            print("✅ Response is valid JSON")
            print(f"📋 JSON keys: {list(parsed.keys())}")
            
            if "main_dart" in parsed:
                main_dart = parsed["main_dart"]
                if "PersonalFinanceApp" in main_dart:
                    print("✅ main_dart contains PersonalFinanceApp")
                else:
                    print("❌ main_dart does not contain PersonalFinanceApp")
                    print(f"Content: {main_dart[:200]}")
            else:
                print("❌ No main_dart in response")
                
        except json.JSONDecodeError as e:
            print(f"❌ Response is not valid JSON: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_simple_llm())
    print(f"Result: {result}")
