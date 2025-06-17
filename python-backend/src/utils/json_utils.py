import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_llm_response(response_text):
    """
    Safely extract JSON from LLM response text which might contain additional text.
    """
    # First try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON within the text
    try:
        # Try to find JSON objects enclosed in {}
        json_pattern = r'({[\s\S]*})'
        matches = re.findall(json_pattern, response_text)
        
        if matches:
            for potential_json in matches:
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    continue
        
        # If we didn't find valid JSON, try fixing common issues
        # 1. Remove markdown code block markers
        cleaned_text = re.sub(r'```(?:json)?\s*|\s*```', '', response_text)
        
        # 2. Try to extract JSON again
        matches = re.findall(json_pattern, cleaned_text)
        if matches:
            for potential_json in matches:
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    continue
        
        logger.error(f"Failed to extract JSON from: {response_text[:100]}...")
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return None
