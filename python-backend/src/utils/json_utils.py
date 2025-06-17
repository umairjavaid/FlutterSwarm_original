import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_llm_response(response_text):
    """
    Safely extract JSON from LLM response text which might contain additional text.
    """
    if not response_text or not response_text.strip():
        logger.error("Empty or None response text provided")
        return None
    
    logger.debug(f"Attempting to extract JSON from response length: {len(response_text)}")
    
    # First try to parse the entire response as JSON
    try:
        parsed = json.loads(response_text.strip())
        logger.info("Successfully parsed entire response as JSON")
        return parsed
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")
    
    # Look for JSON within the text
    try:
        # Remove markdown code block markers first
        cleaned_text = re.sub(r'```(?:json)?\s*|\s*```', '', response_text)
        
        # Try to parse cleaned text directly
        try:
            parsed = json.loads(cleaned_text.strip())
            logger.info("Successfully parsed cleaned response as JSON")
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"Cleaned text parsing failed: {e}")
        
        # More aggressive JSON extraction patterns - look for the main object
        json_patterns = [
            r'^\s*(\{[\s\S]*\})\s*$',  # Entire response is JSON
            r'\{[\s\S]*?"files"[\s\S]*?\}(?=\s*$|\s*```|\s*[A-Z][a-z])',  # JSON with files ending at text/code blocks
            r'\{[\s\S]*?"main_dart"[\s\S]*?\}(?=\s*$|\s*```|\s*[A-Z][a-z])',  # JSON with main_dart
            r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON in markdown blocks
            r'(\{[\s\S]*?"dependencies"[\s\S]*?\})',  # JSON with dependencies
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple balanced braces (non-recursive)
        ]
        
        for i, pattern in enumerate(json_patterns):
            matches = re.findall(pattern, cleaned_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    # Handle case where match is a tuple (from capturing groups)
                    json_str = match if isinstance(match, str) else match[0] if match else ""
                    
                    # Clean up common issues
                    json_str = json_str.strip()
                    if not json_str:
                        continue
                        
                    # Try to fix truncated JSON by balancing braces
                    json_str = _fix_truncated_json(json_str)
                    
                    parsed = json.loads(json_str)
                    logger.info(f"Successfully extracted JSON using pattern {i}")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"Pattern {i} failed: {e}")
                    continue

        # Final fallback: Find the first complete JSON object using brace matching
        start_idx = cleaned_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(cleaned_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if brace_count == 0:  # Found matching braces
                potential_json = cleaned_text[start_idx:end_idx]
                try:
                    parsed = json.loads(potential_json)
                    logger.info("Successfully extracted JSON using brace matching")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"Brace matching failed: {e}")
        
        logger.error(f"Failed to extract JSON from response. First 500 chars: {response_text[:500]}...")
        return None
    
    except Exception as e:
        logger.error(f"Error during JSON extraction: {e}")
        return None


def _fix_truncated_json(json_str):
    """
    Try to fix common JSON truncation issues.
    """
    if not json_str.strip():
        return json_str
    
    json_str = json_str.strip()
    
    # If it doesn't end with }, try to close it properly
    if not json_str.endswith('}'):
        # Look for the last complete key-value pair
        last_complete_positions = [
            json_str.rfind('},'),
            json_str.rfind('],'),
            json_str.rfind('",'),
            json_str.rfind('"')
        ]
        
        # Find the furthest complete position
        best_pos = max([pos for pos in last_complete_positions if pos > 0], default=-1)
        
        if best_pos > 0:
            if json_str[best_pos:best_pos+2] in ['},', '],', '",']:
                json_str = json_str[:best_pos + 1] + '}'
            elif json_str[best_pos] == '"':
                json_str = json_str[:best_pos + 1] + '}'
    
    # Fix common formatting issues
    json_str = re.sub(r',\s*\}', '}', json_str)  # Remove trailing commas
    json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas in arrays
    
    # Fix unterminated strings by adding closing quotes where needed
    if json_str.count('"') % 2 != 0:
        # Odd number of quotes, try to fix
        json_str = json_str + '"}'
    
    return json_str
