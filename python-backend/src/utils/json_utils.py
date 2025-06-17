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
        
    # First try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON within the text
    try:
        # Remove markdown code block markers first
        cleaned_text = re.sub(r'```(?:json)?\s*|\s*```', '', response_text)
        
        # Try to parse cleaned text directly
        try:
            return json.loads(cleaned_text.strip())
        except json.JSONDecodeError:
            pass
        
        # More aggressive JSON extraction patterns
        json_patterns = [
            r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON in markdown blocks
            r'\{[\s\S]*?"main_dart"[\s\S]*?\}(?=\s*```|\s*$|\s*[A-Z])',  # JSON ending before next section
            r'\{[\s\S]*?"files"[\s\S]*?\}(?=\s*```|\s*$|\s*[A-Z])',  # JSON with files object
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Basic nested JSON matching
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_text, re.MULTILINE)
            for match in matches:
                try:
                    # Handle case where match is a tuple (from capturing groups)
                    json_str = match if isinstance(match, str) else match[0] if match else ""
                    
                    # Try to fix truncated JSON
                    json_str = _fix_truncated_json(json_str)
                    
                    parsed = json.loads(json_str)
                    # Validate it's a meaningful JSON object
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        # Try a more aggressive approach - look for lines that start and end with braces
        lines = cleaned_text.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            if not in_json and stripped.startswith('{'):
                in_json = True
                json_lines = [line]
                brace_count = stripped.count('{') - stripped.count('}')
            elif in_json:
                json_lines.append(line)
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    # Try to parse accumulated JSON
                    potential_json = '\n'.join(json_lines)
                    try:
                        # Try to fix any issues with the JSON
                        potential_json = _fix_truncated_json(potential_json)
                        parsed = json.loads(potential_json)
                        if isinstance(parsed, dict) and len(parsed) > 0:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    # Reset for next potential JSON block
                    in_json = False
                    json_lines = []
                    brace_count = 0
        
        logger.error(f"Failed to extract JSON from: {response_text[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
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
