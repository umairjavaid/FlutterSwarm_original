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
        
        # Try to find JSON objects enclosed in {}
        json_pattern = r'({[\s\S]*?})'
        matches = re.findall(json_pattern, cleaned_text)
        
        if matches:
            # Try each match, starting with the longest (most likely to be complete)
            matches_sorted = sorted(matches, key=len, reverse=True)
            for potential_json in matches_sorted:
                try:
                    parsed = json.loads(potential_json)
                    # Validate it's a meaningful JSON object
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed
                except json.JSONDecodeError:
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
