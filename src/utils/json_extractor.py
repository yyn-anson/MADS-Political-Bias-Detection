"""
Robust JSON Extraction Utility for Multi-Agent Discussion System

This module provides robust JSON extraction and parsing capabilities to handle
various formatting issues that occur during LLM discussions.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)


class RobustJSONExtractor:
    """Robust JSON extractor with multiple fallback strategies."""
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text using multiple strategies.
        
        Args:
            text: Raw text that may contain JSON
            
        Returns:
            Extracted JSON as dictionary, or None if extraction fails
        """
        if not text or not isinstance(text, str):
            return None
            
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Strategy 2: Extract JSON block from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON-like structure in text
        # Look for content between first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_candidate = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                # Try to fix common issues
                fixed_json = RobustJSONExtractor._fix_common_json_issues(json_candidate)
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 4: Regex extraction for specific fields
        extracted = RobustJSONExtractor._extract_fields_with_regex(text)
        if extracted:
            return extracted
        
        return None
    
    @staticmethod
    def _fix_common_json_issues(json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Fix single quotes to double quotes (more robust)
        # First, protect escaped quotes
        json_str = json_str.replace(r"\'", "ESCAPED_SINGLE_QUOTE")
        json_str = json_str.replace(r'\"', "ESCAPED_DOUBLE_QUOTE")
        
        # Convert single quotes to double quotes for JSON keys/values
        # This regex is more careful about single quotes
        import re
        # Replace single quotes that are likely JSON delimiters
        json_str = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', json_str)  # Keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Values
        
        # Restore escaped quotes
        json_str = json_str.replace("ESCAPED_SINGLE_QUOTE", "'")
        json_str = json_str.replace("ESCAPED_DOUBLE_QUOTE", '\\"')
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unquoted keys (but avoid already quoted ones)
        json_str = re.sub(r'(?<!")(\w+)(?!"):', r'"\1":', json_str)
        
        return json_str
    
    @staticmethod
    def _extract_fields_with_regex(text: str) -> Optional[Dict[str, Any]]:
        """Extract specific fields using regex patterns."""
        result = {}
        
        # Extract understanding field
        understanding_pattern = r'"understanding"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        match = re.search(understanding_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result['understanding'] = match.group(1).replace('\\"', '"')
        
        # Extract challenge field (handle both string and array)
        # First try string format
        challenge_pattern = r'"challenge"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        match = re.search(challenge_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result['challenge'] = match.group(1).replace('\\"', '"')
        else:
            # Try array format
            challenge_array_pattern = r'"challenge"\s*:\s*\[(.*?)\]'
            match = re.search(challenge_array_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                # Extract strings from array and join them
                array_content = match.group(1)
                strings = re.findall(r'"([^"]*(?:\\.[^"]*)*)"', array_content)
                if strings:
                    result['challenge'] = ' '.join(s.replace('\\"', '"') for s in strings)
        
        # Extract adjusted_lean field
        lean_pattern = r'"adjusted_lean"\s*:\s*(-?\d+(?:\.\d+)?)'
        match = re.search(lean_pattern, text, re.IGNORECASE)
        if match:
            try:
                result['adjusted_lean'] = float(match.group(1))
            except ValueError:
                pass
        
        # Extract acknowledgment field
        ack_pattern = r'"acknowledgment"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        match = re.search(ack_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result['acknowledgment'] = match.group(1).replace('\\"', '"')
        
        # Extract lean field (for responses)
        lean_pattern = r'"lean"\s*:\s*(-?\d+(?:\.\d+)?)'
        match = re.search(lean_pattern, text, re.IGNORECASE)
        if match:
            try:
                result['lean'] = float(match.group(1))
            except ValueError:
                pass
        
        # Extract reason field
        reason_pattern = r'"reason"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        match = re.search(reason_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result['reason'] = match.group(1).replace('\\"', '"')
        
        return result if result else None
    
    @staticmethod
    def extract_challenge_fields(text: str) -> Dict[str, Any]:
        """
        Extract challenge-specific fields from discussion text.
        
        Returns dict with 'understanding', 'challenge', and 'adjusted_lean' fields.
        """
        extracted = RobustJSONExtractor.extract_json(text)
        
        if extracted:
            # Ensure challenge is a string (not array)
            if 'challenge' in extracted:
                if isinstance(extracted['challenge'], list):
                    # Join array elements with space
                    extracted['challenge'] = ' '.join(str(item) for item in extracted['challenge'])
                elif not isinstance(extracted['challenge'], str):
                    # Convert non-string to string
                    extracted['challenge'] = str(extracted['challenge'])
            
            # Ensure adjusted_lean is properly typed as float or None
            if 'adjusted_lean' in extracted and extracted['adjusted_lean'] is not None:
                if isinstance(extracted['adjusted_lean'], str):
                    try:
                        extracted['adjusted_lean'] = float(extracted['adjusted_lean'])
                        logger.warning(f"Converted adjusted_lean from string '{extracted['adjusted_lean']}' to float")
                    except ValueError:
                        logger.error(f"Failed to convert adjusted_lean '{extracted['adjusted_lean']}' to float, setting to None")
                        extracted['adjusted_lean'] = None
                elif not isinstance(extracted['adjusted_lean'], (int, float)):
                    logger.error(f"Unexpected type for adjusted_lean: {type(extracted['adjusted_lean'])}, setting to None")
                    extracted['adjusted_lean'] = None
            
            return extracted
        
        # If full extraction failed, return empty dict with defaults
        return {
            'understanding': '',
            'challenge': text,  # Use full text as fallback
            'adjusted_lean': None
        }
    
    @staticmethod
    def extract_response_fields(text: str) -> Dict[str, Any]:
        """
        Extract response-specific fields from discussion text.
        
        Returns dict with 'acknowledgment', 'lean', and 'reason' fields.
        """
        extracted = RobustJSONExtractor.extract_json(text)
        
        if extracted:
            # Map acknowledgment to reason if reason is missing
            if 'acknowledgment' in extracted and 'reason' not in extracted:
                extracted['reason'] = extracted['acknowledgment']
            
            # Ensure lean is properly typed as float or None
            if 'lean' in extracted and extracted['lean'] is not None:
                if isinstance(extracted['lean'], str):
                    try:
                        extracted['lean'] = float(extracted['lean'])
                        logger.warning(f"Converted lean from string '{extracted['lean']}' to float")
                    except ValueError:
                        logger.error(f"Failed to convert lean '{extracted['lean']}' to float, setting to 0")
                        extracted['lean'] = 0
                elif not isinstance(extracted['lean'], (int, float)):
                    logger.error(f"Unexpected type for lean: {type(extracted['lean'])}, setting to 0")
                    extracted['lean'] = 0
            
            # Also handle final_lean field (used in mistral responses)
            if 'final_lean' in extracted and extracted['final_lean'] is not None:
                if isinstance(extracted['final_lean'], str):
                    try:
                        extracted['final_lean'] = float(extracted['final_lean'])
                        logger.warning(f"Converted final_lean from string '{extracted['final_lean']}' to float")
                    except ValueError:
                        logger.error(f"Failed to convert final_lean '{extracted['final_lean']}' to float, setting to 0")
                        extracted['final_lean'] = 0
                elif not isinstance(extracted['final_lean'], (int, float)):
                    logger.error(f"Unexpected type for final_lean: {type(extracted['final_lean'])}, setting to 0")
                    extracted['final_lean'] = 0
            
            return extracted
        
        # If full extraction failed, return empty dict with defaults
        return {
            'acknowledgment': '',
            'reason': text,  # Use full text as fallback
            'lean': 0
        }


def clean_json_string(text: str) -> str:
    """
    Clean a string that should contain JSON by removing common artifacts.
    
    Args:
        text: Raw text that may contain JSON with artifacts
        
    Returns:
        Cleaned string more likely to parse as JSON
    """
    if not text:
        return text
    
    # Remove markdown code block markers
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    
    # Remove assistant markers and thinking tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:assistant|user|system):', '', text, flags=re.IGNORECASE)
    
    # Remove any text before the first { or [
    json_start = max(text.find('{'), text.find('['))
    if json_start > 0:
        text = text[json_start:]
    
    # Remove any text after the last } or ]
    json_end_brace = text.rfind('}')
    json_end_bracket = text.rfind(']')
    json_end = max(json_end_brace, json_end_bracket)
    if json_end > 0:
        text = text[:json_end + 1]
    
    return text.strip()


def validate_discussion_json(data: Dict[str, Any], expected_fields: List[str]) -> bool:
    """
    Validate that extracted JSON contains expected fields.
    
    Args:
        data: Extracted JSON data
        expected_fields: List of field names that should be present
        
    Returns:
        True if all expected fields are present, False otherwise
    """
    if not data or not isinstance(data, dict):
        return False
    
    for field in expected_fields:
        if field not in data or data[field] is None:
            logger.warning(f"Missing or null field in JSON: {field}")
            return False
    
    return True