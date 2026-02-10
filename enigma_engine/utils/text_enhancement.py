"""
Text Enhancement Utilities
===========================

Utilities for improving user text input, including typo correction,
suggestion generation, and formatting helpers.

Usage:
    from enigma_engine.utils.text_enhancement import correct_typos, suggest_command
    
    corrected = correct_typos("Helo wrld")  # "Hello world"
    suggestion = suggest_command("tran", ["train", "transfer", "translate"])
"""

import re
from difflib import get_close_matches
from typing import Optional


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into another.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (0 = identical strings)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # Use dynamic programming
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_closest_match(
    word: str,
    candidates: list[str],
    cutoff: float = 0.6,
    n: int = 3
) -> list[tuple[str, float]]:
    """
    Find closest matches for a word from a list of candidates.
    
    Args:
        word: Word to match
        candidates: List of candidate words
        cutoff: Minimum similarity ratio (0.0 - 1.0)
        n: Maximum number of matches to return
        
    Returns:
        List of (match, similarity) tuples
    """
    if not word or not candidates:
        return []
    
    # Use difflib for quick matching
    matches = get_close_matches(word.lower(), 
                                [c.lower() for c in candidates],
                                n=n, cutoff=cutoff)
    
    # Calculate similarity scores
    results = []
    for match in matches:
        # Find original case version
        original = next((c for c in candidates if c.lower() == match), match)
        
        # Calculate similarity (inverse of normalized distance)
        distance = levenshtein_distance(word.lower(), match)
        max_len = max(len(word), len(match))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        
        results.append((original, similarity))
    
    return sorted(results, key=lambda x: x[1], reverse=True)


def suggest_command(
    input_cmd: str,
    valid_commands: list[str],
    threshold: float = 0.5
) -> Optional[str]:
    """
    Suggest correct command based on typo.
    
    Args:
        input_cmd: User's input command
        valid_commands: List of valid commands
        threshold: Minimum similarity threshold
        
    Returns:
        Suggested command or None
    """
    matches = find_closest_match(input_cmd, valid_commands, cutoff=threshold, n=1)
    if matches and matches[0][1] >= threshold:
        return matches[0][0]
    return None


def format_did_you_mean(input_word: str, suggestions: list[str]) -> str:
    """
    Format "Did you mean..." message.
    
    Args:
        input_word: User's input
        suggestions: List of suggested words
        
    Returns:
        Formatted suggestion message
    """
    if not suggestions:
        return ""
    
    if len(suggestions) == 1:
        return f"Did you mean '{suggestions[0]}'?"
    elif len(suggestions) == 2:
        return f"Did you mean '{suggestions[0]}' or '{suggestions[1]}'?"
    else:
        # Show first 3 suggestions
        formatted = "', '".join(suggestions[:3])
        return f"Did you mean '{formatted}'?"


# Common English typos dictionary (subset)
COMMON_TYPOS = {
    'teh': 'the',
    'taht': 'that',
    'hte': 'the',
    'fo': 'of',
    'adn': 'and',
    'nad': 'and',
    'tot': 'to',
    'wiht': 'with',
    'thsi': 'this',
    'youe': 'you',
    'yuor': 'your',
    'trian': 'train',
    'modle': 'model',
    'infrence': 'inference',
    'inferece': 'inference',
    'epoc': 'epoch',
    'epocs': 'epochs',
    'epohcs': 'epochs',
    'bathc': 'batch',
    'leraning': 'learning',
    'learnign': 'learning',
}


def correct_typos(text: str, custom_dict: dict = None) -> str:
    """
    Auto-correct common typos in text.
    
    Args:
        text: Input text with potential typos
        custom_dict: Optional custom typo dictionary
        
    Returns:
        Text with typos corrected
    """
    typo_dict = COMMON_TYPOS.copy()
    if custom_dict:
        typo_dict.update(custom_dict)
    
    # Split into words while preserving punctuation
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    corrected = []
    
    for word in words:
        if word.lower() in typo_dict:
            # Preserve original capitalization
            correction = typo_dict[word.lower()]
            if word[0].isupper():
                correction = correction.capitalize()
            corrected.append(correction)
        else:
            corrected.append(word)
    
    # Rejoin text
    result = ''
    for i, word in enumerate(corrected):
        if i > 0 and word not in '.,!?;:\'"\'--':
            result += ' '
        result += word
    
    return result


def validate_parameter(
    value: str,
    param_type: str,
    valid_options: list[str] = None,
    min_val: float = None,
    max_val: float = None
) -> tuple[bool, Optional[str]]:
    """
    Validate a parameter value and provide helpful error messages.
    
    Args:
        value: Parameter value as string
        param_type: Type ('int', 'float', 'choice', 'bool')
        valid_options: Valid options for 'choice' type
        min_val: Minimum value for numeric types
        max_val: Maximum value for numeric types
        
    Returns:
        (is_valid, error_message) tuple
    """
    try:
        if param_type == 'int':
            val = int(value)
            if min_val is not None and val < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and val > max_val:
                return False, f"Value must be <= {max_val}"
            return True, None
        
        elif param_type == 'float':
            val = float(value)
            if min_val is not None and val < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and val > max_val:
                return False, f"Value must be <= {max_val}"
            return True, None
        
        elif param_type == 'choice':
            if valid_options and value not in valid_options:
                # Suggest close matches
                suggestions = find_closest_match(value, valid_options, cutoff=0.4, n=3)
                if suggestions:
                    sugg_list = [s[0] for s in suggestions]
                    return False, f"Invalid choice. {format_did_you_mean(value, sugg_list)}"
                else:
                    return False, f"Invalid choice. Valid options: {', '.join(valid_options)}"
            return True, None
        
        elif param_type == 'bool':
            if value.lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
                return False, "Value must be true/false, yes/no, or 1/0"
            return True, None
        
        else:
            return True, None  # No validation for unknown types
    
    except ValueError as e:
        return False, f"Invalid {param_type}: {str(e)}"


def format_error_message(
    error: str,
    context: dict = None,
    suggestions: list[str] = None
) -> str:
    """
    Format an error message with context and suggestions.
    
    Args:
        error: Base error message
        context: Optional context dict (param name, value, etc.)
        suggestions: Optional list of suggestions
        
    Returns:
        Formatted error message
    """
    message = f"[ERROR] {error}"
    
    if context:
        message += "\n"
        for key, value in context.items():
            message += f"\n  {key}: {value}"
    
    if suggestions:
        message += "\n\nSuggestions:"
        for i, suggestion in enumerate(suggestions, 1):
            message += f"\n  {i}. {suggestion}"
    
    return message


def highlight_error_in_text(text: str, error_pos: int, error_len: int = 1) -> str:
    """
    Highlight error position in text with visual markers.
    
    Args:
        text: Full text
        error_pos: Position of error
        error_len: Length of error
        
    Returns:
        Text with error highlighted
    """
    if error_pos < 0 or error_pos >= len(text):
        return text
    
    before = text[:error_pos]
    error = text[error_pos:error_pos + error_len]
    after = text[error_pos + error_len:]
    
    # Use markers to highlight
    return f"{before}>>>{error}<<<{after}"


# Common parameter mistakes and suggestions
PARAMETER_SUGGESTIONS = {
    'epochs': {
        'typos': ['epoc', 'epoch', 'epohcs', 'epocs'],
        'tips': [
            'Typical values: 10-50 for small datasets, 3-10 for large',
            'More epochs = longer training but potentially better results'
        ]
    },
    'batch_size': {
        'typos': ['batchsize', 'batch', 'bathc_size', 'batch-size'],
        'tips': [
            'Start with 2-4 for CPU, 8-16 for GPU',
            'Larger batch = faster but uses more memory'
        ]
    },
    'learning_rate': {
        'typos': ['learningrate', 'learn_rate', 'lr', 'lernign_rate'],
        'tips': [
            'Typical values: 0.0001 - 0.001',
            'Too high = unstable training, too low = slow learning'
        ]
    },
    'temperature': {
        'typos': ['temp', 'temperatur'],
        'tips': [
            'Range: 0.0 (deterministic) to 2.0 (creative)',
            'Lower = more focused, higher = more random'
        ]
    }
}


def suggest_parameter_fix(param_name: str) -> Optional[dict]:
    """
    Get suggestions for fixing a parameter.
    
    Args:
        param_name: Parameter name (possibly misspelled)
        
    Returns:
        Dict with corrections and tips, or None
    """
    # Check direct match
    if param_name in PARAMETER_SUGGESTIONS:
        return PARAMETER_SUGGESTIONS[param_name]
    
    # Check for typos
    for correct_name, info in PARAMETER_SUGGESTIONS.items():
        if param_name.lower() in info['typos']:
            return {
                'correct_name': correct_name,
                'tips': info['tips']
            }
    
    # Try fuzzy matching
    all_params = list(PARAMETER_SUGGESTIONS.keys())
    matches = find_closest_match(param_name, all_params, cutoff=0.6, n=1)
    if matches:
        correct_name = matches[0][0]
        return {
            'correct_name': correct_name,
            'tips': PARAMETER_SUGGESTIONS[correct_name]['tips']
        }
    
    return None


if __name__ == "__main__":
    # Test examples
    print("Text Enhancement Utilities")
    print("=" * 50)
    
    # Test typo correction
    print("\n1. Typo Correction:")
    test_text = "Teh modle is leraning form teh data"
    corrected = correct_typos(test_text)
    print(f"   Original: {test_text}")
    print(f"   Corrected: {corrected}")
    
    # Test command suggestion
    print("\n2. Command Suggestion:")
    commands = ["train", "inference", "generate", "evaluate"]
    typo = "trian"
    suggestion = suggest_command(typo, commands)
    print(f"   Input: {typo}")
    print(f"   Suggestion: {suggestion}")
    
    # Test parameter validation
    print("\n3. Parameter Validation:")
    is_valid, error = validate_parameter("150", "int", min_val=1, max_val=100)
    print(f"   Value: 150, Range: 1-100")
    print(f"   Valid: {is_valid}, Error: {error}")
    
    # Test "Did you mean"
    print("\n4. Did You Mean:")
    suggestions = ["small", "medium", "large"]
    print(f"   {format_did_you_mean('smal', suggestions)}")
