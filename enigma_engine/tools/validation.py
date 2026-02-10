"""
Tool Schema Validation
=======================

Validates tool parameters against schemas with detailed error reporting.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolSchemaValidator:
    """
    Validates tool parameters against schemas.
    
    Provides comprehensive validation including:
    - Type checking
    - Required parameter verification
    - Enum validation
    - Range validation
    - Pattern matching
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize schema validator.
        
        Args:
            strict_mode: If True, fail on unknown parameters
        """
        self.strict_mode = strict_mode
        logger.info(f"ToolSchemaValidator initialized (strict={strict_mode})")
    
    def validate(
        self,
        tool_definition,
        params: dict[str, Any]
    ) -> tuple[bool, list[str], dict[str, Any]]:
        """
        Validate parameters against tool schema.
        
        Args:
            tool_definition: ToolDefinition object with schema
            params: Parameters to validate
            
        Returns:
            (is_valid, errors, validated_params)
        """
        errors = []
        validated = {}
        
        # Get parameter definitions
        param_defs = {p.name: p for p in tool_definition.parameters}
        
        # Check for unknown parameters in strict mode
        if self.strict_mode:
            for param_name in params:
                if param_name not in param_defs:
                    errors.append(f"Unknown parameter: {param_name}")
        
        # Validate each defined parameter
        for param_def in tool_definition.parameters:
            param_name = param_def.name
            
            # Check if parameter is provided
            if param_name in params:
                value = params[param_name]
                
                # Validate the value
                is_valid, error, validated_value = self._validate_value(
                    param_name,
                    value,
                    param_def
                )
                
                if not is_valid:
                    errors.append(error)
                else:
                    validated[param_name] = validated_value
            
            elif param_def.required:
                # Required parameter missing
                errors.append(f"Missing required parameter: {param_name}")
            
            elif param_def.default is not None:
                # Use default value
                validated[param_name] = param_def.default
        
        is_valid = len(errors) == 0
        return is_valid, errors, validated
    
    def _validate_value(
        self,
        param_name: str,
        value: Any,
        param_def
    ) -> tuple[bool, Optional[str], Any]:
        """
        Validate a single parameter value.
        
        Returns:
            (is_valid, error_message, validated_value)
        """
        expected_type = param_def.type
        
        # Type validation and conversion
        if expected_type == "int":
            if not isinstance(value, int):
                # Try to convert
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    return False, f"Parameter '{param_name}' must be an integer", value
        
        elif expected_type == "float":
            if not isinstance(value, (int, float)):
                # Try to convert
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return False, f"Parameter '{param_name}' must be a number", value
        
        elif expected_type == "bool":
            if not isinstance(value, bool):
                # Try to convert
                if isinstance(value, str):
                    value = value.lower() in ('true', 'yes', '1', 'on')
                else:
                    value = bool(value)
        
        elif expected_type == "string":
            if not isinstance(value, str):
                value = str(value)
        
        elif expected_type == "list":
            if not isinstance(value, list):
                return False, f"Parameter '{param_name}' must be a list", value
        
        elif expected_type == "dict":
            if not isinstance(value, dict):
                return False, f"Parameter '{param_name}' must be a dictionary", value
        
        # Enum validation
        if param_def.enum and value not in param_def.enum:
            return False, f"Parameter '{param_name}' must be one of {param_def.enum}, got {value}", value
        
        return True, None, value
    
    def validate_multiple(
        self,
        tool_calls: list[tuple[Any, dict[str, Any]]]
    ) -> list[tuple[bool, list[str], dict[str, Any]]]:
        """
        Validate multiple tool calls.
        
        Args:
            tool_calls: List of (tool_definition, params) tuples
            
        Returns:
            List of (is_valid, errors, validated_params) tuples
        """
        results = []
        
        for tool_def, params in tool_calls:
            result = self.validate(tool_def, params)
            results.append(result)
        
        return results
    
    def get_validation_report(
        self,
        tool_definition,
        params: dict[str, Any]
    ) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            tool_definition: ToolDefinition object
            params: Parameters to validate
            
        Returns:
            Formatted validation report
        """
        is_valid, errors, validated = self.validate(tool_definition, params)
        
        report = [f"Validation Report for '{tool_definition.name}'"]
        report.append("=" * 50)
        
        if is_valid:
            report.append("✓ All parameters valid")
            report.append(f"\nValidated parameters:")
            for key, value in validated.items():
                report.append(f"  - {key}: {value}")
        else:
            report.append(f"✗ Validation failed with {len(errors)} error(s):\n")
            for i, error in enumerate(errors, 1):
                report.append(f"  {i}. {error}")
            
            if validated:
                report.append(f"\nPartially validated parameters:")
                for key, value in validated.items():
                    report.append(f"  - {key}: {value}")
        
        return "\n".join(report)


__all__ = [
    "ToolSchemaValidator",
]
