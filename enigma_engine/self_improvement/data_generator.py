"""
Training Data Generator for Self-Improvement System

Automatically generates Q&A training pairs from code analysis,
teaching the AI about new features, classes, functions, and GUI elements.

Output format:
Q: How do I use the XyzClass?
A: The XyzClass is used for... Example: ...
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """A question-answer training pair."""
    question: str
    answer: str
    category: str = "general"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    def to_training_format(self) -> str:
        """Convert to training text format."""
        return f"Q: {self.question}\nA: {self.answer}\n"


class TrainingDataGenerator:
    """
    Generates training data from code analysis.
    
    Creates diverse Q&A pairs covering:
    - How to use classes and functions
    - What features do
    - GUI interactions
    - Common patterns and best practices
    
    Usage:
        generator = TrainingDataGenerator()
        pairs = generator.generate_from_analysis(analysis)
        generator.save_to_file(pairs, "training_data.txt")
    """
    
    # Question templates for different types
    CLASS_QUESTION_TEMPLATES = [
        "What is {class_name} used for?",
        "How do I use the {class_name} class?",
        "What does {class_name} do?",
        "Explain the {class_name} class",
        "Can you describe {class_name}?",
        "What are the main features of {class_name}?",
        "How does {class_name} work?",
    ]
    
    FUNCTION_QUESTION_TEMPLATES = [
        "What does the {func_name} function do?",
        "How do I call {func_name}?",
        "What are the parameters for {func_name}?",
        "Explain {func_name}",
        "What does {func_name}() return?",
        "When should I use {func_name}?",
    ]
    
    METHOD_QUESTION_TEMPLATES = [
        "What does {class_name}.{method_name} do?",
        "How do I use the {method_name} method on {class_name}?",
        "Explain {class_name}.{method_name}()",
    ]
    
    GUI_QUESTION_TEMPLATES = [
        "How do I use the {widget_name} in the GUI?",
        "What is the {widget_name} widget for?",
        "Where do I find {widget_name} in the interface?",
        "How does the {widget_name} tab work?",
        "What features does {widget_name} provide?",
    ]
    
    USAGE_QUESTION_TEMPLATES = [
        "Show me an example of using {name}",
        "How do I implement {name}?",
        "Give me a code example for {name}",
        "What's the typical usage pattern for {name}?",
    ]
    
    def __init__(self, variation_factor: float = 0.3):
        """
        Initialize generator.
        
        Args:
            variation_factor: How much variation to add to questions (0-1)
        """
        self.variation_factor = variation_factor
        self.generated_questions: set = set()  # Track to avoid duplicates
    
    def generate_from_analysis(self, analysis: Dict[str, Any]) -> List[TrainingPair]:
        """
        Generate training pairs from code analysis.
        
        Args:
            analysis: Output from CodeAnalyzer.analyze()
            
        Returns:
            List of TrainingPair objects
        """
        pairs = []
        
        # Generate for new classes
        for cls in analysis.get("new_classes", []):
            pairs.extend(self._generate_class_pairs(cls))
        
        # Generate for new functions
        for func in analysis.get("new_functions", []):
            pairs.extend(self._generate_function_pairs(func))
        
        # Generate for new GUI elements
        for gui in analysis.get("new_gui_elements", []):
            pairs.extend(self._generate_gui_pairs(gui))
        
        logger.info(f"Generated {len(pairs)} training pairs")
        return pairs
    
    def _generate_class_pairs(self, cls: Dict[str, Any]) -> List[TrainingPair]:
        """Generate training pairs for a class."""
        pairs = []
        class_name = cls.get("name", "Unknown")
        docstring = cls.get("docstring", "")
        methods = cls.get("methods", [])
        bases = cls.get("bases", [])
        path = cls.get("path", "")
        
        # Skip private/internal classes
        if class_name.startswith("_"):
            return pairs
        
        # Build description from docstring or infer from structure
        if docstring:
            description = self._clean_docstring(docstring)
        else:
            description = self._infer_class_description(cls)
        
        # Main class questions
        templates = random.sample(self.CLASS_QUESTION_TEMPLATES, 
                                  min(3, len(self.CLASS_QUESTION_TEMPLATES)))
        
        for template in templates:
            question = template.format(class_name=class_name)
            if question not in self.generated_questions:
                self.generated_questions.add(question)
                pairs.append(TrainingPair(
                    question=question,
                    answer=description,
                    category="class",
                    metadata={"class_name": class_name, "path": path},
                ))
        
        # Method-specific questions
        for method in methods[:5]:  # Limit to 5 most important methods
            method_name = method.get("name", "")
            if method_name.startswith("_") and method_name != "__init__":
                continue
            
            method_doc = method.get("docstring", "")
            signature = method.get("signature", "")
            
            method_answer = self._build_method_answer(
                class_name, method_name, method_doc, signature
            )
            
            template = random.choice(self.METHOD_QUESTION_TEMPLATES)
            question = template.format(class_name=class_name, method_name=method_name)
            
            if question not in self.generated_questions:
                self.generated_questions.add(question)
                pairs.append(TrainingPair(
                    question=question,
                    answer=method_answer,
                    category="method",
                    metadata={"class_name": class_name, "method_name": method_name},
                ))
        
        # Usage example question
        if methods:
            usage_answer = self._build_usage_example(cls)
            template = random.choice(self.USAGE_QUESTION_TEMPLATES)
            question = template.format(name=class_name)
            
            if question not in self.generated_questions:
                self.generated_questions.add(question)
                pairs.append(TrainingPair(
                    question=question,
                    answer=usage_answer,
                    category="usage",
                    metadata={"class_name": class_name},
                ))
        
        return pairs
    
    def _generate_function_pairs(self, func: Dict[str, Any]) -> List[TrainingPair]:
        """Generate training pairs for a function."""
        pairs = []
        func_name = func.get("name", "Unknown")
        docstring = func.get("docstring", "")
        signature = func.get("signature", "")
        params = func.get("parameters", [])
        return_type = func.get("return_type", "")
        
        # Skip private functions
        if func_name.startswith("_"):
            return pairs
        
        # Build description
        if docstring:
            description = self._clean_docstring(docstring)
        else:
            description = self._infer_function_description(func)
        
        # Add signature info
        full_answer = f"The {func_name} function: {description}"
        
        if params:
            param_desc = ", ".join(
                f"{p.get('name')} ({p.get('type', 'any')})" 
                for p in params if p.get('name') != 'self'
            )
            if param_desc:
                full_answer += f"\n\nParameters: {param_desc}"
        
        if return_type:
            full_answer += f"\n\nReturns: {return_type}"
        
        # Generate questions
        templates = random.sample(self.FUNCTION_QUESTION_TEMPLATES, 
                                  min(2, len(self.FUNCTION_QUESTION_TEMPLATES)))
        
        for template in templates:
            question = template.format(func_name=func_name)
            if question not in self.generated_questions:
                self.generated_questions.add(question)
                pairs.append(TrainingPair(
                    question=question,
                    answer=full_answer,
                    category="function",
                    metadata={"func_name": func_name},
                ))
        
        return pairs
    
    def _generate_gui_pairs(self, gui: Dict[str, Any]) -> List[TrainingPair]:
        """Generate training pairs for a GUI element."""
        pairs = []
        widget_name = gui.get("name", "Unknown")
        widget_type = gui.get("widget_type", "Widget")
        properties = gui.get("properties", {})
        
        # Build description
        description = f"{widget_name} is a {widget_type} in the Enigma AI Engine GUI."
        
        methods = properties.get("methods", [])
        if methods:
            key_methods = [m for m in methods if not m.startswith("_")][:5]
            if key_methods:
                description += f"\n\nKey features: {', '.join(key_methods)}"
        
        # Check if it's a tab
        if "Tab" in widget_name:
            description += "\n\nYou can access this tab from the main window's tab bar."
        
        # Generate questions
        templates = random.sample(self.GUI_QUESTION_TEMPLATES,
                                  min(2, len(self.GUI_QUESTION_TEMPLATES)))
        
        for template in templates:
            question = template.format(widget_name=widget_name)
            if question not in self.generated_questions:
                self.generated_questions.add(question)
                pairs.append(TrainingPair(
                    question=question,
                    answer=description,
                    category="gui",
                    metadata={"widget_name": widget_name, "widget_type": widget_type},
                ))
        
        return pairs
    
    def _clean_docstring(self, docstring: str) -> str:
        """Clean and format docstring for training."""
        # Remove excessive whitespace
        lines = [line.strip() for line in docstring.split('\n')]
        
        # Take first paragraph (until empty line or section header)
        result = []
        for line in lines:
            if not line or line.startswith(':') or line.startswith('Args') or \
               line.startswith('Returns') or line.startswith('Example'):
                break
            result.append(line)
        
        return ' '.join(result)[:500]  # Limit length
    
    def _infer_class_description(self, cls: Dict[str, Any]) -> str:
        """Infer class description from its structure."""
        class_name = cls.get("name", "Unknown")
        bases = cls.get("bases", [])
        methods = cls.get("methods", [])
        
        # Infer from name patterns
        if "Manager" in class_name:
            purpose = "manages and coordinates"
        elif "Controller" in class_name:
            purpose = "controls"
        elif "Handler" in class_name:
            purpose = "handles"
        elif "Generator" in class_name:
            purpose = "generates"
        elif "Parser" in class_name:
            purpose = "parses"
        elif "Tab" in class_name:
            purpose = "provides a GUI interface for"
        elif "Config" in class_name:
            purpose = "stores configuration for"
        else:
            purpose = "provides functionality for"
        
        # Extract subject from name
        subject = self._extract_subject(class_name)
        
        description = f"{class_name} {purpose} {subject} in Enigma AI Engine."
        
        # Add base class info
        if bases:
            description += f" It extends {', '.join(bases)}."
        
        # Add key methods
        public_methods = [m.get("name") for m in methods 
                         if not m.get("name", "").startswith("_")][:5]
        if public_methods:
            description += f" Key methods include: {', '.join(public_methods)}."
        
        return description
    
    def _infer_function_description(self, func: Dict[str, Any]) -> str:
        """Infer function description from its signature."""
        func_name = func.get("name", "unknown")
        params = func.get("parameters", [])
        return_type = func.get("return_type", "")
        
        # Infer from name patterns
        if func_name.startswith("get_"):
            action = "retrieves"
            subject = func_name[4:].replace("_", " ")
        elif func_name.startswith("set_"):
            action = "sets"
            subject = func_name[4:].replace("_", " ")
        elif func_name.startswith("create_"):
            action = "creates"
            subject = func_name[7:].replace("_", " ")
        elif func_name.startswith("load_"):
            action = "loads"
            subject = func_name[5:].replace("_", " ")
        elif func_name.startswith("save_"):
            action = "saves"
            subject = func_name[5:].replace("_", " ")
        elif func_name.startswith("is_") or func_name.startswith("has_"):
            action = "checks if"
            subject = func_name[3:].replace("_", " ")
        else:
            action = "handles"
            subject = func_name.replace("_", " ")
        
        description = f"This function {action} the {subject}."
        
        return description
    
    def _build_method_answer(self, class_name: str, method_name: str, 
                            docstring: str, signature: str) -> str:
        """Build answer for a method question."""
        if docstring:
            description = self._clean_docstring(docstring)
        else:
            # Infer
            if method_name == "__init__":
                description = f"Initializes a new {class_name} instance."
            elif method_name.startswith("get_"):
                description = f"Returns the {method_name[4:].replace('_', ' ')}."
            elif method_name.startswith("set_"):
                description = f"Sets the {method_name[4:].replace('_', ' ')}."
            else:
                description = f"Performs {method_name.replace('_', ' ')} operation."
        
        answer = f"The {class_name}.{method_name} method: {description}"
        
        if signature:
            answer += f"\n\nSignature: {signature}"
        
        return answer
    
    def _build_usage_example(self, cls: Dict[str, Any]) -> str:
        """Build a usage example for a class."""
        class_name = cls.get("name", "Unknown")
        methods = cls.get("methods", [])
        
        # Find __init__ parameters
        init_params = []
        for method in methods:
            if method.get("name") == "__init__":
                init_params = [p.get("name") for p in method.get("parameters", [])
                              if p.get("name") not in ("self", "cls")]
                break
        
        # Build example
        if init_params:
            params_str = ", ".join(f"{p}=..." for p in init_params[:3])
            init_call = f"instance = {class_name}({params_str})"
        else:
            init_call = f"instance = {class_name}()"
        
        # Find a good method to call
        example_methods = [m.get("name") for m in methods 
                         if not m.get("name", "").startswith("_")][:2]
        
        example = f"""Here's how to use {class_name}:

```python
from enigma_engine import {class_name}

# Create instance
{init_call}
"""
        
        if example_methods:
            for method_name in example_methods:
                example += f"\n# Use the {method_name} method\nresult = instance.{method_name}()"
        
        example += "\n```"
        
        return example
    
    def _extract_subject(self, name: str) -> str:
        """Extract subject from class/function name."""
        # Split camelCase and snake_case
        words = re.findall(r'[A-Z][a-z]+|[a-z]+', name)
        
        # Remove common suffixes
        suffixes = ["Manager", "Controller", "Handler", "Generator", 
                   "Parser", "Tab", "Config", "Info", "Result"]
        words = [w for w in words if w not in suffixes]
        
        return " ".join(words).lower() if words else "functionality"
    
    def generate_variations(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Generate question variations for more robust training."""
        variations = []
        
        for pair in pairs:
            # Keep original
            variations.append(pair)
            
            # Add variations based on category
            if pair.category == "class":
                alt_questions = [
                    f"Tell me about {pair.metadata.get('class_name', 'this')}",
                    f"Describe {pair.metadata.get('class_name', 'this')}",
                ]
            elif pair.category == "function":
                alt_questions = [
                    f"What's {pair.metadata.get('func_name', 'this')} for?",
                    f"Purpose of {pair.metadata.get('func_name', 'this')}?",
                ]
            else:
                alt_questions = []
            
            for alt_q in alt_questions:
                if random.random() < self.variation_factor:
                    variations.append(TrainingPair(
                        question=alt_q,
                        answer=pair.answer,
                        category=pair.category,
                        metadata=pair.metadata,
                    ))
        
        return variations
    
    def save_to_file(self, pairs: List[TrainingPair], output_path: str,
                    append: bool = True):
        """
        Save training pairs to file.
        
        Args:
            pairs: Training pairs to save
            output_path: Path to output file
            append: Whether to append or overwrite
        """
        mode = 'a' if append else 'w'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, mode, encoding='utf-8') as f:
            if not append:
                f.write(f"# Auto-generated training data\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            
            for pair in pairs:
                f.write(pair.to_training_format())
                f.write("\n")
        
        logger.info(f"Saved {len(pairs)} training pairs to {output_path}")
    
    def save_as_json(self, pairs: List[TrainingPair], output_path: str):
        """Save training pairs as JSON for inspection."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "generated": datetime.now().isoformat(),
            "count": len(pairs),
            "pairs": [p.to_dict() for p in pairs],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(pairs)} training pairs to {output_path}")


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Data Generator")
    parser.add_argument("--analyze", help="Path to analyze")
    parser.add_argument("--output", default="training_data.txt", help="Output file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Import analyzer
    from .analyzer import CodeAnalyzer
    
    engine_path = args.analyze or str(Path(__file__).parent.parent)
    
    analyzer = CodeAnalyzer(engine_path)
    analysis = analyzer.analyze()
    
    generator = TrainingDataGenerator()
    pairs = generator.generate_from_analysis(analysis)
    pairs = generator.generate_variations(pairs)
    
    if args.json:
        generator.save_as_json(pairs, args.output)
    else:
        generator.save_to_file(pairs, args.output, append=False)
    
    print(f"Generated {len(pairs)} training pairs")


if __name__ == "__main__":
    main()
