"""
ReAct Framework Implementation

Reasoning + Acting framework for tool use.
Allows AI to reason about which tool to use, then act.

FILE: enigma_engine/core/react_framework.py
TYPE: Advanced Reasoning
MAIN CLASSES: ReActAgent, ReActStep, ReActTrace
PAPER: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al. 2022)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of ReAct steps."""
    THOUGHT = "thought"     # Reasoning step
    ACTION = "action"       # Tool invocation
    OBSERVATION = "observation"  # Tool result
    ANSWER = "answer"       # Final answer


@dataclass
class ReActStep:
    """A single step in the ReAct trace."""
    step_num: int
    step_type: StepType
    content: str
    action_name: Optional[str] = None
    action_input: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert to string format."""
        if self.step_type == StepType.THOUGHT:
            return f"Thought {self.step_num}: {self.content}"
        elif self.step_type == StepType.ACTION:
            action_str = f"Action {self.step_num}: {self.action_name}"
            if self.action_input:
                action_str += f"\nAction Input: {self.action_input}"
            return action_str
        elif self.step_type == StepType.OBSERVATION:
            return f"Observation {self.step_num}: {self.content}"
        elif self.step_type == StepType.ANSWER:
            return f"Final Answer: {self.content}"
        return f"{self.step_type.value}: {self.content}"


@dataclass
class ReActTrace:
    """Complete trace of a ReAct session."""
    task: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    
    def add_thought(self, content: str) -> ReActStep:
        """Add a thought step."""
        step = ReActStep(
            step_num=len(self.steps) + 1,
            step_type=StepType.THOUGHT,
            content=content
        )
        self.steps.append(step)
        return step
    
    def add_action(self, action_name: str, action_input: dict = None) -> ReActStep:
        """Add an action step."""
        step = ReActStep(
            step_num=len(self.steps) + 1,
            step_type=StepType.ACTION,
            content=f"Call {action_name}",
            action_name=action_name,
            action_input=action_input or {}
        )
        self.steps.append(step)
        return step
    
    def add_observation(self, content: str) -> ReActStep:
        """Add an observation step."""
        step = ReActStep(
            step_num=len(self.steps) + 1,
            step_type=StepType.OBSERVATION,
            content=content
        )
        self.steps.append(step)
        return step
    
    def set_answer(self, answer: str):
        """Set the final answer."""
        step = ReActStep(
            step_num=len(self.steps) + 1,
            step_type=StepType.ANSWER,
            content=answer
        )
        self.steps.append(step)
        self.final_answer = answer
        self.success = True
        self.finished_at = time.time()
    
    def set_error(self, error: str):
        """Set error state."""
        self.error = error
        self.success = False
        self.finished_at = time.time()
    
    def to_string(self) -> str:
        """Convert trace to string."""
        lines = [f"Task: {self.task}", ""]
        for step in self.steps:
            lines.append(step.to_string())
            lines.append("")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "steps": [
                {
                    "step_num": s.step_num,
                    "type": s.step_type.value,
                    "content": s.content,
                    "action_name": s.action_name,
                    "action_input": s.action_input
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error
        }


@dataclass
class Tool:
    """Definition of a tool the agent can use."""
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)  # name -> description
    execute: Callable[[dict], str] = None
    
    def to_string(self) -> str:
        """Convert to string for prompt."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}: {self.description}. Parameters: {params}"


class ReActAgent:
    """ReAct agent that reasons and acts to solve tasks."""
    
    REACT_PROMPT = """You are a helpful assistant that uses tools to solve tasks.
You follow the ReAct framework: Reason about what to do, then Act, then observe the result.

Available tools:
{tools}

Use the following format:
Task: the task you need to solve
Thought 1: reason about what to do first
Action 1: the action to take (tool name)
Action Input: the input to the action (JSON)
Observation 1: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought N: I now know the final answer
Final Answer: the final answer to the task

Rules:
- Always start with a Thought
- Only use the available tools
- Be concise in your reasoning
- If you cannot complete the task, say so in your Final Answer

Task: {task}
"""

    def __init__(self,
                 generator: Callable[[str], str] = None,
                 tools: list[Tool] = None,
                 max_steps: int = 10):
        """
        Initialize ReAct agent.
        
        Args:
            generator: Function to generate text. Takes prompt, returns completion.
            tools: List of available tools
            max_steps: Maximum reasoning steps
        """
        self._generator = generator or self._default_generator
        self._tools: dict[str, Tool] = {}
        self._max_steps = max_steps
        
        if tools:
            for tool in tools:
                self.add_tool(tool)
                
    def _default_generator(self, prompt: str) -> str:
        """Default generator (placeholder)."""
        return "Thought 1: I need to analyze the task.\nFinal Answer: I cannot complete this task without a language model."
    
    def add_tool(self, tool: Tool):
        """Add a tool."""
        self._tools[tool.name] = tool
        
    def remove_tool(self, name: str):
        """Remove a tool."""
        if name in self._tools:
            del self._tools[name]
            
    def _build_prompt(self, trace: ReActTrace) -> str:
        """Build prompt from current trace state."""
        tools_str = "\n".join(f"- {t.to_string()}" for t in self._tools.values())
        
        # Start with base prompt
        prompt = self.REACT_PROMPT.format(tools=tools_str, task=trace.task)
        
        # Add existing steps
        if trace.steps:
            prompt += "\n" + trace.to_string()
            
        return prompt
    
    def _parse_response(self, response: str) -> tuple[str, Optional[tuple[str, dict]]]:
        """
        Parse model response to extract thought and action.
        
        Returns:
            Tuple of (thought, (action_name, action_input) or None)
        """
        thought = None
        action = None
        action_input = {}
        final_answer = None
        
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract thought
            thought_match = re.match(r'Thought \d+:\s*(.+)', line, re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1)
                
            # Extract action
            action_match = re.match(r'Action \d+:\s*(\w+)', line, re.IGNORECASE)
            if action_match:
                action = action_match.group(1)
                
            # Extract action input
            input_match = re.match(r'Action Input:\s*(.+)', line, re.IGNORECASE)
            if input_match:
                try:
                    import json
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {"input": input_match.group(1)}
                    
            # Extract final answer
            answer_match = re.match(r'Final Answer:\s*(.+)', line, re.IGNORECASE)
            if answer_match:
                final_answer = answer_match.group(1)
        
        if final_answer:
            return thought or "", ("FINAL_ANSWER", {"answer": final_answer})
        elif action:
            return thought or "", (action, action_input)
        else:
            return thought or response, None
    
    def _execute_action(self, action_name: str, action_input: dict) -> str:
        """Execute an action and return observation."""
        if action_name == "FINAL_ANSWER":
            return action_input.get("answer", "")
            
        tool = self._tools.get(action_name)
        if not tool:
            return f"Error: Unknown tool '{action_name}'. Available tools: {list(self._tools.keys())}"
            
        if tool.execute is None:
            return f"Error: Tool '{action_name}' has no execute function."
            
        try:
            result = tool.execute(action_input)
            return str(result)
        except Exception as e:
            return f"Error executing {action_name}: {str(e)}"
    
    def run(self, task: str) -> ReActTrace:
        """
        Run the ReAct loop on a task.
        
        Args:
            task: Task description
            
        Returns:
            ReActTrace with full reasoning trace
        """
        trace = ReActTrace(task=task)
        
        for step in range(self._max_steps):
            # Build prompt and generate
            prompt = self._build_prompt(trace)
            response = self._generator(prompt)
            
            # Parse response
            thought, action_result = self._parse_response(response)
            
            # Add thought
            if thought:
                trace.add_thought(thought)
            
            # Check for action
            if action_result is None:
                # No action found, try to extract answer from thought
                if "final answer" in thought.lower():
                    trace.set_answer(thought)
                    break
                continue
                
            action_name, action_input = action_result
            
            # Check for final answer
            if action_name == "FINAL_ANSWER":
                trace.set_answer(action_input.get("answer", ""))
                break
            
            # Execute action
            trace.add_action(action_name, action_input)
            observation = self._execute_action(action_name, action_input)
            trace.add_observation(observation)
        
        # Check if we hit max steps without answer
        if not trace.final_answer:
            trace.set_error(f"Reached maximum steps ({self._max_steps}) without finding answer")
        
        return trace


# Default tools
def _search_tool(inputs: dict) -> str:
    """Placeholder search tool."""
    query = inputs.get("query", "")
    return f"Search results for '{query}': [No search implementation available]"


def _calculate_tool(inputs: dict) -> str:
    """Simple calculation tool using safe evaluation."""
    expression = inputs.get("expression", "")
    try:
        # Strict validation: only digits, operators, decimals, parens, spaces
        import re
        if not re.match(r'^[\d\s+\-*/().]+$', expression):
            return "Error: Invalid expression (only numbers and +-*/() allowed)"
        
        # Additional safety: no consecutive dots, no empty parens
        if '..' in expression or '()' in expression:
            return "Error: Invalid expression syntax"
        
        # Use compile to validate syntax before eval
        code = compile(expression, '<string>', 'eval')
        
        # Only allow numeric types in result
        result = eval(code, {"__builtins__": {}}, {})
        
        if isinstance(result, (int, float, complex)):
            return str(result)
        return "Error: Result must be numeric"
    except SyntaxError:
        return "Error: Invalid syntax"
    except Exception as e:
        return f"Error: {e}"


DEFAULT_TOOLS = [
    Tool(
        name="search",
        description="Search for information",
        parameters={"query": "The search query"},
        execute=_search_tool
    ),
    Tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={"expression": "Mathematical expression to evaluate"},
        execute=_calculate_tool
    )
]


# Integration with Enigma AI Engine
class ForgeReActAgent(ReActAgent):
    """ReAct agent integrated with Enigma AI Engine."""
    
    def __init__(self, engine=None, **kwargs):
        """
        Initialize with Enigma AI Engine engine.
        
        Args:
            engine: EnigmaEngine instance
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._engine = engine
        
        # Add default tools
        for tool in DEFAULT_TOOLS:
            self.add_tool(tool)
        
    def set_engine(self, engine):
        """Set the inference engine."""
        self._engine = engine
        
    def _generate(self, prompt: str) -> str:
        """Generate using Enigma AI Engine."""
        if self._engine is None:
            return self._default_generator(prompt)
            
        try:
            return self._engine.generate(
                prompt,
                max_tokens=500,
                temperature=0.3,  # Lower for more focused reasoning
                stop=["\nObservation"]  # Stop before observation
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {e}"


# Factory function
def create_react_agent(engine=None, 
                       tools: list[Tool] = None,
                       **kwargs) -> ForgeReActAgent:
    """
    Create a ReAct agent.
    
    Args:
        engine: EnigmaEngine for generation
        tools: Additional tools to add
        **kwargs: Additional arguments
        
    Returns:
        Configured ForgeReActAgent
    """
    agent = ForgeReActAgent(engine=engine, **kwargs)
    if tools:
        for tool in tools:
            agent.add_tool(tool)
    return agent


# Singleton
_react_agent: Optional[ForgeReActAgent] = None


def get_react_agent(engine=None) -> ForgeReActAgent:
    """Get the ReAct agent singleton."""
    global _react_agent
    if _react_agent is None:
        _react_agent = ForgeReActAgent(engine)
    elif engine:
        _react_agent.set_engine(engine)
    return _react_agent


__all__ = [
    'ReActAgent',
    'ReActStep',
    'ReActTrace',
    'Tool',
    'StepType',
    'ForgeReActAgent',
    'create_react_agent',
    'get_react_agent',
    'DEFAULT_TOOLS'
]
