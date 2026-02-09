"""
Agent Templates for Enigma AI Engine

Pre-built agent configurations for common use cases.

Available templates:
- Researcher: Web search, summarization, fact-checking
- Coder: Code generation, review, debugging
- Critic: Analysis, feedback, improvement suggestions
- Writer: Creative writing, documentation, editing
- Assistant: General purpose, task management
- Tutor: Teaching, explanations, quizzes
- Analyst: Data analysis, visualization recommendations
- DevOps: CI/CD, deployment, infrastructure

Usage:
    from enigma_engine.agents.templates import AgentTemplates, create_agent
    
    # Get a template
    template = AgentTemplates.CODER
    
    # Create agent from template
    agent = create_agent(template)
    
    # Or customize
    agent = create_agent(
        template,
        temperature=0.7,
        tools=['web_search', 'code_execution']
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Available agent capabilities."""
    WEB_SEARCH = auto()
    CODE_EXECUTION = auto()
    FILE_ACCESS = auto()
    IMAGE_GENERATION = auto()
    IMAGE_ANALYSIS = auto()
    MEMORY_ACCESS = auto()
    TOOL_USE = auto()
    PLANNING = auto()
    REFLECTION = auto()
    MULTI_TURN = auto()


class AgentPersonality(Enum):
    """Agent personality types."""
    PROFESSIONAL = auto()
    FRIENDLY = auto()
    CONCISE = auto()
    VERBOSE = auto()
    CREATIVE = auto()
    ANALYTICAL = auto()
    SOCRATIC = auto()  # Asks guiding questions


@dataclass
class AgentTemplate:
    """Configuration template for an agent."""
    # Basic info
    name: str
    description: str
    
    # System prompt
    system_prompt: str
    
    # Model settings
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    
    # Tools this agent should use
    tools: List[str] = field(default_factory=list)
    
    # Personality
    personality: AgentPersonality = AgentPersonality.PROFESSIONAL
    
    # Behavior
    max_iterations: int = 10
    auto_plan: bool = False
    self_reflect: bool = False
    
    # Response format
    structured_output: bool = False
    output_format: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"


# Built-in templates
RESEARCHER_TEMPLATE = AgentTemplate(
    name="Researcher",
    description="Research assistant that searches, summarizes, and synthesizes information",
    system_prompt="""You are a thorough research assistant. Your job is to:
1. Search for relevant information on topics
2. Summarize findings clearly and concisely
3. Cross-reference multiple sources
4. Identify gaps in available information
5. Present findings with citations

Always verify information from multiple sources when possible.
Clearly distinguish between facts and interpretations.
If you're uncertain about something, say so.""",
    temperature=0.3,
    capabilities=[
        AgentCapability.WEB_SEARCH,
        AgentCapability.MEMORY_ACCESS,
        AgentCapability.PLANNING,
    ],
    tools=['web_search', 'summarize', 'extract_info'],
    personality=AgentPersonality.ANALYTICAL,
    auto_plan=True,
    tags=['research', 'search', 'information']
)

CODER_TEMPLATE = AgentTemplate(
    name="Coder",
    description="Expert programmer for code generation, review, and debugging",
    system_prompt="""You are an expert software developer. Your job is to:
1. Write clean, efficient, well-documented code
2. Review code for bugs, security issues, and improvements
3. Debug problems systematically
4. Explain technical concepts clearly
5. Follow best practices and conventions

Always consider:
- Edge cases and error handling
- Performance implications
- Security best practices
- Code readability and maintainability

When showing code, use proper formatting and include comments.""",
    temperature=0.2,
    max_tokens=4096,
    capabilities=[
        AgentCapability.CODE_EXECUTION,
        AgentCapability.FILE_ACCESS,
        AgentCapability.REFLECTION,
    ],
    tools=['code_execute', 'file_read', 'file_write', 'lint'],
    personality=AgentPersonality.CONCISE,
    self_reflect=True,
    tags=['coding', 'programming', 'development']
)

CRITIC_TEMPLATE = AgentTemplate(
    name="Critic",
    description="Analytical critic that provides constructive feedback",
    system_prompt="""You are a thoughtful critic. Your job is to:
1. Analyze work objectively
2. Identify strengths and weaknesses
3. Provide specific, actionable feedback
4. Suggest concrete improvements
5. Be constructive, not destructive

When critiquing:
- Start with what works well
- Be specific about issues
- Explain why something is a problem
- Offer alternatives or solutions
- Consider the context and constraints

Balance honesty with helpfulness.""",
    temperature=0.4,
    capabilities=[
        AgentCapability.REFLECTION,
        AgentCapability.PLANNING,
    ],
    tools=['analyze', 'compare'],
    personality=AgentPersonality.ANALYTICAL,
    self_reflect=True,
    tags=['critique', 'review', 'feedback']
)

WRITER_TEMPLATE = AgentTemplate(
    name="Writer",
    description="Creative writer for content, documentation, and editing",
    system_prompt="""You are a skilled writer. Your job is to:
1. Create engaging, well-structured content
2. Adapt tone and style to the audience
3. Edit and improve existing text
4. Maintain consistent voice throughout
5. Use clear, precise language

Writing principles:
- Show, don't tell
- Vary sentence structure
- Use active voice when possible
- Eliminate unnecessary words
- Structure content logically

Match the formality level to the context.""",
    temperature=0.8,
    max_tokens=4096,
    capabilities=[
        AgentCapability.MEMORY_ACCESS,
        AgentCapability.REFLECTION,
    ],
    tools=['thesaurus', 'grammar_check', 'readability'],
    personality=AgentPersonality.CREATIVE,
    tags=['writing', 'content', 'editing', 'creative']
)

ASSISTANT_TEMPLATE = AgentTemplate(
    name="Assistant",
    description="General-purpose helpful assistant",
    system_prompt="""You are a helpful assistant. Your job is to:
1. Understand and clarify requests
2. Provide accurate, relevant information
3. Complete tasks efficiently
4. Ask clarifying questions when needed
5. Adapt to the user's preferences

Be helpful, harmless, and honest.
Focus on being useful rather than impressive.
Admit when you don't know something.""",
    temperature=0.7,
    capabilities=[
        AgentCapability.TOOL_USE,
        AgentCapability.MEMORY_ACCESS,
        AgentCapability.MULTI_TURN,
    ],
    tools=['search', 'calculate', 'reminder'],
    personality=AgentPersonality.FRIENDLY,
    tags=['general', 'assistant', 'help']
)

TUTOR_TEMPLATE = AgentTemplate(
    name="Tutor",
    description="Educational tutor that teaches and explains concepts",
    system_prompt="""You are a patient, effective tutor. Your job is to:
1. Explain concepts at the appropriate level
2. Use examples and analogies
3. Check understanding frequently
4. Adapt to the student's learning style
5. Encourage and motivate

Teaching approach:
- Break complex topics into steps
- Build on existing knowledge
- Use Socratic questioning
- Provide practice opportunities
- Celebrate progress

Never make students feel bad for not knowing something.""",
    temperature=0.6,
    capabilities=[
        AgentCapability.MEMORY_ACCESS,
        AgentCapability.MULTI_TURN,
        AgentCapability.PLANNING,
    ],
    tools=['quiz', 'explain', 'example'],
    personality=AgentPersonality.SOCRATIC,
    auto_plan=True,
    tags=['education', 'teaching', 'learning', 'tutor']
)

ANALYST_TEMPLATE = AgentTemplate(
    name="Analyst",
    description="Data analyst for analysis, visualization, and insights",
    system_prompt="""You are a data analyst. Your job is to:
1. Analyze data systematically
2. Identify patterns and trends
3. Generate actionable insights
4. Recommend appropriate visualizations
5. Communicate findings clearly

Analysis approach:
- Start with exploratory analysis
- Check data quality first
- Use appropriate statistical methods
- Consider confounding factors
- Present uncertainty honestly

Always explain your methodology.""",
    temperature=0.3,
    capabilities=[
        AgentCapability.CODE_EXECUTION,
        AgentCapability.FILE_ACCESS,
        AgentCapability.PLANNING,
    ],
    tools=['data_analyze', 'visualize', 'statistics'],
    personality=AgentPersonality.ANALYTICAL,
    structured_output=True,
    tags=['data', 'analysis', 'statistics', 'visualization']
)

DEVOPS_TEMPLATE = AgentTemplate(
    name="DevOps",
    description="DevOps engineer for CI/CD, deployment, and infrastructure",
    system_prompt="""You are a DevOps engineer. Your job is to:
1. Design and maintain CI/CD pipelines
2. Manage infrastructure as code
3. Ensure system reliability
4. Automate repetitive tasks
5. Monitor and respond to incidents

Principles:
- Automate everything possible
- Make systems observable
- Plan for failure
- Document thoroughly
- Security is not optional

Always consider: cost, scalability, security, and maintainability.""",
    temperature=0.2,
    capabilities=[
        AgentCapability.CODE_EXECUTION,
        AgentCapability.FILE_ACCESS,
        AgentCapability.TOOL_USE,
    ],
    tools=['shell', 'docker', 'kubernetes', 'terraform'],
    personality=AgentPersonality.CONCISE,
    tags=['devops', 'infrastructure', 'automation', 'deployment']
)


class AgentTemplates(Enum):
    """Available agent templates."""
    RESEARCHER = RESEARCHER_TEMPLATE
    CODER = CODER_TEMPLATE
    CRITIC = CRITIC_TEMPLATE
    WRITER = WRITER_TEMPLATE
    ASSISTANT = ASSISTANT_TEMPLATE
    TUTOR = TUTOR_TEMPLATE
    ANALYST = ANALYST_TEMPLATE
    DEVOPS = DEVOPS_TEMPLATE


# Template registry for custom templates
_custom_templates: Dict[str, AgentTemplate] = {}


def register_template(name: str, template: AgentTemplate):
    """Register a custom agent template."""
    _custom_templates[name.lower()] = template
    logger.info(f"Registered custom template: {name}")


def get_template(name: str) -> Optional[AgentTemplate]:
    """
    Get a template by name.
    
    Args:
        name: Template name (case-insensitive)
        
    Returns:
        AgentTemplate or None if not found
    """
    # Check built-in templates
    name_upper = name.upper()
    if hasattr(AgentTemplates, name_upper):
        return getattr(AgentTemplates, name_upper).value
    
    # Check custom templates
    if name.lower() in _custom_templates:
        return _custom_templates[name.lower()]
    
    return None


def list_templates() -> List[str]:
    """Get list of all available template names."""
    builtin = [t.name for t in AgentTemplates]
    custom = list(_custom_templates.keys())
    return builtin + custom


def create_agent(
    template: AgentTemplate | str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[str]] = None,
    extra_system_prompt: Optional[str] = None
) -> 'Agent':
    """
    Create an agent from a template.
    
    Args:
        template: AgentTemplate or template name
        model: Override model to use
        temperature: Override temperature
        max_tokens: Override max tokens
        tools: Override or extend tools
        extra_system_prompt: Additional system prompt text
        
    Returns:
        Configured Agent instance
    """
    # Get template
    if isinstance(template, str):
        template = get_template(template)
        if template is None:
            raise ValueError(f"Unknown template: {template}")
    elif isinstance(template, AgentTemplates):
        template = template.value
    
    # Build configuration
    config = {
        'name': template.name,
        'system_prompt': template.system_prompt,
        'temperature': temperature if temperature is not None else template.temperature,
        'max_tokens': max_tokens if max_tokens is not None else template.max_tokens,
        'top_p': template.top_p,
        'tools': tools if tools is not None else template.tools,
        'max_iterations': template.max_iterations,
        'auto_plan': template.auto_plan,
        'self_reflect': template.self_reflect,
    }
    
    if extra_system_prompt:
        config['system_prompt'] += f"\n\n{extra_system_prompt}"
    
    if model:
        config['model'] = model
    
    # Create agent
    return Agent(**config)


@dataclass
class Agent:
    """
    Configured agent instance.
    
    This is a simple implementation - can be extended with
    actual inference capabilities.
    """
    name: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    auto_plan: bool = False
    self_reflect: bool = False
    model: Optional[str] = None
    
    _history: List[Dict[str, str]] = field(default_factory=list, repr=False)
    
    def chat(self, message: str) -> str:
        """
        Chat with the agent.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        self._history.append({'role': 'user', 'content': message})
        
        try:
            # Try to use inference engine
            from enigma_engine.core.inference import get_engine
            engine = get_engine()
            
            # Build messages
            messages = [
                {'role': 'system', 'content': self.system_prompt}
            ] + self._history
            
            response = engine.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            
            self._history.append({'role': 'assistant', 'content': response})
            return response
            
        except ImportError:
            # Fallback message
            response = f"[{self.name} agent would respond here]"
            self._history.append({'role': 'assistant', 'content': response})
            return response
    
    def reset(self):
        """Clear conversation history."""
        self._history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return list(self._history)
    
    def run_task(self, task: str) -> str:
        """
        Run a task with the agent.
        
        For auto_plan agents, this will create a plan first.
        
        Args:
            task: Task description
            
        Returns:
            Task result
        """
        if self.auto_plan:
            # Ask for a plan first
            plan_response = self.chat(f"Create a step-by-step plan to: {task}")
            
            # Execute the plan
            return self.chat("Now execute this plan step by step.")
        else:
            return self.chat(task)


# Quick access functions
def researcher(query: str) -> str:
    """Quick researcher agent."""
    agent = create_agent(AgentTemplates.RESEARCHER)
    return agent.run_task(query)


def code_review(code: str) -> str:
    """Quick code review."""
    agent = create_agent(AgentTemplates.CODER)
    return agent.chat(f"Please review this code:\n\n```\n{code}\n```")


def critique(work: str) -> str:
    """Quick critique."""
    agent = create_agent(AgentTemplates.CRITIC) 
    return agent.chat(f"Please provide constructive feedback on:\n\n{work}")


def explain(topic: str, level: str = "beginner") -> str:
    """Quick explanation."""
    agent = create_agent(AgentTemplates.TUTOR)
    return agent.chat(f"Explain {topic} at a {level} level.")
