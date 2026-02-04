"""
Universal Action System for ForgeAI.

Allows the AI to perform ANY action dynamically, not just predefined tools.
This makes ForgeAI truly limitless - it can adapt to any user request.

The system includes:
- Dynamic tool creation from natural language
- Universal command execution (with safety)
- Learning new actions from demonstrations
- Composing complex actions from primitives
- Fallback to code generation for novel tasks

Usage:
    from forge_ai.tools.universal_action import UniversalAction, ActionPlanner
    
    action = UniversalAction()
    result = action.do("organize my desktop by file type")
    result = action.do("create a python script that monitors CPU usage")
    result = action.do("make the avatar wave while saying hello")
"""

import os
import re
import json
import logging
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Safety: Blocked commands and patterns
BLOCKED_COMMANDS = {
    'rm -rf /', 'rm -rf /*', 'mkfs', 'dd if=/dev/zero',
    ':(){:|:&};:', 'chmod -R 777 /', 'mv /* /dev/null',
}
BLOCKED_PATTERNS = [
    r'rm\s+-rf\s+/', r'sudo\s+rm', r'>\s*/dev/sd',
    r'chmod\s+777\s+/', r'curl.*\|\s*bash', r'wget.*\|\s*sh',
]


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    action_type: str = "unknown"
    steps_taken: List[str] = field(default_factory=list)
    time_taken: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "action_type": self.action_type,
            "steps": self.steps_taken,
            "time_taken": self.time_taken,
        }


@dataclass  
class LearnedAction:
    """An action learned from demonstration or definition."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    use_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "created_at": self.created_at,
            "use_count": self.use_count,
        }


class ActionPlanner:
    """Plans complex actions by breaking them into steps."""
    
    def __init__(self, inference_engine=None):
        self.inference = inference_engine
        self._primitives = self._register_primitives()
    
    def _register_primitives(self) -> Dict[str, Callable]:
        """Register primitive actions that can be composed."""
        return {
            # File operations
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "create_dir": self._create_dir,
            "move_file": self._move_file,
            "copy_file": self._copy_file,
            "delete_file": self._delete_file,
            
            # System operations
            "run_command": self._run_command,
            "get_env": self._get_env,
            "set_env": self._set_env,
            
            # Communication
            "http_get": self._http_get,
            "http_post": self._http_post,
            
            # UI/Output
            "notify": self._notify,
            "speak": self._speak,
            "display": self._display,
            
            # Code execution
            "run_python": self._run_python,
            "run_javascript": self._run_javascript,
            
            # Wait/timing
            "wait": self._wait,
            "schedule": self._schedule,
        }
    
    def plan(self, request: str) -> List[Dict[str, Any]]:
        """
        Create an action plan for a natural language request.
        
        Returns list of steps, each with:
        - action: primitive action name
        - params: parameters for the action
        - condition: optional condition for execution
        """
        # Try to match known patterns first
        plan = self._pattern_match(request)
        if plan:
            return plan
        
        # Use AI to generate plan if available
        if self.inference:
            return self._ai_plan(request)
        
        # Fallback: single command execution
        return [{"action": "run_command", "params": {"command": request}}]
    
    def _pattern_match(self, request: str) -> Optional[List[Dict]]:
        """Match common request patterns."""
        request_lower = request.lower()
        
        # File organization
        if "organize" in request_lower and ("file" in request_lower or "folder" in request_lower):
            path = self._extract_path(request) or os.path.expanduser("~/Desktop")
            return [
                {"action": "list_dir", "params": {"path": path}, "store": "files"},
                {"action": "run_python", "params": {
                    "code": f"""
import os, shutil
from pathlib import Path
path = Path('{path}')
for f in path.iterdir():
    if f.is_file():
        ext = f.suffix[1:] or 'no_extension'
        dest = path / ext
        dest.mkdir(exist_ok=True)
        shutil.move(str(f), str(dest / f.name))
print(f"Organized files in {{path}}")
"""
                }},
            ]
        
        # Create file/script
        if re.search(r"create\s+(a\s+)?(python|script|file)", request_lower):
            filename = self._extract_filename(request) or "script.py"
            purpose = request.replace("create", "").replace("python", "").replace("script", "").strip()
            return [
                {"action": "generate_code", "params": {"purpose": purpose, "language": "python"}, "store": "code"},
                {"action": "write_file", "params": {"path": filename, "content": "{code}"}},
                {"action": "notify", "params": {"message": f"Created {filename}"}},
            ]
        
        # Avatar actions
        if "avatar" in request_lower:
            return self._plan_avatar_action(request)
        
        # Open application
        if re.search(r"open\s+\w+", request_lower):
            app = re.search(r"open\s+(\w+)", request_lower).group(1)
            return [{"action": "run_command", "params": {"command": f"xdg-open {app} || open {app} || start {app}"}}]
        
        return None
    
    def _plan_avatar_action(self, request: str) -> List[Dict]:
        """Plan avatar-related actions."""
        steps = []
        request_lower = request.lower()
        
        # Parse simultaneous actions
        if "while" in request_lower or "and" in request_lower:
            parts = re.split(r'\s+(?:while|and)\s+', request_lower)
            for part in parts:
                if any(w in part for w in ["wave", "gesture", "nod", "shake"]):
                    gesture = next((w for w in ["wave", "nod", "shake", "bow"] if w in part), "wave")
                    steps.append({"action": "avatar_gesture", "params": {"gesture": gesture}})
                elif any(w in part for w in ["say", "speak", "tell"]):
                    text = re.sub(r"^(say|speak|tell)\s*", "", part).strip()
                    steps.append({"action": "speak", "params": {"text": text}})
        else:
            if "wave" in request_lower:
                steps.append({"action": "avatar_gesture", "params": {"gesture": "wave"}})
            if "speak" in request_lower or "say" in request_lower:
                text = re.sub(r".*(?:say|speak)\s*", "", request_lower).strip()
                steps.append({"action": "speak", "params": {"text": text}})
        
        return steps or [{"action": "avatar_respond", "params": {"request": request}}]
    
    def _ai_plan(self, request: str) -> List[Dict]:
        """Use AI to generate action plan."""
        prompt = f"""Break down this request into primitive actions.
Available primitives: {list(self._primitives.keys())}

Request: {request}

Respond with JSON array of steps, each with "action" and "params"."""
        
        try:
            response = self.inference.generate(prompt, max_tokens=500)
            # Extract JSON from response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"AI planning failed: {e}")
        
        return [{"action": "interpret", "params": {"request": request}}]
    
    def _extract_path(self, text: str) -> Optional[str]:
        """Extract file path from text."""
        patterns = [
            r'["\']([^"\']+)["\']',
            r'(?:in|from|at)\s+(\S+)',
            r'(~?/[\w/.-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return os.path.expanduser(match.group(1))
        return None
    
    def _extract_filename(self, text: str) -> Optional[str]:
        """Extract filename from text."""
        match = re.search(r'(\w+\.\w+)', text)
        return match.group(1) if match else None
    
    # Primitive implementations
    def _read_file(self, path: str) -> str:
        return Path(path).read_text()
    
    def _write_file(self, path: str, content: str) -> bool:
        Path(path).write_text(content)
        return True
    
    def _list_dir(self, path: str) -> List[str]:
        return [str(p) for p in Path(path).iterdir()]
    
    def _create_dir(self, path: str) -> bool:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    
    def _move_file(self, src: str, dst: str) -> bool:
        import shutil
        shutil.move(src, dst)
        return True
    
    def _copy_file(self, src: str, dst: str) -> bool:
        import shutil
        shutil.copy2(src, dst)
        return True
    
    def _delete_file(self, path: str) -> bool:
        Path(path).unlink()
        return True
    
    def _run_command(self, command: str, timeout: int = 30) -> str:
        # Safety check
        if not self._is_safe_command(command):
            return "Command blocked for safety"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    
    def _is_safe_command(self, command: str) -> bool:
        if command in BLOCKED_COMMANDS:
            return False
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        return True
    
    def _get_env(self, name: str) -> Optional[str]:
        return os.environ.get(name)
    
    def _set_env(self, name: str, value: str) -> bool:
        os.environ[name] = value
        return True
    
    def _http_get(self, url: str) -> str:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode()
    
    def _http_post(self, url: str, data: dict) -> str:
        import urllib.request
        req = urllib.request.Request(url, json.dumps(data).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()
    
    def _notify(self, message: str, title: str = "ForgeAI") -> bool:
        try:
            from forge_ai.utils.notifications import notify
            notify(message, title)
        except ImportError:
            logger.info(f"Notification: {title} - {message}")
        return True
    
    def _speak(self, text: str) -> bool:
        try:
            from forge_ai.voice.voice_generator import speak
            speak(text)
        except ImportError:
            logger.info(f"Speak: {text}")
        return True
    
    def _display(self, content: Any) -> bool:
        print(content)
        return True
    
    def _run_python(self, code: str) -> Any:
        exec_globals = {}
        exec(code, exec_globals)
        return exec_globals.get('result', True)
    
    def _run_javascript(self, code: str) -> str:
        # Use Node.js if available
        result = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=30)
        return result.stdout
    
    def _wait(self, seconds: float) -> bool:
        import time
        time.sleep(seconds)
        return True
    
    def _schedule(self, action: str, delay_seconds: float) -> bool:
        def delayed():
            import time
            time.sleep(delay_seconds)
            self._primitives.get(action, lambda: None)()
        threading.Thread(target=delayed, daemon=True).start()
        return True


class UniversalAction:
    """
    Execute any action requested in natural language.
    
    This is the core of ForgeAI's limitless capability.
    """
    
    def __init__(self, inference_engine=None, allow_learning: bool = True):
        self.planner = ActionPlanner(inference_engine)
        self.inference = inference_engine
        self.allow_learning = allow_learning
        
        # Learned actions
        self._learned_actions: Dict[str, LearnedAction] = {}
        self._actions_file = Path.home() / ".forge_ai" / "learned_actions.json"
        self._load_learned_actions()
        
        # Execution context
        self._context: Dict[str, Any] = {}
    
    def do(self, request: str) -> ActionResult:
        """
        Execute any action from natural language.
        
        Args:
            request: What you want ForgeAI to do
        
        Returns:
            ActionResult with success/failure and results
        """
        import time
        start_time = time.time()
        steps_taken = []
        
        try:
            # Check for learned action
            learned = self._find_learned_action(request)
            if learned:
                learned.use_count += 1
                plan = learned.steps
                action_type = "learned"
            else:
                # Generate plan
                plan = self.planner.plan(request)
                action_type = "planned"
            
            # Execute plan
            result = None
            for step in plan:
                action = step.get("action")
                params = step.get("params", {})
                
                # Substitute variables from context
                params = self._substitute_vars(params)
                
                steps_taken.append(f"{action}: {params}")
                
                # Execute primitive
                if action in self.planner._primitives:
                    result = self.planner._primitives[action](**params)
                elif action == "generate_code":
                    result = self._generate_code(**params)
                elif action == "avatar_gesture":
                    result = self._avatar_gesture(**params)
                elif action == "avatar_respond":
                    result = self._avatar_respond(**params)
                elif action == "interpret":
                    result = self._interpret(**params)
                else:
                    logger.warning(f"Unknown action: {action}")
                    continue
                
                # Store result if requested
                if "store" in step:
                    self._context[step["store"]] = result
            
            return ActionResult(
                success=True,
                result=result,
                action_type=action_type,
                steps_taken=steps_taken,
                time_taken=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"Action failed: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                action_type="failed",
                steps_taken=steps_taken,
                time_taken=time.time() - start_time,
            )
    
    def learn(self, name: str, description: str, steps: List[Dict]) -> bool:
        """
        Learn a new action from steps.
        
        Args:
            name: Action name for later recall
            description: Natural language description
            steps: List of action steps
        """
        if not self.allow_learning:
            return False
        
        self._learned_actions[name] = LearnedAction(
            name=name,
            description=description,
            steps=steps,
        )
        self._save_learned_actions()
        return True
    
    def teach(self, name: str, description: str, demonstration: Callable) -> bool:
        """
        Learn an action by observing a demonstration.
        
        The demonstration function is executed while recording
        what primitives are called.
        """
        # Record primitive calls during demonstration
        recorded_steps = []
        original_primitives = self.planner._primitives.copy()
        
        def make_recorder(action_name, original_func):
            def recorder(*args, **kwargs):
                recorded_steps.append({
                    "action": action_name,
                    "params": kwargs if kwargs else {"args": args}
                })
                return original_func(*args, **kwargs)
            return recorder
        
        # Wrap primitives with recorders
        for name_p, func in original_primitives.items():
            self.planner._primitives[name_p] = make_recorder(name_p, func)
        
        try:
            demonstration()
        finally:
            self.planner._primitives = original_primitives
        
        if recorded_steps:
            return self.learn(name, description, recorded_steps)
        return False
    
    def forget(self, name: str) -> bool:
        """Remove a learned action."""
        if name in self._learned_actions:
            del self._learned_actions[name]
            self._save_learned_actions()
            return True
        return False
    
    def list_learned(self) -> List[Dict]:
        """List all learned actions."""
        return [a.to_dict() for a in self._learned_actions.values()]
    
    def _find_learned_action(self, request: str) -> Optional[LearnedAction]:
        """Find a learned action matching the request."""
        request_lower = request.lower()
        
        # Exact name match
        for name, action in self._learned_actions.items():
            if name.lower() in request_lower:
                return action
        
        # Description similarity (simple keyword matching)
        for action in self._learned_actions.values():
            desc_words = set(action.description.lower().split())
            req_words = set(request_lower.split())
            overlap = len(desc_words & req_words)
            if overlap >= 2:
                return action
        
        return None
    
    def _substitute_vars(self, params: Dict) -> Dict:
        """Substitute {var} placeholders with context values."""
        result = {}
        for key, value in params.items():
            if isinstance(value, str) and "{" in value:
                for ctx_key, ctx_val in self._context.items():
                    value = value.replace(f"{{{ctx_key}}}", str(ctx_val))
            result[key] = value
        return result
    
    def _generate_code(self, purpose: str, language: str = "python") -> str:
        """Generate code for a purpose."""
        if self.inference:
            prompt = f"Write {language} code to: {purpose}\nOnly output the code, no explanation."
            return self.inference.generate(prompt, max_tokens=1000)
        return f"# TODO: {purpose}\npass"
    
    def _avatar_gesture(self, gesture: str) -> bool:
        """Execute avatar gesture."""
        try:
            from forge_ai.avatar.controller import get_controller
            controller = get_controller()
            if controller:
                controller.play_gesture(gesture)
                return True
        except ImportError:
            pass
        logger.info(f"Avatar gesture: {gesture}")
        return True
    
    def _avatar_respond(self, request: str) -> bool:
        """Have avatar respond to request."""
        try:
            from forge_ai.avatar.autonomous import get_autonomous
            auto = get_autonomous()
            if auto:
                auto.process_request(request)
                return True
        except ImportError:
            pass
        return False
    
    def _interpret(self, request: str) -> str:
        """Interpret and execute a free-form request."""
        if self.inference:
            # Ask AI how to accomplish this
            prompt = f"""User wants: {request}

How should I accomplish this? Respond with either:
1. A shell command to run
2. Python code to execute
3. A description of manual steps needed

Be concise and actionable."""
            response = self.inference.generate(prompt, max_tokens=300)
            
            # Try to execute if it looks like code/command
            if response.strip().startswith("```"):
                code = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
                if code:
                    try:
                        return self.planner._run_python(code.group(1))
                    except Exception:
                        return self.planner._run_command(code.group(1))
            
            return response
        
        return f"Cannot interpret request without AI: {request}"
    
    def _load_learned_actions(self) -> None:
        """Load learned actions from file."""
        if self._actions_file.exists():
            try:
                data = json.loads(self._actions_file.read_text())
                for name, action_data in data.items():
                    self._learned_actions[name] = LearnedAction(**action_data)
            except Exception as e:
                logger.warning(f"Failed to load learned actions: {e}")
    
    def _save_learned_actions(self) -> None:
        """Save learned actions to file."""
        self._actions_file.parent.mkdir(parents=True, exist_ok=True)
        data = {name: action.to_dict() for name, action in self._learned_actions.items()}
        self._actions_file.write_text(json.dumps(data, indent=2))


# Global instance
_universal_action: Optional[UniversalAction] = None


def get_universal_action(inference_engine=None) -> UniversalAction:
    """Get the universal action handler."""
    global _universal_action
    if _universal_action is None:
        _universal_action = UniversalAction(inference_engine)
    return _universal_action


def do(request: str) -> ActionResult:
    """Convenience function to do anything."""
    return get_universal_action().do(request)
