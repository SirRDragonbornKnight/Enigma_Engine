"""Tests for command system, sanitization, code sandbox, and blocked paths."""
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestCommandSystem:
    """Test the command processing system."""

    def test_parse_commands(self):
        from enigma_engine.core.commands import parse_commands
        text = "Hello [CMD]system.info[/CMD] world"
        clean, commands = parse_commands(text)
        assert len(commands) == 1
        assert commands[0] == "system.info"
        assert "[CMD]" not in clean
        assert "Hello" in clean and "world" in clean

    def test_parse_multiple_commands(self):
        from enigma_engine.core.commands import parse_commands
        text = "[CMD]gui.tab.switch chat[/CMD] and [CMD]system.info[/CMD]"
        _, commands = parse_commands(text)
        assert len(commands) == 2

    def test_registry(self):
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        available = registry.list_commands()
        assert len(available) > 0


class TestCodeSandbox:
    """Verify code.run command registration and safety checks."""

    def test_code_run_registered(self):
        """code.run command should be registered in builtin commands."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        assert "code.run" in reg._commands

    def test_code_run_executes_simple_code(self):
        """code.run should execute simple Python and capture output."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute('code.run print("hello world")')
        assert result.success
        assert "hello world" in result.message

    def test_code_run_blocks_forbidden_ops(self):
        """code.run should block dangerous operations."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run os.remove('important.txt')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_blocks_dunder_import(self):
        """code.run should block __import__ bypass attempts."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run __import__('os').system('whoami')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_blocks_importlib(self):
        """code.run should block importlib-based imports."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run importlib.import_module('os').system('ls')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_blocks_open_write_outside_outputs(self):
        """code.run should block open() for writing outside outputs/."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run open('/etc/passwd', 'w').write('bad')")
        assert not result.success or "restricted" in result.message.lower() or "PermissionError" in result.message

    def test_code_run_blocks_compile_exec(self):
        """code.run should block compile() bypass."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run x = compile('1+1', '', 'eval')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_empty_args(self):
        """code.run with no code should return error."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run")
        assert not result.success

    def test_code_run_blocks_os_system(self):
        """code.run should block os.system() calls (S740)."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run os.system('whoami')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_blocks_os_popen(self):
        """code.run should block os.popen() calls (S740)."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run os.popen('ls').read()")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_safe_open_path_traversal(self):
        """_safe_open should block writes to dirs starting with 'outputs' but outside it (S739)."""
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        # Verify the sandbox uses relative_to() instead of startswith()
        assert "relative_to(outputs)" in source
        assert "startswith(str(outputs))" not in source

    def test_execute_preserves_raw_code_for_code_run(self):
        """CommandRegistry.execute should pass raw code as one arg for code.run."""
        from enigma_engine.core.commands import CommandRegistry, CommandResult

        reg = CommandRegistry()
        captured: list[str] = []

        def capture(args, _ctx):
            captured.extend(args)
            return CommandResult(True, "[OK]")

        reg.register("code.run", capture, "capture", "code.run <python_code>")
        reg.execute("code.run ```python\nprint('a;b|c')\n```")

        assert len(captured) == 1
        assert "```python" in captured[0]
        assert "a;b|c" in captured[0]

    def test_code_run_accepts_fenced_multiline_code(self):
        """code.run should execute fenced multiline Python without syntax mangling."""
        from enigma_engine.core.commands import get_registry

        reg = get_registry()
        cmd = (
            "code.run ```python\n"
            "print('hello from fenced code')\n"
            "```"
        )
        result = reg.execute(cmd)

        assert result.success
        assert "hello from fenced code" in result.message


class TestConfigSetTypeConversion:
    """Tests for config.set robust type conversion (int→float→string)."""

    def _run_config_set(self, key, value):
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "config.set")
        ctx = {"config": {}}
        result = cmd.handler([key, value], ctx)
        return result, ctx["config"].get(key)

    def test_set_integer(self):
        """config.set should convert '42' to int."""
        result, val = self._run_config_set("key", "42")
        assert result.success
        assert val == 42
        assert isinstance(val, int)

    def test_set_float(self):
        """config.set should convert '3.14' to float."""
        result, val = self._run_config_set("key", "3.14")
        assert result.success
        assert isinstance(val, float)
        assert abs(val - 3.14) < 0.001

    def test_set_multi_dot_string(self):
        """config.set should keep '1.2.3' as string — no crash."""
        result, val = self._run_config_set("key", "1.2.3")
        assert result.success
        assert val == "1.2.3"
        assert isinstance(val, str)

    def test_set_bool_true(self):
        """config.set should convert 'true' to bool True."""
        result, val = self._run_config_set("key", "true")
        assert result.success
        assert val is True

    def test_set_bool_false(self):
        """config.set should convert 'FALSE' to bool False."""
        result, val = self._run_config_set("key", "FALSE")
        assert result.success
        assert val is False

    def test_set_plain_string(self):
        """config.set should keep 'hello' as string."""
        result, val = self._run_config_set("key", "hello")
        assert result.success
        assert val == "hello"

    def test_set_negative_int(self):
        """config.set should handle '-5'."""
        result, val = self._run_config_set("key", "-5")
        assert result.success
        assert val == -5

    def test_set_missing_args(self):
        """config.set with <2 args returns error."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "config.set")
        result = cmd.handler(["only_key"], {})
        assert not result.success


class TestCommandSanitization:
    """Verify shell metacharacter sanitization in command args."""

    def test_sanitize_args_removes_metacharacters(self):
        """sanitize_args strips ; | & ` { } from args."""
        from enigma_engine.core.commands import sanitize_args
        dirty = ["hello;world", "foo|bar", "a&b", "`rm -rf`", "{x}", "clean"]
        result = sanitize_args(dirty)
        assert result == ["helloworld", "foobar", "ab", "rm -rf", "x", "clean"]

    def test_sanitize_args_preserves_clean(self):
        """sanitize_args passes through clean args unchanged."""
        from enigma_engine.core.commands import sanitize_args
        clean = ["hello", "world", "foo.bar", "path/to/file", "arg-with-dashes"]
        assert sanitize_args(clean) == clean

    def test_sanitize_args_empty_list(self):
        """sanitize_args handles empty input."""
        from enigma_engine.core.commands import sanitize_args
        assert sanitize_args([]) == []

    def test_execute_sanitizes_before_handler(self):
        """CommandRegistry.execute() sanitizes args before passing to handler."""
        from enigma_engine.core.commands import CommandRegistry, CommandResult
        reg = CommandRegistry()
        received_args = []
        def capture_handler(args, ctx):
            received_args.extend(args)
            return CommandResult(True, "[OK]")
        reg.register("test.cap", capture_handler, "capture", "test.cap")
        reg.execute("test.cap hello;world foo|bar")
        assert received_args == ["helloworld", "foobar"]

    def test_shell_metacharacters_constant(self):
        """SHELL_METACHARACTERS is a frozenset with expected chars."""
        from enigma_engine.core.commands import SHELL_METACHARACTERS
        assert isinstance(SHELL_METACHARACTERS, frozenset)
        for ch in ";|&`{}*?<>()[]":
            assert ch in SHELL_METACHARACTERS
        # Backslash, $, ! removed to preserve Windows paths and filenames
        for ch in "\\$!":
            assert ch not in SHELL_METACHARACTERS

    def test_sanitize_args_does_not_mutate_input(self):
        """sanitize_args returns a new list, not mutating the original."""
        from enigma_engine.core.commands import sanitize_args
        original = ["a;b", "c|d"]
        result = sanitize_args(original)
        assert original == ["a;b", "c|d"]  # unchanged
        assert result == ["ab", "cd"]


class TestBlockedPathEnforcement:
    """Verify file commands enforce blocked_paths/blocked_patterns from config."""

    def test_blocked_pattern_matches(self):
        """_check_blocked_path blocks files matching config patterns."""
        from enigma_engine.core.builtin_commands import register_builtin_commands
        from enigma_engine.core.commands import CommandRegistry
        reg = CommandRegistry()
        register_builtin_commands(reg)
        reg.set_context("config", {"blocked_patterns": ["*.pem", "*secret*"]})
        result = reg.execute("file.read server.pem")
        assert not result.success
        assert "blocked pattern" in result.message.lower()

    def test_default_blocked_patterns_has_sensitive_extensions(self):
        """Default blocked_patterns should block sensitive file types."""
        from enigma_engine.config.defaults import CONFIG
        patterns = CONFIG.get("blocked_patterns", [])
        # Should block executables, keys, and sensitive files
        assert "*.exe" in patterns
        assert "*.pem" in patterns
        assert "*.key" in patterns


class TestShellMetacharNotBlocked:
    """Verify shell command no longer blocks metacharacters (shell=False is sufficient)."""

    def test_no_metacharacter_error_in_shell_handler(self):
        """shell command handler does not contain metacharacter blocking."""
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        # Find the shell command handler
        shell_idx = source.index("ALLOWED_COMMANDS")
        shell_body = source[shell_idx:shell_idx + 1000]
        assert "Shell metacharacters are not allowed" not in shell_body


class TestRunCommandRemovedFromDefaults:
    """Verify run_command is not in default GGUF tools (security)."""

    def test_guard_no_run_command_in_gguf_loader(self):
        """Guard test: run_command must not appear in gguf_loader.py at all."""
        source_path = Path(__file__).resolve().parent.parent / "enigma_engine" / "core" / "gguf_loader.py"
        source = source_path.read_text(encoding="utf-8")
        assert "run_command" not in source


# =========================================================================
# Suggestion #5 — Plugin Loader Security
# =========================================================================


# ================================================================
# Tensor contiguity: .reshape() over .view()
# ================================================================


# ================================================================
# Command handler tests
# ================================================================



def _get_handler(name):
    """Get a command handler by name from the registry."""
    from enigma_engine.core.commands import get_registry
    reg = get_registry()
    cmd = next(c for c in reg.list_commands() if c.name == name)
    return cmd.handler


class TestConfigGetHandler:
    """Tests for config.get command handler."""

    def test_get_existing_key(self):
        handler = _get_handler("config.get")
        ctx = {"config": {"temperature": 0.7}}
        result = handler(["temperature"], ctx)
        assert result.success
        assert "0.7" in result.message

    def test_get_missing_key(self):
        handler = _get_handler("config.get")
        ctx = {"config": {}}
        result = handler(["nope"], ctx)
        assert not result.success

    def test_get_no_args(self):
        handler = _get_handler("config.get")
        result = handler([], {})
        assert not result.success
        assert "Usage" in result.message


class TestConfigListHandler:
    """Tests for config.list command handler."""

    def test_list_empty(self):
        handler = _get_handler("config.list")
        result = handler([], {"config": {}})
        assert result.success
        assert "No config" in result.message

    def test_list_with_values(self):
        handler = _get_handler("config.list")
        ctx = {"config": {"a": 1, "b": "hello"}}
        result = handler([], ctx)
        assert result.success
        assert "a = 1" in result.message
        assert "b = hello" in result.message


class TestFileListHandler:
    """Tests for file.list command handler."""

    def test_list_existing_dir(self, tmp_path):
        handler = _get_handler("file.list")
        (tmp_path / "test.txt").write_text("hi")
        result = handler([str(tmp_path)], {})
        assert result.success
        assert "test.txt" in result.message

    def test_list_nonexistent(self, tmp_path):
        handler = _get_handler("file.list")
        result = handler([str(tmp_path / "nope")], {})
        assert not result.success

    def test_list_empty_dir(self, tmp_path):
        handler = _get_handler("file.list")
        result = handler([str(tmp_path)], {})
        assert result.success
        assert "empty" in result.message

    def test_list_hides_dotfiles(self, tmp_path):
        handler = _get_handler("file.list")
        (tmp_path / ".hidden").write_text("x")
        (tmp_path / "visible.txt").write_text("x")
        result = handler([str(tmp_path)], {})
        assert "visible.txt" in result.message
        assert ".hidden" not in result.message

    def test_list_file_shows_size(self, tmp_path):
        handler = _get_handler("file.list")
        f = tmp_path / "data.bin"
        f.write_bytes(b"12345")
        result = handler([str(f)], {})
        assert result.success
        assert "5 bytes" in result.message


class TestFileReadHandler:
    """Tests for file.read command handler."""

    def test_read_existing_file(self, tmp_path):
        handler = _get_handler("file.read")
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        result = handler([str(f)], {})
        assert result.success
        assert "hello world" in result.message

    def test_read_missing_file(self, tmp_path):
        handler = _get_handler("file.read")
        result = handler([str(tmp_path / "nope.txt")], {})
        assert not result.success

    def test_read_directory_fails(self, tmp_path):
        handler = _get_handler("file.read")
        result = handler([str(tmp_path)], {})
        assert not result.success
        assert "directory" in result.message.lower()

    def test_read_no_args(self):
        handler = _get_handler("file.read")
        result = handler([], {})
        assert not result.success
        assert "Usage" in result.message

    def test_read_truncates_long_content(self, tmp_path):
        handler = _get_handler("file.read")
        f = tmp_path / "big.txt"
        f.write_text("x" * 2000, encoding="utf-8")
        result = handler([str(f)], {})
        assert result.success
        assert "chars total" in result.message


class TestFileWriteHandler:
    """Tests for file.write command handler."""

    def test_write_new_file(self, tmp_path):
        handler = _get_handler("file.write")
        target = tmp_path / "out.txt"
        result = handler([str(target), "hello", "world"], {})
        assert result.success
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_write_creates_dirs(self, tmp_path):
        handler = _get_handler("file.write")
        target = tmp_path / "sub" / "dir" / "file.txt"
        result = handler([str(target), "content"], {})
        assert result.success
        assert target.exists()

    def test_write_no_args(self):
        handler = _get_handler("file.write")
        result = handler([], {})
        assert not result.success

    def test_write_blocked_path(self, tmp_path):
        handler = _get_handler("file.write")
        target = tmp_path / "secret.key"
        ctx = {"config": {"blocked_patterns": ["*.key"]}}
        result = handler([str(target), "data"], ctx)
        assert not result.success
        assert "blocked" in result.message.lower()


class TestFileAppendHandler:
    """Tests for file.append command handler."""

    def test_append_to_existing(self, tmp_path):
        handler = _get_handler("file.append")
        f = tmp_path / "log.txt"
        f.write_text("line1\n", encoding="utf-8")
        result = handler([str(f), "line2"], {})
        assert result.success
        content = f.read_text(encoding="utf-8")
        assert "line1" in content
        assert "line2" in content

    def test_append_adds_newline(self, tmp_path):
        handler = _get_handler("file.append")
        f = tmp_path / "log.txt"
        f.write_text("no newline at end", encoding="utf-8")
        handler([str(f), "next"], {})
        content = f.read_text(encoding="utf-8")
        assert "no newline at end\nnext" in content

    def test_append_no_args(self):
        handler = _get_handler("file.append")
        result = handler([], {})
        assert not result.success


class TestMemorySaveLoadList:
    """Tests for memory.save, memory.load, memory.list handlers."""

    def test_save_and_load(self, tmp_path):
        save = _get_handler("memory.save")
        load = _get_handler("memory.load")
        ctx = {
            "memory_dir": tmp_path,
            "chat_messages": [{"role": "user", "content": "hi"}],
        }
        result = save(["test_convo"], ctx)
        assert result.success

        ctx2 = {"memory_dir": tmp_path}
        result = load(["test_convo"], ctx2)
        assert result.success
        assert len(ctx2["chat_messages"]) == 1

    def test_save_no_name(self):
        handler = _get_handler("memory.save")
        result = handler([], {})
        assert not result.success

    def test_save_empty_conversation(self, tmp_path):
        handler = _get_handler("memory.save")
        ctx = {"memory_dir": tmp_path, "chat_messages": []}
        result = handler(["empty"], ctx)
        assert not result.success

    def test_save_path_traversal_blocked(self, tmp_path):
        handler = _get_handler("memory.save")
        ctx = {"memory_dir": tmp_path, "chat_messages": [{"role": "user", "content": "x"}]}
        result = handler([".."], ctx)
        assert not result.success

    def test_load_missing(self, tmp_path):
        handler = _get_handler("memory.load")
        ctx = {"memory_dir": tmp_path}
        result = handler(["nonexistent"], ctx)
        assert not result.success

    def test_list_empty(self, tmp_path):
        handler = _get_handler("memory.list")
        ctx = {"memory_dir": tmp_path}
        result = handler([], ctx)
        assert result.success
        assert "No saved" in result.message

    def test_list_with_memories(self, tmp_path):
        import json
        (tmp_path / "chat1.json").write_text(
            json.dumps({"messages": [1, 2, 3], "message_count": 3}),
            encoding="utf-8")
        handler = _get_handler("memory.list")
        ctx = {"memory_dir": tmp_path}
        result = handler([], ctx)
        assert result.success
        assert "chat1" in result.message


class TestPersistentMemoryHandlers:
    """Tests for memory.remember, forget, notes, clear, search."""

    def test_remember_and_notes(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()

        remember = _get_handler("memory.remember")
        notes = _get_handler("memory.notes")

        result = remember(["test", "fact", "123"], {})
        assert result.success

        result = notes([], {})
        assert result.success
        assert "test fact 123" in result.message

        mem.clear()

    def test_remember_duplicate(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()

        remember = _get_handler("memory.remember")
        remember(["unique_fact"], {})
        result = remember(["unique_fact"], {})
        assert "Already known" in result.message

        mem.clear()

    def test_forget(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()
        mem.add("forgettable item")

        forget = _get_handler("memory.forget")
        result = forget(["forgettable"], {})
        assert result.success

        mem.clear()

    def test_forget_not_found(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()

        forget = _get_handler("memory.forget")
        result = forget(["nothing_here"], {})
        assert not result.success

    def test_search_finds_match(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()
        mem.add("the sky is blue")

        search = _get_handler("memory.search")
        result = search(["sky"], {})
        assert result.success
        assert "sky is blue" in result.message

        mem.clear()

    def test_search_no_match(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()
        mem.add("something else")

        search = _get_handler("memory.search")
        result = search(["zzz_not_here"], {})
        assert "No memories found" in result.message

        mem.clear()

    def test_clear_notes(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()
        mem.add("to be cleared")

        clear = _get_handler("memory.clear_notes")
        result = clear([], {})
        assert result.success
        assert mem.count == 0

    def test_notes_empty(self):
        from enigma_engine.core.memory import get_memory
        mem = get_memory()
        mem.clear()

        notes = _get_handler("memory.notes")
        result = notes([], {})
        assert "No persistent" in result.message

    def test_search_no_args(self):
        search = _get_handler("memory.search")
        result = search([], {})
        assert not result.success

    def test_remember_no_args(self):
        remember = _get_handler("memory.remember")
        result = remember([], {})
        assert not result.success

    def test_forget_no_args(self):
        forget = _get_handler("memory.forget")
        result = forget([], {})
        assert not result.success


class TestHistoryHandler:
    """Tests for history command handler."""

    def test_history_no_registry(self):
        handler = _get_handler("history")
        result = handler([], {})
        assert result.success
        assert "No history" in result.message

    def test_history_with_registry(self):
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        # Execute a command to create history
        reg.execute("config.list")
        handler = _get_handler("history")
        result = handler([], {"registry": reg})
        assert result.success


class TestHelpHandler:
    """Tests for help command handler."""

    def test_help_all(self):
        from enigma_engine.core.commands import get_registry
        handler = _get_handler("help")
        result = handler([], {"registry": get_registry()})
        assert result.success

    def test_help_specific_command(self):
        from enigma_engine.core.commands import get_registry
        handler = _get_handler("help")
        result = handler(["config.get"], {"registry": get_registry()})
        assert result.success

    def test_help_no_registry(self):
        handler = _get_handler("help")
        result = handler([], {})
        assert result.success


class TestTrainDataHandlers:
    """Tests for train.data.add and train.data.list."""

    def test_add_training_data(self, tmp_path):
        handler = _get_handler("train.data.add")
        ctx = {"data_dir": tmp_path}
        result = handler(["Q:", "hello", "A:", "world"], ctx)
        assert result.success
        content = (tmp_path / "training.txt").read_text(encoding="utf-8")
        assert "Q: hello A: world" in content

    def test_add_no_args(self):
        handler = _get_handler("train.data.add")
        result = handler([], {})
        assert not result.success

    def test_list_no_file(self, tmp_path):
        handler = _get_handler("train.data.list")
        ctx = {"data_dir": tmp_path}
        result = handler([], ctx)
        assert result.success
        assert "No training data" in result.message

    def test_list_with_data(self, tmp_path):
        (tmp_path / "training.txt").write_text(
            "Q: what\nA: that\n", encoding="utf-8")
        handler = _get_handler("train.data.list")
        ctx = {"data_dir": tmp_path}
        result = handler([], ctx)
        assert result.success
        assert "Q: what" in result.message


class TestStopHandler:
    """Tests for stop command handler."""

    def test_stop_no_engine(self):
        handler = _get_handler("stop")
        result = handler([], {})
        assert result.success

    def test_stop_sets_cancel_flag_when_generation_active(self):
        """Pass 156z9ff (Pass B): stop_cmd sets the flag only when
        _generation_lock is currently held (a generation is in flight)."""
        import threading

        handler = _get_handler("stop")

        class FakeEngine:
            def __init__(self):
                self._cancel_generation = False
                self._generation_lock = threading.Lock()

        engine = FakeEngine()
        # Hold the lock to simulate active generation.
        with engine._generation_lock:
            result = handler([], {"engine": engine})
        assert result.success
        assert engine._cancel_generation, "cancel flag must be set"
        assert "Stop signal sent" in result.message

    def test_stop_noop_when_no_generation_active(self):
        """Pass 156z9ff (Pass B): unlocked lock means idle engine; stop
        must not set the flag (would silently kill the next generation)."""
        import threading

        handler = _get_handler("stop")

        class FakeEngine:
            def __init__(self):
                self._cancel_generation = False
                self._generation_lock = threading.Lock()

        engine = FakeEngine()
        result = handler([], {"engine": engine})
        assert result.success
        assert not engine._cancel_generation, "cancel flag must NOT be set"
        assert "No active generation" in result.message

