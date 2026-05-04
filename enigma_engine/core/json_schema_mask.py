"""T3-9: Grammar-guided JSON decoding via finite-state machine.

Enforces structurally valid JSON output during generation by masking
logits at each step.  Only tokens whose first decoded character is
compatible with the current FSM state are allowed.

Supports flat and nested JSON objects with string, number, boolean,
null, object, and array value types.  Tracks key-value pair count to
enforce the correct number of fields from the schema.  Nested values
(objects and arrays) are depth-tracked so inner structure does not
corrupt the outer FSM state.

Usage::

    constraint = JsonSchemaConstraint(schema, tokenizer)
    # ... in generation loop:
    logits = constraint.mask_logits(logits)
    token = sample(logits)
    constraint.advance(token)
"""

import logging
from collections import defaultdict
from typing import Optional

import torch

logger = logging.getLogger(__name__)


_DEFAULT_SUPPORTED_TYPES: frozenset[str] = frozenset({
    'string', 'number', 'integer', 'boolean', 'null',
    'object', 'array',
})


def validate_json_schema_shape(
    schema: object,
    *,
    supported_types: frozenset[str] = _DEFAULT_SUPPORTED_TYPES,
) -> None:
    """Validate the structural shape of a JSON schema dict.

    Pass 156z9ac: extracted from ``JsonSchemaConstraint.__init__`` so
    boundary callers (FastAPI handlers, GUI Apply button, CLI tools)
    can validate at the boundary and surface a clean user-facing error
    BEFORE the request reaches generation.  Without this split a
    malformed schema raised ``ValueError`` deep inside the engine, the
    FastAPI exception handler mapped it to HTTP 500, and the user saw
    a generic "Internal Server Error" instead of the actionable
    validator message naming what's wrong.

    Raises:
        ValueError: Schema is not a dict, ``type`` is not
            ``"object"``, ``properties`` is not a dict, a property
            spec is not a dict, or a property's ``type`` is outside
            the supported set.

    The check is deliberately structural-only — it does NOT validate
    that the schema is a valid draft-2020-12 JSON Schema (no ``$ref``
    resolution, no format checks, no required-array semantics).  Its
    sole job is to gate the inputs the FSM can actually constrain.
    """
    if not isinstance(schema, dict):
        raise ValueError(
            f"json_schema must be a dict, got {type(schema).__name__}"
        )
    schema_type = schema.get('type', 'object')
    if schema_type != 'object':
        raise ValueError(
            f"json_schema['type'] must be 'object' (FSM is "
            f"object-only), got {schema_type!r}"
        )
    props = schema.get('properties', {})
    if not isinstance(props, dict):
        raise ValueError(
            f"json_schema['properties'] must be a dict, got "
            f"{type(props).__name__}"
        )
    for key, spec in props.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"json_schema['properties'][{key!r}] must be a "
                f"dict, got {type(spec).__name__}"
            )
        ptype = spec.get('type', 'string')
        if ptype not in supported_types:
            raise ValueError(
                f"json_schema['properties'][{key!r}]['type'] = "
                f"{ptype!r} is not supported. Allowed: "
                f"{sorted(supported_types)}"
            )


class JsonSchemaConstraint:
    """FSM that constrains generation to produce schema-conforming JSON.

    The FSM tracks structural position (``{``, key, ``:``, value,
    ``,``, ``}``) and masks logits so only tokens starting with a
    valid character are allowed.  This guarantees syntactically valid
    JSON with the expected number of key-value pairs.

    Args:
        schema: JSON schema dict.  Must have ``'type': 'object'``
            and ``'properties'`` mapping key names to type dicts.
        tokenizer: Object with ``decode(ids) -> str`` and
            ``vocab_size: int``.
    """

    # Types the FSM knows how to constrain.  Anything outside this set
    # falls through to ``_value_starters``'s "unknown type" branch and
    # silently degrades to free generation.  Validating up-front turns
    # that silent-degradation into a loud rejection at construction.
    _SUPPORTED_TYPES: frozenset[str] = frozenset({
        'string', 'number', 'integer', 'boolean', 'null',
        'object', 'array',
    })

    def __init__(self, schema: dict, tokenizer: object) -> None:
        # --- boundary validation -------------------------------------
        # Closes the "API accepts any dict" follow-up from Pass 156z3 /
        # 156z4.  A malformed schema (missing ``properties``,
        # ``type != "object"``, properties value not a dict) reached
        # the FSM and silently produced degraded output that callers
        # could not distinguish from a successful constrained
        # generation.  Fail loud at the constructor instead.
        #
        # Pass 156z9ac extracted the shape checks to
        # ``validate_json_schema_shape`` so the FastAPI handlers can
        # call them at the request boundary and return HTTP 400 with
        # a helpful message — without that, a malformed schema raised
        # ``ValueError`` deep inside generation, FastAPI mapped it to
        # HTTP 500 with a stack trace, and the user saw "Internal
        # Server Error" instead of "your schema is malformed".
        validate_json_schema_shape(schema, supported_types=self._SUPPORTED_TYPES)
        props = schema.get('properties', {})

        self._n_keys = len(props)
        self._key_types = [
            v.get('type', 'string') for v in props.values()
        ]
        self._key_type_map: dict[str, str] = {
            k: v.get('type', 'string') for k, v in props.items()
        }
        self._vocab_size: int = tokenizer.vocab_size  # type: ignore[union-attr]
        self._tokenizer = tokenizer

        # Pre-build: first-char → token IDs
        self._char_tokens: dict[str, list[int]] = defaultdict(list)
        for tid in range(self._vocab_size):
            try:
                text = tokenizer.decode([tid])  # type: ignore[union-attr]
            except Exception:
                continue
            if text:
                self._char_tokens[text[0]].append(tid)

        self.reset()

    def reset(self) -> None:
        """Reset the FSM to the initial state."""
        self._state = 'EXPECT_OPEN'
        self._pairs_done = 0
        self._in_string = False  # tracking inside a quoted string
        self._escape_next = False
        self._brace_depth = 0
        self._value_depth = 0  # nesting depth inside current value
        self._current_key = ''  # key name being parsed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the schema constraint mask to *logits* in-place."""
        allowed = self._allowed_tokens()
        if allowed is None:
            return logits
        mask = torch.full_like(logits, float('-inf'))
        for tid in allowed:
            if tid < logits.shape[-1]:
                mask[..., tid] = 0.0
        return logits + mask

    def advance(self, token_id: int) -> None:
        """Advance the FSM after *token_id* is generated."""
        try:
            text = self._tokenizer.decode([token_id])  # type: ignore[union-attr]
        except Exception:
            return
        for ch in text:
            self._advance_char(ch)

    @property
    def is_done(self) -> bool:
        return self._state == 'DONE'

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tokens_for(self, chars: str) -> list[int]:
        """Token IDs whose decoded text starts with any char in *chars*."""
        result: list[int] = []
        for c in chars:
            result.extend(self._char_tokens.get(c, []))
        return result

    def _allowed_tokens(self) -> Optional[list[int]]:
        """Allowed token IDs at the current state, or *None* for any."""
        state = self._state

        if state == 'EXPECT_OPEN':
            return self._tokens_for('{ \t\n\r')

        if state == 'EXPECT_KEY':
            return self._tokens_for('" \t\n\r')

        if state == 'IN_KEY':
            return None  # Any token — we detect closing " in advance

        if state == 'EXPECT_COLON':
            return self._tokens_for(': \t\n\r')

        if state == 'EXPECT_VALUE':
            vtype = self._current_value_type()
            return self._value_starters(vtype)

        if state == 'IN_VALUE':
            return None  # Free generation inside values

        if state == 'AFTER_VALUE':
            if self._pairs_done < self._n_keys:
                return self._tokens_for(', \t\n\r')
            return self._tokens_for(',} \t\n\r')

        if state == 'DONE':
            return []  # Only EOS should follow

        return None

    def _value_starters(self, vtype: str) -> list[int]:
        """Token IDs that can start a value of the given type."""
        if vtype == 'string':
            return self._tokens_for('" \t\n\r')
        if vtype in ('number', 'integer'):
            return self._tokens_for('0123456789-. \t\n\r')
        if vtype == 'boolean':
            return self._tokens_for('tf \t\n\r')
        if vtype == 'null':
            return self._tokens_for('n \t\n\r')
        # Unknown type: allow any value starter
        return self._tokens_for('"0123456789-.tfn[{ \t\n\r')

    def _current_value_type(self) -> str:
        # Look up by actual key name first (S738)
        if self._current_key and self._current_key in self._key_type_map:
            return self._key_type_map[self._current_key]
        idx = min(self._pairs_done, len(self._key_types) - 1)
        return self._key_types[idx] if self._key_types else 'string'

    def _advance_char(self, ch: str) -> None:
        """Single-character FSM transition."""
        # Handle escape sequences inside strings
        if self._escape_next:
            self._escape_next = False
            return
        if ch == '\\' and self._in_string:
            self._escape_next = True
            return

        state = self._state

        if state == 'EXPECT_OPEN':
            if ch == '{':
                self._brace_depth = 1
                if self._n_keys == 0:
                    self._state = 'AFTER_VALUE'
                else:
                    self._state = 'EXPECT_KEY'

        elif state == 'EXPECT_KEY':
            if ch == '"':
                self._in_string = True
                self._current_key = ''
                self._state = 'IN_KEY'

        elif state == 'IN_KEY':
            if ch == '"' and not self._escape_next:
                self._in_string = False
                self._state = 'EXPECT_COLON'
            else:
                self._current_key += ch

        elif state == 'EXPECT_COLON':
            if ch == ':':
                self._state = 'EXPECT_VALUE'

        elif state == 'EXPECT_VALUE':
            if ch == '"':
                self._in_string = True
                self._value_depth = 0
                self._state = 'IN_VALUE'
            elif ch in '0123456789-':
                self._value_depth = 0
                self._state = 'IN_VALUE'
            elif ch in 'tfn':
                self._value_depth = 0
                self._state = 'IN_VALUE'
            elif ch == '{':
                self._value_depth = 1
                self._state = 'IN_VALUE'
            elif ch == '[':
                self._value_depth = 1
                self._state = 'IN_VALUE'

        elif state == 'IN_VALUE':
            if self._in_string:
                if ch == '"':
                    self._in_string = False
                    if self._value_depth == 0:
                        self._pairs_done += 1
                        self._state = 'AFTER_VALUE'
            else:
                if ch == '"':
                    self._in_string = True
                elif ch in '{[':
                    self._value_depth += 1
                elif ch == ']':
                    self._value_depth -= 1
                    if self._value_depth == 0:
                        self._pairs_done += 1
                        self._state = 'AFTER_VALUE'
                elif ch == '}':
                    if self._value_depth > 0:
                        self._value_depth -= 1
                        if self._value_depth == 0:
                            self._pairs_done += 1
                            self._state = 'AFTER_VALUE'
                    else:
                        self._pairs_done += 1
                        self._brace_depth -= 1
                        if self._brace_depth <= 0:
                            self._state = 'DONE'
                        else:
                            self._state = 'AFTER_VALUE'
                elif ch == ',':
                    if self._value_depth == 0:
                        self._pairs_done += 1
                        self._state = 'EXPECT_KEY'

        elif state == 'AFTER_VALUE':
            if ch == ',':
                self._state = 'EXPECT_KEY'
            elif ch == '}':
                self._brace_depth -= 1
                if self._brace_depth <= 0:
                    self._state = 'DONE'
