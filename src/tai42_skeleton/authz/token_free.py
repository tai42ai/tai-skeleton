"""The token-free-evaluable rule for a jq policy condition: which conditions a TOKENLESS
background execution can be authorized against.

A fire's jq context carries the token claims REDUCED to the key's owner, making ``identity``
the SOLE field a condition may not depend on — depending on it is a fail-OPEN, since
``.identity.X != v`` evaluates TRUE under an absent claim. The scan is structural, not by
sampling (``. | tojson`` exfiltrates claims without spelling ``identity``): only
``.identity.owner_user_id``, with nothing further applied, may project ``identity``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from string import ascii_letters, digits

from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_kit.utils.data.jq_util import get_compiled_jq

# The one context field whose content a fire cannot reproduce, and the one claim inside
# it that IS readable at a fire (from the execution key's stored policy data).
_IDENTITY_FIELD = "identity"
_OWNER_REFERENCE = f".{_IDENTITY_FIELD}.{OWNER_USER_ID_CLAIM}"

# Allowlisted builtins and their permitted arities. Each reads ONLY its input and
# arguments, which is what lets the taint rule treat a call as "input+args SAFE ⇒ result
# SAFE". Anything absent is refused; adding one is a security decision.
_BUILTIN_ARITIES: Mapping[str, frozenset[int]] = {
    "not": frozenset({0}),
    "length": frozenset({0}),
    "ascii_downcase": frozenset({0}),
    "startswith": frozenset({1}),
    "endswith": frozenset({1}),
    "contains": frozenset({1}),
    "test": frozenset({1, 2}),
    "IN": frozenset({1, 2}),
    "map": frozenset({1}),
    "any": frozenset({0, 1, 2}),
    "all": frozenset({0, 1, 2}),
}

# Names the allowlist already refuses; this table only makes the refusal say WHY.
_NAMED_REFUSALS: Mapping[str, str] = {
    "def": "defines a function, which can rename any builtin and so defeats the per-builtin rule",
    "env": "reads the process environment, which is not part of the auth context",
    "input": "reads a value from outside the auth context",
    "inputs": "reads values from outside the auth context",
    "now": "reads the wall clock, which is nondeterministic and outside the auth context; "
    "a fire reads the clock from .system.time",
    "localtime": "is outside the allowlisted builtins; express date arithmetic as a "
    "comparison against .system.*, which a fire can read",
    "gmtime": "is outside the allowlisted builtins; express date arithmetic as a "
    "comparison against .system.*, which a fire can read",
}
_NAMED_VARIABLE_REFUSALS: Mapping[str, str] = {
    "ENV": "reads the process environment, which is not part of the auth context",
    "__loc__": "reads the program's own source location, which is not part of the auth context",
}

_KEYWORD_LITERALS = frozenset({"true", "false", "null"})
_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_ADDITIVE = frozenset({"+", "-"})
_MULTIPLICATIVE = frozenset({"*", "/", "%"})

_IDENT_START = frozenset(ascii_letters + "_")
_IDENT_CHARS = _IDENT_START | frozenset(digits)
_DIGITS = frozenset(digits)

# The only token separator; every other whitespace character is already refused
# pre-lexically by :func:`_assert_source_shape`.
_WHITESPACE = frozenset(" ")

# The printable-ASCII band the source-shape gate accepts, both ends inclusive.
_PRINTABLE_ASCII_START = " "
_PRINTABLE_ASCII_END = "~"

# Punctuation, longest first so a prefix never shadows a longer operator.
_OPERATORS = (
    "?//",
    "//=",
    "|=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "==",
    "!=",
    "<=",
    ">=",
    "//",
    "..",
    "=",
    "<",
    ">",
    "|",
    ",",
    "+",
    "-",
    "*",
    "/",
    "%",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ":",
    ";",
    "?",
    ".",
)

# The escapes a string literal is decoded through. ``\uXXXX`` is deliberately absent and
# refused: it is the one escape that mints a character the source-shape gate never saw.
_STRING_ESCAPES: Mapping[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

_EXCERPT_LENGTH = 32


class TokenFreeConditionError(Exception):
    """A jq policy condition cannot be shown evaluable by a background execution.

    Raised by :func:`assert_token_free_evaluable` naming the offending construct and where
    it sits, and by :func:`~tai42_skeleton.authz.execution.assert_execution_key_evaluable`
    when a condition does not render at all.
    """


# Analysis-REFUSAL bounds, not limits; headroom over the shipped conditions is small (pinned by test).
_MAX_TOKENS = 256
_MAX_NESTING_DEPTH = 32


def _excerpt(condition_text: str, position: int) -> str:
    """A bounded, readable slice of ``condition_text`` starting at the offending
    construct — enough to locate it without echoing the whole condition."""
    return condition_text[position : position + _EXCERPT_LENGTH]


def _refusal(condition_text: str, position: int, detail: str) -> TokenFreeConditionError:
    """The one refusal shape: what was refused, and where to find it."""
    return TokenFreeConditionError(f"{detail} at offset {position} ({_excerpt(condition_text, position)!r})")


def _assert_source_shape(condition_text: str) -> None:
    """Assert that ``condition_text`` is written in the character subset where this
    module's lexer and jq's cannot disagree about where a token begins and ends:
    printable ASCII (space through ``~``), on a single line, with no ``#`` anywhere.

    **PRE-LEXICAL, and must stay that way** — reading raw characters only. A gate
    expressed over tokens would inherit the very lexing assumptions it exists to test.

    Outside this subset a disagreement that parses SUCCESSFULLY INTO A DIFFERENT PROGRAM
    certifies a program jq never runs while jq runs one never analyzed: a fail-OPEN.
    Refusing the whole class removes that shape.
    """
    for position, char in enumerate(condition_text):
        if char == "#":
            raise _refusal(
                condition_text,
                position,
                "condition contains '#', which opens a jq comment whose extent the token-free scan does not "
                "adjudicate; delete the comment (a '#' inside a string literal is refused too, and has to be "
                "rewritten out)",
            )
        if char in "\n\r":
            raise _refusal(
                condition_text,
                position,
                "condition spans more than one line; write it on a single line",
            )
        if char == "\t":
            raise _refusal(
                condition_text,
                position,
                "condition contains a tab; separate tokens with spaces",
            )
        if char < _PRINTABLE_ASCII_START or char == "\x7f":
            raise _refusal(
                condition_text,
                position,
                f"condition contains the control character {char!r}; remove it",
            )
        if char > _PRINTABLE_ASCII_END:
            raise _refusal(
                condition_text,
                position,
                f"condition contains the non-ASCII character {char!r}; the token-free scan reads printable ASCII "
                "only, so rewrite the condition without it",
            )


@dataclass
class _Budget:
    """The SINGLE allowance one scan of one condition draws from.

    Threaded through the lexer, the parser and every ``\\(...)`` body they descend into,
    so an interpolation cannot mint a fresh allowance and the depth counter tracks the
    real interpreter stack.
    """

    tokens: int = 0
    depth: int = 0

    def spend(self, condition_text: str, position: int) -> None:
        """Account for one lexed token, refusing past :data:`_MAX_TOKENS`."""
        self.tokens += 1
        if self.tokens > _MAX_TOKENS:
            raise _refusal(
                condition_text,
                position,
                f"condition is longer than the {_MAX_TOKENS} tokens the token-free scan will analyze; shorten it "
                "or move part of it into a named scope/role",
            )

    def descend(self, condition_text: str, position: int) -> None:
        """Enter one level of nesting, refusing past :data:`_MAX_NESTING_DEPTH`. Always
        paired with a ``finally`` that calls :meth:`ascend`."""
        self.depth += 1
        if self.depth > _MAX_NESTING_DEPTH:
            raise _refusal(
                condition_text,
                position,
                f"condition nests more than {_MAX_NESTING_DEPTH} levels deep, past what the token-free scan will "
                "analyze",
            )

    def ascend(self) -> None:
        """Leave the level entered by :meth:`descend`."""
        self.depth -= 1


class _Kind(Enum):
    """The lexical classes the condition grammar is built from."""

    FIELD = "field"
    IDENT = "identifier"
    VARIABLE = "variable"
    NUMBER = "number"
    STRING = "string"
    FORMAT = "format"
    OPERATOR = "operator"
    END = "end"


@dataclass(frozen=True)
class _Interpolation:
    """A ``\\(...)`` hole in a string literal, as a half-open span of the condition
    text. Its body is jq CODE and is parsed and analyzed as such."""

    start: int
    end: int


@dataclass(frozen=True)
class _Token:
    kind: _Kind
    text: str
    position: int
    interpolations: tuple[_Interpolation, ...] = ()


def _describe(token: _Token) -> str:
    """How a token reads in a refusal message."""
    return "the end of the condition" if token.kind is _Kind.END else repr(token.text)


def _lex_string(condition_text: str, start: int, budget: _Budget) -> tuple[_Token, int]:
    """Lex the string literal opening at ``start``, returning it and the offset just
    past its closing quote. ``text`` is the DECODED literal value, meaningful only when
    the string carries no interpolation.

    Mutually recursive with :func:`_scan_interpolation` once per nested ``\\(...)`` hole,
    so it takes a level of ``budget``."""
    budget.descend(condition_text, start)
    try:
        parts: list[str] = []
        interpolations: list[_Interpolation] = []
        index = start + 1
        while index < len(condition_text):
            char = condition_text[index]
            if char == '"':
                return _Token(_Kind.STRING, "".join(parts), start, tuple(interpolations)), index + 1
            if char != "\\":
                parts.append(char)
                index += 1
                continue
            index += 1
            if index >= len(condition_text):
                break
            escape = condition_text[index]
            if escape == "(":
                body_start = index + 1
                body_end = _scan_interpolation(condition_text, body_start, budget)
                interpolations.append(_Interpolation(body_start, body_end))
                index = body_end + 1
                continue
            if escape not in _STRING_ESCAPES:
                raise _refusal(condition_text, index - 1, f"condition contains the unsupported escape '\\{escape}'")
            parts.append(_STRING_ESCAPES[escape])
            index += 1
        raise _refusal(condition_text, start, "condition contains an unterminated string literal")
    finally:
        budget.ascend()


def _scan_interpolation(condition_text: str, start: int, budget: _Budget) -> int:
    """The offset of the ``)`` closing the interpolation body that opens at ``start``.
    Nested strings are skipped exactly as the lexer skips them, so a parenthesis inside
    one never closes the body.

    Mutually recursive with :func:`_lex_string` over those nested strings, so it takes a
    level of ``budget`` too."""
    budget.descend(condition_text, start - 2)
    try:
        depth = 1
        index = start
        while index < len(condition_text):
            char = condition_text[index]
            if char == '"':
                _, index = _lex_string(condition_text, index, budget)
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        raise _refusal(condition_text, start - 2, "condition contains an unterminated string interpolation")
    finally:
        budget.ascend()


def _lex(condition_text: str, budget: _Budget, start: int = 0, end: int | None = None) -> list[_Token]:
    """Tokenize ``condition_text[start:end]``, with every token carrying its offset in
    the WHOLE condition so a refusal points at the real text. Every token emitted is
    spent from ``budget``, whether it sits in the outer text or inside an interpolation
    body."""
    stop = len(condition_text) if end is None else end
    tokens: list[_Token] = []

    def emit(token: _Token) -> None:
        budget.spend(condition_text, token.position)
        tokens.append(token)

    index = start
    while index < stop:
        char = condition_text[index]
        if char in _WHITESPACE:
            index += 1
            continue
        if char == '"':
            token, index = _lex_string(condition_text, index, budget)
            emit(token)
            continue
        if char == "$":
            name_end = index + 1
            while name_end < stop and condition_text[name_end] in _IDENT_CHARS:
                name_end += 1
            if name_end == index + 1:
                raise _refusal(condition_text, index, "condition contains a nameless variable")
            emit(_Token(_Kind.VARIABLE, condition_text[index + 1 : name_end], index))
            index = name_end
            continue
        if char == "@":
            name_end = index + 1
            while name_end < stop and condition_text[name_end] in _IDENT_CHARS:
                name_end += 1
            emit(_Token(_Kind.FORMAT, condition_text[index:name_end], index))
            index = name_end
            continue
        if char in _DIGITS:
            token, index = _lex_number(condition_text, index, stop)
            emit(token)
            continue
        if char in _IDENT_START:
            name_end = index
            while name_end < stop and condition_text[name_end] in _IDENT_CHARS:
                name_end += 1
            emit(_Token(_Kind.IDENT, condition_text[index:name_end], index))
            index = name_end
            continue
        if char == "." and index + 1 < stop and condition_text[index + 1] in _IDENT_START:
            name_end = index + 1
            while name_end < stop and condition_text[name_end] in _IDENT_CHARS:
                name_end += 1
            emit(_Token(_Kind.FIELD, condition_text[index + 1 : name_end], index))
            index = name_end
            continue
        if char == "." and index + 1 < stop and condition_text[index + 1] == '"':
            quoted, index = _lex_string(condition_text, index + 1, budget)
            if quoted.interpolations:
                raise _refusal(condition_text, quoted.position, "condition names a field by an interpolated string")
            emit(_Token(_Kind.FIELD, quoted.text, quoted.position - 1))
            continue
        operator = next((candidate for candidate in _OPERATORS if condition_text.startswith(candidate, index)), None)
        if operator is None:
            raise _refusal(condition_text, index, f"condition contains the unrecognized character {char!r}")
        emit(_Token(_Kind.OPERATOR, operator, index))
        index += len(operator)
    emit(_Token(_Kind.END, "", stop))
    return tokens


def _lex_number(condition_text: str, start: int, stop: int) -> tuple[_Token, int]:
    """Lex the numeric literal at ``start``, returning it and the offset just past it."""
    index = start
    while index < stop and condition_text[index] in _DIGITS:
        index += 1
    if index < stop and condition_text[index] == "." and index + 1 < stop and condition_text[index + 1] in _DIGITS:
        index += 1
        while index < stop and condition_text[index] in _DIGITS:
            index += 1
    if index < stop and condition_text[index] in "eE":
        exponent = index + 1
        if exponent < stop and condition_text[exponent] in "+-":
            exponent += 1
        if exponent < stop and condition_text[exponent] in _DIGITS:
            while exponent < stop and condition_text[exponent] in _DIGITS:
                exponent += 1
            index = exponent
    return _Token(_Kind.NUMBER, condition_text[start:index], start), index


@dataclass(frozen=True)
class _Node:
    """A parsed construct, carrying the offset a refusal about it reports."""

    position: int


@dataclass(frozen=True)
class _Root(_Node):
    """``.`` — the value flowing into the current expression."""


@dataclass(frozen=True)
class _Constant(_Node):
    """A number or a ``true``/``false``/``null`` literal."""


@dataclass(frozen=True)
class _String(_Node):
    """A string literal. ``value`` is the decoded text and is a usable static field name
    only when ``interpolations`` is empty."""

    value: str
    interpolations: tuple[_Node, ...] = ()


@dataclass(frozen=True)
class _Variable(_Node):
    name: str


@dataclass(frozen=True)
class _Field(_Node):
    """A ``.name`` / ``."name"`` projection suffix."""

    name: str


@dataclass(frozen=True)
class _Index(_Node):
    """A ``[expr]`` suffix, or ``[]`` (iterate all values) when ``key`` is ``None``."""

    key: _Node | None


@dataclass(frozen=True)
class _Optional(_Node):
    """A ``?`` suffix: the same value, with errors suppressed."""


@dataclass(frozen=True)
class _Projection(_Node):
    """``source`` with a chain of suffixes applied to it."""

    source: _Node
    suffixes: tuple[_Field | _Index | _Optional, ...]


@dataclass(frozen=True)
class _Call(_Node):
    """An application of an allowlisted builtin to the current input."""

    name: str
    arguments: tuple[_Node, ...] = ()


@dataclass(frozen=True)
class _Binary(_Node):
    operator: str
    left: _Node
    right: _Node


@dataclass(frozen=True)
class _Negate(_Node):
    operand: _Node


@dataclass(frozen=True)
class _Bind(_Node):
    """``source as $variable | body``."""

    source: _Node
    variable: str
    body: _Node


@dataclass(frozen=True)
class _Try(_Node):
    """``try body`` with an optional ``catch handler``."""

    body: _Node
    handler: _Node | None


@dataclass(frozen=True)
class _ObjectConstruction(_Node):
    """``{key: value, ...}``. The keys are static names, so only the VALUES can carry a
    value derived from the auth context."""

    values: tuple[_Node, ...] = ()


@dataclass(frozen=True)
class _ArrayConstruction(_Node):
    """``[expr]``, or ``[]`` when ``element`` is ``None``."""

    element: _Node | None


class _Parser:
    """A recursive-descent parser for the allowlisted condition grammar.

    It recognizes exactly the constructs the taint analysis can reason about and raises
    :class:`TokenFreeConditionError` on everything else, so an unknown construct is
    refused before any question of information flow is asked. Precedence follows jq's,
    lowest first: ``|``, ``,``, ``//``, ``or``, ``and``, comparison, ``+``/``-``,
    ``*``/``/``/``%``, unary, then postfix suffixes on a term.
    """

    def __init__(self, condition_text: str, tokens: Sequence[_Token], budget: _Budget) -> None:
        self._text = condition_text
        self._tokens = tokens
        self._index = 0
        self._budget = budget

    def parse(self) -> _Node:
        """The whole token stream as one expression; trailing tokens are a refusal."""
        node = self._pipe()
        token = self._peek()
        if token.kind is not _Kind.END:
            raise self._refuse(token, f"condition does not parse from {_describe(token)}")
        return node

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _at_operator(self, *texts: str) -> bool:
        token = self._peek()
        return token.kind is _Kind.OPERATOR and token.text in texts

    def _at_keyword(self, keyword: str) -> bool:
        token = self._peek()
        return token.kind is _Kind.IDENT and token.text == keyword

    def _expect_operator(self, text: str) -> _Token:
        if not self._at_operator(text):
            token = self._peek()
            raise self._refuse(token, f"condition is missing the {text!r} expected before {_describe(token)}")
        return self._advance()

    def _refuse(self, token: _Token, detail: str) -> TokenFreeConditionError:
        return _refusal(self._text, token.position, detail)

    def _descend(self) -> None:
        """Enter one level of nesting on the shared budget.

        Paired with a ``finally`` that leaves the level again. The three self-recursive
        rules — the pipe, the alternative and the unary prefixes — call it, and every
        other nested construct (parentheses, brackets, object values, ``as`` bodies)
        re-enters the grammar through the pipe, so one bound covers them all."""
        self._budget.descend(self._text, self._peek().position)

    def _pipe(self) -> _Node:
        self._descend()
        try:
            left = self._comma()
            if self._at_operator("|"):
                token = self._advance()
                return _Binary(token.position, "|", left, self._pipe())
            return left
        finally:
            self._budget.ascend()

    def _comma(self) -> _Node:
        left = self._alternative()
        while self._at_operator(","):
            token = self._advance()
            left = _Binary(token.position, ",", left, self._alternative())
        return left

    def _alternative(self) -> _Node:
        self._descend()
        try:
            left = self._disjunction()
            if self._at_operator("//"):
                token = self._advance()
                return _Binary(token.position, "//", left, self._alternative())
            return left
        finally:
            self._budget.ascend()

    def _disjunction(self) -> _Node:
        left = self._conjunction()
        while self._at_keyword("or"):
            token = self._advance()
            left = _Binary(token.position, "or", left, self._conjunction())
        return left

    def _conjunction(self) -> _Node:
        left = self._comparison()
        while self._at_keyword("and"):
            token = self._advance()
            left = _Binary(token.position, "and", left, self._comparison())
        return left

    def _comparison(self) -> _Node:
        left = self._additive()
        if self._at_operator(*_COMPARISONS):
            token = self._advance()
            return _Binary(token.position, token.text, left, self._additive())
        return left

    def _additive(self) -> _Node:
        left = self._multiplicative()
        while self._at_operator(*_ADDITIVE):
            token = self._advance()
            left = _Binary(token.position, token.text, left, self._multiplicative())
        return left

    def _multiplicative(self) -> _Node:
        left = self._unary()
        while self._at_operator(*_MULTIPLICATIVE):
            token = self._advance()
            left = _Binary(token.position, token.text, left, self._unary())
        return left

    def _unary(self) -> _Node:
        self._descend()
        try:
            if self._at_operator("-"):
                token = self._advance()
                return _Negate(token.position, self._unary())
            if self._at_keyword("try"):
                token = self._advance()
                body = self._unary()
                handler = None
                if self._at_keyword("catch"):
                    self._advance()
                    handler = self._unary()
                return _Try(token.position, body, handler)
            return self._postfix()
        finally:
            self._budget.ascend()

    def _postfix(self) -> _Node:
        start = self._peek()
        term = self._primary()
        suffixes: list[_Field | _Index | _Optional] = []
        while True:
            token = self._peek()
            if token.kind is _Kind.FIELD:
                self._advance()
                suffixes.append(_Field(token.position, token.text))
                continue
            if token.kind is _Kind.OPERATOR and token.text == "[":
                self._advance()
                if self._at_operator("]"):
                    self._advance()
                    suffixes.append(_Index(token.position, None))
                    continue
                key = self._pipe()
                self._expect_operator("]")
                suffixes.append(_Index(token.position, key))
                continue
            if token.kind is _Kind.OPERATOR and token.text == "?":
                self._advance()
                suffixes.append(_Optional(token.position))
                continue
            break
        node = _Projection(start.position, term, tuple(suffixes)) if suffixes else term
        if self._at_keyword("as"):
            token = self._advance()
            variable = self._peek()
            if variable.kind is not _Kind.VARIABLE:
                raise self._refuse(variable, "condition binds to something other than a plain variable name")
            self._advance()
            self._expect_operator("|")
            return _Bind(token.position, node, variable.text, self._pipe())
        return node

    def _primary(self) -> _Node:
        token = self._peek()
        if token.kind is _Kind.FIELD:
            # The suffix loop consumes the field itself; a leading field is a projection
            # of the value flowing in.
            return _Root(token.position)
        if token.kind is _Kind.STRING:
            self._advance()
            return _String(
                token.position, token.text, tuple(self._interpolation(part) for part in token.interpolations)
            )
        if token.kind is _Kind.NUMBER:
            self._advance()
            return _Constant(token.position)
        if token.kind is _Kind.VARIABLE:
            self._advance()
            named = _NAMED_VARIABLE_REFUSALS.get(token.text)
            if named is not None:
                raise self._refuse(token, f"condition uses '${token.text}', which {named}")
            return _Variable(token.position, token.text)
        if token.kind is _Kind.FORMAT:
            raise self._refuse(
                token, f"condition uses the format string {token.text!r}, which serializes its whole input"
            )
        if token.kind is _Kind.IDENT:
            return self._identifier()
        if token.kind is _Kind.OPERATOR:
            return self._operator_term()
        raise self._refuse(token, "condition ends before the expression is complete")

    def _operator_term(self) -> _Node:
        token = self._peek()
        if token.text == ".":
            self._advance()
            return _Root(token.position)
        if token.text == "..":
            raise self._refuse(
                token, "condition uses recursive descent '..', which reads every value in the auth context"
            )
        if token.text == "(":
            self._advance()
            node = self._pipe()
            self._expect_operator(")")
            return node
        if token.text == "[":
            self._advance()
            if self._at_operator("]"):
                self._advance()
                return _ArrayConstruction(token.position, None)
            element = self._pipe()
            self._expect_operator("]")
            return _ArrayConstruction(token.position, element)
        if token.text == "{":
            return self._object()
        raise self._refuse(token, f"condition uses the unsupported operator {token.text!r}")

    def _object(self) -> _Node:
        token = self._expect_operator("{")
        values: list[_Node] = []
        if not self._at_operator("}"):
            while True:
                key = self._peek()
                is_static_key = key.kind is _Kind.IDENT or (key.kind is _Kind.STRING and not key.interpolations)
                if not is_static_key:
                    raise self._refuse(key, "condition builds an object under a key that is not a static name")
                self._advance()
                self._expect_operator(":")
                values.append(self._alternative())
                if not self._at_operator(","):
                    break
                self._advance()
        self._expect_operator("}")
        return _ObjectConstruction(token.position, tuple(values))

    def _identifier(self) -> _Node:
        token = self._advance()
        if token.text in _KEYWORD_LITERALS:
            return _Constant(token.position)
        named = _NAMED_REFUSALS.get(token.text)
        if named is not None:
            raise self._refuse(token, f"condition uses {token.text!r}, which {named}")
        arguments: list[_Node] = []
        if self._at_operator("("):
            self._advance()
            while True:
                arguments.append(self._pipe())
                if not self._at_operator(";"):
                    break
                self._advance()
            self._expect_operator(")")
        arities = _BUILTIN_ARITIES.get(token.text)
        if arities is None:
            raise self._refuse(
                token,
                f"condition uses the builtin {token.text!r}, which is outside the set a background execution can be "
                "authorized against",
            )
        if len(arguments) not in arities:
            raise self._refuse(token, f"condition calls the builtin {token.text!r} with {len(arguments)} arguments")
        return _Call(token.position, token.text, tuple(arguments))

    def _interpolation(self, interpolation: _Interpolation) -> _Node:
        """The body of a ``\\(...)`` hole, parsed as the jq code it is — under the
        ENCLOSING scan's budget, so its tokens are spent from the same allowance and its
        nesting continues from the depth the hole sits at."""
        self._budget.descend(self._text, interpolation.start)
        try:
            tokens = _lex(self._text, self._budget, interpolation.start, interpolation.end)
            return _Parser(self._text, tokens, self._budget).parse()
        finally:
            self._budget.ascend()


class _Taint(Enum):
    """What a value may carry.

    ``ROOT`` is the whole auth context — the value whose ``identity`` field a fire
    cannot present. ``SAFE`` is a value derived from it in a way that provably cannot
    depend on those claims.
    """

    ROOT = "root"
    SAFE = "safe"


@dataclass(frozen=True)
class _Scope:
    """The taint environment an expression is analyzed in: what ``.`` carries, and what
    each ``as``-bound variable carries."""

    subject: _Taint
    variables: Mapping[str, _Taint] = field(default_factory=dict)


class _TaintAnalysis:
    """The information-flow rule over a parsed condition.

    One rule decides the whole property: a ``ROOT`` value may only be PROJECTED by a
    static field name. Reaching a builtin, a comparison, an arithmetic operator, a
    constructor, an index expression, a string interpolation or the condition's own
    result is refused, because each of those turns the context into a value the
    condition's outcome depends on — and a fire cannot reproduce that context.
    """

    def __init__(self, condition_text: str) -> None:
        self._text = condition_text

    def assert_safe(self, node: _Node) -> None:
        """Assert that the condition's RESULT does not depend on the identity claims."""
        self._require_safe(self._taint(node, _Scope(_Taint.ROOT)), node.position, "the condition's result")

    def _require_safe(self, taint: _Taint, position: int, consumer: str) -> None:
        if taint is _Taint.ROOT:
            raise _refusal(
                self._text,
                position,
                f"condition lets the whole auth context reach {consumer}; a background execution presents no token, "
                f"so the context may only be read through a statically named field (and {_OWNER_REFERENCE!r} is the "
                "only readable identity claim)",
            )

    def _taint(self, node: _Node, scope: _Scope) -> _Taint:
        if isinstance(node, _Root):
            return scope.subject
        if isinstance(node, _Constant):
            return _Taint.SAFE
        if isinstance(node, _String):
            for part in node.interpolations:
                self._require_safe(self._taint(part, scope), part.position, "a string interpolation")
            return _Taint.SAFE
        if isinstance(node, _Variable):
            taint = scope.variables.get(node.name)
            if taint is None:
                raise _refusal(self._text, node.position, f"condition reads the unbound variable '${node.name}'")
            return taint
        if isinstance(node, _Projection):
            return self._projection(node, scope)
        if isinstance(node, _Call):
            return self._call(node, scope)
        if isinstance(node, _Binary):
            return self._binary(node, scope)
        if isinstance(node, _Negate):
            self._require_safe(self._taint(node.operand, scope), node.position, "arithmetic negation")
            return _Taint.SAFE
        if isinstance(node, _Bind):
            variables = {**scope.variables, node.variable: self._taint(node.source, scope)}
            return self._taint(node.body, _Scope(scope.subject, variables))
        if isinstance(node, _Try):
            return self._try(node, scope)
        if isinstance(node, _ObjectConstruction):
            for value in node.values:
                self._require_safe(self._taint(value, scope), value.position, "an object being constructed")
            return _Taint.SAFE
        if isinstance(node, _ArrayConstruction):
            if node.element is not None:
                self._require_safe(self._taint(node.element, scope), node.element.position, "an array being collected")
            return _Taint.SAFE
        raise _refusal(self._text, node.position, "condition uses a construct the token-free scan cannot decide")

    def _projection(self, node: _Projection, scope: _Scope) -> _Taint:
        """The taint of a suffix chain — the ONE place a ``ROOT`` value is allowed to be
        consumed, and only by a static field name."""
        taint = self._taint(node.source, scope)
        suffixes = node.suffixes
        position = 0
        while position < len(suffixes):
            suffix = suffixes[position]
            if taint is _Taint.SAFE:
                # Every field of a safe value is safe, and so is every element of it —
                # but WHICH element is selected must not depend on the context either.
                if isinstance(suffix, _Index) and suffix.key is not None:
                    self._require_safe(self._taint(suffix.key, scope), suffix.position, "an index expression")
                position += 1
                continue
            name = _static_name(suffix)
            if name is None:
                raise _refusal(
                    self._text,
                    suffix.position,
                    "condition reaches into the auth context without naming a field; only a statically named field "
                    "can be read at a fire",
                )
            if name == _IDENTITY_FIELD:
                remaining = suffixes[position + 1 :]
                if len(remaining) != 1 or _static_name(remaining[0]) != OWNER_USER_ID_CLAIM:
                    raise _refusal(
                        self._text,
                        suffix.position,
                        f"condition reads an identity claim beyond {_OWNER_REFERENCE!r}, which is the only one a "
                        "background execution can present",
                    )
                return _Taint.SAFE
            taint = _Taint.SAFE
            position += 1
        return taint

    def _call(self, node: _Call, scope: _Scope) -> _Taint:
        """A builtin reads its input and its arguments, so both must be safe — and its
        result is then derived from safe values only."""
        self._require_safe(scope.subject, node.position, f"the builtin {node.name!r}")
        # jq evaluates every argument against the call's own input, and the filter
        # arguments of ``map``/``any``/``all`` against an element of it. That input is
        # safe by the check above, so each argument is analyzed under a safe subject.
        arguments = _Scope(_Taint.SAFE, scope.variables)
        for argument in node.arguments:
            self._require_safe(
                self._taint(argument, arguments), argument.position, f"an argument of the builtin {node.name!r}"
            )
        return _Taint.SAFE

    def _binary(self, node: _Binary, scope: _Scope) -> _Taint:
        if node.operator == "|":
            piped = _Scope(self._taint(node.left, scope), scope.variables)
            return self._taint(node.right, piped)
        if node.operator in (",", "//"):
            # Both sides can reach the output, so the stream carries the taint of either.
            left = self._taint(node.left, scope)
            right = self._taint(node.right, scope)
            return _Taint.SAFE if left is _Taint.SAFE and right is _Taint.SAFE else _Taint.ROOT
        consumer = f"the {node.operator!r} operator"
        self._require_safe(self._taint(node.left, scope), node.left.position, consumer)
        self._require_safe(self._taint(node.right, scope), node.right.position, consumer)
        return _Taint.SAFE

    def _try(self, node: _Try, scope: _Scope) -> _Taint:
        body = self._taint(node.body, scope)
        if node.handler is None:
            return body
        # A jq error message quotes the values that produced it, so the handler's input
        # is treated as the context itself.
        handler = self._taint(node.handler, _Scope(_Taint.ROOT, scope.variables))
        return _Taint.SAFE if body is _Taint.SAFE and handler is _Taint.SAFE else _Taint.ROOT


def _static_name(suffix: _Field | _Index | _Optional) -> str | None:
    """The field name a suffix statically projects, or ``None`` when it projects
    something only evaluation could name (``[]``, a computed or interpolated index, an
    error-suppressing ``?``)."""
    if isinstance(suffix, _Field):
        return suffix.name
    if isinstance(suffix, _Index) and isinstance(suffix.key, _String) and not suffix.key.interpolations:
        return suffix.key.value
    return None


def assert_token_free_evaluable(condition_text: str) -> None:
    """Assert that ``condition_text`` can be evaluated for a background execution.

    Returns on an evaluable condition; raises :class:`TokenFreeConditionError` naming the
    offending construct and its offset otherwise, including text that does not compile.
    See the module docstring for the rule.

    Gate ORDER is load-bearing: compile first, so non-jq text is named as such rather than
    refused for its characters; then the raw-text source-shape gate, so everything below
    it works over a condition whose token boundaries jq cannot read differently.
    """
    try:
        get_compiled_jq(condition_text)
    except ValueError as exc:
        raise TokenFreeConditionError(
            f"condition does not compile as a jq program ({exc}), so it cannot be shown evaluable for a background "
            "execution"
        ) from exc
    _assert_source_shape(condition_text)
    budget = _Budget()
    tokens = _lex(condition_text, budget)
    program = _Parser(condition_text, tokens, budget).parse()
    _TaintAnalysis(condition_text).assert_safe(program)
