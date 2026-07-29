"""Which jq policy conditions a TOKENLESS background execution can be authorized against.

A fire carries the caller's claims reduced to the one readable from the store (the key's
owner), so ``identity`` is the SOLE context field a condition may not depend on. Depending
on it is a fail-OPEN, not merely an unevaluable: ``.identity.X != v`` evaluates TRUE
against the absent claim and ALLOWS a fire a real request would have denied.

The root value ``.`` carries the whole context, so ``. | tojson``, ``"\\(.)"`` or
``[recurse]`` leak claims without spelling ``identity``; the scan therefore allowlists a
minimal grammar and refuses any flow of the context into anything but a static field
projection. Unknown defaults to refuse. The source-shape gate on the raw text is what makes
the scan's lexer agreeing with jq's unnecessary, so it is pinned from both sides: refusals
per excluded character class, plus a differential check running every ACCEPTED condition
through the real linked libjq under a full and a reduced claim set.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from tai42_kit.utils.data.jq_util import get_compiled_jq

from tai42_skeleton.access_control.roles import EDITOR_JQ, VIEWER_JQ
from tai42_skeleton.authz.token_free import (
    _MAX_NESTING_DEPTH,
    _MAX_TOKENS,
    TokenFreeConditionError,
    _Budget,
    _lex,
    assert_token_free_evaluable,
)

# Conditions a fire CAN evaluate: every context field but ``identity``, plus the exact
# owner reference — the one identity claim readable from the execution key's policy.
_EVALUABLE = [
    # The seeded base-tier role conditions: the common case, so they must bind freely.
    EDITOR_JQ,
    VIEWER_JQ,
    # The owner reference, the only readable identity claim.
    '.identity.owner_user_id == "alice"',
    '.identity.owner_user_id != "banned"',
    '."identity"."owner_user_id" == "alice"',
    '.["identity"]["owner_user_id"] == "alice"',
    ".identity.owner_user_id | length > 0",
    ".identity.owner_user_id as $owner | .sub == $owner",
    # The context fields a fire populates exactly as a request does.
    '.sub == "k1"',
    '.sub != "banned"',
    '.sub | startswith("svc-")',
    '.scopes[0] == "a"',
    '.scopes[] == "hooks"',
    '.policy.tier == "gold"',
    ".policy.tier == 1",
    '.context.plan == "pro"',
    '.request.method == "POST"',
    '.request.path | startswith("/api")',
    '.request.path | startswith("/api/hooks")',
    '.request.path | endswith("/wipe") | not',
    '.request.path | ascii_downcase | test("^/api")',
    '.request.path | test("api"; "i")',
    '.request.method | IN("GET","HEAD","OPTIONS")',
    ".system.time > 0",
    ".system.hour < 18",
    ".sub != null",
    "true",
    # A statically named index cannot reach outside the subpath it names.
    '.["policy"].tier == "gold"',
    '.policy["tier"] == "gold"',
    "(.scopes | length) == 5",
    '.scopes | map(ascii_downcase) | all(. == "a")',
    '.scopes | any(. == "hooks")',
    '.scopes | contains(["hooks"])',
    # Value-level operators over safe values.
    ".policy.limit + 1 > 2",
    "-1 < .policy.tier",
    '(.policy.tier // "free") == "gold"',
    "try (.policy.tier == 1) catch false",
    "{tier: .policy.tier} | length == 1",
    "[.sub] | length == 1",
    '.request.path == "/api/\\(.request.method)"',
    ".sub as $s | .policy.owner == $s",
    # Inside a builtin argument ``.`` is the builtin's own input, not the auth context.
    '.policy | test("\\(.identity)")',
]

# Conditions a fire CANNOT evaluate — each names, or silently reaches, an identity claim
# beyond the owner, or is a construct the scan cannot decide.
_REFUSED = [
    # The direct forms.
    '.identity.description == "ops"',
    ".identity.mfa",
    '.identity.email | endswith("@corp")',
    ".identity",
    '.identity["team"]',
    ".identity | keys",
    '.identity.department != "eng"',
    '.identity | has("department") | not',
    # The owner reference with anything applied to it is no longer the owner claim.
    ".identity.owner_user_id_extra",
    ".identity.owner_user_id.sub",
    ".identity.owner_user_idx",
    # A NEGATIVE predicate — the fail-open case the rule exists to close.
    ".identity.suspended != true",
    # SERIALIZING the whole context reaches every claim without naming one.
    '(. | tojson) | test("engineering")',
    '(. | tojson | test("engineering")) | not',
    '(. | @json) | test("x")',
    '(. | tostring) | test("x")',
    '(. | @text) | test("x")',
    '(. | @base64) | test("x")',
    '(. | tojson | ascii_downcase) | test("eng")',
    '(.|tojson) | splits("a") | length>0',
    'tostring | test("suspended") | not',
    '@base64 | test("x")',
    ". | tojson",
    # ENUMERATING the context reaches the same values by key rather than by name.
    "..",
    '.. | select(. == "x")',
    ".identity.owner_user_id as $o | .. | select(. == $o)",
    '[recurse] | tojson | test("eng")',
    'getpath(["identity","x"])',
    "to_entries | map(.key)",
    'with_entries(select(.key != "identity"))',
    "keys | length > 0",
    'walk(if type == "string" then ascii_downcase else . end)',
    "tostream | length",
    "paths | length",
    "leaf_paths | length",
    # The root value reaching a builtin, constructor, comparison or interpolation.
    "map(.) | length > 0",
    "{a:1} | inside(.)",
    '"\\(.)" | test("eng")',
    "{a: .} | length == 1",
    "[.] | length == 1",
    ". == {}",
    ". | length",
    ".",
    "try . catch .",
    '. as $c | "\\($c)" | test("x")',
    # Indexing the ROOT with anything but a static field name reaches ``identity``.
    ".[]",
    ".[$field]",
    '.["ident" + "ity"].team',
    '(.)[("iden"+"tity")].suspended != true',
    "(.)[] | length",
    '. as $c | $c["ident" + "ity"].team',
    ". as $c | $c[$k]",
    ". as $c | $c[]",
    '. as {("iden"+"tity"): $i} | ($i.suspended // false) | not',
    # A safe value indexed by an identity claim still decides on that claim.
    ".policy[.identity.dept]",
    # Renaming defeats any per-builtin reasoning, so defining a function is refused.
    "def leak: tojson; . | leak",
    'def d: .identity.dept; d == "eng"',
    # Reading outside the auth context, and non-determinism: both break bind-time ≈ fire-time.
    "$ENV.HOME",
    "env.HOME",
    "now > 0",
    "localtime | length",
    "input | length",
    "inputs | length",
    "$__loc__ | length",
    # Not a jq program at all — unrendered or malformed text.
    "",
    "   ",
    "this is ( not jq",
    '.request.path | startswith("/api"',
]

# Constructs the scan cannot decide: refused whether or not they could leak. Each has a
# repair inside the grammar, noted beside it.
_REFUSED_OUTSIDE_THE_MINIMAL_GRAMMAR = [
    # → ``.scopes | any(. == "hooks")``
    '.scopes | index("hooks")',
    # → ``any(...)`` / ``all(...)`` over a named subpath
    'any(.scopes[]; . == "hooks")',
    # → ``.policy.tier == "gold" or .policy.tier == "silver"``
    'if .policy.tier == "gold" then true else false end',
    # → ``.scopes | any(. == "hooks")``
    '.scopes | map(select(. == "hooks")) | length > 0',
    # → ``.request.path | startswith("/api")``
    '.request.path[0:4] == "/api"',
    # → ``.policy.tier == "gold"``
    '.policy | has("tier")',
]

# Refused by the SOURCE-SHAPE gate on the raw text, before a token is read. The first four
# are working fail-opens: a ``#`` comment ending in a backslash continues across the newline
# for jq, so jq runs an identity predicate the scan would read as a safe primary.
_REFUSED_BY_THE_SOURCE_SHAPE_GATE = [
    '# \\\n"a"\n.identity.suspended != true',
    "# \\\n1\n.identity.suspended != true",
    "# \\\ntrue\n.identity.suspended != true",
    '.sub == "k" and (# \\\n"a"\n.identity.suspended != true)',
    # The gate keys on the CHARACTER, not the comment's extent — deciding the extent is
    # the thing it refuses to do.
    '.sub == "a" # tojson recurse\n',
    '.sub == "a#b"',
    '.request.path | startswith("/api") # trailing',
    # Line structure of every spelling.
    '.sub ==\n"a"',
    '.sub ==\r\n"a"',
    '.sub == "a\\nb" and\n.policy.tier == 1',
    # Whitespace/control characters. NUL is the sharpest: libjq reads a C string and stops
    # there, so everything after it is text jq never compiles and the scan would.
    '.sub\t== "a"',
    '.sub == "a\x0bb"',
    '.sub == "a"\x00 and .identity.suspended != true',
    '.sub == "a\x7fb"',
    # Non-ASCII anywhere: a Cyrillic homoglyph for ``identity``'s 'y', a zero-width joiner.
    # Written as Python escapes so this file stays ASCII; the gate sees the decoded chars.
    '.sub == "caf\u00e9"',
    '."identit\u0443".owner_user_id == "alice"',
    '.sub == "a\u200db"',
]


@pytest.mark.parametrize("condition", _EVALUABLE)
def test_a_token_free_condition_is_accepted(condition: str) -> None:
    assert_token_free_evaluable(condition)  # no raise


@pytest.mark.parametrize("condition", _REFUSED)
def test_a_condition_needing_absent_claims_is_refused(condition: str) -> None:
    with pytest.raises(TokenFreeConditionError):
        assert_token_free_evaluable(condition)


@pytest.mark.parametrize("condition", _REFUSED_OUTSIDE_THE_MINIMAL_GRAMMAR)
def test_a_construct_outside_the_grammar_is_refused(condition: str) -> None:
    # Unknown defaults to REFUSE: over-refusal is repairable, under-refusal is a silent allow.
    with pytest.raises(TokenFreeConditionError):
        assert_token_free_evaluable(condition)


@pytest.mark.parametrize("condition", [EDITOR_JQ, VIEWER_JQ])
def test_the_seeded_role_conditions_bind_freely(condition: str) -> None:
    # Every non-admin role holder's policy carries these, so their keys must be bindable.
    # Both embed ``"/api/auth/api-keys"``, which lexes as a string and not the ``keys`` builtin.
    assert_token_free_evaluable(condition)


@pytest.mark.parametrize(("condition", "tokens", "depth"), [(EDITOR_JQ, 84, 16), (VIEWER_JQ, 135, 22)])
def test_the_seeded_role_conditions_cost_what_the_budget_comment_says(condition: str, tokens: int, depth: int) -> None:
    # The seeded conditions are the platform's real worst case (VIEWER_JQ spends 53% of the
    # token allowance, 69% of the depth). An edit eating the margin must fail here rather
    # than as a PermissionDenied on every fire under a viewer-role key.
    budget = _Budget()
    peak = 0
    descend = _Budget.descend

    def _record_depth(self: _Budget, text: str, position: int) -> None:
        nonlocal peak
        descend(self, text, position)
        peak = max(peak, self.depth)

    with mock.patch.object(_Budget, "descend", _record_depth):
        assert_token_free_evaluable(condition)
    _lex(condition, budget)
    assert (budget.tokens, peak) == (tokens, depth)
    assert budget.tokens < _MAX_TOKENS
    assert peak < _MAX_NESTING_DEPTH


@pytest.mark.parametrize("condition", _REFUSED_BY_THE_SOURCE_SHAPE_GATE)
def test_a_condition_outside_the_accepted_source_shape_is_refused_before_it_is_lexed(condition: str) -> None:
    # The gate reads raw characters only, so it holds where the scan's lexer and jq's would
    # part company. The message is asserted to be the GATE's, not the parser's: a case that
    # started dying at the parser would mean the gate had stopped covering it.
    with pytest.raises(TokenFreeConditionError, match=r"comment|single line|tab|control character|non-ASCII"):
        assert_token_free_evaluable(condition)


def test_a_newline_in_a_condition_is_refused_at_the_bind_gate() -> None:
    # The single-line requirement, pinned on its own: a newline is the seam a ``#`` comment
    # can be continued across, so every spelling of it is refused before a token is lexed.
    for newline in ("\n", "\r\n", "\r"):
        with pytest.raises(TokenFreeConditionError, match=r"single line"):
            assert_token_free_evaluable(f'.sub =={newline}"a"')

    # The exact fail-open the gate closes: jq runs the continued line's identity predicate,
    # while a scan that ended the comment at the newline would accept.
    with pytest.raises(TokenFreeConditionError, match=r"comment|single line"):
        assert_token_free_evaluable('# \\\n"a"\n.identity.suspended != true')

    # Over-refusal guard: the seeded single-line conditions must still bind.
    assert_token_free_evaluable(EDITOR_JQ)
    assert_token_free_evaluable(VIEWER_JQ)


def test_the_compile_gate_answers_before_the_source_shape_gate() -> None:
    # An unrendered template must not read as a formatting complaint.
    with pytest.raises(TokenFreeConditionError, match="does not compile"):
        assert_token_free_evaluable("# \\\n{{ tier }}\n(")


# Builtin names inside string literals are text, and a field of a named subpath is a field.
_WORDS_INSIDE_STRING_LITERALS = [
    '.request.path == "/api/auth/api-keys"',
    '.request.path | startswith("/api/auth/api-keys")',
    ".policy.paths | length > 0",
    '.policy.keys == "none"',
    '.sub == "walk to_entries getpath .."',
]


@pytest.mark.parametrize("condition", _WORDS_INSIDE_STRING_LITERALS)
def test_a_word_inside_a_string_literal_is_a_string(condition: str) -> None:
    assert_token_free_evaluable(condition)


def test_a_string_interpolation_is_analyzed_as_code() -> None:
    # ``\\(...)`` re-enters CODE context, so the same rule applies inside the hole.
    assert_token_free_evaluable('.request.path == "/api/\\(.request.method)"')
    for condition in ('"\\(.)" | test("eng")', '"\\(.identity.dept)" == "eng"', '.sub == "\\(.identity)"'):
        with pytest.raises(TokenFreeConditionError):
            assert_token_free_evaluable(condition)


def test_a_condition_that_does_not_compile_is_refused_loudly() -> None:
    # The compile gate runs first: an unrendered or truncated condition must never pass.
    for condition in ("", "{{ tier }}", ".request.path | startswith(", "|"):
        with pytest.raises(TokenFreeConditionError, match="does not compile"):
            assert_token_free_evaluable(condition)


def test_the_refusal_names_the_offending_reference() -> None:
    # The refusal quotes the text it refused on, so the author knows which clause tripped it.
    with pytest.raises(TokenFreeConditionError) as ei:
        assert_token_free_evaluable('.sub == "svc" and .identity.description == "ops"')
    assert ".identity.description" in str(ei.value)
    assert "owner_user_id" in str(ei.value)

    with pytest.raises(TokenFreeConditionError) as ei:
        assert_token_free_evaluable('.sub == "svc" and (. | tojson | test("eng"))')
    assert "tojson" in str(ei.value)

    with pytest.raises(TokenFreeConditionError) as ei:
        assert_token_free_evaluable('.sub == "svc" and ("\\(.)" | test("eng"))')
    assert "interpolation" in str(ei.value)


def test_a_negative_identity_predicate_is_refused_not_merely_unevaluable() -> None:
    # A negative predicate evaluates TRUE against the absent claim and ALLOWS; the positive
    # form fails closed yet is refused with it — over-refusing, never under-refusing.
    for condition in (".identity.suspended != true", ".identity.suspended == true"):
        with pytest.raises(TokenFreeConditionError):
            assert_token_free_evaluable(condition)


def test_the_serialized_context_cannot_be_pattern_matched() -> None:
    # A real request's claims serialize to include the string and DENY; a fire's reduced
    # identity does not and ALLOWS. Every spelling of "serialize the input" is refused.
    for builtin in ("tojson", "tostring", "@json", "@text", "@base64", "@csv", "@uri", "@sh"):
        with pytest.raises(TokenFreeConditionError):
            assert_token_free_evaluable(f'(. | {builtin} | test("engineering")) | not')


# The rule follows the VALUE, not the spelling: a rebound ``.`` inside a filter is a safe
# element, so its ``identity`` field is not the auth context's.
_REBOUND_EVALUABLE = [
    '. as $c | $c.request.path | startswith("/api")',
    '. as $c | $c.identity.owner_user_id == "alice"',
    '.policy.items | map(.identity) | all(. == "x")',
]


@pytest.mark.parametrize("condition", _REBOUND_EVALUABLE)
def test_a_rebound_input_is_analyzed_as_the_value_it_carries(condition: str) -> None:
    assert_token_free_evaluable(condition)


def test_rebinding_the_input_moves_the_rule_with_it() -> None:
    # Binding the context to a variable carries the refusal onto the variable.
    for condition in (
        ". as $c | $c.identity.dept",
        '. as $c | $c | tojson | test("eng")',
        ". as $c | [$c] | length",
        '. as $c | .sub == "\\($c.identity.dept)"',
        ". as $c | .scopes | map($c) | length == 1",
    ):
        with pytest.raises(TokenFreeConditionError):
            assert_token_free_evaluable(condition)


def test_a_field_name_is_compared_after_its_escapes_are_decoded() -> None:
    # A statically named index IS the field it decodes to, so an escape hides nothing.
    assert_token_free_evaluable('."identity"."owner_user_id" == "alice"')
    assert_token_free_evaluable('.policy."a\\/b" == 1')
    for condition in ('.["identity"].team', '."identity".dept'):
        with pytest.raises(TokenFreeConditionError):
            assert_token_free_evaluable(condition)


def test_a_unicode_escape_is_refused_rather_than_decoded() -> None:
    # ``\\uXXXX`` is the one escape that conjures a character the source-shape gate never saw
    # in the raw text, so refusing it keeps the gate's guarantee about raw characters a
    # guarantee about the values the scan reasons over.
    for condition in (
        '."\\u0069dentity"."owner_user_id" == "alice"',
        '.["\\u0069dentity"].team',
        '.sub == "\\u0041"',
    ):
        with pytest.raises(TokenFreeConditionError, match="unsupported escape"):
            assert_token_free_evaluable(condition)


def test_every_code_position_inside_a_string_is_analyzed() -> None:
    # Interpolation nests, and an object key is a place code can hide: both are parsed as code.
    for condition in ('"\\("\\(.)")" | test("x")', '{"\\(.identity.dept)": 1} | length == 1'):
        with pytest.raises(TokenFreeConditionError):
            assert_token_free_evaluable(condition)


# Past the LENGTH bound: each compiles, so each must be refused for being long rather than
# by exhausting the interpreter stack. The last two are long only inside a ``\\(...)`` hole,
# which draws from the same allowance as the text around it.
_PAST_THE_LENGTH_BOUND = [
    " | ".join([".sub"] * 900),
    "(" * 400 + ".sub" + ")" * 400,
    "+".join(["1"] * 900),
    "-" * 900 + "1",
    '"\\(' + "+".join(["1"] * 1000) + ')" == "x"',
    '"' + "\\(1+1)" * 2000 + '" | length > 0',
]

# Past the NESTING bound: each is short enough to pass the length bound, so only the depth
# counter can refuse it. The last two nest inside interpolation, where the lexer recurses.
_PAST_THE_NESTING_BOUND = [
    "(" * 100 + ".sub" + ")" * 100,
    "try " * 100 + ".sub",
    ". as $a | " * 40 + ".sub",
    '"' + '\\("' * 200 + "x" + '")' * 200 + '"' + ' == "x"',
    '.sub == "' + '\\("' * 60 + "x" + '")' * 60 + '"',
]


@pytest.mark.parametrize("condition", _PAST_THE_LENGTH_BOUND)
def test_an_overlong_condition_is_refused_rather_than_overflowing_the_stack(condition: str) -> None:
    # A caller can author and bind its own key's condition, so the answer must be the
    # ordinary refusal, never a RecursionError escaping the bind door as a 500. The message
    # is asserted so a case cannot silently migrate onto the other bound.
    with pytest.raises(TokenFreeConditionError, match="longer than"):
        assert_token_free_evaluable(condition)


@pytest.mark.parametrize("condition", _PAST_THE_NESTING_BOUND)
def test_a_deeply_nested_condition_is_refused_rather_than_overflowing_the_stack(condition: str) -> None:
    # The scan recurses on nesting in the parser AND the lexer, and the depth counter is
    # shared by both, so a hole in a string cannot restart it at zero.
    with pytest.raises(TokenFreeConditionError, match="nests more than"):
        assert_token_free_evaluable(condition)


# Over-refusal guard for the bounds: ordinary policy expressions, deeply parenthesised or
# interpolated, must still pass.
_WITHIN_THE_BOUNDS = [
    '((((.sub != "banned")))) and (.identity.owner_user_id == "alice" or (.scopes | any(. == "hooks")))',
    '.request.path == "/api/\\(.request.method)/\\(.policy.tier)"',
    '.sub == "\\("k-\\(.policy.tier)")"',
]


@pytest.mark.parametrize("condition", _WITHIN_THE_BOUNDS)
def test_the_bounds_sit_far_above_a_real_condition(condition: str) -> None:
    assert_token_free_evaluable(condition)


# Every condition this module asserts the scan ACCEPTS, so the differential check below
# cannot fall behind the suite's evaluable set.
_ACCEPTED_CORPUS = [
    *_EVALUABLE,
    *_WORDS_INSIDE_STRING_LITERALS,
    *_REBOUND_EVALUABLE,
    *_WITHIN_THE_BOUNDS,
    '."identity"."owner_user_id" == "alice"',
    '.policy."a\\/b" == 1',
]

# The two contexts a condition is enforced over. They must differ in ``identity`` and
# nothing else, so any behavioural difference between them IS a dependence on the claims.
_FULL_CLAIMS = {
    "sub": "svc-k1",
    "scopes": ["hooks", "a", "b", "c", "d"],
    "policy": {
        "tier": "gold",
        "limit": 5,
        "owner": "svc-k1",
        "paths": ["/api"],
        "keys": "none",
        "items": [{"identity": "x"}],
        "a/b": 1,
    },
    "context": {"plan": "pro"},
    "request": {"method": "GET", "path": "/api/auth/api-keys"},
    "system": {"time": 1700000000, "hour": 9},
    "identity": {
        "owner_user_id": "alice",
        "email": "alice@corp",
        "department": "engineering",
        "description": "ops",
        "suspended": True,
        "mfa": True,
        "team": "core",
    },
}
_REDUCED_CLAIMS = {**_FULL_CLAIMS, "identity": {"owner_user_id": "alice"}}


def _jq_outcome(condition: str, context: dict[str, Any]) -> object:
    """What real libjq does with ``condition`` over ``context``: the whole output stream, or
    the error. Errors count — a jq error message quotes the values that produced it."""
    try:
        return ("values", get_compiled_jq(condition).input(context).all())
    except Exception as exc:
        return ("error", f"{type(exc).__name__}: {exc}")


@pytest.mark.parametrize("condition", _ACCEPTED_CORPUS)
def test_an_accepted_condition_decides_identically_under_the_reduced_claim_set(condition: str) -> None:
    # Run against the real libjq the enforcer evaluates with, not the scan's reading of the
    # text: sampling cannot prove independence, but a disagreement proves a defect.
    assert_token_free_evaluable(condition)  # no raise: the corpus is the ACCEPTED set
    assert _jq_outcome(condition, _FULL_CLAIMS) == _jq_outcome(condition, _REDUCED_CLAIMS)


def test_the_differential_check_would_catch_a_condition_that_decides_on_a_claim() -> None:
    # A refused condition must come back DIFFERENT, or a green differential run would mean
    # only that the two contexts are too alike to tell anything apart.
    for condition in (".identity.suspended != true", '.identity.department == "engineering"'):
        assert _jq_outcome(condition, _FULL_CLAIMS) != _jq_outcome(condition, _REDUCED_CLAIMS)
