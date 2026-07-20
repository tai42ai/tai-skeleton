"""Fixture extensions exercising apply-site kind enforcement against the base
tool ``shout(text: str)`` in :mod:`tests.app._fixtures.tools_b`.

Covers every branch of ``ToolBinding._enforce_extension_schema``: wrapper
schema-transparency (with ``reserved_params``), transformer concrete-signature
requirement, and the backend no-rule case. Several fixtures use a
``*args/**kwargs`` IMPLEMENTATION body behind a concrete presented signature —
these pin that the apply site inspects the presented ``__signature__``, never
the impl body.
"""

import functools

from makefun import create_function
from tai_contract.app import tai_app
from tai_contract.extensions import ExtensionKind

# --- wrappers (preserves_schema) --------------------------------------------


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="argswrap")
def argswrap(func, name, desc, config=None):
    """Preserve schema via ``functools.wraps`` behind a ``*args/**kwargs`` body;
    upper-case the string result. Presented signature stays ``func``'s."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result

    variant.__name__ = f"{name}_argswrap"
    variant.__qualname__ = variant.__name__
    return variant


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="renamep")
def renamep(func, name, desc, config=None):
    """Rename the input param ``text`` -> ``txt`` — a schema change a wrapper
    must not make."""

    def impl(*args, **kwargs):
        return func(*args, **kwargs)

    return create_function(f"{name}_renamep(txt: str)", impl, func_name=f"{name}_renamep")


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="cachereq")
def cachereq(func, name, desc, config=None):
    """Append a default-less reserved control param ``exp`` (lands in
    ``required`` too), mirroring cache's append shape."""

    def impl(*args, **kwargs):
        kwargs.pop("exp", None)
        return func(*args, **kwargs)

    return create_function(f"{name}_cachereq(text: str, exp: float)", impl, func_name=f"{name}_cachereq")


cachereq.reserved_params = frozenset({"exp"})


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="proxylike")
def proxylike(func, name, desc, config=None):
    """Inject a reserved list param, mirroring proxy/vpn's list-inject shape."""

    def impl(*args, **kwargs):
        kwargs.pop("proxies", None)
        return func(*args, **kwargs)

    return create_function(f"{name}_proxylike(text: str, proxies: list[str])", impl, func_name=f"{name}_proxylike")


proxylike.reserved_params = frozenset({"proxies"})


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="collidewrap")
def collidewrap(func, name, desc, config=None):
    """Declare a reserved name that ALSO exists on the input signature — the
    exclusion would mask a real change, so this must raise at apply time."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        return func(*args, **kwargs)

    variant.__name__ = f"{name}_collidewrap"
    variant.__qualname__ = variant.__name__
    return variant


collidewrap.reserved_params = frozenset({"text"})


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="driftreserved")
def driftreserved(func, name, desc, config=None):
    """Tolerated reserved ``exp`` PLUS an untolerated rename ``text`` -> ``txt``:
    the reserved subtraction must not hide the real drift, so this still raises."""

    def impl(*args, **kwargs):
        kwargs.pop("exp", None)
        return func(*args, **kwargs)

    return create_function(f"{name}_driftreserved(txt: str, exp: float)", impl, func_name=f"{name}_driftreserved")


driftreserved.reserved_params = frozenset({"exp"})


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="onlyreserved")
def onlyreserved(func, name, desc, config=None):
    """Append a default-less reserved param to a tool that has NO required
    params, so subtracting it empties the branch's ``required`` list — exercises
    the empty-``required`` normalization."""

    def impl(*args, **kwargs):
        kwargs.pop("exp", None)
        return func(*args, **kwargs)

    return create_function(f"{name}_onlyreserved(exp: float)", impl, func_name=f"{name}_onlyreserved")


onlyreserved.reserved_params = frozenset({"exp"})


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="marka")
def marka(func, name, desc, config=None):
    """Order marker: append ``|a`` to the string result; schema-preserving."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        return f"{func(*args, **kwargs)}|a"

    variant.__name__ = f"{name}_marka"
    variant.__qualname__ = variant.__name__
    return variant


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="localwrap", requires_body_locality=True)
def localwrap(func, name, desc, config=None):
    """A locality-requiring wrapper (its wrapper only works in the process that
    runs the tool body): schema-preserving, and the bind engine must place it
    INSIDE any execution-relocating (BACKEND) extension in a stacked combo."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        return func(*args, **kwargs)

    variant.__name__ = f"{name}_localwrap"
    variant.__qualname__ = variant.__name__
    return variant


@tai_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="markb")
def markb(func, name, desc, config=None):
    """Order marker: append ``|b`` to the string result; schema-preserving."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        return f"{func(*args, **kwargs)}|b"

    variant.__name__ = f"{name}_markb"
    variant.__qualname__ = variant.__name__
    return variant


# --- transformers (declares_schema) -----------------------------------------


@tai_app.extensions.extension(kind=ExtensionKind.TRANSFORMER, name="concretetf")
def concretetf(func, name, desc, config=None):
    """Present a concrete OWN schema via makefun behind a ``*args/**kwargs``
    body — the valid transformer shape."""

    def impl(*args, **kwargs):
        return kwargs

    return create_function(f"{name}_concretetf(a: int, b: str = 'z')", impl, func_name=f"{name}_concretetf")


@tai_app.extensions.extension(kind=ExtensionKind.TRANSFORMER, name="baretf")
def baretf(func, name, desc, config=None):
    """A raw passthrough presenting a bare ``(*args, **kwargs)`` signature — no
    concrete schema, so a transformer must reject it."""

    def variant(*args, **kwargs):
        return func(*args, **kwargs)

    variant.__name__ = f"{name}_baretf"
    variant.__qualname__ = variant.__name__
    return variant


@tai_app.extensions.extension(kind=ExtensionKind.TRANSFORMER, name="reservedtf")
def reservedtf(func, name, desc, config=None):
    """A concrete transformer carrying ``reserved_params`` — which is ignored on
    a non-wrapper kind (a transformer owns its whole schema)."""

    def impl(*args, **kwargs):
        return kwargs

    return create_function(f"{name}_reservedtf(a: int)", impl, func_name=f"{name}_reservedtf")


reservedtf.reserved_params = frozenset({"a"})


# --- backends (no schema rule) ----------------------------------------------


@tai_app.extensions.extension(kind=ExtensionKind.BACKEND, name="backendswap")
def backendswap(func, name, desc, config=None):
    """A single backend strategy that alters the schema — legal, because backend
    branches carry no schema rule."""

    def impl(*args, **kwargs):
        return func(*args, **kwargs)

    return create_function(f"{name}_backendswap(other: int)", impl, func_name=f"{name}_backendswap")


@tai_app.extensions.extension(kind=ExtensionKind.BACKEND, name="backendx")
def backendx(func, name, desc, config=None):
    @functools.wraps(func)
    def variant(*args, **kwargs):
        return func(*args, **kwargs)

    variant.__name__ = f"{name}_backendx"
    variant.__qualname__ = variant.__name__
    return variant


@tai_app.extensions.extension(kind=ExtensionKind.BACKEND, name="backendy")
def backendy(func, name, desc, config=None):
    @functools.wraps(func)
    def variant(*args, **kwargs):
        return func(*args, **kwargs)

    variant.__name__ = f"{name}_backendy"
    variant.__qualname__ = variant.__name__
    return variant
