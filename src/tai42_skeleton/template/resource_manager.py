import asyncio
import base64
import logging
import mimetypes
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from async_lru import alru_cache
from jinja2 import FunctionLoader, meta
from jinja2 import Template as JinjaTemplate
from jinja2.sandbox import SandboxedEnvironment
from jinja2schema import JSONSchemaDraft4Encoder, infer, to_json_schema
from tai42_contract.storage import Storage
from tai42_kit.clients import shutdown_all_clients
from tai42_kit.net import fetch_url

from tai42_skeleton.template.media import ContentPart, MediaBlock
from tai42_skeleton.template.path_guard import safe_template_path
from tai42_skeleton.template.settings import template_cache_settings

logger = logging.getLogger(__name__)


class TemplateNotFoundError(Exception):
    """A requested stored template does not exist.

    Raised by the read paths when a ``template_id`` resolves to no stored
    content, so an HTTP surface can map a missing template to 404 instead of
    leaking storage's ``FileNotFoundError`` as a 500. Genuine storage/transport
    failures are NOT this type and keep propagating.
    """


def _decode_data_uri(uri: str) -> tuple[bytes, str | None]:
    """Decode a ``data:`` URI into ``(bytes, mime)``.

    Parses the inline ``data:[<mediatype>][;base64],<payload>`` form — no network
    or storage access. A URI without the required comma separator is malformed and
    raises loudly.
    """
    header, sep, payload = uri[len("data:") :].partition(",")
    if not sep:
        raise ValueError(f"Malformed data URI (missing ','): {uri[:64]!r}")
    is_base64 = header.endswith(";base64")
    mediatype = header[: -len(";base64")] if is_base64 else header
    data = base64.b64decode(payload) if is_base64 else unquote_to_bytes(payload)
    mime = mediatype.split(";", 1)[0] or None
    return data, mime


def _data_uri(data: bytes, mime: str) -> str:
    """Build a base64 ``data:`` URI that always carries a resolved ``mime``."""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _assert_image(mime: str | None, source: object) -> None:
    """Guard the image-only media boundary: a resolved ``image/*`` mime is required."""
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"normalize_media requires an image resource; resolved mime {mime!r} for {source!r}")


class ResourceManager:
    def __init__(self, provider: Storage | None):
        self._provider = provider
        # Sandboxed engine: template text is an authoring surface (hook
        # condition/expr, stored/inline renders), so rendering blocks dunder and
        # unsafe-callable access — a traversal like
        # ``{{ ''.__class__.__mro__[1].__subclasses__() }}`` raises SecurityError
        # instead of reaching host primitives. Normal ``{{ var }}``, filters, and
        # ``{% include %}`` are unaffected.
        self._env = SandboxedEnvironment(loader=FunctionLoader(self._sync_loader))

        # Per-render event loop bound to the rendering thread so every
        # {% include %}/{% extends %} dependency resolves through one loop — hence
        # one storage client per render instead of one opened+closed per include.
        self._render_loop = threading.local()

        settings = template_cache_settings()

        # Compile-once cache for INLINE (content) templates. The stored-template
        # path is cached by id via ``_get_compiled_template``; without this an
        # inline condition/expr (e.g. a hook rendered on every fire) is re-parsed
        # on every render.
        self._compile_inline = lru_cache(maxsize=settings.max_size)(self._env.from_string)

        # Template ids currently compiled into the cache. Tracked so a delete can
        # evict just the affected key(s) instead of dropping the whole cache. It
        # stays bounded by what the cache actually holds: a delete prunes stale
        # ids, and ``_fetch_and_compile`` runs an amortized sweep whenever the set
        # has grown past the live cache size (an id the cache dropped on its own
        # via TTL/LRU), so the registry never outgrows the cache itself.
        self._cached_template_ids: set[str] = set()

        # Only cache if we have BOTH storage space (>0) AND retention time (>0).
        # If ttl is 0, items expire instantly, making caching useless.
        self._cache_enabled = (settings.max_size is None or settings.max_size > 0) and (
            settings.ttl is None or settings.ttl > 0
        )
        if self._cache_enabled:
            self._get_compiled_template = alru_cache(maxsize=settings.max_size, ttl=settings.ttl)(
                self._fetch_and_compile
            )

            logger.info(f"ResourceManager initialized with cache_size={settings.max_size} ttl={settings.ttl}")
        else:
            # Bypass cache wrapper entirely
            self._get_compiled_template = self._fetch_and_compile
            logger.info("ResourceManager initialized with caching disabled")

    @property
    def provider(self) -> Storage:
        """The active Storage provider, or a loud failure when none is registered.

        The skeleton ships no backend (dead by default): an unconfigured manager
        must never silently no-op, so every use that needs storage raises here.
        """
        if self._provider is None:
            raise RuntimeError(
                "ResourceManager has no Storage provider registered. The skeleton "
                "ships no storage backend (dead by default); register one via "
                "tai42_app.storage.register_storage before fetching or rendering a "
                "stored resource."
            )
        return self._provider

    @contextmanager
    def _render_scope(self) -> Iterator[None]:
        """Mark this render thread as inside a render so every dependency it
        resolves shares one event loop — the storage client opened for the first
        {% include %}/{% extends %} is reused and closed once when the scope exits,
        instead of a fresh loop + connection per dependency.

        The loop is created lazily by ``_sync_loader`` on the first dependency, so
        an include-free render creates no loop at all.
        """
        self._render_loop.active = True
        self._render_loop.loop = None
        try:
            yield
        finally:
            self._render_loop.active = False
            loop = self._render_loop.loop
            self._render_loop.loop = None
            if loop is not None:
                # Close the render loop's pooled clients once. A cleanup failure is
                # logged loudly but must not mask the render's result/exception.
                try:
                    loop.run_until_complete(shutdown_all_clients())
                except Exception:
                    logger.exception("Error closing pooled clients after template render")
                loop.close()

    async def _load_by_id(self, template_id: str) -> str:
        """Read a template by its bare storage id, root-contained.

        Every by-id read — render/fetch/schema/``{% include %}``, not only the media
        ``load`` seam — funnels through here so the id (a storage path) is passed
        through :func:`safe_template_path` before it reaches the provider. That is the
        app-layer defense against traversal for a backend that does not self-guard: an
        id like ``../../x`` (from an untrusted backup or an LLM-driven tool) raises
        :class:`~tai42_skeleton.template.path_guard.UnsafeTemplatePathError` instead of
        reading outside the template root.
        """
        safe_template_path(template_id)
        return await self.provider.load(template_id)

    def _sync_loader(self, template_id: str) -> str | None:
        """Synchronous bridge for Jinja2's FunctionLoader, letting {% extends %}
        and {% include %} fetch their dependency from async storage.

        A genuinely missing dependency becomes ``None`` so Jinja raises
        ``TemplateNotFound`` (keeping ``{% include ... ignore missing %}``
        working); every other failure propagates loudly instead of being mistaken
        for a missing template.
        """
        if getattr(self._render_loop, "active", False):
            # Inside a render scope (the normal path): reuse the per-render loop so
            # sibling includes share one storage client. Create it on first use.
            loop = self._render_loop.loop
            if loop is None:
                loop = asyncio.new_event_loop()
                self._render_loop.loop = loop
            try:
                return loop.run_until_complete(self._load_by_id(template_id))
            except FileNotFoundError:
                return None

        # Standalone fallback (no active render scope): a one-shot loop that
        # closes its own pooled client, mirroring the render-scope teardown.
        async def _load() -> str | None:
            try:
                return await self._load_by_id(template_id)
            finally:
                try:
                    await shutdown_all_clients()
                except Exception:
                    logger.exception("Error closing pooled clients after template load")

        try:
            return asyncio.run(_load())
        except FileNotFoundError:
            return None

    async def _fetch_and_compile(self, template_id: str) -> JinjaTemplate:
        logger.debug(f"Fetching template '{template_id}'")
        try:
            content = await self._load_by_id(template_id)
        except FileNotFoundError as exc:
            # Genuinely missing -> raise clearly. ``Storage.load`` contractually
            # raises ``FileNotFoundError`` only for absent content; present-but-
            # empty content returns "" and is a valid template that renders to
            # empty, so it must NOT be treated as missing.
            logger.error(f"Template '{template_id}' lookup failed.")
            raise TemplateNotFoundError(f"Template '{template_id}' not found.") from exc

        if self._cache_enabled:
            self._cached_template_ids.add(template_id)
            # Amortized prune: the set only exceeds the cache's live entry count
            # when the cache has evicted ids on its own — LRU eviction under a
            # bounded max_size AND ttl expiry under an unbounded one — so drop any
            # tracked id the cache no longer holds. Runs only when there is
            # something to prune; between sweeps the set tracks the live cache size.
            cache_info = getattr(self._get_compiled_template, "cache_info", None)
            contains = getattr(self._get_compiled_template, "cache_contains", None)
            if (
                cache_info is not None
                and contains is not None
                and len(self._cached_template_ids) > cache_info().currsize
            ):
                for tracked in list(self._cached_template_ids):
                    if not contains(tracked):
                        self._cached_template_ids.discard(tracked)
        return self._env.from_string(content)

    async def fetch_template(self, template_id: str) -> str:
        """Return the raw (uncompiled) template text for ``template_id``.

        A missing template raises :class:`TemplateNotFoundError` (mapped to 404 by
        the HTTP surface) rather than leaking storage's ``FileNotFoundError``.
        """
        try:
            return await self._load_by_id(template_id)
        except FileNotFoundError as exc:
            raise TemplateNotFoundError(f"Template '{template_id}' not found.") from exc

    async def load(self, source: str, *, with_mime: bool = True) -> tuple[bytes, str | None]:
        """Resolve ``source`` to ``(bytes, mime)`` — the unified content resolver.

        The source kind is disambiguated UP FRONT by ``urlsplit(source).scheme``,
        never by a try/fallback (which would swallow storage's loud
        ``FileNotFoundError``):

        - ``http`` / ``https`` -> fetched over the SSRF-pinned :func:`fetch_url`.
        - ``data`` -> the inline ``data:`` URI is decoded (no network/storage).
        - empty scheme -> a storage ``id``/path (``Storage.load_bytes`` for the
          bytes, ``Storage.stat`` for the mime). The empty-scheme branch runs the
          bare id through :func:`safe_template_path` first, enforcing root
          containment on the id (an ``UnsafeTemplatePathError`` propagates loudly),
          so every read that funnels here is guarded at this one seam.

        Any other scheme (``file:``, ``ftp:``) or an id containing ``://`` is
        ambiguous and raises loudly rather than being guessed. Each branch's own
        errors propagate.

        ``with_mime=False`` skips the storage ``Storage.stat`` lookup and returns
        ``mime`` as ``None`` for a storage id — for a caller that re-detects the
        type itself (:meth:`load_file` via Magika), so a metadata-bearing backend
        (e.g. S3, whose ``stat`` is a ``head_object`` round-trip) makes no extra
        call. http/data sources carry their mime inline, so the flag is a no-op
        for them.
        """
        scheme = urlsplit(source).scheme
        if scheme in ("http", "https"):
            return await fetch_url(source)
        if scheme == "data":
            return _decode_data_uri(source)
        if scheme == "" and "://" not in source:
            # A bare storage id is a caller-supplied logical key; enforce root
            # containment before it reaches the provider, so this one seam guards
            # every unguarded read (get_resource_by_id, file_loader, normalize_media's
            # storage branch, and any future caller of ``load``).
            safe_template_path(source)
            data = await self.provider.load_bytes(source)
            mime = (await self.provider.stat(source)).content_type if with_mime else None
            return data, mime
        raise ValueError(
            f"Cannot resolve source {source!r}: only http(s) URLs, data URIs, and bare "
            "storage ids are supported (a file:/ftp: scheme or an embedded '://' is rejected)."
        )

    async def load_file(self, source: str) -> str | MediaBlock:
        """Load ``source`` and decode it to extracted text OR a ``MediaBlock``.

        Resolves the bytes via :meth:`load` (``with_mime=False`` — the type is
        re-detected here with Magika, so the storage mime lookup is skipped),
        then dispatches through the type registry (the opt-in ``files`` extra).
        An unknown type raises loudly.
        """
        from tai42_skeleton.template.file_loading import load_content

        data, _mime = await self.load(source, with_mime=False)
        return await asyncio.to_thread(load_content, data, source)

    async def normalize_media(self, source: str | bytes) -> ContentPart:
        """Normalize an image reference to a model-ready LangChain ``ContentPart``.

        Image-only: the resolved mime must be ``image/*`` or it raises. Per source
        kind, without a wasteful download:

        - a public ``http(s)`` URL is passed through unchanged (the model
          dereferences it); a non-``image/*`` URL suffix raises here, an
          indeterminable one passes through.
        - a ``data:`` URI's mediatype is asserted ``image/*`` and passed through.
        - a storage ``id`` (or raw ``bytes``) is loaded and emitted as a base64
          ``data:`` URI that always carries a resolved image mime.
        """
        if isinstance(source, bytes):
            from tai42_skeleton.template.file_loading import detect_mime

            mime = detect_mime(source)
            _assert_image(mime, "<bytes>")
            assert mime is not None
            return {"type": "image_url", "image_url": {"url": _data_uri(source, mime)}}

        parts = urlsplit(source)
        scheme = parts.scheme
        if scheme in ("http", "https"):
            mime, _ = mimetypes.guess_type(parts.path)
            if mime is not None and not mime.startswith("image/"):
                raise ValueError(
                    f"normalize_media requires an image resource; URL suffix mime is {mime!r} for {source!r}"
                )
            return {"type": "image_url", "image_url": {"url": source}}
        if scheme == "data":
            mediatype = source[len("data:") :].partition(",")[0]
            mime = mediatype.split(";", 1)[0] or None
            _assert_image(mime, source)
            return {"type": "image_url", "image_url": {"url": source}}

        # Storage id (empty scheme) or a bad scheme -> load() validates and raises.
        data, mime = await self.load(source)
        _assert_image(mime, source)
        assert mime is not None
        return {"type": "image_url", "image_url": {"url": _data_uri(data, mime)}}

    def clear_cache(self) -> None:
        """Drop every compiled template from the cache (no-op when caching is off)."""
        # ``_get_compiled_template`` is the alru_cache wrapper only when caching is
        # enabled; otherwise it's the bare method with no cache controls.
        cache_clear = getattr(self._get_compiled_template, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
            self._cached_template_ids.clear()
            logger.info("ResourceManager cache cleared.")

    def _evict_compiled(self, template_id: str) -> None:
        """Evict a single compiled template from the cache (no-op if caching is
        off)."""
        invalidate = getattr(self._get_compiled_template, "cache_invalidate", None)
        if invalidate is not None:
            invalidate(template_id)
        self._cached_template_ids.discard(template_id)

    def _evict_compiled_prefix(self, prefix: str) -> None:
        """Evict every compiled template whose id falls under ``prefix``.

        Tracked ids that the cache has already dropped (TTL/LRU) are pruned in
        the same pass, keeping the registry bounded by what is actually cached.
        """
        invalidate = getattr(self._get_compiled_template, "cache_invalidate", None)
        if invalidate is None:
            # Caching disabled: nothing is cached, so nothing to track.
            self._cached_template_ids.clear()
            return
        contains = getattr(self._get_compiled_template, "cache_contains", None)
        for template_id in list(self._cached_template_ids):
            if template_id.startswith(prefix):
                invalidate(template_id)
                self._cached_template_ids.discard(template_id)
            elif contains is not None and not contains(template_id):
                self._cached_template_ids.discard(template_id)

    def get_cache_info(self) -> Any:
        """Return the compiled-template cache statistics, or ``None`` when caching
        is disabled."""
        cache_info = getattr(self._get_compiled_template, "cache_info", None)
        if cache_info is not None:
            return cache_info()
        return None

    async def render_by_id(self, template_id: str, kwargs: dict[str, Any] | None = None) -> str:
        template = await self._get_compiled_template(template_id)
        context = kwargs or {}

        def _render() -> str:
            with self._render_scope():
                return template.render(**context)

        return await asyncio.to_thread(_render)

    async def render_by_id_or_content(
        self,
        content: str | None = None,
        template_id: str | None = None,
        kwargs: dict[str, Any] | None = None,
        allow_empty: bool = True,
    ) -> str:
        if content is not None and template_id is not None:
            raise ValueError("Provide either 'content' OR 'template_id', not both.")

        context = kwargs or {}

        if content is not None:

            def _parse_and_render() -> str:
                with self._render_scope():
                    return self._compile_inline(content).render(**context)

            return await asyncio.to_thread(_parse_and_render)

        elif template_id is not None:
            return await self.render_by_id(template_id, context)

        elif allow_empty:
            return ""

        else:
            raise ValueError("You must provide either a template or a template_id.")

    async def get_template_schema(self, content: str | None = None, template_id: str | None = None) -> dict[str, Any]:
        if content is not None and template_id is not None:
            raise ValueError("Provide either 'content' OR 'template_id', not both.")

        if content is None:
            if template_id is None:
                raise ValueError("You must provide either a template or a template_id.")
            try:
                content = await self._load_by_id(template_id)
            except FileNotFoundError as exc:
                # Missing -> raise clearly; present-but-empty is a valid (empty)
                # template, inferred as an empty schema rather than rejected.
                raise TemplateNotFoundError(f"Template '{template_id}' not found.") from exc

        def _infer_schema():
            return to_json_schema(infer(content), jsonschema_encoder=JSONSchemaDraft4Encoder)

        return await asyncio.to_thread(_infer_schema)

    async def find_undeclared_variables(self, content: str | None = None, template_id: str | None = None) -> set[str]:
        if content is not None and template_id is not None:
            raise ValueError("Provide either 'content' OR 'template_id', not both.")

        if content is None:
            if template_id is None:
                raise ValueError("You must provide either a template or a template_id.")
            try:
                content = await self._load_by_id(template_id)
            except FileNotFoundError as exc:
                # Missing -> raise clearly; present-but-empty is a valid (empty)
                # template with no undeclared variables rather than rejected.
                raise TemplateNotFoundError(f"Template '{template_id}' not found.") from exc

        def _find_vars():
            # ``Storage.load`` contractually returns ``str`` (present-but-empty is
            # ""), and the block above raised on a missing template, so ``content``
            # is a real string here.
            assert content is not None
            ast = self._env.parse(content)
            return meta.find_undeclared_variables(ast)

        return await asyncio.to_thread(_find_vars)

    async def upload_template(self, path: str, content: str) -> None:
        """Store ``content`` at ``path`` and evict its compiled entry.

        The compiled-template cache keys entries by ``path`` and holds them until
        the cache TTL, so the single key is evicted after a successful upload to
        keep renders consistent with storage — the next render recompiles the new
        content.
        """
        await self.provider.upload(path, content)
        self._evict_compiled(path)

    async def list_resources(self) -> list[str]:
        """List every stored resource path in the storage provider."""
        return await self.provider.list()

    async def delete_template(self, path: str) -> None:
        """Delete the stored template at ``path`` and evict its compiled entry."""
        result = await self.provider.delete(path)
        # The compiled cache retains an entry per path until TTL; evict only the
        # deleted key so its stale compilation is dropped while other templates
        # stay cached.
        self._evict_compiled(path)
        return result

    async def delete_template_dir(self, path: str) -> None:
        """Delete every stored template under ``path/`` and evict their compiled
        entries (eviction runs even on a partial-delete failure)."""
        # Match the provider's prefix semantics: everything under ``path/``.
        prefix = path.rstrip("/") + "/"
        try:
            return await self.provider.delete_dir(path)
        finally:
            # Dir-delete is non-atomic: a failure can still leave some files
            # deleted, so evict regardless of outcome — never serve content under
            # this dir after a partial delete. Only keys under the prefix are
            # dropped; templates outside it stay cached.
            self._evict_compiled_prefix(prefix)
