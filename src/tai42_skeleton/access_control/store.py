"""Postgres-backed access-control POLICY store.

The enforced authorization rules: per-user policy bodies and the route/scope +
dynamic-pattern mappings the auth gate reads and the management surface writes.
Postgres is the SOLE store for these rules; there is no second backend and no
selection knob.

Scope of this store — POLICY RULES ONLY. Two live counter surfaces stay in Redis
and never touch this store: the plain-Redis ``ac:policy_version`` cache-buster and
the plain-Redis ``ac:context:*`` per-user live-counter hashes (deleted by the
revoke orchestration in ``management``; created by the first counter write). The
api-key IDENTITY record (key hash -> ``{user_id, description}``) is owned by the
identity provider's own storage, reached only through the ``ApiKeyIdentityProvider``
API — never here.

Mirrors :class:`~tai42_skeleton.versioning.store.PostgresVersionedStore`: Postgres is
reached through the app-pooled ``PostgresClient`` via ``client_ctx`` so it shares
one pool per DSN, and every multi-row mutation runs inside one ``conn.transaction()``
(real ACID), so consistency rides ordinary transactional SQL with no optimistic-lock
retry loop. Identities are column VALUES (parameterized,
injection-safe), so there is no identifier charset guard: an OIDC subject
containing ``:``/``@``/unicode is a plain bound value. Scope membership is DERIVED
(``WHERE scope_id = %s``), with no reverse index to keep consistent, so a scope-strip
cascade is one ``UPDATE ... array_remove`` and can never corrupt an index.

The grant-vs-remove interleave (grant a scope while its last route is being removed)
is closed by ROW-LOCK COUPLING: a grant locks the granted scopes' route rows
``FOR SHARE`` while it validates and writes the
policy, and a removal deletes those route rows (an exclusive row lock) BEFORE it
strips policies. The two lock modes conflict, so the operations serialize — a grant
either commits before the removal (which then strips the freshly-granted scope) or
runs after it and fails validation because the scope has no live route. A durable
policy therefore never holds a scope with no backing route.

The version-aware ``alru_cache`` layers stay in the verifier/enforcer on top of the
RAW reads this store exposes; the store does no caching of its own.
"""

from __future__ import annotations

from typing import Any

from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import Json, PostgresClient

from tai42_skeleton.access_control.settings import AccessControlSettings, access_control_settings
from tai42_skeleton.access_control.store_settings import access_control_store_settings

# The policy body shape enforcement reads and the management surface writes.
_POLICY_COLUMNS = "scopes, policy_data, condition, condition_id, condition_kwargs"


def _normalize_url(url: str) -> str:
    """Canonicalize a route url to the exact form the verifier resolves against.

    ``AccessControlVerifier.resolve_resource_ids`` strips a trailing slash before
    every lookup (when the path is longer than ``/``), so a route stored with a
    trailing slash could never match a request. Normalizing on write with the same
    rule keeps one canonical form on both sides."""
    if len(url) > 1 and url.endswith("/"):
        return url.rstrip("/")
    return url


def _policy_body(row: tuple[Any, ...]) -> dict[str, Any]:
    """Assemble the canonical policy body from a ``_POLICY_COLUMNS`` row."""
    scopes, policy_data, condition, condition_id, condition_kwargs = row
    return {
        "scopes": list(scopes or []),
        "policy_data": policy_data or {},
        "condition": condition,
        "condition_id": condition_id,
        "condition_kwargs": condition_kwargs,
    }


class PostgresAccessControlStore:
    """Postgres implementation of the access-control policy store."""

    def _settings(self) -> AccessControlSettings:
        return access_control_settings()

    # -- route / scope enumeration -------------------------------------------

    async def get_all_existing_scopes(self) -> dict[str, str]:
        """Every NON-public route mapping as ``{url: scope_id}``. The public marker
        names no scope, so a public route is excluded."""
        public = self._settings().public_resource_id
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT url, scope_id FROM access_control_routes WHERE scope_id <> %s",
                (public,),
            )
            rows = await cur.fetchall()
        return dict(rows)

    async def get_all_route_mappings(self) -> dict[str, str]:
        """Every route mapping as ``{url: scope_id}``, INCLUDING public routes whose
        value is the public marker (which ``get_all_existing_scopes`` filters out).
        The full faithful set a backup needs so an explicit public mapping
        round-trips instead of silently reverting to protected on restore."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT url, scope_id FROM access_control_routes ORDER BY url")
            rows = await cur.fetchall()
        return dict(rows)

    async def get_all_existing_patterns(self) -> dict[str, str]:
        """Every dynamic route's ``{url: pattern}`` mapping — the regex an operator
        registered for a url so the verifier matches request paths onto that url's
        scope. Empty when no url carries a dynamic pattern."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT url, pattern FROM access_control_routes WHERE pattern IS NOT NULL ORDER BY url")
            rows = await cur.fetchall()
        return dict(rows)

    # -- runtime reads (backing the verifier's version-keyed caches) ---------

    async def fetch_route(self, path: str) -> str | None:
        """The scope id a request ``path`` maps to (exact route), or ``None`` when
        the path has no mapping (a legitimately-unknown route → denied downstream)."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT scope_id FROM access_control_routes WHERE url = %s", (path,))
            row = await cur.fetchone()
        return row[0] if row is not None else None

    async def fetch_dynamic_patterns(self) -> dict[str, str]:
        """Every dynamic pattern as ``{regex: url_template}`` — the verifier compiles
        each regex and, on a full match, resolves the url template's route. Empty
        when no dynamic pattern is registered."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT pattern, url FROM access_control_routes WHERE pattern IS NOT NULL")
            rows = await cur.fetchall()
        return dict(rows)

    # -- route / scope mutations ---------------------------------------------

    async def add_url_to_scope(self, scope_id: str, url: str, pattern: str | None = None) -> None:
        """Map ``url`` to ``scope_id``, optionally registering the dynamic route
        ``pattern`` the verifier matches request paths against. When ``scope_id`` is
        the public marker the url is mapped public (the marker is an ordinary column
        value, not a scope). Any prior binding for ``url`` — its scope AND its
        pattern — is replaced by the upsert, so a re-point, a pattern change, and a
        pattern drop all leave exactly what this call writes. The url is normalized
        to the verifier's canonical form; the ``pattern`` regex is stored verbatim."""
        url = _normalize_url(url)
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "INSERT INTO access_control_routes (url, scope_id, pattern) VALUES (%s, %s, %s) "
                "ON CONFLICT (url) DO UPDATE SET scope_id = EXCLUDED.scope_id, pattern = EXCLUDED.pattern",
                (url, scope_id, pattern),
            )

    async def remove_url_from_scope(self, url: str) -> tuple[bool, list[tuple[str, dict[str, Any]]]]:
        """Unmap ``url``. If its scope has no urls left afterwards, cascade the scope
        out of every token policy that references it.

        Returns ``(existed, [(user_id, committed_body), …])`` — whether ``url`` was
        mapped at all (so the caller can 404 a typo'd unmap) and, for each token the
        cascade rewrote, the exact policy body committed so the caller records it as
        a new durable policy version. Empty list when no cascade fired. A url mapped
        to the public marker owns no scope, so its removal skips the cascade. The
        delete, the emptiness check, and the cascade all commit in one transaction."""
        url = _normalize_url(url)
        public = self._settings().public_resource_id
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                await cur.execute("SELECT scope_id FROM access_control_routes WHERE url = %s", (url,))
                row = await cur.fetchone()
                if row is None:
                    return False, []
                scope_id = row[0]
                await cur.execute("DELETE FROM access_control_routes WHERE url = %s", (url,))

                affected: list[tuple[str, dict[str, Any]]] = []
                if scope_id != public:
                    await cur.execute(
                        "SELECT 1 FROM access_control_routes WHERE scope_id = %s LIMIT 1",
                        (scope_id,),
                    )
                    if await cur.fetchone() is None:
                        affected = await self._strip_scope_from_policies(cur, scope_id)
            return True, affected

    # -- public route pins ---------------------------------------------------

    async def get_public_route_pins(self) -> list[str]:
        """The sorted urls whose route value is the public marker — the same rows
        ``get_all_existing_scopes`` filters out. Empty when no route is pinned
        public."""
        public = self._settings().public_resource_id
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT url FROM access_control_routes WHERE scope_id = %s ORDER BY url",
                (public,),
            )
            rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def pin_route_public(self, url: str, pattern: str | None = None) -> None:
        """Pin ``url`` public: set its route value to the public marker, replacing any
        prior scope binding AND any prior dynamic pattern in one upsert. The marker is
        an ordinary column value, not a scope, so NO scope-membership row is written;
        re-pointing a previously scope-mapped url OFF its scope keeps a later
        ``remove_scope`` on that scope from blind-deleting the now-public route. When
        ``pattern`` is given the dynamic-pattern pair is registered exactly as
        ``add_url_to_scope`` does; without it any prior pattern is cleared. The url is
        normalized to the verifier's canonical form. Committed in one transaction.

        Raises ``ValueError`` when ``url`` falls under a reserved management prefix
        (``reserved_public_pin_prefixes``): the access-control control plane must not be
        pinnable public, so the sole public-pin writer refuses to de-authenticate it.
        The verifier also drops the public marker for a reserved-prefix path at
        resolution, so the invariant holds even against a pattern or a direct write that
        does not pass through this door — this loud reject is the operator-facing half."""
        settings = self._settings()
        url = _normalize_url(url)
        for prefix in settings.reserved_public_pin_prefixes:
            if url == prefix or url.startswith(f"{prefix}/"):
                raise ValueError(
                    f"{url!r} is under the reserved access-control management prefix "
                    f"{prefix!r} and cannot be pinned public"
                )
        public = settings.public_resource_id
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(
                "INSERT INTO access_control_routes (url, scope_id, pattern) VALUES (%s, %s, %s) "
                "ON CONFLICT (url) DO UPDATE SET scope_id = EXCLUDED.scope_id, pattern = EXCLUDED.pattern",
                (url, public, pattern),
            )

    async def unpin_public_route(self, url: str) -> bool:
        """Unpin ``url``: delete its route row (and, with it, any dynamic pattern) only
        when its stored value is EXACTLY the public marker. Returns ``False`` — writing
        nothing — for an absent or scope-mapped url (those are not unpinnable through
        this path; the router 404s). Never touches scope rows. Committed in one
        transaction.

        The row is locked ``FOR UPDATE`` before the marker check so a concurrent
        re-point of the same url cannot commit between the check and the delete: the
        stricter "only when public" contract holds even under a racing writer."""
        url = _normalize_url(url)
        public = self._settings().public_resource_id
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                await cur.execute("SELECT scope_id FROM access_control_routes WHERE url = %s FOR UPDATE", (url,))
                row = await cur.fetchone()
                if row is None or row[0] != public:
                    return False
                await cur.execute("DELETE FROM access_control_routes WHERE url = %s", (url,))
            return True

    async def remove_scope(self, scope_id: str) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
        """Delete a scope: strip it from every token policy and delete every route
        that maps to it (their dynamic patterns go with the rows).

        Returns ``(deleted_count, [(user_id, committed_body), …])`` — the count of
        route rows deleted PLUS token policies stripped (0 = the scope was truly
        absent, so the caller 404s; a scope with token references but no url mapping
        still strips those references, so the count is non-zero and the caller treats
        it as found), and, for each token the cascade rewrote, the exact committed
        body. The public marker names no scope, so removing it raises ``ValueError``.
        All of it commits as one transaction."""
        public = self._settings().public_resource_id
        if scope_id == public:
            raise ValueError(f"{scope_id!r} is the public marker, not a removable scope")
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                # Delete the routes FIRST: the exclusive row lock this takes conflicts
                # with the ``FOR SHARE`` lock a concurrent grant of this scope holds, so
                # the delete waits for that grant to commit. Stripping AFTER the delete
                # therefore sees the freshly-granted scope and removes it too — the
                # grant cannot slip a stale scope past the cascade.
                await cur.execute("DELETE FROM access_control_routes WHERE scope_id = %s", (scope_id,))
                routes_deleted = cur.rowcount
                affected = await self._strip_scope_from_policies(cur, scope_id)
            return routes_deleted + len(affected), affected

    async def _strip_scope_from_policies(self, cur: Any, scope_id: str) -> list[tuple[str, dict[str, Any]]]:
        """Strip ``scope_id`` from every policy that lists it in ONE
        ``UPDATE ... array_remove ... RETURNING`` and return ``[(user_id,
        committed_body), …]`` — the exact rows the strip changed, each the caller
        records as a new version. Deriving the affected set from the UPDATE's own
        RETURNING (not a prior SELECT) keeps the recorded history in lockstep with the
        rows the strip actually committed, even under a concurrent grant."""
        await cur.execute(
            f"UPDATE access_control_policies SET scopes = array_remove(scopes, %s) "
            f"WHERE %s = ANY(scopes) RETURNING user_id, {_POLICY_COLUMNS}",
            (scope_id, scope_id),
        )
        rows = await cur.fetchall()
        return [(user_id, _policy_body(tuple(policy_cols))) for user_id, *policy_cols in rows]

    # -- policy reads / writes -----------------------------------------------

    async def get_policy_body(self, user_id: str) -> dict[str, Any] | None:
        """The full policy body enforcement serves for ``user_id`` (the canonical
        record the durable version history mirrors verbatim), or ``None`` when the
        user has no policy row."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                f"SELECT {_POLICY_COLUMNS} FROM access_control_policies WHERE user_id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
        return _policy_body(row) if row is not None else None

    async def policy_exists(self, user_id: str) -> bool:
        """Whether ``user_id`` has a policy row — the mint duplicate-user pre-check."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT 1 FROM access_control_policies WHERE user_id = %s", (user_id,))
            return await cur.fetchone() is not None

    async def create_policy(
        self,
        user_id: str,
        scopes: list[str],
        policy_data: dict[str, Any] | None = None,
        condition: str | None = None,
        condition_id: str | None = None,
        condition_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write the policy row for a freshly-minted key and return the committed
        body. The ``UNIQUE (user_id)`` constraint rejects a racing duplicate mint as
        the authority; that surfaces loudly (never a silent second row). Any supplied
        scope is validated against live routes and its route rows are locked
        ``FOR SHARE`` in the same transaction, so a scope's last route cannot be
        removed between the check and the write — the grant-vs-remove race is closed.
        Raises ``ValueError`` if any supplied scope has no live route."""
        body = {
            "scopes": scopes,
            "policy_data": policy_data or {},
            "condition": condition,
            "condition_id": condition_id,
            "condition_kwargs": condition_kwargs,
        }
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            if scopes:
                await self._lock_and_validate_scopes(cur, scopes)
            await cur.execute(
                "INSERT INTO access_control_policies "
                "(user_id, scopes, policy_data, condition, condition_id, condition_kwargs) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    user_id,
                    scopes,
                    Json(policy_data or {}),
                    condition,
                    condition_id,
                    Json(condition_kwargs),
                ),
            )
        return body

    async def update_policy_fields(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Partially update a key's POLICY fields in place and return the committed
        body, or ``None`` if ``user_id`` has no policy row (a falsy sentinel the
        route's 404 guard tests). ``updates`` carries only the fields the caller
        actually supplied (keys ⊆ ``scopes``/``policy_data``/``condition``/
        ``condition_id``/``condition_kwargs``); an absent field keeps its stored
        value. Supplied scopes are validated against live routes with their route
        rows locked ``FOR SHARE`` in the same transaction, so a concurrent removal
        of a scope's last route serializes rather than committing against a dead
        scope. Raises ``ValueError`` if any supplied scope does not exist."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                if updates.get("scopes"):
                    await self._lock_and_validate_scopes(cur, updates["scopes"])
                await cur.execute(
                    f"SELECT {_POLICY_COLUMNS} FROM access_control_policies WHERE user_id = %s FOR UPDATE",
                    (user_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                body = _policy_body(row)
                if "scopes" in updates:
                    body["scopes"] = updates["scopes"]
                if "policy_data" in updates:
                    body["policy_data"] = updates["policy_data"] or {}
                if "condition" in updates:
                    body["condition"] = updates["condition"]
                if "condition_id" in updates:
                    body["condition_id"] = updates["condition_id"]
                if "condition_kwargs" in updates:
                    body["condition_kwargs"] = updates["condition_kwargs"]
                await self._write_policy_body(cur, user_id, body)
            return body

    async def restore_policy_body(self, user_id: str, body: dict[str, Any]) -> dict[str, Any] | None:
        """Write a prior policy ``body`` back as the enforced policy — the store side
        of a version rollback. Returns the restored body, or ``None`` if ``user_id``
        has no live policy row (a revoke deletes it, so a rollback after revoke 404s
        rather than resurrecting a revoked key). Unlike ``update_policy_fields`` the
        restored scopes are NOT re-validated against live routes: a historical body
        must restore verbatim even if a scope's route was removed afterwards (such a
        scope is inert at enforcement). Committed in one transaction."""
        resolved = {
            "scopes": list(body.get("scopes") or []),
            "policy_data": body.get("policy_data") or {},
            "condition": body.get("condition"),
            "condition_id": body.get("condition_id"),
            "condition_kwargs": body.get("condition_kwargs"),
        }
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM access_control_policies WHERE user_id = %s FOR UPDATE",
                    (user_id,),
                )
                if await cur.fetchone() is None:
                    return None
                await self._write_policy_body(cur, user_id, resolved)
            return resolved

    async def delete_policy(self, user_id: str) -> bool:
        """Delete ``user_id``'s policy row (the store half of revoke). Returns
        whether a row existed. A real backend error propagates loudly."""
        async with (
            client_ctx(PostgresClient, access_control_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("DELETE FROM access_control_policies WHERE user_id = %s", (user_id,))
            return cur.rowcount > 0

    # -- helpers -------------------------------------------------------------

    async def _lock_and_validate_scopes(self, cur: Any, scopes: list[str]) -> None:
        """Lock every route row backing the granted ``scopes`` ``FOR SHARE`` and
        reject any scope not backed by a live (non-public) route mapping.

        The universal wildcard ``"*"`` is ALWAYS valid to write, mirroring the read
        side (``middleware`` treats ``"*"`` as "everything"): it names no routed scope,
        so validating it against the route table would reject every role template (all
        carry ``["*"]``) and brick the first-owner bootstrap. This is a scope-typo
        validator, meaningless for the wildcard; it is NOT the security boundary for
        ``"*"`` (the route-level mint rules are).

        The ``FOR SHARE`` lock is what closes the grant-vs-remove race: it conflicts
        with the exclusive lock a concurrent ``remove_scope``/``remove_url_from_scope``
        takes when it deletes those route rows, so the grant and the removal
        serialize instead of interleaving. A grant that runs after a removal sees no
        route for the scope and raises here; a grant that commits first is caught by
        the removal's post-delete cascade. Callers run this inside their own
        transaction so the lock is held until the policy write commits."""
        public = self._settings().public_resource_id
        concrete = [scope for scope in scopes if scope != "*"]
        if not concrete:
            return
        await cur.execute(
            "SELECT scope_id FROM access_control_routes WHERE scope_id = ANY(%s) AND scope_id <> %s FOR SHARE",
            (concrete, public),
        )
        live = {row[0] for row in await cur.fetchall()}
        for scope in concrete:
            if scope not in live:
                raise ValueError(f"scope {scope!r} does not exist or has no urls assigned")

    async def _write_policy_body(self, cur: Any, user_id: str, body: dict[str, Any]) -> None:
        await cur.execute(
            "UPDATE access_control_policies SET scopes = %s, policy_data = %s, condition = %s, "
            "condition_id = %s, condition_kwargs = %s WHERE user_id = %s",
            (
                body["scopes"],
                Json(body["policy_data"]),
                body["condition"],
                body["condition_id"],
                Json(body["condition_kwargs"]),
                user_id,
            ),
        )


def access_control_store() -> PostgresAccessControlStore:
    """Return the active access-control policy store (Postgres, the only backend)."""
    return PostgresAccessControlStore()
