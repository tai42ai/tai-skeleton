-- ============================================================
-- Tai Platform — Connector framework PostgreSQL Schema
-- ============================================================
-- Framework connector-engine tables only: the de-tenanted token
-- store, the no-auth market catalog, its categories, and the
-- allowed discovery sources. Product tables live in product repos.
-- Database: tai
-- ============================================================

-- ------------------------------------------------------------
-- Connectors — per-connection token store (durable source of truth)
-- ------------------------------------------------------------
-- Durable home for hub OAuth connections. One row per connection, holding
-- the SAME AES-GCM-encrypted blob the redis-pg ConnectorTokenStore caches in Redis
-- (`encrypted_blob` is opaque ciphertext — the DB never sees plaintext). The
-- store reads Redis first and falls back to this table on a cache miss, so a
-- large fleet of connections never opens a Postgres connection per request.
--
-- `connection_id` is the single global identity (uuid4, globally unique) and
-- the whole primary key: the store is single-namespace, with no tenant
-- partition. `session_expires_at` is the effective session/refresh-token expiry
-- the API computed (provider expiry, capped by CONNECTORS_MAX_SESSION_TTL); it
-- lets a cache-miss repopulate Redis with the remaining TTL. Scopes and
-- per-token expiries live INSIDE the encrypted blob, not as columns — the store
-- treats the blob as opaque and nothing queries by them.
--
-- `provider_id` and `alias` are the ONLY plaintext record fields kept as
-- columns, and only to back the `UNIQUE (provider_id, alias)` constraint: the
-- alias is a user-typed display label (not a secret), so the database enforces
-- per-provider alias uniqueness durably. A racing pair of same-alias connects
-- can no longer both pass a list-scan check and both persist — the second INSERT
-- hits the unique constraint and is rejected as the authority.
--
-- `cache_version` is a per-row monotonic counter bumped on every durable write.
-- The redis-pg store carries it alongside the cached blob and applies a cache
-- write only when the incoming version is newer, so a slow cache-miss populate
-- holding an old snapshot can never overwrite a fresher entry a concurrent
-- writer already installed (cache-coherence fence).
CREATE TABLE IF NOT EXISTS connector_connections (
    connection_id      UUID         NOT NULL,
    provider_id        TEXT         NOT NULL,
    alias              TEXT         NOT NULL,
    encrypted_blob     BYTEA        NOT NULL,
    session_expires_at TIMESTAMPTZ,
    cache_version      BIGINT       NOT NULL DEFAULT 1,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (connection_id),
    CONSTRAINT connector_connections_provider_alias_unique UNIQUE (provider_id, alias)
);
-- The `connection_id` primary key is the sole point-read path
-- (`WHERE connection_id = %s`); the listing read enumerates the table. The
-- `UNIQUE (provider_id, alias)` constraint additionally indexes those two
-- columns to arbitrate concurrent same-alias connects.

-- ------------------------------------------------------------
-- Connector categories — grouping for the Connectors UI tabs. GLOBAL. One row
-- per category; `sort_order` drives the display order inside each tab
-- (`other` carries a large sentinel so it always sorts last). Seeded below;
-- new categories are created at runtime only through the provider-add agent
-- tool's category-create path.
CREATE TABLE IF NOT EXISTS connector_category (
    id            TEXT         NOT NULL,
    display_name  TEXT         NOT NULL,
    sort_order    INTEGER      NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

INSERT INTO connector_category (id, display_name, sort_order) VALUES
    ('communication', 'Communication', 1),
    ('productivity',  'Productivity',  2),
    ('dev-tools',     'Dev Tools',     3),
    ('data',          'Data',          4),
    ('ai-ml',         'AI & ML',       5),
    ('other',         'Other',         1000)
ON CONFLICT (id) DO NOTHING;

-- ------------------------------------------------------------
-- Market catalog of no-auth MCP connectors. GLOBAL: a single shared market.
-- One row per offered no-auth
-- provider; `descriptor` is the full no-auth ProviderDescriptor serialized as
-- JSON (id, display_name, icon, mcp_server, sub_services, config_fields,
-- kind="none") WITHOUT origin/category — those live only in the columns, the
-- single source of truth. It is a public template — it holds NO secrets.
-- Per-connection values (env/headers filled at connect time) live encrypted in
-- connector_connections, never here.
--
-- `origin` records who curated the row: 'system' for ops-inserted rows,
-- 'community' for rows added at runtime through the verified agent-tool add
-- path (connectors.store.catalog_write) — those also carry `added_by` (the
-- adding caller) and `source_url` (where the provider was discovered).
--
-- The api and mcp processes load enabled rows into an in-memory cache at
-- startup and on reload (connectors.store.catalog_store.refresh_catalog).
-- Ops add a system MCP by INSERTing a row (ops/SQL); community adds insert
-- through catalog_write and trigger a fleet reload. `enabled` hides a row
-- without deleting it. DDL is applied only by the API (run_schema_ddl), the
-- same deploy-ordering assumption connector_connections already relies on.
CREATE TABLE IF NOT EXISTS connector_catalog (
    provider_id   TEXT         NOT NULL,
    descriptor    JSONB        NOT NULL,
    origin        TEXT         NOT NULL CHECK (origin IN ('system', 'community')),
    category      TEXT         NOT NULL REFERENCES connector_category (id),
    source_url    TEXT,
    added_by      TEXT,
    enabled       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (provider_id)
);

-- ------------------------------------------------------------
-- Allowed discovery sources for the MCP-finder agent. GLOBAL. The discovery
-- tools search/fetch ONLY enabled rows here — any other URL raises. Runtime
-- read-only: ops extend the list by editing this table directly in Postgres;
-- no tool, route, or UI writes it. `enabled` disables a source without
-- deleting it.
CREATE TABLE IF NOT EXISTS connector_allowed_source (
    id            TEXT         NOT NULL,
    url           TEXT         NOT NULL,
    enabled       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

INSERT INTO connector_allowed_source (id, url) VALUES
    ('github',       'https://github.com'),
    ('mcp-registry', 'https://registry.modelcontextprotocol.io'),
    ('smithery',     'https://smithery.ai')
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- Versioned-document store — the generic versioning primitive
-- ============================================================
-- A `kind`-discriminated, body-opaque persistence primitive: append-only
-- version rows over an opaque JSONB body, an active-version pointer, and
-- rollback. The store knows NOTHING about what any `kind` holds — presets
-- (`kind='preset'`), AC policies (`kind='ac_policy'`), authored agents, and any
-- future kind are typed VIEWS layered on top. Identity is `(kind, name)`
-- throughout; every write is one transaction. Single-tenant: no tenant column
-- (each user runs their own full stack).

-- ------------------------------------------------------------
-- versioned_documents — one live record per `(kind, name)`, holding the active
-- pointer. `active_version` names the version currently served; rollback
-- re-points it WITHOUT copying data. `is_active` is the soft-delete flag: a
-- soft-deleted document keeps its version history (audit) but drops out of the
-- active listing. The partial-unique index enforces one ACTIVE row per
-- `(kind, name)` while allowing soft-deleted ghosts to coexist (so a name can be
-- created again after a soft delete, and repeated soft deletes each leave their
-- own ghost) — a plain full unique would wrongly reject the recreate.
CREATE TABLE IF NOT EXISTS versioned_documents (
    id             BIGSERIAL    NOT NULL,
    kind           TEXT         NOT NULL,
    name           TEXT         NOT NULL,
    active_version INTEGER      NOT NULL,
    is_active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);
CREATE UNIQUE INDEX IF NOT EXISTS versioned_documents_active_name_unique
    ON versioned_documents (kind, name) WHERE is_active;

-- ------------------------------------------------------------
-- versioned_document_versions — immutable, append-only version rows. A version
-- is written once and never changes (`version`, `body`, `tags`, `created_at`).
-- `body` is the opaque JSONB the store never inspects; `tags` is the generic
-- per-version grouping label (kind-agnostic, no product meaning). The FK
-- `ON DELETE CASCADE` removes a document's whole version set with it on a hard
-- delete of the active row. `UNIQUE (document_id, version)` keeps the append
-- monotonic; its btree index also backs the `MAX(version)` read that computes
-- the next version number (a btree scans either direction, so `version DESC`
-- needs no separate index).
CREATE TABLE IF NOT EXISTS versioned_document_versions (
    id          BIGSERIAL    NOT NULL,
    document_id BIGINT       NOT NULL REFERENCES versioned_documents (id) ON DELETE CASCADE,
    version     INTEGER      NOT NULL,
    body        JSONB        NOT NULL,
    tags        TEXT[]       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT versioned_document_versions_doc_version_unique UNIQUE (document_id, version)
);

-- ------------------------------------------------------------
-- role_audit append-only enforcement — DB-level immutability for the
-- security audit trail.
-- ------------------------------------------------------------
-- The role-edit audit trail rides the generic store under `kind='role_audit'`
-- (the RoleAuditView). For every other kind the store's full surface (edit,
-- soft-delete, hard-delete + FK cascade, rename, rollback) is legitimate, but an
-- audit trail must never be rewritten. These triggers make `kind='role_audit'`
-- documents append-only IN THE DATABASE, so no code path or bug can silently alter
-- the security record — a comment is not enforcement. A legitimate audit write is
-- only ever an INSERT of a new version row plus the `active_version` pointer bump,
-- and both stay allowed. (This guards against application code, not a hostile
-- superuser/DBA, who can disable triggers — that is WORM territory, out of scope.)
--
-- The functions RAISE loudly (matching the store's fail-loud posture); psycopg
-- surfaces the raise as an error that propagates rather than being swallowed. The
-- DDL is idempotent: CREATE OR REPLACE FUNCTION plus DROP TRIGGER IF EXISTS before
-- each CREATE TRIGGER.

-- Version rows are written once and never change: block UPDATE or DELETE of any
-- version row whose parent document is a role_audit. (The FK ON DELETE CASCADE can
-- only reach these rows via a delete of the parent document, which the next trigger
-- already forbids for role_audit, so a cascade never strips a role_audit history.)
CREATE OR REPLACE FUNCTION versioned_document_versions_role_audit_immutable()
    RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT kind FROM versioned_documents WHERE id = OLD.document_id) = 'role_audit' THEN
        RAISE EXCEPTION
            'role_audit version rows are immutable and append-only: % on document_id=% version=% is forbidden',
            TG_OP, OLD.document_id, OLD.version;
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_versioned_document_versions_role_audit_immutable ON versioned_document_versions;
CREATE TRIGGER trg_versioned_document_versions_role_audit_immutable
    BEFORE UPDATE OR DELETE ON versioned_document_versions
    FOR EACH ROW EXECUTE FUNCTION versioned_document_versions_role_audit_immutable();

-- A role_audit document can never be deleted: a hard delete would drop the whole
-- audit history via the FK cascade. Blocking DELETE here also stops the cascade
-- from ever reaching the version rows above.
CREATE OR REPLACE FUNCTION versioned_documents_role_audit_no_delete()
    RETURNS TRIGGER AS $$
BEGIN
    IF OLD.kind = 'role_audit' THEN
        RAISE EXCEPTION
            'role_audit documents cannot be deleted: the security audit trail is append-only (id=%, name=%)',
            OLD.id, OLD.name;
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_versioned_documents_role_audit_no_delete ON versioned_documents;
CREATE TRIGGER trg_versioned_documents_role_audit_no_delete
    BEFORE DELETE ON versioned_documents
    FOR EACH ROW EXECUTE FUNCTION versioned_documents_role_audit_no_delete();

-- On a role_audit document, ONLY `active_version` may change — the legitimate
-- pointer bump an append (or a rollback) performs. Any other column change is
-- forbidden: this blocks the soft-delete `is_active` flip and the rename, which
-- would each corrupt or hide the audit trail.
CREATE OR REPLACE FUNCTION versioned_documents_role_audit_guard_update()
    RETURNS TRIGGER AS $$
BEGIN
    IF OLD.kind = 'role_audit' AND (
        NEW.id IS DISTINCT FROM OLD.id
        OR NEW.kind IS DISTINCT FROM OLD.kind
        OR NEW.name IS DISTINCT FROM OLD.name
        OR NEW.is_active IS DISTINCT FROM OLD.is_active
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    ) THEN
        RAISE EXCEPTION
            'role_audit documents are append-only: only active_version may change (id=%, name=%)',
            OLD.id, OLD.name;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_versioned_documents_role_audit_guard_update ON versioned_documents;
CREATE TRIGGER trg_versioned_documents_role_audit_guard_update
    BEFORE UPDATE ON versioned_documents
    FOR EACH ROW EXECUTE FUNCTION versioned_documents_role_audit_guard_update();

-- ============================================================
-- Access-control policy store — the enforced authz rules
-- ============================================================
-- The authorization POLICY RULES the auth gate enforces: per-user policy bodies
-- and the route/scope + dynamic-pattern mappings. Postgres is the SOLE store for
-- these rules. Two live counter surfaces stay in Redis and are NOT tables here:
-- the plain-Redis `ac:policy_version` cache-buster and the plain-Redis `ac:context:*`
-- per-user live-counter hashes; and the api-key IDENTITY record (key hash ->
-- `{user_id, description}`) is owned by the identity provider's own storage.
-- Single-tenant: no tenant column (each user runs their own full stack).

-- ------------------------------------------------------------
-- access_control_policies — one row per provisioned `user_id`, holding the policy
-- body the gate enforces: the granted `scopes`, the opaque `policy_data`, and the
-- optional jq authorization `condition` (inline `condition` or a stored
-- `condition_id`, plus `condition_kwargs`). Identities are COLUMN VALUES
-- (`user_id`), so an OIDC subject containing `:`/`@`/unicode is a plain
-- parameterized value with no key-encoding hazard. There is deliberately NO
-- `description` column and NO api-key-hash column: both belong to the identity
-- record, whose single home is the provider's storage — duplicating them here
-- would create a second, stale-able copy.
CREATE TABLE IF NOT EXISTS access_control_policies (
    id               BIGSERIAL    NOT NULL,
    user_id          TEXT         NOT NULL,
    scopes           TEXT[]       NOT NULL DEFAULT '{}',
    policy_data      JSONB        NOT NULL DEFAULT '{}',
    condition        TEXT,
    condition_id     TEXT,
    condition_kwargs JSONB,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT access_control_policies_user_id_unique UNIQUE (user_id)
);
-- `user_id` is the sole point-read path (`WHERE user_id = %s`) and the mint
-- duplicate-user guard; the `UNIQUE (user_id)` constraint indexes it and rejects
-- a racing second mint of the same user as the authority. The scope-cascade read
-- (`WHERE %s = ANY(scopes)`) scans the table — the policy set is operator-sized,
-- not a per-request hot path (enforcement serves from the in-process version-keyed
-- cache and reads a single row by `user_id` only on a miss).

-- ------------------------------------------------------------
-- access_control_routes — one row per mapped `url`, giving the `scope_id` a
-- request path resolves to. The public marker (`settings.public_resource_id`) is
-- an ORDINARY `scope_id` value, not a glob key: a public route stores that marker
-- verbatim and is excluded from scope enumeration by a plain `scope_id <> marker`
-- filter, while `remove_scope(<marker>)` is rejected in the store. `pattern` is the
-- optional dynamic-route regex the verifier full-matches a request path against;
-- when set, the row's `url` is the template the matched path resolves through
-- (NULL for a plain exact route).
CREATE TABLE IF NOT EXISTS access_control_routes (
    id         BIGSERIAL    NOT NULL,
    url        TEXT         NOT NULL,
    scope_id   TEXT         NOT NULL,
    pattern    TEXT,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT access_control_routes_url_unique UNIQUE (url)
);
-- `url` is the exact-route point-read path (`WHERE url = %s`) and the upsert
-- conflict target; the `UNIQUE (url)` constraint indexes it. Scope membership is
-- DERIVED (`WHERE scope_id = %s`) with no reverse-index bookkeeping — a scope
-- "exists" only while at least one non-public route maps to it.

-- ============================================================
-- Marketplace attribution — the installed-plugin record
-- ============================================================
-- One row per marketplace-installed plugin: which listing (`ref` =
-- "namespace/name"), which exact version, from which artifact channel, and the
-- full PluginSpec that version shipped (`spec`, JSONB) — stored locally so
-- `installed`/`update`/`advisories` answer from local truth and so UNINSTALL
-- can unpatch the manifest without asking the registry (the spec that was
-- installed, not whatever the registry currently serves). Written only by the
-- skeleton's own installer; deleted on uninstall; version+spec replaced on
-- update. Single-tenant: no tenant column (each user runs their own stack).
CREATE TABLE IF NOT EXISTS marketplace_installs (
    ref            TEXT         NOT NULL,
    version        TEXT         NOT NULL,
    source         TEXT         NOT NULL CHECK (source IN ('pypi', 'github')),
    -- The git repository URL and tag the github pin came from (both NULL for a
    -- pypi source), kept for display/provenance. The install itself does NOT
    -- clone this tag: it downloads `artifact_ref` and verifies `sha256` (below).
    -- These values came from resolve (NOT from `spec`), so they are stored here,
    -- never re-derived.
    repository_url TEXT,
    tag            TEXT,
    -- The github artifact tarball URL the verified install fetched, and the
    -- sha256 the registry captured for it at ingest (both NULL for a pypi
    -- source, whose immutable `package==version` needs no fetch). Update's unwind
    -- reinstalls the old pin by re-fetching `artifact_ref` and re-checking
    -- `sha256`, so a re-pointed release tag can never smuggle unverified code
    -- back in during a rollback.
    artifact_ref   TEXT,
    sha256         TEXT,
    spec           JSONB        NOT NULL,
    installed_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (ref)
);
-- `ref` is the sole point-read path (`WHERE ref = %s`); the installed listing
-- enumerates the table. Operator-sized (tens of rows), no further indexes.
