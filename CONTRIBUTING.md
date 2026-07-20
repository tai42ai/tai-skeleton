# Contributing to tai-skeleton

`tai-skeleton` is the framework body that implements `tai-contract`: the concrete
`TaiMCP` server and the runtime engines (tool registry + adapters, agent registry,
OAuth connector engine, access-control middleware, hooks router, template/storage
manager, manifest loader, transport layer) behind the protocols `tai-contract`
declares. The app is constructed once as `tai_skeleton.app.instance.app` (a
`TaiMCP`) and exposed as the `tai_app` contract handle; tools, agents, and
extensions register against it (e.g. the `tai_app.bind` / `@app.tool` decorators).
Providers ship as separate plugins that register through `tai_app` when the
manifest loads them — no plugin imports the skeleton.

The hard rule (the leaf rule): **among tai-* packages it depends only on
`tai-contract` and `tai-kit`.** `tai-contract` is the pure interface it
implements; `tai-kit` is the generic leaf helpers, settings primitives, pooled
clients, and LLM factories it builds on. It imports no other tai-* package.

## Ground rules

- **Among tai-* packages, import `tai_contract` and `tai_kit` only.** Nothing
  else (no other tai-* package, no downstream plugins):
  ```bash
  grep -rnE '(from|import)\s+tai_' src/ | grep -vE 'tai_contract|tai_kit|tai_skeleton'   # expect no output
  ```
- **Providers stay out of the core.** OAuth connectors, storage backends, config
  providers, worker backends, and monitoring exporters ship as plugins that
  register through `tai_app` at manifest load — they are never imported by the
  skeleton.
- **Optional client drivers stay optional.** Redis and Postgres reach the engine
  through `tai-kit[redis,postgres]`; the skeleton never imports their drivers
  directly.
- **Errors surface loudly.** A missing module, a malformed config entry, or a
  failed sub-step raises and propagates — no silent skip, swallow, or degrade.
- **Typed package** (`py.typed`). Pyright runs clean.

## Layout

Each `tai_skeleton.<feature>` package implements the matching `tai-contract`
protocol (where one exists):

- `tools` — tool registry + adapters (MCP tool ↔ callable)
- `agent` — agent registry
- `extensions` — wrapper / transformer / backend tool extensions
- `connectors` — OAuth connector engine (providers, runtime, store, service)
- `hooks` — event hooks router + in-memory / redis managers
- `interactions` — the `ask_user` human-in-the-loop capability (helper + Redis store + settings)
- `channels` — the named-channel registry `ask_user` delivers through, plus the notification sink and notify helper
- `webhooks` — webhook-security surface: verifier registry, per-topic bindings, builtin `shared_secret` verifier
- `backend` — worker-backend dispatch + callback chaining
- `access_control` — auth adapter + ASGI middleware
- `middleware` — app-level ASGI middleware (the public-door rate limiter)
- `manifest` — manifest model + loader
- `config` — config-mode + manager seam (file / k8s providers plug in)
- `storage` — storage manager
- `backup` — backup-section registry (the concrete `AppBackup`) + the host's core sections
- `template` — Jinja template rendering + render mixins
- `monitoring` — monitoring writer seam
- `plugins` — Studio-plugin registry + bundle validation and serving
- `presets` — the preset kernel (bind a base tool to fixed kwargs)
- `sql` — centralized SQL schema (DDL) loader
- `authz` — tool-edge authorization: the single permission decision from `access_control`, installed on every MCP-serving instance
- `operations` — the operations layer: typed async functions + declared metadata that the routes, OpenAPI spec, CLI, and MCP tool surface all derive from
- `marketplace` — marketplace client, plugin installer, attribution store, and advisory polling
- `sub_mcp` — durable, cross-worker sub-MCP app registrations backing the `/app/{slug}` routes
- `versioning` — the versioned-document store package and its single construction point

Framework infrastructure lives in `app` (the `TaiMCP` server + lifecycle + facets),
`cli` (the unified `tai` CLI — `tai serve` / `tai backend` / `tai metrics` and the
remote/OpenAPI subcommands), `core`,
`routers`, `settings`, `exceptions`, and `utils`. `asgi.py` is the public
`create_app` factory for embedding the server in a host-owned ASGI process, and
`data/` holds shipped package data (`ecosystem.yml`).

## Dev

The dev venv resolves `tai-contract`, `tai-kit`, `tai-toolbox`, and
`tai-identity-redis` from sibling checkouts (see `[tool.uv.sources]`):

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
```

For local cross-repo work, `make dev` editable-installs the sibling `tai-*`
checkouts this package builds on into the venv. While `[tool.uv.sources]` pins
those siblings to local paths, `uv sync` already installs them editable and
`make dev` changes nothing; once the lock resolves them from the registry,
`uv sync` / `uv run` installs the published builds instead, so re-run
`make dev` afterward to restore the editable links.

Before any commit, run a secret scan over `src/` and `tests/` (e.g.
`detect-secrets scan`).

## License

By contributing you agree your contributions are licensed under Apache-2.0.
