# tai42-skeleton

[![CI](https://github.com/tai42ai/tai-skeleton/actions/workflows/ci.yml/badge.svg)](https://github.com/tai42ai/tai-skeleton/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The open-source implementation of `tai42-contract` — the extended MCP server that
hosts tools, agents, extensions, connectors, hooks, and storage for the TAI
ecosystem. It provides the concrete `TaiMCP` server and the runtime engines
(tool registry and adapters, the agent registry, the OAuth connector engine, the
access-control middleware, the hooks router, the template/storage manager, the
manifest loader, and the transport layer) that implement the protocols declared
in `tai42-contract`.

Providers — OAuth connectors, storage backends, config providers, worker
backends, monitoring — ship as separate plugins that register through the
`tai42_app` contract handle when the manifest loads them; no plugin imports the
skeleton.

## Position in the ecosystem

TAI is an open-source runtime for MCP tools, agents, and workflows — the server
that hosts a capability and supplies the operational layer around it (manifest
loading, access control, OAuth connectors, background execution, monitoring,
storage, and human-in-the-loop steps).

Three packages; each depends only on the ones to its left:

```
tai42-contract  <--  tai42-kit  <--  tai42-skeleton
(interfaces)      (helpers)     (the server)
```

`tai42-skeleton` is the server at the end of the chain: it depends on **only**
`tai42-contract` (the pure interface package) and `tai42-kit` (generic leaf
helpers). It is the runnable body every plugin plugs into.

## Install

Requires **Python 3.13+**. Nothing is on PyPI yet, so install from source: clone
the three repos **side by side** — this repo's `[tool.uv.sources]` points at
`../tai-contract` and `../tai-kit`:

```bash
git clone https://github.com/tai42ai/tai-contract
git clone https://github.com/tai42ai/tai-kit
git clone https://github.com/tai42ai/tai-skeleton
cd tai-skeleton
uv sync
```

Add the `toolbox` extra for batteries — it pulls in the `tai42-toolbox` contrib
package, whose composition tool extensions (`chain`, `batch`) and generic tool
collection load from the manifest (see [`examples/toolbox`](examples/toolbox)):

```bash
uv sync --extra toolbox
```

Once the packages are published, `pip install tai42-skeleton[toolbox]` will work
too.

## Run it

The hello-world app in [`examples/hello`](examples/hello) boots one local `greet`
tool, no external services. From the repo root:

```bash
ACCESS_CONTROL_ENABLE=false \
PYTHONPATH=examples/hello \
uv run tai serve --manifest-path examples/hello/manifest.yml --port 8765
```

The MCP endpoint is then `http://127.0.0.1:8765/mcp`. `tai backend` runs the
agent/worker backend process and `tai metrics` serves the Prometheus endpoint;
the full command surface is in the CLI reference below.

`tai serve`, `tai backend`, and `tai metrics` form one run family over a single
shared Prometheus multiprocess directory (`PROMETHEUS_MULTIPROC_DIR`, default a
fixed absolute path under the host temp dir; any override must be absolute).
Point all three at the same directory and restart them together: `tai serve`
clears the directory once at boot, so restarting it mid-run while a backend
worker is still writing orphans that worker's counters until the whole family
restarts.
[`examples/README.md`](examples/README.md) walks through the examples, and
[`examples/manifest.yml`](examples/manifest.yml) is the commented manifest
reference.

## The worker bus

When a deployment runs more than one process — several `tai serve` workers, a
`tai backend` runtime alongside the server, or multiple pods sharing one config —
a manifest edit on one process must reach the others, or siblings serve stale
state. The **worker bus** is how every process converges: each subscribes to one
Redis channel at startup, a mutation applies locally and is then broadcast, and
the response carries a per-origin report of how every worker fared.

The bus is **internal app infrastructure, like the reload gate — it is NOT a
plugin.** Nothing about it is registrable, swappable, or user-selectable; there is
exactly one bus and no manifest field chooses an implementation. It is configured
only by environment: set `TAI_BUS_REDIS_URL` (plus the optional `TAI_BUS_*` knobs)
to turn it on. A single-worker, file-mode, no-backend deployment needs no bus and
runs on a no-op local variant; a multi-worker, backend-bearing, or `k8s`-mode boot
refuses to start without one, naming `TAI_BUS_REDIS_URL`. On a shared Redis,
`TAI_BUS_NAMESPACE` must diverge per stack — Redis pub/sub is server-global, so
co-tenant deployments would otherwise cross-talk.

## Development

The dev venv resolves `tai42-contract`, `tai42-kit`, `tai42-toolbox`, and
`tai42-identity-redis` from sibling checkouts (see `[tool.uv.sources]`):

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
```

See `CONTRIBUTING.md` for the rules.

## Documentation

The whole platform — the quickstart, concepts, guides, and the generated
reference — lives in the unified documentation site:

- Getting started & install: https://tai42.ai/getting-started/installation
- Quickstart: https://tai42.ai/getting-started/quickstart
- Concepts: https://tai42.ai/concepts
- Guides: https://tai42.ai/guides
- Built-in tools & extensions: https://tai42.ai/concepts/tools-and-extensions
- Reference (HTTP API, CLI, Python SDK): https://tai42.ai/reference/cli

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
