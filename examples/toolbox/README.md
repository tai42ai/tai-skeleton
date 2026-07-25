# Toolbox starter

Batteries-included: this example wires content from the
[`tai42-toolbox`](../../../tai-toolbox) contrib package — composition **tool
extensions** (`batch`, `chain`) and two generic **tools** (`generate_uuid`,
`current_time_info`) — entirely from the manifest. The skeleton never imports
tai42-toolbox; the manifest loader pulls it in.

```
toolbox/
└── manifest.yml        # wires two toolbox extensions + two toolbox tools
```

There is no `myapp/` here: every module is an installed-package import path
(tai42-toolbox ships them), so no `PYTHONPATH` is needed — but tai42-toolbox must be
installed.

## Install

tai42-toolbox rides along on the `toolbox` extra:

```bash
pip install tai42-skeleton[toolbox]     # or, from a source checkout: uv sync --extra dev
```

Toolbox keeps its base install light and gates heavier modules behind their own
extras. The `chain` extension needs jq (`tai42-toolbox[chain]`); the skeleton
already ships `tai42-kit[jq]`, so `chain` imports cleanly here. A module whose
required extra is missing fails **loudly** at import with a
`pip install 'tai42-toolbox[extra]'` hint — never a silent skip.

## Run it

From the repo root:

```bash
ACCESS_CONTROL_ENABLE=false \
uv run tai serve --manifest-path examples/toolbox/manifest.yml --port 8765
```

`ACCESS_CONTROL_ENABLE=false` turns off the request-auth gate (on by default, it
checks every request against Redis, which this example does not use). The
manifest declares no connectors, so startup completes immediately:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

## What you get

The two `extensions_modules` register the `batch` and `chain` extensions, and
the `tools:` entries branch `generate_uuid` into extension variants. The server
ends up exposing:

- `generate_uuid` — the plain tool.
- `generate_uuid_batch` — runs many `generate_uuid` calls at once (from `batch`).
- `generate_uuid_chain` — calls `generate_uuid`, transforms its output with a jq
  expression, then calls a second registered tool with the result (from `chain`).
- `current_time_info` — a second plain tool; a `chain` call can name it as its
  `next_tool_name`.

The MCP endpoint is `http://127.0.0.1:8765/mcp`. From a second terminal in the
repo root:

```bash
uv run python - <<'EOF'
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8765/mcp") as c:
        print("tools:", sorted(t.name for t in await c.list_tools()))
        result = await c.call_tool("generate_uuid", {})
        print("generate_uuid ->", result.content[0].text)

asyncio.run(main())
EOF
```

Expected output (tool order aside):

```
tools: ['current_time_info', 'generate_uuid', 'generate_uuid_batch', 'generate_uuid_chain']
generate_uuid -> <a random uuid>
```

From here, browse the full toolbox catalog (the `http`, `files`, `proxy`, `vpn`,
`prometheus` tools and extensions, each behind its own extra) in the tai42-toolbox
package, and add the modules you want the same way.
