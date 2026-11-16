"""Run Opus 5.5/xhigh with all tools from one local HTTP MCP server."""
import asyncio
from datetime import datetime, timezone
import json
import os
import sys

import anthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams


def api_tool(tool):
    # Claude rejects root combinators. Keep them explicit in the description;
    # the MCP server still validates the original schema on every actual call.
    schema = dict(tool.input_schema)
    constraints = {key: schema.pop(key) for key in ("oneOf", "anyOf", "allOf", "not") if key in schema}
    description = tool.description or ""
    if constraints:
        description += "\nArgument constraints enforced by the MCP server: " + json.dumps(constraints)
    return tool.model_copy(update={"input_schema": schema, "description": description})


async def run(prompt, *, url="http://127.0.0.1:3000/mcp", system="",
              max_tokens=16384, max_rounds=16, checkpoint=None, model="claude-opus-5-5"):
    trace = {"messages": [], "tool_calls": [], "model": model, "effort": "xhigh",
             "max_tokens_per_turn":max_tokens,"max_rounds":max_rounds}

    def save():
        if checkpoint:
            checkpoint(trace)

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1], read_timeout_seconds=600) as session:
            initialized = await session.initialize()
            trace["server"] = initialized.server_info.model_dump(mode="json")
            definitions, cursor = [], None
            while True:
                page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
                definitions.extend(page.tools)
                cursor = page.next_cursor
                if not cursor:
                    break
            trace["available_tools"] = [t.name for t in definitions]

            class LoggedSession:
                async def call_tool(self, name, arguments):
                    call = {"name": name, "arguments": arguments,
                            "started_at": datetime.now(timezone.utc).isoformat()}
                    trace["tool_calls"].append(call)
                    save()
                    try:
                        result = await session.call_tool(name, arguments)
                        call["result"] = result.model_dump(mode="json", by_alias=True)
                        return result
                    except Exception as exc:
                        call["error"] = str(exc)
                        raise
                    finally:
                        save()

            tools = [async_mcp_tool(api_tool(t), LoggedSession()) for t in definitions]
            async with anthropic.AsyncAnthropic(timeout=600, max_retries=5) as client:
                runner = client.beta.messages.tool_runner(
                    model=trace["model"], output_config={"effort": trace["effort"]},
                    max_tokens=max_tokens, max_iterations=max_rounds, stream=True,
                    system="\n\n".join(filter(None, [initialized.instructions, system])),
                    messages=[{"role": "user", "content": prompt}], tools=tools,
                )
                async for stream in runner:
                    message = await stream.get_final_message()
                    trace["messages"].append(message.model_dump(mode="json"))
                    save()
                if not trace["messages"] or not any(b["type"]=="text" and b.get("text") for b in trace["messages"][-1]["content"]):
                    raise RuntimeError("Generation produced no final text; see saved trace")
    trace["text"] = "\n".join(b["text"] for b in trace["messages"][-1]["content"] if b["type"] == "text")
    trace["stop_reason"] = trace["messages"][-1]["stop_reason"]
    save()
    return trace


if __name__ == "__main__":
    load_dotenv()
    result = asyncio.run(run(" ".join(sys.argv[1:]) or sys.stdin.read(),
                             url=os.getenv("MCP_URL", "http://127.0.0.1:3000/mcp")))
    print(result["text"])
