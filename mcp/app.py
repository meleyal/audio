"""
mcp
Connect to Ableton Live → control via Claude or GPT.
"""

import asyncio
import json
import sys
import threading
import zipfile
from pathlib import Path
from typing import Generator

import gradio as gr

# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent / "AbletonMCP"
SCRIPT_ZIP = Path(__file__).parent / "AbletonMCP.zip"

# Build the zip once at startup: AbletonMCP/__init__.py
with zipfile.ZipFile(SCRIPT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(SCRIPT_DIR / "__init__.py", arcname="AbletonMCP/__init__.py")

ANTHROPIC_MODEL = "claude-opus-4-6"
OPENAI_MODEL    = "gpt-4o"

SYSTEM_PROMPT = (
    "You are a music production assistant with direct control over Ableton Live. "
    "Use the provided tools to fulfil the user's requests. Always check session "
    "state before making changes. Be concise."
)

# ── MCP client ────────────────────────────────────────────────────────────────

class MCPClient:
    """
    Maintains a persistent MCP session with the ableton-mcp server in a
    background thread, exposing a synchronous interface to Gradio handlers.
    """

    def __init__(self):
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session    = None
        self._stdio_cm   = None
        self._session_cm = None
        self._tools      = []

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    async def _connect(self):
        from mcp.client.stdio import stdio_client

        from mcp import ClientSession, StdioServerParameters

        # Tear down any existing session first
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).parent / "server.py")],
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools = result.tools

    def connect(self):
        self._run(self._connect())

    def tools(self):
        return self._tools

    async def _call_tool(self, name: str, args: dict) -> str:
        result = await self._session.call_tool(name, args)
        parts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(parts)

    def call_tool(self, name: str, args: dict) -> str:
        return self._run(self._call_tool(name, args))


_client = MCPClient()


# ── Tool schema conversion ────────────────────────────────────────────────────

def _anthropic_tools():
    return [
        {
            "name":         t.name,
            "description":  t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in _client.tools()
    ]


def _openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description or "",
                "parameters":  t.inputSchema,
            },
        }
        for t in _client.tools()
    ]


# ── Connection check ──────────────────────────────────────────────────────────

def check_connection() -> str:
    try:
        _client.connect()
        n = len(_client.tools())
        return f"● Connected ({n} tools)"
    except Exception as e:
        return f"○ Failed: {e}"


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def dispatch_tool(name: str, args: dict) -> str:
    try:
        return _client.call_tool(name, args)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Provider helpers ──────────────────────────────────────────────────────────

def _call_anthropic(api_msgs: list[dict], api_key: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=_anthropic_tools(),
        messages=api_msgs,
    ) as stream:
        final = stream.get_final_message()
    text  = "\n".join(b.text for b in final.content if b.type == "text") or None
    calls = [b for b in final.content if b.type == "tool_use"]
    return text, calls, final.content


def _call_openai(api_msgs: list[dict], api_key: str):
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=api_msgs,
        tools=_openai_tools(),
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    return msg.content, msg.tool_calls or [], msg


# ── Agentic loop ──────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    provider: str,
    api_key: str,
) -> Generator:
    if not api_key.strip():
        yield history + [{"role": "assistant", "content": "⚠ Please enter an API key."}], ""
        return

    if not _client.tools():
        yield history + [{"role": "assistant", "content": "⚠ Not connected — click 'Check connection' first."}], ""
        return

    display  = list(history) + [{"role": "user", "content": user_message}]
    api_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in display
        if m["role"] in ("user", "assistant") and isinstance(m["content"], str)
    ]

    yield display + [{"role": "assistant", "content": "…"}], "Thinking…"

    while True:
        try:
            if provider == "Anthropic":
                text, tool_calls, raw = _call_anthropic(api_msgs, api_key)
                api_msgs.append({"role": "assistant", "content": raw})
            else:
                text, tool_calls, raw = _call_openai(api_msgs, api_key)
                api_msgs.append(raw)
        except Exception as e:
            display.append({"role": "assistant", "content": f"❌ {e}"})
            yield display, "Error."
            return

        if tool_calls:
            for tc in tool_calls:
                name  = tc.name  if provider == "Anthropic" else tc.function.name
                inp   = tc.input if provider == "Anthropic" else json.loads(tc.function.arguments)
                tc_id = tc.id

                display.append({"role": "assistant", "content": f"`{name}`\n```json\n{json.dumps(inp, indent=2)}\n```"})
                yield display, f"Calling {name}…"

                result = dispatch_tool(name, inp)

                display.append({"role": "assistant", "content": f"```json\n{result}\n```"})
                yield display, f"{name} done."

                if provider == "Anthropic":
                    api_msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": result}]})
                else:
                    api_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": result})
        else:
            if text:
                display.append({"role": "assistant", "content": text})
            yield display, "Done."
            return


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="mcp") as demo:
    gr.Markdown(
        "# mcp\n"
        "Connect to Ableton Live → control via Claude or GPT."
    )

    with gr.Sidebar(position="right"):
        provider_sel = gr.Radio(
            ["Anthropic", "OpenAI"],
            value="Anthropic",
            label="Provider",
        )
        api_key_in = gr.Textbox(
            placeholder="sk-ant-... or sk-...",
            label="API key",
            type="password",
            elem_id="api-key-input",
        )
        conn_status = gr.Textbox(
            value="",
            label="Ableton connection",
            interactive=False,
        )
        check_btn = gr.Button("Check connection", variant="secondary")

        with gr.Accordion("Remote script", open=False):
            gr.Markdown(
                "Download and extract `AbletonMCP.zip` into your Ableton Remote "
                "Scripts folder. Then select `AbletonMCP` as a Control Surface in "
                "Live's MIDI preferences."
            )
            gr.File(value=str(SCRIPT_ZIP), label="AbletonMCP.zip", interactive=False)

    chatbot = gr.Chatbot(
        label="Chat",
        height=500,
        layout="bubble",
        buttons=["copy"],
    )
    with gr.Row():
        msg_in   = gr.Textbox(placeholder="Ask Claude or GPT to control Ableton…", label="", scale=5)
        send_btn = gr.Button("Send →", variant="primary", scale=1)

    status_bar = gr.Textbox(interactive=False, show_label=False, placeholder="Status…")

    check_btn.click(fn=check_connection, inputs=[], outputs=[conn_status])

    # Persist API key in localStorage
    demo.load(
        fn=None,
        outputs=[api_key_in],
        js="() => { return [localStorage.getItem('mcp_api_key') || '']; }",
    )
    api_key_in.change(
        fn=None,
        inputs=[api_key_in],
        js="(key) => { localStorage.setItem('mcp_api_key', key); }",
    )

    def on_send(user_msg, history, provider, api_key):
        yield "", history, "Sending…"
        for new_history, status in chat(user_msg, history, provider, api_key):
            yield "", new_history, status

    send_btn.click(
        fn=on_send,
        inputs=[msg_in, chatbot, provider_sel, api_key_in],
        outputs=[msg_in, chatbot, status_bar],
    )
    msg_in.submit(
        fn=on_send,
        inputs=[msg_in, chatbot, provider_sel, api_key_in],
        outputs=[msg_in, chatbot, status_bar],
    )


if __name__ == "__main__":
    demo.launch()
