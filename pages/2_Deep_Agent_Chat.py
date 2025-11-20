from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st
from stock_analysis.agent import build_agent
from app_navigation import render_sidebar_nav


def render_deep_agent_chat_page():
    st.set_page_config(page_title="Stock KB - Deep Agent Chat", layout="wide")
    render_sidebar_nav()
    st.title("Deep Agent Chat")
    st.caption("Chat with the deep agent. Uses principles from principles.txt if available.")

    principles_path = Path.cwd() / "principles.txt"
    principles_text = ""
    if principles_path.exists():
        try:
            principles_text = principles_path.read_text(encoding="utf-8")
            st.caption(f"Using principles from {principles_path.name}.")
        except (OSError, UnicodeDecodeError) as e:
            st.warning(f"Could not read {principles_path.name}: {e}")
    else:
        st.caption("No principles.txt found. The agent will run without injected principles.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "deep_agent" not in st.session_state:
        st.session_state["deep_agent"] = None
    if "chat_running" not in st.session_state:
        st.session_state["chat_running"] = False
    if "chat_cancel_requested" not in st.session_state:
        st.session_state["chat_cancel_requested"] = False

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(str(msg.get("content", "")))

    pending_user_input: str | None = None
    if st.session_state["chat_running"]:
        stop_signal = st.chat_input(
            "⏹️ Stop current chat (click send to halt)",
            key="chat_stop_input",
        )
        if stop_signal is not None:
            st.session_state["chat_cancel_requested"] = True
            st.info("Stop requested. Finishing current response...")
    else:
        user_input = st.chat_input(
            "Ask the Deep Agent about stocks, sources, or reports...",
            key="chat_main_input",
        )
        pending_user_input = (user_input or "").strip()

    if pending_user_input:
        st.session_state["chat_messages"].append(
            {"role": "user", "content": pending_user_input}
        )
        with st.chat_message("user"):
            st.markdown(pending_user_input)

        if st.session_state["deep_agent"] is None:
            with st.spinner("Initializing Deep Agent..."):
                async def _build():
                    return await build_agent(principles=principles_text or None)
                try:
                    st.session_state["deep_agent"] = asyncio.run(_build())
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        st.session_state["deep_agent"] = loop.run_until_complete(_build())
                    finally:
                        loop.close()

        agent = st.session_state["deep_agent"]
        st.session_state["chat_running"] = True
        last = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()

            def _format_assistant_chunk(message_obj) -> str:
                get_attr = getattr
                content = None
                additional = {}
                response_meta = None
                tool_calls = None

                if isinstance(message_obj, dict):
                    content = message_obj.get("content")
                    additional = message_obj.get("additional_kwargs", {}) or {}
                    response_meta = message_obj.get("response_metadata")
                    tool_calls = message_obj.get("tool_calls") or additional.get("tool_calls")
                else:
                    content = getattr(message_obj, "content", None)
                    additional = getattr(message_obj, "additional_kwargs", {}) or {}
                    response_meta = getattr(message_obj, "response_metadata", None)
                    tool_calls = getattr(message_obj, "tool_calls", None) or additional.get("tool_calls")

                if isinstance(content, str) and content.strip():
                    return content

                lines = []
                if tool_calls:
                    lines.append("Planning tasks:")
                    for tc in tool_calls:
                        name = ""
                        args = {}
                        if isinstance(tc, dict):
                            name = str(tc.get("name", "") or "")
                            args = tc.get("args", {}) or {}
                        else:
                            name = str(get_attr(tc, "name", "") or "")
                            args = get_attr(tc, "args", {}) or {}
                        subagent = str(args.get("subagent_type", "") or "").strip()
                        desc = str(args.get("description", "") or "").strip()
                        first_line = desc.splitlines()[0] if desc else ""
                        title = f"- {name}"
                        if subagent:
                            title += f" ({subagent})"
                        lines.append(title)
                        if first_line:
                            lines.append(f"  - summary: {first_line}")
                        try:
                            args_json = json.dumps(args, ensure_ascii=False, indent=2)[:4000]
                        except (TypeError, ValueError):
                            args_json = str(args)[:4000]
                        lines.append("  - args:")
                        lines.append("")
                        lines.append("```json")
                        lines.append(args_json)
                        lines.append("```")

                if response_meta and isinstance(response_meta, dict):
                    model = response_meta.get("model_name") or response_meta.get("model")
                    tokens = response_meta.get("token_usage", {}).get("total_tokens")
                    meta_line = "Metadata:"
                    if model:
                        meta_line += f" model={model}"
                    if tokens is not None:
                        meta_line += f" total_tokens={tokens}"
                    lines.append(meta_line)

                return "\n".join(lines) if lines else ""

            async def _astream(a, history, ph):
                last_text = ""
                async for chunk in a.astream({"messages": history}, stream_mode="values"):
                    if st.session_state.get("chat_cancel_requested"):
                        ph.info("Stop requested. Halting response...")
                        break
                    if "messages" in chunk:
                        msg = chunk["messages"][-1]
                        formatted = _format_assistant_chunk(msg)
                        last_text = formatted or str(msg)
                        ph.markdown(last_text)
                return last_text

            try:
                last = asyncio.run(
                    _astream(agent, st.session_state["chat_messages"], placeholder)
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    last = loop.run_until_complete(
                        _astream(agent, st.session_state["chat_messages"], placeholder)
                    )
                finally:
                    loop.close()
        st.session_state["chat_running"] = False
        stop_requested = st.session_state.get("chat_cancel_requested")
        st.session_state["chat_cancel_requested"] = False

        if last and not stop_requested:
            st.session_state["chat_messages"].append({"role": "assistant", "content": last})
        elif stop_requested:
            st.info("Chat stopped before completion.")


if __name__ == "__main__":
    render_deep_agent_chat_page()


