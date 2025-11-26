from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT / "stock_analysis" / "workspace"
REPORT_TEMPLATE_PATH = PROJECT_ROOT / "stock_analysis" / "templates" / "report_template.md"
RESET_MESSAGE_KEY = "workspace_reset_flash"
RESET_MODAL_KEY = "workspace_reset_modal_open"
RESET_STYLE_KEY = "workspace_reset_style"
RESET_ANCHOR_ID = "workspace-reset-anchor"
CHAT_RESET_KEYS = [
    "chat_messages",
    "deep_agent",
    "chat_running",
    "chat_cancel_requested",
    "chat_main_input",
    "chat_stop_input",
]


def _rerun_app():
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
    else:
        experimental_rerun = getattr(st, "experimental_rerun", None)
        if experimental_rerun is not None:  # pragma: no cover - fallback for older Streamlit
            experimental_rerun()


def _clear_workspace():
    if WORKSPACE_ROOT.exists():
        try:
            shutil.rmtree(WORKSPACE_ROOT)
        except OSError as exc:
            raise RuntimeError(f"Unable to remove workspace folder {WORKSPACE_ROOT}: {exc}") from exc
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def reset_workspace() -> tuple[bool, str]:
    try:
        _clear_workspace()
        for key in CHAT_RESET_KEYS:
            st.session_state.pop(key, None)
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Workspace reset failed: {exc}"
    return True, "Workspace reset complete. The workspace directory is now empty."


def render_workspace_reset_button():
    def _inject_styles():
        if st.session_state.get(RESET_STYLE_KEY):
            return
        st.markdown(
            f"""
            <style>
            #{RESET_ANCHOR_ID} ~ div[data-testid="stButton"] button {{
                background-color: #c0392b !important;
                border: 1px solid #922b21 !important;
                color: #ffffff !important;
                padding: 0.15rem 0.8rem !important;
                font-size: 0.85rem !important;
            }}
            #{RESET_ANCHOR_ID} ~ div[data-testid="stButton"] button:hover {{
                background-color: #922b21 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state[RESET_STYLE_KEY] = True

    _inject_styles()

    message = st.session_state.pop(RESET_MESSAGE_KEY, None)
    if message:
        st.success(message)

    _, button_col = st.columns([1, 0.16])
    with button_col:
        st.markdown(f'<div id="{RESET_ANCHOR_ID}"></div>', unsafe_allow_html=True)
        if st.button(
            "Start a new research",
            key="workspace_reset_btn",
            help="Delete everything under stock_analysis/workspace and recreate default folders.",
        ):
            st.session_state[RESET_MODAL_KEY] = True

    if not st.session_state.get(RESET_MODAL_KEY):
        return

    def _render_confirmation_body():
        st.warning(
            "This will permanently delete the entire `stock_analysis/workspace` folder, "
            "including all notes, reports, and figures. This action cannot be undone.",
            icon="⚠️",
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Yes, reset workspace", type="primary", key="workspace_reset_confirm"):
                ok, detail = reset_workspace()
                if ok:
                    st.session_state[RESET_MESSAGE_KEY] = detail
                    st.session_state[RESET_MODAL_KEY] = False
                    _rerun_app()
                else:
                    st.error(detail)
        with cancel_col:
            if st.button("Cancel", key="workspace_reset_cancel"):
                st.session_state[RESET_MODAL_KEY] = False
                _rerun_app()

    if hasattr(st, "modal"):
        with st.modal("Start a new research", key="workspace_reset_modal"):
            _render_confirmation_body()
    else:
        st.markdown("---")
        st.subheader("Start a new research")
        _render_confirmation_body()

