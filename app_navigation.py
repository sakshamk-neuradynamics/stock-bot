from __future__ import annotations

import streamlit as st


def render_sidebar_nav():
    with st.sidebar:
        st.page_link("main.py", label="Data Extraction", icon="📥")
        st.page_link(
            "pages/1_File_Viewer.py",
            label="Workspace File Viewer",
            icon="📁",
        )
        st.page_link(
            "pages/2_Deep_Agent_Chat.py",
            label="Deep Agent Chat",
            icon="💬",
        )
        st.page_link(
            "pages/3_Settings.py",
            label="Settings",
            icon="⚙️",
        )


