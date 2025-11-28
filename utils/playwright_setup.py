"""Utility to ensure Playwright and Chromium are installed."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

# Fix for Windows - Playwright requires ProactorEventLoopPolicy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def ensure_playwright_chromium(silent: bool = True) -> bool:
    """Ensure Playwright Chromium browser is installed.
    
    Checks if Playwright is available and if Chromium is installed.
    If not, attempts to install Chromium programmatically.
    
    Args:
        silent: If True, don't show Streamlit messages (for startup).
    
    Returns:
        bool: True if Playwright Chromium is available, False otherwise.
    """
    # Check if Playwright is installed
    try:
        import playwright  # noqa: F401
    except ImportError:
        if not silent:
            import streamlit as st
            st.warning("Playwright is not installed. PDF export will not be available.")
        return False
    
    # Check if Chromium is already installed by checking the file system
    # This avoids launching Playwright which causes Windows event loop issues
    try:
        import os
        import platform
        from pathlib import Path
        
        # Playwright stores browsers in platform-specific locations
        system = platform.system().lower()
        home = Path.home()
        
        if system == "windows":
            # Windows: %USERPROFILE%\AppData\Local\ms-playwright
            chromium_base = home / "AppData" / "Local" / "ms-playwright"
        elif system == "darwin":
            # macOS: ~/Library/Caches/ms-playwright
            chromium_base = home / "Library" / "Caches" / "ms-playwright"
        else:
            # Linux: ~/.cache/ms-playwright
            chromium_base = home / ".cache" / "ms-playwright"
        
        # Check if any chromium directory exists
        if chromium_base.exists():
            chromium_dirs = list(chromium_base.glob("chromium-*"))
            if chromium_dirs:
                # Check if the executable exists
                for chromium_dir in chromium_dirs:
                    if system == "windows":
                        exe_path = chromium_dir / "chrome-win" / "chrome.exe"
                    elif system == "darwin":
                        exe_path = chromium_dir / "chrome-mac" / "Chromium.app" / "Contents" / "MacOS" / "Chromium"
                    else:
                        exe_path = chromium_dir / "chrome" / "chrome"
                    
                    if exe_path.exists():
                        return True
    except Exception as e:
        # If we can't check, proceed with installation attempt
        logger.debug(f"Could not verify Chromium installation: {e}")
        pass
    
    # Install Chromium
    try:
        import streamlit as st
        
        if "playwright_chromium_installing" in st.session_state:
            # Already installing, skip
            return False
        
        st.session_state["playwright_chromium_installing"] = True
        
        if not silent:
            with st.spinner("Installing Playwright Chromium browser (this may take a few minutes)..."):
                result = _install_chromium()
        else:
            result = _install_chromium()
        
        del st.session_state["playwright_chromium_installing"]
        
        if result:
            if not silent:
                st.success("✅ Playwright Chromium installed successfully!")
            return True
        else:
            if not silent:
                st.error("❌ Failed to install Chromium. Please install manually: `playwright install chromium`")
            return False
    except subprocess.TimeoutExpired:
        if not silent:
            import streamlit as st
            st.error("⏱️ Chromium installation timed out. Please install manually: `playwright install chromium`")
        if "playwright_chromium_installing" in st.session_state:
            del st.session_state["playwright_chromium_installing"]
        return False
    except Exception as e:
        logger.error(f"Error installing Chromium: {e}")
        if not silent:
            import streamlit as st
            st.error(f"❌ Error installing Chromium: {e}")
        if "playwright_chromium_installing" in st.session_state:
            del st.session_state["playwright_chromium_installing"]
        return False
    
    return False


def _install_chromium() -> bool:
    """Install Chromium using playwright CLI.
    
    Returns:
        bool: True if installation succeeded, False otherwise.
    """
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    
    return result.returncode == 0


def check_playwright_on_startup():
    """Check and install Playwright Chromium on app startup.
    
    This should be called once when the app starts. It uses session state
    to ensure it only runs once per session. Runs silently to avoid blocking
    the UI on startup.
    """
    try:
        import streamlit as st
        
        if "playwright_checked" not in st.session_state:
            # Run silently on startup
            ensure_playwright_chromium(silent=True)
            st.session_state["playwright_checked"] = True
    except Exception as e:
        # Silently fail on startup - will show error when user tries to use PDF export
        logger.debug(f"Playwright check on startup failed: {e}")
        try:
            import streamlit as st
            st.session_state["playwright_checked"] = True
        except Exception:
            pass

