from __future__ import annotations

import asyncio
import base64
import csv
import html
import io
import json
import mimetypes
import re
import sys
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import streamlit as st
from markdown import markdown as md_to_html

try:  # pragma: no cover - optional dependency guard
    from playwright.sync_api import Error as PlaywrightError, sync_playwright  # type: ignore[import-not-found]
    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime with message
    PlaywrightError = RuntimeError  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]
    PLAYWRIGHT_AVAILABLE = False

from app_navigation import render_sidebar_nav
from workspace_reset import render_workspace_reset_button
from utils.playwright_setup import check_playwright_on_startup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "stock_analysis" / "workspace"
ROOT_LABEL = "stock_analysis/workspace"
SELECTED_FILE_KEY = "file_viewer_selected"
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
GITHUB_MARKDOWN_CSS_PATH = PROJECT_ROOT / "assets" / "styles" / "github-markdown-light.css"
TEXT_EXTENSIONS = {
    ".txt",
    ".py",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".yml",
    ".yaml",
    ".lock",
    ".log",
    ".gitignore",
    ".csv",
    ".json",
    ".jsonl",
}
MAX_PREVIEW_BYTES = 2_000_000
MAX_CSV_ROWS = 200
MAX_JSONL_RECORDS = 200
MARKDOWN_PAGE_CSS = """
@page {
    size: A4;
    margin: 1in;
}

body {
    margin: 0;
    font-family: 'DejaVu Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #ffffff;
    -webkit-print-color-adjust: exact;
}

.markdown-body {
    box-sizing: border-box;
    max-width: 900px;
    margin: 0 auto;
    padding: 24px 32px 48px;
}

code,
pre {
    font-family: 'DejaVu Sans', 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

table {
    page-break-inside: avoid;
}
"""

MARKDOWN_FONT_FAMILY = "DejaVuSans"
MARKDOWN_FONT_FILES = {
    "": PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
    "B": PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf",
    "I": PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans-Oblique.ttf",
    "BI": PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans-BoldOblique.ttf",
}
VALID_FONT_HEADERS = {b"\x00\x01\x00\x00", b"OTTO", b"ttcf", b"true"}
IMAGE_TAG_PATTERN = re.compile(r'(<img[^>]+src=["\'])([^"\']+)(["\'][^>]*>)', re.IGNORECASE)
EDIT_MODE_PREFIX = "fv_mode::"
EDITOR_WIDGET_PREFIX = "fv_editor_widget::"
EDITOR_PATH_PREFIX = "fv_editor_path::"
EDITOR_PENDING_VALUE_PREFIX = "fv_editor_pending_value::"
EDITOR_MESSAGE_PREFIX = "fv_editor_message::"

if sys.platform.startswith("win"):
    # Playwright relies on asyncio subprocess support, which requires the proactor policy on Windows.
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


@lru_cache(maxsize=1)
def ensure_markdown_fonts() -> str:
    """Ensure the DejaVu font files exist and return @font-face CSS rules."""
    missing = [style for style, path in MARKDOWN_FONT_FILES.items() if not path.exists()]
    if missing:
        missing_desc = ", ".join(sorted(style or "regular" for style in missing))
        raise FileNotFoundError(
            f"Markdown PDF font files are missing ({missing_desc}). "
            "Ensure assets/fonts contains the DejaVu Sans family."
        )

    font_rules: list[str] = []
    for style, font_path in MARKDOWN_FONT_FILES.items():
        try:
            header = font_path.read_bytes()[:4]
        except OSError as exc:
            raise FileNotFoundError(f"Unable to read font file {font_path}: {exc}") from exc
        if header not in VALID_FONT_HEADERS:
            raise ValueError(
                f"Font file {font_path} is not a valid TrueType/OpenType font. "
                "Re-download the DejaVu Sans fonts (ttf)."
            )
        is_bold = "B" in style.upper()
        is_italic = "I" in style.upper()
        weight = "700" if is_bold else "400"
        font_style = "italic" if is_italic else "normal"
        font_rules.append(
            f"""
@font-face {{
    font-family: '{MARKDOWN_FONT_FAMILY}';
    font-style: {font_style};
    font-weight: {weight};
    src: url('{font_path.as_uri()}') format('truetype');
}}
"""
        )
    return "\n".join(font_rules)


@lru_cache(maxsize=1)
def load_github_markdown_css() -> str:
    """Load the GitHub Markdown CSS used to style rendered documents."""
    try:
        return GITHUB_MARKDOWN_CSS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            "GitHub Markdown CSS not found. Download github-markdown-light.css "
            "into assets/styles/ to enable styled PDF exports."
        ) from exc


def render_html_to_pdf(document: str) -> bytes:
    """Render HTML into a PDF using Playwright/Chromium."""
    if not PLAYWRIGHT_AVAILABLE or sync_playwright is None:
        raise ValueError(
            "Playwright is required for Markdown PDF export. "
            "Install it (`uv pip install playwright`) and run `playwright install chromium`."
        )

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(document, wait_until="networkidle")
            pdf_bytes = page.pdf(
                format="A4",
                margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
                print_background=True,
            )
            browser.close()
            return pdf_bytes
    except PlaywrightError as exc:  # pragma: no cover - depends on local browser install
        raise RuntimeError(
            "Unable to render Markdown as PDF via Playwright. "
            "Ensure Chromium is installed with `playwright install chromium`."
        ) from exc


def markdown_to_pdf_bytes(
    markdown_text: str,
    title: str | None = None,
    source_path: Path | None = None,
) -> bytes:
    """Convert Markdown text into a rendered PDF and return the raw bytes."""
    markdown_html = md_to_html(
        markdown_text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
        ],
        output_format="html5",
    )
    if source_path is not None:
        markdown_html = inline_local_image_sources(markdown_html, source_path)
    css_block = "\n".join(
        [
            ensure_markdown_fonts(),
            load_github_markdown_css(),
            MARKDOWN_PAGE_CSS,
        ]
    )
    safe_title = html.escape(title or "Markdown Document")
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{safe_title}</title>
  <style>
  {css_block}
  </style>
</head>
<body>
  <article class="markdown-body">
    {markdown_html}
  </article>
</body>
</html>
"""
    return render_html_to_pdf(document)


def render_markdown_export_button(file_path: Path):
    """Render a download button that exports the selected Markdown file as PDF."""
    try:
        markdown_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        st.error(f"Unable to read Markdown file for export: {exc}")
        return

    try:
        pdf_bytes = markdown_to_pdf_bytes(markdown_text, title=file_path.name, source_path=file_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pylint: disable=broad-except  # pragma: no cover - defensive
        st.error(f"Unable to render Markdown as PDF: {exc}")
        return

    st.download_button(
        "Export Markdown to PDF",
        data=pdf_bytes,
        file_name=f"{file_path.stem}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=False,
        help="Downloads a rendered PDF version of the entire Markdown file.",
    )
    st.caption("Exports the Markdown with headings, lists, tables, and code blocks rendered.")


def list_directory_entries(directory: Path) -> Tuple[list[Path], list[Path]]:
    try:
        children = list(directory.iterdir())
    except OSError as exc:
        st.error(f"Unable to read directory {directory}: {exc}")
        return [], []

    dirs = sorted([child for child in children if child.is_dir()], key=lambda p: p.name.lower())
    files = sorted([child for child in children if child.is_file()], key=lambda p: p.name.lower())
    return dirs, files


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def read_text_preview(path: Path, max_bytes: int = MAX_PREVIEW_BYTES) -> Tuple[str, bool]:
    with path.open("rb") as handle:
        data = handle.read(max_bytes)
        truncated = bool(handle.read(1))
    text = data.decode("utf-8", errors="replace")
    return text, truncated


def render_csv_preview(file_path: Path):
    st.markdown("**CSV Preview**")
    rows: list[list[str]] = []
    truncated = False
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx + 1 >= MAX_CSV_ROWS:
                    truncated = True
                    break
    except UnicodeDecodeError as exc:
        st.error(f"Unable to decode CSV as UTF-8: {exc}")
        return

    if not rows:
        st.info("CSV file is empty.")
        return

    num_cols = max(len(r) for r in rows)
    padded_rows = [row + [""] * (num_cols - len(row)) for row in rows]
    header = padded_rows[0]
    data_rows = padded_rows[1:] if len(padded_rows) > 1 else []

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in data_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    st.markdown("\n".join(md_lines))

    if truncated:
        st.caption(f"Showing first {MAX_CSV_ROWS} rows.")


def render_json_preview(file_path: Path):
    st.markdown("**JSON Preview**")
    try:
        obj = json.loads(file_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.error(f"Unable to parse JSON: {exc}")
        return
    st.json(obj)


def render_jsonl_preview(file_path: Path):
    st.markdown("**JSONL Preview**")
    records = []
    truncated = False
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON on line {idx + 1}: {exc}")
                    return
                if len(records) >= MAX_JSONL_RECORDS:
                    truncated = True
                    break
    except UnicodeDecodeError as exc:
        st.error(f"Unable to decode JSONL as UTF-8: {exc}")
        return

    if not records:
        st.info("JSONL file is empty.")
        return
    st.json(records)
    if truncated:
        st.caption(f"Showing first {MAX_JSONL_RECORDS} records.")


def render_generic_text_preview(file_path: Path, language: str = "text"):
    text, truncated = read_text_preview(file_path)
    st.code(text, language=language)
    if truncated:
        st.caption(f"Preview truncated after {format_bytes(MAX_PREVIEW_BYTES)}.")


def inline_local_image_sources(markdown_html: str, markdown_path: Path) -> str:
    """Replace <img> tags that reference local files with data URIs so Streamlit can render them."""
    workspace_root = WORKSPACE_ROOT.resolve()

    def replace_src(match: re.Match[str]) -> str:
        prefix, src, suffix = match.groups()
        parsed = urlparse(src)
        if parsed.scheme or src.startswith("data:"):
            return match.group(0)

        candidate = (markdown_path.parent / src).resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError:
            return match.group(0)

        if not candidate.is_file():
            return match.group(0)

        mime_type, _ = mimetypes.guess_type(candidate.name)
        mime_type = mime_type or "application/octet-stream"
        try:
            encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
        except OSError:
            return match.group(0)

        return f"{prefix}data:{mime_type};base64,{encoded}{suffix}"

    return IMAGE_TAG_PATTERN.sub(replace_src, markdown_html)


def trigger_rerun():
    """Trigger a Streamlit rerun compatible with newer and older versions."""
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
    else:  # pragma: no cover - fallback for older Streamlit
        st.experimental_rerun()  # type: ignore[attr-defined]


def is_editable_suffix(suffix: str) -> bool:
    """Return True if the file extension should be editable in the UI."""
    return suffix in MARKDOWN_EXTENSIONS or suffix in TEXT_EXTENSIONS or suffix in {".json", ".jsonl"} or suffix == ""


def render_file_editor(file_path: Path, rel_key: str):
    """Render a textarea editor for UTF-8 text files with save/discard controls."""
    editor_key = f"{EDITOR_WIDGET_PREFIX}{rel_key}"
    path_key = f"{EDITOR_PATH_PREFIX}{rel_key}"
    pending_key = f"{EDITOR_PENDING_VALUE_PREFIX}{rel_key}"
    message_key = f"{EDITOR_MESSAGE_PREFIX}{rel_key}"
    try:
        current_disk_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        st.error("This file is not UTF-8 encoded, so inline editing is disabled.")
        return
    except OSError as exc:
        st.error(f"Unable to read file for editing: {exc}")
        return

    if st.session_state.get(path_key) != str(file_path):
        st.session_state[path_key] = str(file_path)
        st.session_state[pending_key] = current_disk_text

    if pending_key in st.session_state:
        st.session_state[editor_key] = st.session_state.pop(pending_key)
    elif editor_key not in st.session_state:
        st.session_state[editor_key] = current_disk_text

    message = st.session_state.pop(message_key, None)
    if message:
        st.info(message)

    edited_text = st.text_area(
        "Edit file contents",
        key=editor_key,
        height=400,
        help="Changes are stored in the text box until you click Save.",
    )

    col_save, col_discard = st.columns(2)
    with col_save:
        if st.button("💾 Save changes", key=f"{editor_key}::save", use_container_width=True):
            try:
                file_path.write_text(edited_text, encoding="utf-8")
            except OSError as exc:
                st.error(f"Unable to save file: {exc}")
            else:
                st.session_state[pending_key] = edited_text
                st.session_state[path_key] = str(file_path)
                st.session_state[message_key] = "File updated."
                trigger_rerun()
    with col_discard:
        if st.button("↩️ Discard & reload", key=f"{editor_key}::discard", use_container_width=True):
            try:
                refreshed = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                st.error(f"Unable to reload file: {exc}")
            else:
                st.session_state[pending_key] = refreshed
                st.session_state[path_key] = str(file_path)
                st.session_state[message_key] = "Re-loaded from disk; unsaved changes were discarded."
                trigger_rerun()


def render_markdown_preview(file_path: Path):
    """Render Markdown with GitHub-style formatting and inline local images."""
    text, truncated = read_text_preview(file_path)
    markdown_html = md_to_html(
        text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
        ],
        output_format="html5",
    )
    markdown_html = inline_local_image_sources(markdown_html, file_path)

    try:
        css = load_github_markdown_css()
    except FileNotFoundError:
        css = ""

    html_block = (
        "<style>"
        f"{css}"
        "</style>"
        '<article class="markdown-body">'
        f"{markdown_html}"
        "</article>"
    )
    st.markdown(html_block, unsafe_allow_html=True)
    if truncated:
        st.caption(f"Preview truncated after {format_bytes(MAX_PREVIEW_BYTES)}.")


def render_file_metadata(file_path: Path):
    try:
        stat = file_path.stat()
    except OSError as exc:
        st.error(f"Unable to read file stats: {exc}")
        return False
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Size: {format_bytes(stat.st_size)} • Modified: {mtime}")
    return True


def render_file_preview(file_path: Path, rel_path: str | None = None):
    if not file_path.exists():
        st.error("Selected file no longer exists.")
        return

    if not render_file_metadata(file_path):
        return

    suffix = file_path.suffix.lower()
    try:
        rel_key = rel_path or file_path.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        rel_key = file_path.as_posix()

    mode_key = f"{EDIT_MODE_PREFIX}{rel_key}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Preview"
    mode = st.radio(
        "File mode",
        ["Preview", "Edit"],
        key=mode_key,
        horizontal=True,
        help="Switch between previewing the file or editing it inline.",
    )

    if mode == "Edit":
        if is_editable_suffix(suffix):
            render_file_editor(file_path, rel_key)
        else:
            st.warning("Editing is only supported for UTF-8 text files. Switch back to preview mode for this file.")
        return

    if suffix in MARKDOWN_EXTENSIONS:
        render_markdown_export_button(file_path)
        render_markdown_preview(file_path)
    elif suffix == ".csv":
        render_csv_preview(file_path)
    elif suffix == ".json":
        render_json_preview(file_path)
    elif suffix == ".jsonl":
        render_jsonl_preview(file_path)
    elif suffix in TEXT_EXTENSIONS or suffix == "":
        render_generic_text_preview(file_path)
    else:
        st.info("Binary or unsupported format. Download or open manually to inspect this file.")


def folder_state_key(rel_path: Path) -> str:
    rel = rel_path.as_posix()
    return f"fv_expanded::{rel if rel != '.' else '__root__'}"


def render_file_entry(file_path: Path, rel_path: Path, depth: int):
    selected = st.session_state.get(SELECTED_FILE_KEY)
    is_selected = selected == rel_path.as_posix()
    indent = " " * (depth * 2)
    label = f"{indent}📄 {file_path.name}"
    if st.button(
        label,
        key=f"fv_file::{rel_path.as_posix()}",
        use_container_width=True,
        type="primary" if is_selected else "secondary",
    ):
        st.session_state[SELECTED_FILE_KEY] = rel_path.as_posix()


def render_directory_node(directory: Path, rel_path: Path, depth: int, display_name: str | None = None):
    state_key = folder_state_key(rel_path)
    if state_key not in st.session_state:
        st.session_state[state_key] = depth == 0
    expanded = st.session_state[state_key]

    indent = " " * (depth * 2)
    caret = "▼" if expanded else "▶"
    name = display_name or f"{directory.name}/"
    label = f"{indent}{caret} 📁 {name}"

    if st.button(label, key=f"{state_key}::btn", use_container_width=True):
        st.session_state[state_key] = not expanded
        expanded = not expanded

    if not expanded:
        return

    subdirs, files = list_directory_entries(directory)
    if not subdirs and not files and depth == 0:
        st.caption("This workspace folder is currently empty.")
        return

    for child in subdirs:
        render_directory_node(child, rel_path / child.name, depth + 1)
    for file_path in files:
        render_file_entry(file_path, rel_path / file_path.name, depth + 1)


def create_workspace_zip(directory: Path) -> bytes:
    """Create a ZIP file of the workspace directory in memory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                archive_name = file_path.relative_to(directory)
                zip_file.write(file_path, arcname=archive_name)
    return zip_buffer.getvalue()


def render_file_viewer_page():
    st.set_page_config(page_title="Stock KB - File Viewer", layout="wide")
    
    # Ensure Playwright Chromium is installed on startup
    check_playwright_on_startup()
    
    render_sidebar_nav()
    
    # Download Workspace ZIP logic
    zip_bytes = create_workspace_zip(WORKSPACE_ROOT) if WORKSPACE_ROOT.exists() else None
    download_btn = {
        "label": "📦 Download Workspace",
        "data": zip_bytes,
        "file_name": "workspace_backup.zip",
        "mime": "application/zip",
        "help": "Download the entire workspace directory as a compressed ZIP file.",
    }

    render_workspace_reset_button(download_button=download_btn if zip_bytes else None)
    
    st.title("Workspace File Viewer")
    st.caption(
        "Browse everything under stock_analysis/workspace and open files with a single click."
    )

    if not WORKSPACE_ROOT.exists():
        st.warning(f"Expected folder `{ROOT_LABEL}` was not found.")
        return
    
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Folders & Files")
        st.caption(f"Root: `{ROOT_LABEL}`")
        render_directory_node(WORKSPACE_ROOT, Path("."), depth=0, display_name=f"{ROOT_LABEL}/")

    with right:
        st.subheader("Preview")
        selected_rel = st.session_state.get(SELECTED_FILE_KEY)
        if not selected_rel:
            st.info("Select a file from the left panel to preview its contents.")
            return

        selected_path = (WORKSPACE_ROOT / selected_rel).resolve()
        try:
            selected_path.relative_to(WORKSPACE_ROOT.resolve())
        except ValueError:
            st.error("Selected path is outside the workspace root.")
            return

        st.markdown(f"**Selected:** `{selected_rel}`")
        render_file_preview(selected_path, selected_rel)


if __name__ == "__main__":
    render_file_viewer_page()


