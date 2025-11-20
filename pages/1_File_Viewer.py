from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import streamlit as st

from app_navigation import render_sidebar_nav

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "stock_analysis" / "workspace"
ROOT_LABEL = "stock_analysis/workspace"
SELECTED_FILE_KEY = "file_viewer_selected"
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
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


def render_file_metadata(file_path: Path):
    try:
        stat = file_path.stat()
    except OSError as exc:
        st.error(f"Unable to read file stats: {exc}")
        return False
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Size: {format_bytes(stat.st_size)} • Modified: {mtime}")
    return True


def render_file_preview(file_path: Path):
    if not file_path.exists():
        st.error("Selected file no longer exists.")
        return

    if not render_file_metadata(file_path):
        return

    suffix = file_path.suffix.lower()
    if suffix in MARKDOWN_EXTENSIONS:
        text, truncated = read_text_preview(file_path)
        st.markdown(text)
        if truncated:
            st.caption(f"Preview truncated after {format_bytes(MAX_PREVIEW_BYTES)}.")
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


def render_file_viewer_page():
    st.set_page_config(page_title="Stock KB - File Viewer", layout="wide")
    render_sidebar_nav()
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
        render_file_preview(selected_path)


if __name__ == "__main__":
    render_file_viewer_page()


