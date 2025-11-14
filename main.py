"""Streamlit app to manage files in a local `temp` directory.

This app lets you upload files (saved into `temp`), view details, select files
individually or in bulk, and delete single or multiple files.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
from core_principles.graph import graph as core_graph
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from fpdf import FPDF


def ensure_temp_dir() -> Path:
    """Ensure the local `temp` directory exists and return its path.

    Returns:
        Path: Absolute path to the `temp` directory within the project.
    """
    base_dir = Path(__file__).parent
    temp_dir = base_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def format_bytes(num_bytes: int) -> str:
    """Convert a byte count into a human‑readable string.

    Args:
        num_bytes: File size in bytes.

    Returns:
        A string like "123.45 KB" or "1.23 GB".
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def unique_destination_path(directory: Path, filename: str) -> Path:
    """Return a non‑conflicting path for a given filename inside directory.

    If `filename` already exists, append an incrementing suffix before the
    extension, e.g., `file.txt` -> `file_1.txt`, `file_2.txt`, etc.

    Args:
        directory: Target directory.
        filename: Desired filename.

    Returns:
        Path: A unique path that does not currently exist.
    """
    destination = directory / filename
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_uploaded_files(uploaded_files: Iterable) -> List[Path]:
    """Persist uploaded files to the `temp` directory.

    Args:
        uploaded_files: Iterable of Streamlit `UploadedFile` objects.

    Returns:
        List[Path]: Paths of files successfully saved.
    """
    temp_dir = ensure_temp_dir()
    saved_paths: List[Path] = []
    for uploaded in uploaded_files:
        if uploaded is None or uploaded.size == 0:
            continue
        destination = unique_destination_path(temp_dir, uploaded.name)
        with open(destination, "wb") as f:
            f.write(uploaded.read())
        saved_paths.append(destination)
    return saved_paths


def list_temp_files() -> List[Path]:
    """List files currently present in the `temp` directory.

    Returns:
        List[Path]: File paths sorted lexicographically (case‑insensitive).
    """
    temp_dir = ensure_temp_dir()
    return sorted(
        [p for p in temp_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower()
    )


def delete_files(paths: Iterable[Path]) -> int:
    """Delete files provided by their paths.

    Args:
        paths: Iterable of `Path` objects to delete.

    Returns:
        int: Number of files successfully deleted.
    """
    deleted = 0
    for path in paths:
        if path.exists() and path.is_file():
            try:
                path.unlink()
                deleted += 1
            except PermissionError:
                # Skip files that cannot be deleted due to permissions
                continue
    return deleted


def render_uploader():
    """Render the uploader UI and handle saving uploaded files."""
    st.subheader("Upload Files")
    if "uploader_counter" not in st.session_state:
        st.session_state["uploader_counter"] = 0

    uploaded_files = st.file_uploader(
        label="Choose files to upload",
        type=None,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_counter']}",
        help=(
            "Uploaded files are saved to the 'temp' folder in this project"
        ),
    )

    is_disabled = not uploaded_files
    if st.button("Save to temp", type="primary", disabled=is_disabled):
        saved = save_uploaded_files(uploaded_files)
        if saved:
            st.success(f"Uploaded {len(saved)} file(s) to temp.")
        # Reset the uploader by changing its key
        st.session_state["uploader_counter"] += 1
        st.rerun()


def render_file_manager():
    """Render the file list with selection and deletion controls."""
    st.subheader("Temp Folder Files")
    files = list_temp_files()

    file_names = [f.name for f in files]

    # Header controls (even spacing): count | select all | clear | delete selected
    selected_names_pre = [
        name
        for idx, name in enumerate(file_names)
        if st.session_state.get(f"ck_{name}_{idx}", False)
    ]
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        count_ph = st.empty()
        count_ph.caption(f"Selected: {len(selected_names_pre)}")
    with h2:
        if st.button(
            "Select All", use_container_width=True, disabled=len(file_names) == 0
        ):
            for idx, name in enumerate(file_names):
                st.session_state[f"ck_{name}_{idx}"] = True
            st.rerun()
    with h3:
        if st.button(
            "Clear Selection",
            use_container_width=True,
            disabled=len(file_names) == 0,
        ):
            for idx, name in enumerate(file_names):
                st.session_state[f"ck_{name}_{idx}"] = False
            st.rerun()
    with h4:
        disabled_bulk_header = len(selected_names_pre) == 0
        if st.button(
            "Delete Selected",
            type="primary",
            use_container_width=True,
            disabled=disabled_bulk_header,
        ):
            targets = [p for p in files if p.name in selected_names_pre]
            count = delete_files(targets)
            if count > 0:
                st.success(f"Deleted {count} file(s).")
            st.rerun()

    st.divider()

    if not files:
        st.info("No files in temp. Upload some to get started.")
        return

    header_cols = st.columns([6, 2, 2, 1])
    header_cols[0].markdown("**File**")
    header_cols[1].markdown("**Size**")
    header_cols[2].markdown("**Modified**")
    header_cols[3].markdown("**Delete**")

    for idx, fpath in enumerate(files):
        cols = st.columns([6, 2, 2, 1])

        with cols[0]:
            st.checkbox(
                label=fpath.name,
                key=f"ck_{fpath.name}_{idx}",
            )

        with cols[1]:
            try:
                size = format_bytes(fpath.stat().st_size)
            except FileNotFoundError:
                size = "-"
            st.text(size)

        with cols[2]:
            try:
                mtime = datetime.fromtimestamp(
                    fpath.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
            except FileNotFoundError:
                mtime = "-"
            st.text(mtime)

        with cols[3]:
            if st.button("🗑️", key=f"del_{fpath.name}_{idx}"):
                delete_files([fpath])
                st.rerun()

    st.divider()

    # Update header counter after rendering checkboxes so it reflects current state
    selected_names_post = [
        name
        for idx, name in enumerate(file_names)
        if st.session_state.get(f"ck_{name}_{idx}", False)
    ]
    count_ph.caption(f"Selected: {len(selected_names_post)}")


def render_principles_extractor():
    """Render UI to run the core-principles graph and display results."""
    st.subheader("Extract Core Principles")

    if "principles_result" not in st.session_state:
        st.session_state["principles_result"] = None
    if "prompt_result" not in st.session_state:
        st.session_state["prompt_result"] = None

    investor = st.text_input(
        "Investor name",
        key="investor_name_input",
        placeholder="e.g., Warren Buffett",
    )

    run_disabled = investor.strip() == ""
    if st.button("Extract principles", type="primary", disabled=run_disabled):
        with st.spinner("Extracting core principles from temp documents..."):
            configure_hf_cache()
            result_state = core_graph.invoke(
                {"investor_name": investor.strip()},
                config={"configurables": {"batch_size": 5}},
            )
            # Handle possible shapes of outputs defensively
            principles_value = result_state.get("core_principles")
            output_value = result_state.get("output", "")

            st.session_state["principles_result"] = principles_value
            st.session_state["prompt_result"] = output_value
            st.success("Extraction complete.")

    principles_value = st.session_state.get("principles_result")
    prompt_value = st.session_state.get("prompt_result")

    if principles_value is not None or prompt_value:
        st.divider()

    if principles_value is not None:
        st.markdown("**Core Principles**")
        # Normalize to list[str]
        if isinstance(principles_value, str):
            items = [
                line.strip(" -\t")
                for line in principles_value.splitlines()
                if line.strip()
            ]
        elif isinstance(principles_value, list):
            items = [str(x).strip() for x in principles_value if str(x).strip()]
        else:
            items = [str(principles_value).strip()]

        # Render list and copy-friendly code block
        for idx, item in enumerate(items, start=1):
            st.write(f"{idx}. {item}")
        st.code("\n".join(items))

    if prompt_value:
        st.markdown("**Generated Prompt**")
        st.code(str(prompt_value))


def configure_hf_cache():
    """Configure Hugging Face cache to avoid symlink privilege issues on Windows.

    - Forces a local cache directory inside the project (".hf_cache").
    - Disables use of symlinks, which require elevated privileges on Windows.
    """
    cache_dir = Path.cwd() / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def extract_yt_video_id(url: str) -> str | None:
    """Extract YouTube video ID from common URL formats."""
    try:
        parsed = urlparse(url.strip())
        if not parsed.netloc:
            return None
        host = parsed.netloc.lower()
        # youtu.be/<id>
        if "youtu.be" in host:
            vid = parsed.path.lstrip("/").split("/")[0]
            return vid or None
        # youtube.com/watch?v=<id>
        if "youtube.com" in host:
            if parsed.path.startswith("/watch"):
                qs = parse_qs(parsed.query or "")
                vals = qs.get("v", [])
                return vals[0] if vals else None
            # youtube.com/shorts/<id>
            if parsed.path.startswith("/shorts/"):
                parts = parsed.path.split("/")
                return parts[2] if len(parts) > 2 else None
        return None
    except Exception:
        return None


def transcript_snippets_to_text(snippets) -> str:
    """Convert transcript snippets to a readable text block."""
    lines: List[str] = []
    for snip in snippets:
        # each snippet has text, start, duration
        text = (snip.text or "").strip()
        if not text:
            continue
        lines.append(text)
    # Join with line breaks to keep natural pacing
    return "\n".join(lines)


def write_pdf_from_text(text: str, out_path: Path, header: str | None = None):
    """Write plain text (and optional header) to a simple PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=14)
    if header:
        pdf.multi_cell(0, 8, header)
        pdf.ln(4)
    pdf.set_font("Times", size=12)
    # Split into paragraphs to avoid extremely long cells
    for paragraph in text.split("\n\n"):
        pdf.multi_cell(0, 6, paragraph.strip())
        pdf.ln(1)
    pdf.output(str(out_path))


def save_youtube_transcripts(urls: List[str], languages: List[str] | None = None) -> List[Path]:
    """Fetch transcripts for provided YouTube URLs and save each to a PDF in temp/."""
    temp_dir = ensure_temp_dir()
    api = YouTubeTranscriptApi()
    saved: List[Path] = []
    langs = languages or ["en"]
    for raw_url in urls:
        url = (raw_url or "").strip()
        if not url:
            continue
        vid = extract_yt_video_id(url)
        if not vid:
            st.warning(f"Could not parse video ID: {url}")
            continue
        try:
            fetched = api.fetch(vid, languages=langs)
            text = transcript_snippets_to_text(fetched)
            if not text.strip():
                st.warning(f"No transcript text found for: {vid}")
                continue
            filename = f"yt_{vid}.pdf"
            out_path = unique_destination_path(temp_dir, filename)
            header = f"YouTube Transcript\nVideo ID: {vid}\nURL: {url}\n"
            write_pdf_from_text(text, out_path, header=header)
            saved.append(out_path)
        except (TranscriptsDisabled, NoTranscriptFound):
            st.error(f"Transcript not available for: {vid}")
        except Exception as e:
            st.error(f"Failed to fetch transcript for {vid}: {e}")
    return saved


def render_youtube_transcriber():
    """Render UI to accept YouTube links and save transcripts as PDFs to temp/."""
    st.subheader("YouTube → Transcript to PDF")
    st.caption("Paste one or more YouTube links (one per line). Transcripts will be saved into temp/ as PDFs.")
    default_langs = "en"
    urls_text = st.text_area("YouTube URLs (one per line)", height=120, placeholder="https://www.youtube.com/watch?v=VIDEO_ID\nhttps://youtu.be/VIDEO_ID2")
    lang_codes = st.text_input("Languages (priority order, comma-separated)", value=default_langs, help="Example: en,de will try English first then German.")
    urls = [u.strip() for u in (urls_text.splitlines() if urls_text else []) if u.strip()]
    langs = [c.strip() for c in (lang_codes.split(",") if lang_codes else []) if c.strip()]
    disabled = len(urls) == 0
    if st.button("Fetch transcripts to PDF", type="primary", disabled=disabled):
        with st.spinner("Fetching transcripts and generating PDFs..."):
            saved = save_youtube_transcripts(urls, languages=langs or ["en"])
            if saved:
                st.success(f"Saved {len(saved)} transcript PDF(s) to temp/")
            else:
                st.info("No transcripts saved.")


def main():
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="Stock KB - File Manager", layout="wide")
    st.title("Stock KB - Temp File Manager")
    st.caption(
        (
            "Upload files to the temp directory, select, and delete "
            "individually or in bulk."
        )
    )

    ensure_temp_dir()

    left, right = st.columns([1, 2])
    with left:
        render_uploader()
        st.divider()
        render_youtube_transcriber()
    with right:
        render_file_manager()

    st.divider()
    render_principles_extractor()


if __name__ == "__main__":
    main()
