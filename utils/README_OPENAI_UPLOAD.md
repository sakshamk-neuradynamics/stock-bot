
# OpenAI Recursive File Uploader (Files API or Vector Store)

This utility can:
- Upload all files under `LEE _ CUSTOM AI STOCK AGENT` to the Files API with `purpose=user_data`, or
- Upload all files to an existing Vector Store (chunked and embedded for Retrieval),
while skipping unchanged files using a manifest (`.openai_upload_manifest.json`).

## Prerequisites

- Python 3.9+
- Set `OPENAI_API_KEY` in your environment

## Install

```bash
pip install -r requirements.txt
```

## Dry Run

```bash
python openai_upload.py --dry-run
```

## Upload to Files API

```bash
python openai_upload.py
```

## Delete previously uploaded Files API files (from manifest)

```bash
python openai_upload.py --delete-files-api
```

Add `--dry-run` to preview deletions.

## Upload to an existing Vector Store

```bash
python openai_upload.py --vector-store-id vs_XXXXX
```

## Options

- `--root PATH` folder to upload (default: `LEE _ CUSTOM AI STOCK AGENT`)
- `--max-workers N` parallel workers (default: 4)
- `--max-size-mb MB` skip files larger than this (default: 512)
- `--dry-run` list planned actions without uploading

## How skipping works

The script stores for each relative file path the SHA-256, size, and mtime. If unchanged, it skips. For Files API, it also builds a best-effort remote index by `(filename, bytes)` to avoid re-uploads if the local manifest is missing.
