import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI, NotFoundError, APIStatusError, BadRequestError
except ImportError:  # pragma: no cover
    print(
        "The 'openai' package is required. Install with: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise


MANIFEST_FILENAME = ".openai_upload_manifest.json"
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_SIZE_MB = 512


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest(manifest_path: Path) -> Dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt manifest, back it up and start fresh
            backup = manifest_path.with_suffix(manifest_path.suffix + ".bak")
            try:
                backup.write_text(
                    manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
            except OSError:
                pass
    return {"files": {}, "last_updated": int(time.time())}


def save_manifest(manifest_path: Path, manifest: Dict) -> None:
    manifest["last_updated"] = int(time.time())
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_remote_user_data_index(client: OpenAI) -> Dict[Tuple[str, int], str]:
    """Builds an index of remote files keyed by (filename, bytes) -> file_id for purpose=user_data.

    Note: Server-side de-duplication is not guaranteed. This is a best-effort heuristic
    to help skip obvious duplicates if the local manifest is missing.
    """
    index: Dict[Tuple[str, int], str] = {}
    try:
        files = client.files.list(purpose="user_data")
        # The client returns a list-like object with .data in newer SDKs
        items = getattr(files, "data", files)
        for f in items:
            name = getattr(f, "filename", None) or getattr(f, "name", None)
            size = getattr(f, "bytes", None)
            fid = getattr(f, "id", None)
            if isinstance(name, str) and isinstance(size, int) and isinstance(fid, str):
                index[(name, size)] = fid
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        # Don't block on listing failures; continue with local manifest only
        pass
    return index


def should_skip(local_rel_path: str, file_path: Path, manifest: Dict) -> bool:
    record = manifest.get("files", {}).get(local_rel_path)
    if not record:
        return False
    try:
        current_hash = compute_sha256(file_path)
        return (
            record.get("sha256") == current_hash
            and record.get("size") == file_path.stat().st_size
        )
    except OSError:
        return False


def should_skip_for_key(
    local_rel_path: str, file_path: Path, manifest: Dict, required_key: str
) -> bool:
    record = manifest.get("files", {}).get(local_rel_path)
    if not record or required_key not in record:
        return False
    try:
        current_hash = compute_sha256(file_path)
        return (
            record.get("sha256") == current_hash
            and record.get("size") == file_path.stat().st_size
        )
    except OSError:
        return False


def upload_single_file(
    client: OpenAI,
    root_dir: Path,
    file_path: Path,
    manifest: Dict,
    remote_index: Dict[Tuple[str, int], str],
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """Uploads one file. Returns (relative_path, uploaded_file_id or None)."""
    rel_path = str(file_path.relative_to(root_dir)).replace("\\", "/")
    file_size = file_path.stat().st_size

    if should_skip(rel_path, file_path, manifest):
        return rel_path, manifest["files"][rel_path].get("uploaded_file_id")

    # Best-effort remote duplicate check by (filename, bytes)
    remote_id = remote_index.get((file_path.name, file_size))
    if remote_id:
        manifest.setdefault("files", {})[rel_path] = {
            "uploaded_file_id": remote_id,
            "sha256": compute_sha256(file_path),
            "size": file_size,
            "mtime": int(file_path.stat().st_mtime),
            "filename": file_path.name,
        }
        return rel_path, remote_id

    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            with file_path.open("rb") as f:
                resp = client.files.create(file=f, purpose="user_data")
            file_id = getattr(resp, "id", None)
            if not isinstance(file_id, str):
                raise RuntimeError("Upload succeeded but no file id returned")
            manifest.setdefault("files", {})[rel_path] = {
                "uploaded_file_id": file_id,
                "sha256": compute_sha256(file_path),
                "size": file_size,
                "mtime": int(file_path.stat().st_mtime),
                "filename": file_path.name,
            }
            return rel_path, file_id
        except (OSError, RuntimeError) as exc:
            last_err = exc
            time.sleep(min(2**attempt, 10))

    # Give up after retries
    print(f"FAILED: {rel_path}: {last_err}", file=sys.stderr)
    return rel_path, None


def upload_single_file_to_vector_store(
    client: OpenAI,
    root_dir: Path,
    file_path: Path,
    manifest: Dict,
    vector_store_id: str,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """Uploads one file to a vector store. Returns (relative_path, vector_store_file_id or None)."""
    rel_path = str(file_path.relative_to(root_dir)).replace("\\", "/")
    file_size = file_path.stat().st_size

    if should_skip_for_key(rel_path, file_path, manifest, required_key="vs_file_id"):
        return rel_path, manifest["files"][rel_path].get("vs_file_id")

    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            with file_path.open("rb") as f:
                resp = client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id,
                    file=f,
                )
            vs_file_id = getattr(resp, "id", None)
            if not isinstance(vs_file_id, str):
                raise RuntimeError(
                    "Vector store upload succeeded but no file id returned"
                )
            manifest.setdefault("files", {})[rel_path] = {
                "vs_file_id": vs_file_id,
                "vs_id": vector_store_id,
                "sha256": compute_sha256(file_path),
                "size": file_size,
                "mtime": int(file_path.stat().st_mtime),
                "filename": file_path.name,
            }
            return rel_path, vs_file_id
        except BadRequestError as exc:
            # Unsupported file type or similar request errors – skip and record
            manifest.setdefault("files", {})[rel_path] = {
                "skipped_unsupported": True,
                "reason": "bad_request_unsupported",
                "error": str(exc),
                "size": file_size,
                "mtime": int(file_path.stat().st_mtime),
                "filename": file_path.name,
            }
            print(
                f"SKIP UNSUPPORTED: {rel_path} ({(file_path.suffix or '').lower() or 'no-ext'}) - {exc}"
            )
            return rel_path, None
        except (OSError, RuntimeError) as exc:
            last_err = exc
            time.sleep(min(2**attempt, 10))

    print(f"FAILED (vector): {rel_path}: {last_err}", file=sys.stderr)
    return rel_path, None


def delete_files_api_from_manifest(
    client: OpenAI, ids: Iterable[str]
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for fid in ids:
        try:
            client.files.delete(fid)
            results[fid] = True
        except NotFoundError:
            # Treat missing remote file as already deleted
            results[fid] = True
        except APIStatusError as exc:
            # Handle other API errors, but mark as failure
            # If a 404 propagates as APIStatusError, still treat as deleted
            status_code = getattr(exc, "status_code", None)
            results[fid] = status_code == 404
        except (RuntimeError, ValueError, OSError):
            results[fid] = False
    return results


def iter_files(root_dir: Path):
    for path in root_dir.rglob("*"):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively upload files either to Files API (purpose=user_data) or to a Vector Store, with skip logic. Also supports deleting previously uploaded Files API files recorded in the manifest."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path("LEE _ CUSTOM AI STOCK AGENT")),
        help="Root directory to upload recursively.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel upload workers.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=DEFAULT_MAX_SIZE_MB,
        help="Skip files larger than this many MiB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded or skipped without making API calls.",
    )
    parser.add_argument(
        "--vector-store-id",
        type=str,
        default=None,
        help="If provided, upload files to this vector store instead of Files API.",
    )
    parser.add_argument(
        "--delete-files-api",
        action="store_true",
        help="Delete all Files API items recorded in the manifest (uploaded_file_id).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY is not set. Set it in your environment and retry.",
            file=sys.stderr,
        )
        return 2

    root_dir = Path(args.root).resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"Root directory not found: {root_dir}", file=sys.stderr)
        return 2

    manifest_path = (
        root_dir.parent / MANIFEST_FILENAME
        if root_dir.name
        else Path.cwd() / MANIFEST_FILENAME
    )
    manifest = load_manifest(manifest_path)

    all_files = list(iter_files(root_dir))
    max_bytes = args.max_size_mb * 1024 * 1024
    files_to_process = [p for p in all_files if p.stat().st_size <= max_bytes]
    too_large = [p for p in all_files if p.stat().st_size > max_bytes]

    if args.dry_run:
        to_upload = []
        to_skip = []
        for p in files_to_process:
            rel = str(p.relative_to(root_dir)).replace("\\", "/")
            if should_skip(rel, p, manifest):
                to_skip.append(rel)
            else:
                to_upload.append(rel)
        print(f"Root: {root_dir}")
        print(f"Would upload: {len(to_upload)} files")
        print(f"Would skip (already uploaded): {len(to_skip)} files")
        if too_large:
            print(f"Skipping {len(too_large)} files larger than {args.max_size_mb} MiB")
        return 0

    client = OpenAI()

    # Optional deletion step for previously uploaded Files API files
    if args.delete_files_api:
        file_ids = []
        for rec in manifest.get("files", {}).values():
            fid = rec.get("uploaded_file_id")
            if isinstance(fid, str):
                file_ids.append(fid)
        if args.dry_run:
            print(f"Would delete {len(file_ids)} Files API objects")
        else:
            results = delete_files_api_from_manifest(client, file_ids)
            # Remove uploaded_file_id from manifest for successfully deleted ones
            for rel, rec in list(manifest.get("files", {}).items()):
                fid = rec.get("uploaded_file_id")
                if isinstance(fid, str) and results.get(fid):
                    rec.pop("uploaded_file_id", None)
            save_manifest(manifest_path, manifest)
            deleted = sum(1 for ok in results.values() if ok)
            failed_del = sum(1 for ok in results.values() if not ok)
            print(f"Deleted Files API: {deleted}, failed: {failed_del}")
        # If only deletion requested and no upload target provided, exit
        if args.vector_store_id is None:
            return 0

    # Determine mode: vector store vs Files API
    use_vector_store = args.vector_store_id is not None
    remote_index = {} if use_vector_store else list_remote_user_data_index(client)

    completed = 0
    skipped = 0
    skipped_unsupported = 0
    failed = 0
    start = time.time()

    def task(p: Path):
        if use_vector_store:
            rel, fid = upload_single_file_to_vector_store(
                client, root_dir, p, manifest, args.vector_store_id
            )
            return rel, fid
        rel, fid = upload_single_file(client, root_dir, p, manifest, remote_index)
        return rel, fid

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(task, p) for p in files_to_process]
        for fut in concurrent.futures.as_completed(futures):
            rel_path, file_id = fut.result()
            if file_id is None:
                # Distinguish unsupported-type skips vs genuine failures
                rec = manifest["files"].get(rel_path, {})
                if rec.get("skipped_unsupported"):
                    skipped_unsupported += 1
                    skipped += 1
                else:
                    failed += 1
            elif (
                use_vector_store
                and manifest["files"].get(rel_path, {}).get("sha256")
                and manifest["files"][rel_path].get("vs_file_id") == file_id
            ):
                completed += 1
            elif (
                not use_vector_store
                and manifest["files"].get(rel_path, {}).get("sha256")
                and manifest["files"][rel_path].get("uploaded_file_id") == file_id
            ):
                # Could be skip or successful upload; we treat as completed
                completed += 1

    # Count additional skips from manifest that were not in files_to_process (e.g., too large are not counted)
    for p in files_to_process:
        rel = str(p.relative_to(root_dir)).replace("\\", "/")
        if use_vector_store:
            if should_skip_for_key(rel, p, manifest, required_key="vs_file_id"):
                skipped += 1
        else:
            if should_skip(rel, p, manifest):
                skipped += 1

    if too_large:
        print(f"Skipped {len(too_large)} files exceeding {args.max_size_mb} MiB")
    if skipped_unsupported:
        print(f"Skipped {skipped_unsupported} files due to unsupported type")

    save_manifest(manifest_path, manifest)
    elapsed = time.time() - start
    print(
        f"Done. Uploaded/confirmed: {completed}, skipped (unchanged): {skipped}, failed: {failed} in {elapsed:.1f}s"
    )
    print(f"Manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
