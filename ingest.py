"""Read PDFs from the pdfs folder and store page chunks in ChromaDB.

Run this once after adding or changing PDFs:

    python ingest.py
"""

import hashlib
from pathlib import Path

import pymupdf

import config
from store import embed_texts, open_collection


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into chunks of about chunk_size characters.

    Neighboring chunks share `overlap` characters. Whitespace is collapsed so
    a page number and a line break do not create tiny fragments.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(cleaned):
        chunks.append(cleaned[start : start + chunk_size])
        if start + chunk_size >= len(cleaned):
            break
        start += step
    return chunks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_id(filename, file_hash, page, index) -> str:
    raw = f"{filename}|{file_hash}|{page}|{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_pages(path: Path):
    """Return (pages, error). pages is a list of (page_number, text).

    Page numbers are 1-based, matching the page numbers a reader sees.
    """
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        return None, f"could not open file ({exc})"

    try:
        if document.is_encrypted and document.authenticate("") == 0:
            return None, "file is password-protected"
        pages = []
        for number, page in enumerate(document, start=1):
            try:
                text = page.get_text() or ""
            except Exception:
                text = ""
            pages.append((number, text))
        return pages, None
    except Exception as exc:
        return None, f"could not read pages ({exc})"
    finally:
        document.close()


def _pdf_paths(pdf_dir: Path):
    return sorted(
        path
        for path in pdf_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def _existing_for_file(collection, filename):
    found = collection.get(where={"filename": filename}, include=["metadatas"])
    ids = found.get("ids") or []
    metadatas = found.get("metadatas") or []
    return ids, metadatas


def _embed(texts, embedder):
    if embedder is None:
        return embed_texts(texts)
    return embedder(texts)


def ingest(pdf_dir=None, chroma_dir=None, embedder=None):
    """Index every PDF in pdf_dir. Re-running updates changed files only."""
    pdf_directory = config.pdf_dir() if pdf_dir is None else Path(pdf_dir)
    if not pdf_directory.is_dir():
        raise FileNotFoundError(
            f"The pdfs folder was not found at {pdf_directory}. "
            "Create it and add PDF files, then run python ingest.py again."
        )

    collection = open_collection(chroma_dir)
    pdf_paths = _pdf_paths(pdf_directory)
    current_names = {path.name for path in pdf_paths}

    summary = {
        "pdfs_found": len(pdf_paths),
        "pdfs_processed": 0,
        "pdfs_unchanged": 0,
        "pdfs_unreadable": 0,
        "pdfs_without_text": 0,
        "pages_seen": 0,
        "pages_with_text": 0,
        "pages_without_text": 0,
        "chunks_written": 0,
        "chunks_removed": 0,
    }

    if not pdf_paths:
        print(f"No PDF files found in {pdf_directory}.", flush=True)

    for path in pdf_paths:
        _ingest_one(path, collection, embedder, summary)

    summary["chunks_removed"] += _remove_stale_files(collection, current_names)
    return summary


def _ingest_one(path, collection, embedder, summary):
    filename = path.name
    pages, error = extract_pages(path)
    if error:
        summary["pdfs_unreadable"] += 1
        print(f"{filename}: unreadable ({error})", flush=True)
        return

    file_hash = file_sha256(path)
    chunks = []
    for page_number, text in pages:
        summary["pages_seen"] += 1
        pieces = chunk_text(text)
        if not pieces:
            summary["pages_without_text"] += 1
            continue
        summary["pages_with_text"] += 1
        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": chunk_id(filename, file_hash, page_number, index),
                    "text": piece,
                    "metadata": {
                        "filename": filename,
                        "page": int(page_number),
                        "chunk_index": int(index),
                        "file_sha256": file_hash,
                    },
                }
            )

    existing_ids, existing_meta = _existing_for_file(collection, filename)
    existing_hashes = {item.get("file_sha256") for item in existing_meta}

    if not chunks:
        summary["pdfs_without_text"] += 1
        if existing_ids:
            collection.delete(ids=existing_ids)
            summary["chunks_removed"] += len(existing_ids)
        print(f"{filename}: no extractable text", flush=True)
        return

    unchanged = existing_hashes == {file_hash} and len(existing_ids) == len(chunks)
    if unchanged:
        summary["pdfs_unchanged"] += 1
        print(f"{filename}: unchanged, skipped", flush=True)
        return

    if existing_ids:
        collection.delete(ids=existing_ids)
        summary["chunks_removed"] += len(existing_ids)

    vectors = _embed([item["text"] for item in chunks], embedder)
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedding failed for {filename}: unexpected vector count.")

    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=batch_vectors,
            metadatas=[item["metadata"] for item in batch],
        )

    summary["pdfs_processed"] += 1
    summary["chunks_written"] += len(chunks)
    print(
        f"{filename}: {summary_pages(pages)} pages, {len(chunks)} chunks written",
        flush=True,
    )


def summary_pages(pages) -> int:
    return len(pages)


def _remove_stale_files(collection, current_names):
    found = collection.get(include=["metadatas"])
    ids = found.get("ids") or []
    metadatas = found.get("metadatas") or []
    stale_ids = [
        chunk_id
        for chunk_id, metadata in zip(ids, metadatas)
        if metadata and metadata.get("filename") not in current_names
    ]
    if not stale_ids:
        return 0

    removed_names = sorted(
        {
            metadata.get("filename")
            for metadata in metadatas
            if metadata and metadata.get("filename") not in current_names
        }
    )
    collection.delete(ids=stale_ids)
    for name in removed_names:
        print(f"{name}: removed from the index (file is no longer in pdfs/)", flush=True)
    return len(stale_ids)


def format_summary(summary) -> str:
    return "\n".join(
        [
            "Ingestion summary",
            f"PDFs found: {summary['pdfs_found']}",
            f"PDFs processed: {summary['pdfs_processed']}",
            f"PDFs unchanged: {summary['pdfs_unchanged']}",
            f"PDFs unreadable: {summary['pdfs_unreadable']}",
            f"PDFs with no text: {summary['pdfs_without_text']}",
            f"Pages seen: {summary['pages_seen']}",
            f"Pages with text: {summary['pages_with_text']}",
            f"Pages without text: {summary['pages_without_text']}",
            f"Chunks written: {summary['chunks_written']}",
            f"Chunks removed: {summary['chunks_removed']}",
        ]
    )


def main():
    try:
        summary = ingest()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return 1
    print()
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
