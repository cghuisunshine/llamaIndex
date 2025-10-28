#!/usr/bin/env python3
"""Insert a single document into the existing persistent index."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from incremental_index import (
    collect_existing_doc_ids,
    configure_models,
    create_storage_context,
    load_existing_index,
)


def build_index_from_document(
    doc_path: Path, persist_dir: Path, chroma_dir: Path
) -> VectorStoreIndex:
    """Create and persist a new index containing only the provided document."""
    reader = SimpleDirectoryReader(
        input_files=[str(doc_path)],
        filename_as_id=True,
    )
    docs = reader.load_data()
    if not docs:
        raise ValueError(f"Document could not be loaded: {doc_path}")

    storage_ctx = create_storage_context(persist_dir, chroma_dir, expect_existing=False)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
    storage_ctx.persist(persist_dir=str(persist_dir))
    return index


def insert_single_document(index: VectorStoreIndex, doc_path: Path) -> int:
    """Insert the document if it is not already indexed."""
    resolved_path = doc_path.resolve()
    existing_ids = collect_existing_doc_ids(index)
    if str(resolved_path) in existing_ids:
        return 0

    reader = SimpleDirectoryReader(
        input_files=[str(resolved_path)],
        filename_as_id=True,
    )
    docs = reader.load_data()
    if not docs:
        raise ValueError(f"Document could not be loaded: {doc_path}")

    for doc in docs:
        index.insert(doc)

    return len(docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert a single document into the persistent VectorStoreIndex."
    )
    parser.add_argument(
        "--doc-path",
        type=Path,
        required=True,
        help="Path to the document to insert.",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("./storage"),
        help="Directory where the index is stored.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("./storage/chroma_db"),
        help="Directory used by the persistent Chroma vector store.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI chat model identifier (needed for Settings compatibility).",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="Embedding model used when adding the document.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index storage and rebuild using only the provided document.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_models(args.llm_model, args.embed_model)

    if not args.doc_path.exists():
        sys.exit(f"Document does not exist: {args.doc_path}")

    if args.rebuild:
        if args.storage_dir.exists():
            shutil.rmtree(args.storage_dir)
        if args.chroma_dir.exists():
            shutil.rmtree(args.chroma_dir)
        index = build_index_from_document(args.doc_path, args.storage_dir, args.chroma_dir)
        print("Rebuilt index with the provided document.")
        print(f"Indexed {len(collect_existing_doc_ids(index))} document(s).")
        return

    index = load_existing_index(args.storage_dir, args.chroma_dir)

    if index is None:
        if args.storage_dir.exists() or args.chroma_dir.exists():
            print("Existing index storage missing required files. Resetting...")
            if args.storage_dir.exists():
                shutil.rmtree(args.storage_dir)
            if args.chroma_dir.exists():
                shutil.rmtree(args.chroma_dir)
        print("No existing index found. Building a new one with the provided document...")
        index = build_index_from_document(args.doc_path, args.storage_dir, args.chroma_dir)
        print(f"Indexed {len(collect_existing_doc_ids(index))} document(s).")
        return

    try:
        added = insert_single_document(index, args.doc_path)
    except ValueError as exc:
        sys.exit(str(exc))

    if added == 0:
        print("Document already indexed. Storage remains unchanged.")
        return

    index.storage_context.persist(persist_dir=str(args.storage_dir))
    print(
        f"Indexed {added} document(s) from {args.doc_path} "
        f"and persisted storage to {args.storage_dir}."
    )


if __name__ == "__main__":
    main()
