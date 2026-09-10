"""Ingestion: raw documents in, stable citable chunks out.

Two decisions in here are worth calling out, because both of them are the kind of
thing that is invisible until retrieval quality is already bad in production.

1. Chunk ids are `{doc_id}#{ordinal}` and are derived from position, not from a
   hash of the text. A content hash looks tidier but it changes when someone
   fixes a typo, which silently invalidates every citation already stored in a
   log, a feedback table or a support ticket. Position-derived ids stay valid.
2. Exact-duplicate chunk text is dropped, keeping the first occurrence. Real
   corpora are full of repeated boilerplate (licence footers, nav blurbs, shared
   preambles). Duplicated chunks do not add information but they do crowd out
   the top-k slots, so the retriever spends its budget on the same sentence
   three times.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from llmkit import Chunk, Document, chunk_text, normalize_ws
from llmkit import corpus as builtin_corpus

TEXT_EXTENSIONS = (".txt", ".md")


@dataclass
class IngestConfig:
    """Chunking parameters.

    `target_tokens=220` with `overlap_tokens=40` is a deliberate default: roughly
    a long paragraph, which is small enough that a citation points at something a
    human can verify in a couple of seconds, and large enough that a fact and the
    sentence qualifying it usually survive in the same chunk. The overlap is the
    insurance policy for the times they do not.
    """

    target_tokens: int = 220
    overlap_tokens: int = 40
    min_chunk_chars: int = 30
    drop_exact_duplicates: bool = True


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    duplicates_dropped: int = 0
    too_short_dropped: int = 0
    sources: Dict[str, int] = field(default_factory=dict)

    def as_lines(self) -> List[str]:
        return [
            f"documents ingested   : {self.documents}",
            f"chunks produced      : {self.chunks}",
            f"duplicate chunks     : {self.duplicates_dropped}",
            f"below min length     : {self.too_short_dropped}",
            f"sources              : {', '.join(f'{k}={v}' for k, v in sorted(self.sources.items())) or 'none'}",
        ]


class Ingestor:
    """Collects documents from the built-in corpus and from local text files."""

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()
        self.stats = IngestStats()

    # -- sources ---------------------------------------------------------
    def from_builtin_corpus(self) -> List[Document]:
        """The fictional Meridian docs shipped with llmkit.

        Fictional on purpose. A public model cannot have memorised it, so any
        answer quality measured here is retrieval quality, not recall of
        pretraining data.
        """
        docs = builtin_corpus.documents()
        self.stats.sources["builtin"] = self.stats.sources.get("builtin", 0) + len(docs)
        return docs

    def from_path(self, path: str) -> List[Document]:
        """Load a `.txt`/`.md` file, or walk a directory for them.

        Unreadable or non-text files are skipped rather than raising: an ingest
        job that dies on the one bad file in a 10,000 file crawl is worse than
        one that reports what it skipped.
        """
        docs: List[Document] = []
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in sorted(files):
                    if name.lower().endswith(TEXT_EXTENSIONS):
                        docs.extend(self._read_file(os.path.join(root, name)))
        elif os.path.isfile(path):
            docs.extend(self._read_file(path))
        else:
            raise FileNotFoundError(path)
        if docs:
            self.stats.sources["files"] = self.stats.sources.get("files", 0) + len(docs)
        return docs

    def _read_file(self, filepath: str) -> List[Document]:
        try:
            with open(filepath, encoding="utf-8", errors="replace") as handle:
                raw = handle.read()
        except OSError:
            return []
        body = normalize_ws(raw)
        if not body:
            return []
        stem = os.path.splitext(os.path.basename(filepath))[0]
        title = self._title_from_markdown(raw) or stem.replace("_", " ").replace("-", " ")
        return [Document(id=stem, text=body, metadata={"title": title, "path": filepath})]

    @staticmethod
    def _title_from_markdown(raw: str) -> Optional[str]:
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped:
                break
        return None

    # -- chunking --------------------------------------------------------
    def chunk(self, documents: Sequence[Document]) -> List[Chunk]:
        seen_text: Dict[str, str] = {}
        chunks: List[Chunk] = []
        for doc in documents:
            self.stats.documents += 1
            pieces = chunk_text(
                doc.text,
                target_tokens=self.config.target_tokens,
                overlap_tokens=self.config.overlap_tokens,
            )
            ordinal = 0
            for piece in pieces:
                text = piece.strip()
                if len(text) < self.config.min_chunk_chars:
                    self.stats.too_short_dropped += 1
                    continue
                key = text.lower()
                if self.config.drop_exact_duplicates and key in seen_text:
                    self.stats.duplicates_dropped += 1
                    continue
                seen_text[key] = doc.id
                metadata = dict(doc.metadata)
                metadata["doc_title"] = metadata.get("title", doc.id)
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{ordinal}",
                        doc_id=doc.id,
                        text=text,
                        ordinal=ordinal,
                        metadata=metadata,
                    )
                )
                ordinal += 1
        self.stats.chunks = len(chunks)
        return chunks

    def ingest(self, paths: Iterable[str] = (), include_builtin: bool = True) -> List[Chunk]:
        documents: List[Document] = []
        if include_builtin:
            documents.extend(self.from_builtin_corpus())
        for path in paths:
            documents.extend(self.from_path(path))
        if not documents:
            raise ValueError("nothing to ingest: no builtin corpus and no readable paths")
        return self.chunk(documents)
