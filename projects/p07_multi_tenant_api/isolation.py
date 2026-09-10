"""Tenant-namespaced document storage.

This is the file the whole project exists for. Multi-tenant data leaks are
almost never an exotic exploit; they are a query that forgot a WHERE clause.
The defence used here is structural rather than disciplinary: the tenant id is
part of the storage key, so there is no code path that can address a record
without naming the tenant that owns it. A developer cannot forget the filter
because there is no unfiltered accessor to call.

Two further choices:

* Document ids are random (uuid4), not sequential. Sequential ids across
  tenants leak volume and invite enumeration.
* A read for a document owned by another tenant returns "not found", not
  "forbidden". Returning 403 confirms the id exists, which turns the endpoint
  into an existence oracle that lets tenant B map tenant A's id space.

`tests/test_isolation.py` proves both, including the case where tenant B
presents an id it copied verbatim from tenant A.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StoredDocument:
    doc_id: str
    tenant_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_public(self) -> Dict[str, Any]:
        """The wire shape. tenant_id is intentionally not echoed back."""
        return {
            "id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": round(self.created_at, 3),
        }


class TenantStore:
    """Key is always the (tenant_id, doc_id) pair. There is no doc_id-only read."""

    def __init__(self) -> None:
        self._docs: Dict[Tuple[str, str], StoredDocument] = {}
        self._lock = threading.Lock()

    def put(self, tenant_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> StoredDocument:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        doc = StoredDocument(uuid.uuid4().hex, tenant_id, text, dict(metadata or {}))
        with self._lock:
            self._docs[(tenant_id, doc.doc_id)] = doc
        return doc

    def get(self, tenant_id: str, doc_id: str) -> Optional[StoredDocument]:
        """Returns None for both "does not exist" and "belongs to someone else"."""
        with self._lock:
            return self._docs.get((tenant_id, doc_id))

    def delete(self, tenant_id: str, doc_id: str) -> bool:
        with self._lock:
            return self._docs.pop((tenant_id, doc_id), None) is not None

    def list(self, tenant_id: str) -> List[StoredDocument]:
        with self._lock:
            docs = [d for (t, _), d in self._docs.items() if t == tenant_id]
        return sorted(docs, key=lambda d: d.created_at)

    def search(self, tenant_id: str, query: str, limit: int = 5) -> List[StoredDocument]:
        """Substring search, scoped. Even the scan starts from the tenant's own rows."""
        q = (query or "").lower()
        hits = [d for d in self.list(tenant_id) if q in d.text.lower()]
        return hits[:limit]

    def count(self, tenant_id: Optional[str] = None) -> int:
        with self._lock:
            if tenant_id is None:
                return len(self._docs)
            return sum(1 for (t, _) in self._docs if t == tenant_id)
