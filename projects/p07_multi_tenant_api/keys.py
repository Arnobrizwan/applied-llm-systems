"""API key issuance, verification and revocation.

Three properties this file exists to guarantee, all of them things that get got
wrong in real gateways:

1. The secret is shown exactly once. Only a salted hash is retained, so a dump
   of the key table is not a dump of every customer's credentials.
2. The key carries an identifiable, non-secret prefix. When a key leaks into a
   log line, a stack trace or a support ticket, the operator can tell which
   tenant and which key to revoke without knowing the secret. The prefix also
   makes secret scanners (GitHub push protection style) able to match on shape.
3. Verification is O(1). The key embeds its own record id, so the gateway looks
   up one row and compares one hash. The naive design, hashing the presented
   secret against every stored key, turns authentication into a table scan and
   is what makes a key table quietly become a latency problem at ten thousand
   tenants.

Hashing choice: a single SHA-256 over salt plus secret, not PBKDF2 or scrypt.
Slow KDFs exist to make low-entropy human passwords expensive to brute force.
These secrets are 32 hex characters from `secrets.token_hex`, which is 128 bits
of entropy, so an offline attacker gains nothing from a fast hash and the
gateway avoids paying a KDF on every single request. Comparison is constant
time via `hmac.compare_digest` regardless.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

KEY_PREFIX = "mtk"
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# The scopes the gateway understands. Anything outside this set is rejected at
# issue time rather than silently granting nothing at request time, because a
# typo in a scope name should fail loudly when the key is minted, not six weeks
# later when a customer's integration returns 403 in production.
VALID_SCOPES = frozenset({"documents:read", "documents:write", "llm:complete", "usage:read"})


class AuthError(Exception):
    """Raised when a presented key is absent, malformed, unknown or revoked."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")[:24] or "tenant"


@dataclass
class ApiKeyRecord:
    """What the server keeps. Deliberately does not contain the secret."""

    key_id: str
    tenant_id: str
    salt: str
    secret_hash: str
    scopes: FrozenSet[str]
    label: str = ""
    created_at: float = field(default_factory=time.time)
    revoked_at: Optional[float] = None
    last_used_at: Optional[float] = None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    @property
    def public_id(self) -> str:
        """The safe-to-log identifier: everything but the secret."""
        return f"{KEY_PREFIX}_{_slug(self.tenant_id)}_{self.key_id}"

    def allows(self, scope: str) -> bool:
        return scope in self.scopes


def redact(presented: str) -> str:
    """Turn a full key into something safe to write to a log.

    Keeps the prefix, tenant slug and key id (all non-secret) and masks the
    secret. A log line is then actionable, an operator can revoke exactly the
    right key, without the log itself becoming a credential store.
    """
    parts = (presented or "").split("_")
    if len(parts) < 4:
        return "<malformed-key>"
    return "_".join(parts[:3]) + "_" + "*" * 8


def _hash(salt: str, secret: str) -> str:
    return hashlib.sha256((salt + secret).encode("utf-8")).hexdigest()


class ApiKeyStore:
    """In-memory key table. Swap the dict for a row store and nothing else moves."""

    def __init__(self) -> None:
        self._by_id: Dict[str, ApiKeyRecord] = {}

    def issue(self, tenant_id: str, scopes: Iterable[str], label: str = "") -> Tuple[str, ApiKeyRecord]:
        """Mint a key. Returns (secret_shown_once, stored_record).

        The caller is responsible for showing the first element to the customer
        and then dropping it. Nothing in this class can recover it afterwards,
        which is the whole point.
        """
        wanted = frozenset(scopes)
        unknown = wanted - VALID_SCOPES
        if unknown:
            raise ValueError(f"unknown scope(s): {sorted(unknown)}")
        if not wanted:
            raise ValueError("a key with no scopes can do nothing; refusing to issue it")

        key_id = secrets.token_hex(4)
        salt = secrets.token_hex(8)
        secret = secrets.token_hex(16)
        record = ApiKeyRecord(
            key_id=key_id,
            tenant_id=tenant_id,
            salt=salt,
            secret_hash=_hash(salt, secret),
            scopes=wanted,
            label=label,
        )
        self._by_id[key_id] = record
        presented = f"{record.public_id}_{secret}"
        return presented, record

    def verify(self, presented: Optional[str]) -> ApiKeyRecord:
        """Resolve a presented key to its record, or raise AuthError.

        Every failure path returns the same generic message. Distinguishing
        "no such key" from "wrong secret" hands an attacker a key-id oracle for
        free, and there is no operational reason a caller needs the difference.
        """
        if not presented:
            raise AuthError("missing Authorization bearer key", 401)
        parts = presented.split("_")
        if len(parts) != 4 or parts[0] != KEY_PREFIX:
            raise AuthError("invalid api key", 401)
        _, _, key_id, secret = parts
        record = self._by_id.get(key_id)
        if record is None:
            raise AuthError("invalid api key", 401)
        if not hmac.compare_digest(record.secret_hash, _hash(record.salt, secret)):
            raise AuthError("invalid api key", 401)
        if record.revoked:
            # 401 rather than 403: a revoked key is not a credential at all.
            raise AuthError("api key has been revoked", 401)
        record.last_used_at = time.time()
        return record

    def revoke(self, key_id: str) -> bool:
        """Revoke by key id. Idempotent, returns False if already revoked."""
        record = self._by_id.get(key_id)
        if record is None or record.revoked:
            return False
        record.revoked_at = time.time()
        return True

    def for_tenant(self, tenant_id: str) -> List[ApiKeyRecord]:
        return [r for r in self._by_id.values() if r.tenant_id == tenant_id]

    def __len__(self) -> int:
        return len(self._by_id)
