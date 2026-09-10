"""Key lifecycle: one-time secrets, hashed storage, scopes, revocation."""
import pytest

from projects.p07_multi_tenant_api.gateway import Gateway
from projects.p07_multi_tenant_api.keys import ApiKeyStore, AuthError, redact


def test_secret_is_never_recoverable_from_the_stored_record():
    store = ApiKeyStore()
    secret, record = store.issue("acme", ["documents:read"])
    tail = secret.split("_")[-1]
    assert tail not in record.secret_hash
    assert not hasattr(record, "secret")
    assert tail not in repr(record)
    assert store.verify(secret) is record


def test_key_prefix_identifies_the_tenant_without_exposing_the_secret():
    store = ApiKeyStore()
    secret, record = store.issue("globex-industries", ["usage:read"])
    assert secret.startswith("mtk_globex-industries_")
    assert record.public_id in secret
    masked = redact(secret)
    assert masked.startswith("mtk_globex-industries_")
    assert secret.split("_")[-1] not in masked
    assert redact("garbage") == "<malformed-key>"


def test_verify_rejects_missing_malformed_unknown_and_tampered_keys():
    store = ApiKeyStore()
    secret, _ = store.issue("acme", ["documents:read"])
    # Flip the last character deterministically so the tampered key always differs.
    tampered = secret[:-1] + ("1" if secret[-1] == "0" else "0")
    for bad in (None, "", "not-a-key", "mtk_acme_00000000_" + "0" * 32, tampered):
        with pytest.raises(AuthError) as exc:
            store.verify(bad)
        assert exc.value.status == 401
    # every failure gives the same shape, so there is no key-id oracle
    assert store.verify(secret) is not None


def test_revocation_is_immediate_and_idempotent():
    store = ApiKeyStore()
    secret, record = store.issue("acme", ["documents:read"])
    assert store.verify(secret) is record
    assert store.revoke(record.key_id) is True
    with pytest.raises(AuthError):
        store.verify(secret)
    assert store.revoke(record.key_id) is False
    assert store.revoke("nosuchid") is False


def test_issue_rejects_unknown_and_empty_scope_sets():
    store = ApiKeyStore()
    with pytest.raises(ValueError):
        store.issue("acme", ["documents:delete"])
    with pytest.raises(ValueError):
        store.issue("acme", [])


def test_scope_enforcement_returns_403_not_401():
    """A valid credential with insufficient permission is a different problem."""
    g = Gateway()
    _, key, _ = g.register_tenant("initech", "Initech", ["documents:read"], monthly_tokens=1000)
    write = g.handle("POST", "/v1/documents", {"Authorization": f"Bearer {key}"}, {"text": "x"})
    assert write.status == 403
    assert write.body["required_scope"] == "documents:write"
    read = g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {key}"})
    assert read.status == 200


def test_two_tenants_never_share_a_key_id_or_bucket():
    g = Gateway()
    _, k1, r1 = g.register_tenant("a", "A", ["usage:read"], monthly_tokens=10)
    _, k2, r2 = g.register_tenant("b", "B", ["usage:read"], monthly_tokens=10)
    assert r1.key_id != r2.key_id
    assert g.limiter.bucket("a") is not g.limiter.bucket("b")
