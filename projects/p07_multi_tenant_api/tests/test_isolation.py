"""Tenant isolation. This is the test the project exists to pass."""
import json
import urllib.error
import urllib.request

import pytest

from projects.p07_multi_tenant_api.gateway import Gateway
from projects.p07_multi_tenant_api.isolation import TenantStore
from projects.p07_multi_tenant_api.server import serve

ALL = ["documents:read", "documents:write", "llm:complete", "usage:read"]


@pytest.fixture()
def two_tenants():
    g = Gateway()
    _, a_key, _ = g.register_tenant("alpha", "Alpha", ALL, monthly_tokens=10_000)
    _, b_key, _ = g.register_tenant("bravo", "Bravo", ALL, monthly_tokens=10_000)
    return g, a_key, b_key


def test_tenant_b_cannot_read_tenant_a_document_even_with_the_exact_id(two_tenants):
    g, a_key, b_key = two_tenants
    created = g.handle("POST", "/v1/documents", {"Authorization": f"Bearer {a_key}"},
                       {"text": "alpha payroll export"})
    assert created.status == 201
    doc_id = created.body["document"]["id"]

    # Alpha can read its own document.
    own = g.handle("GET", f"/v1/documents/{doc_id}", {"Authorization": f"Bearer {a_key}"})
    assert own.status == 200
    assert own.body["document"]["text"] == "alpha payroll export"

    # Bravo presents the identical, correct id and gets nothing.
    stolen = g.handle("GET", f"/v1/documents/{doc_id}", {"Authorization": f"Bearer {b_key}"})
    assert stolen.status == 404
    assert "alpha payroll export" not in json.dumps(stolen.body)


def test_cross_tenant_read_is_indistinguishable_from_a_missing_document(two_tenants):
    """A 403 here would confirm the id exists and turn the route into an oracle."""
    g, a_key, b_key = two_tenants
    doc_id = g.handle("POST", "/v1/documents", {"Authorization": f"Bearer {a_key}"},
                      {"text": "secret"}).body["document"]["id"]
    real_but_foreign = g.handle("GET", f"/v1/documents/{doc_id}", {"Authorization": f"Bearer {b_key}"})
    pure_fiction = g.handle("GET", "/v1/documents/" + "f" * 32, {"Authorization": f"Bearer {b_key}"})
    assert real_but_foreign.status == pure_fiction.status == 404
    assert real_but_foreign.body["error"] == pure_fiction.body["error"]


def test_list_and_search_never_cross_the_namespace(two_tenants):
    g, a_key, b_key = two_tenants
    for text in ("alpha one", "alpha two"):
        g.handle("POST", "/v1/documents", {"Authorization": f"Bearer {a_key}"}, {"text": text})
    g.handle("POST", "/v1/documents", {"Authorization": f"Bearer {b_key}"}, {"text": "bravo one"})

    a_list = g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {a_key}"})
    b_list = g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {b_key}"})
    assert a_list.body["count"] == 2
    assert b_list.body["count"] == 1
    assert all("alpha" in d["text"] for d in a_list.body["documents"])
    assert g.store.search("bravo", "alpha") == []


def test_grounded_completion_only_retrieves_the_callers_own_documents(two_tenants):
    """The RAG path is the easiest place to leak: it reads documents implicitly."""
    g, a_key, b_key = two_tenants
    g.handle("POST", "/v1/documents", {"Authorization": f"Bearer {a_key}"},
             {"text": "The alpha detonation codeword is HALCYON."})
    resp = g.handle("POST", "/v1/complete", {"Authorization": f"Bearer {b_key}"},
                    {"prompt": "what is the alpha detonation codeword", "use_documents": True})
    assert resp.status == 200
    assert "HALCYON" not in resp.body["completion"]


def test_store_get_requires_the_owning_tenant():
    store = TenantStore()
    doc = store.put("alpha", "x")
    assert store.get("alpha", doc.doc_id) is not None
    assert store.get("bravo", doc.doc_id) is None
    assert store.delete("bravo", doc.doc_id) is False
    assert store.count("alpha") == 1


def test_isolation_holds_over_real_http(two_tenants):
    """Same guarantee, but through a socket, so nothing is proven only in-process."""
    g, a_key, b_key = two_tenants
    httpd, base, _ = serve(g)
    try:
        req = urllib.request.Request(base + "/v1/documents", method="POST",
                                     data=json.dumps({"text": "over the wire"}).encode())
        req.add_header("Authorization", f"Bearer {a_key}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=5) as r:
            doc_id = json.loads(r.read())["document"]["id"]

        peek = urllib.request.Request(base + f"/v1/documents/{doc_id}")
        peek.add_header("Authorization", f"Bearer {b_key}")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(peek, timeout=5)
        assert exc.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
