"""AC-05: tenant A can never reach tenant B's data by swapping identifiers (FR-002, FR-024)."""


def test_ac05_cross_tenant_access_denied(env, client):
    a, b = env.tenant("store-a"), env.tenant("store-b")
    sid_b = env.upload(b["headers"], "b.txt", b"# Secret\nStore B returns take 99 days.", "text/plain",
                       "B Policy").json()["id"]
    conv_b = env.conversation("store-b")
    env.say(conv_b, "I want to talk to a human")                    # creates ticket in B
    tid_b = client.get("/api/v1/tickets", headers=b["headers"]).json()["items"][0]["id"]

    # sources
    assert client.get(f"/api/v1/sources/{sid_b}", headers=a["headers"]).status_code == 404
    assert client.delete(f"/api/v1/sources/{sid_b}", headers=a["headers"]).status_code == 404
    assert client.get("/api/v1/sources", headers=a["headers"]).json()["total"] == 0
    # conversations / messages
    for path in (f"/api/v1/conversations/{conv_b['id']}", f"/api/v1/conversations/{conv_b['id']}/messages"):
        assert client.get(path, headers=a["headers"]).status_code == 404
    assert client.get("/api/v1/conversations", headers=a["headers"]).json()["total"] == 0
    assert client.patch(f"/api/v1/conversations/{conv_b['id']}", headers=a["headers"],
                        json={"status": "closed"}).status_code == 404
    assert client.post(f"/api/v1/conversations/{conv_b['id']}/escalate", headers=a["headers"]).status_code == 404
    # tickets
    assert client.get(f"/api/v1/tickets/{tid_b}", headers=a["headers"]).status_code == 404
    assert client.patch(f"/api/v1/tickets/{tid_b}", headers=a["headers"], json={"status": "resolved"}).status_code == 404
    assert client.get("/api/v1/tickets", headers=a["headers"]).json()["total"] == 0
    # B's own admin still sees everything
    assert client.get(f"/api/v1/sources/{sid_b}", headers=b["headers"]).status_code == 200
    assert client.get(f"/api/v1/tickets/{tid_b}", headers=b["headers"]).status_code == 200


def test_customer_sessions_are_isolated(env, client):
    env.tenant("store-a"); env.tenant("store-b")
    c1, c2, cb = env.conversation("store-a"), env.conversation("store-a"), env.conversation("store-b")
    env.say(c1, "hello")
    # same tenant, different conversation
    assert client.get(f"/api/v1/conversations/{c1['id']}/messages", headers=c2["headers"]).status_code == 404
    # different tenant
    assert client.get(f"/api/v1/conversations/{c1['id']}/messages", headers=cb["headers"]).status_code == 404
    assert client.post(f"/api/v1/conversations/{c1['id']}/messages", headers=cb["headers"],
                       json={"message": "hi"}).status_code == 404
    # no token at all
    assert client.get(f"/api/v1/conversations/{c1['id']}/messages").status_code == 401


def test_retrieval_never_crosses_tenants(env):
    a, b = env.tenant("store-a"), env.tenant("store-b")
    env.upload(b["headers"], "b.txt", b"# Warranty\nStore B warranty lasts 99 months.", "text/plain", "B Warranty")
    reply = env.say(env.conversation("store-a"), "How long does the warranty last?")
    assert reply["status"] == "fallback" and "99" not in reply["answer"]
    reply_b = env.say(env.conversation("store-b"), "How long does the warranty last?")
    assert reply_b["status"] == "answered" and "99 months" in reply_b["answer"]


def test_operator_can_review_own_tenant_conversations(env, client):
    env.tenant("store-a")
    op = env.tenant("store-a", role="operator", email="op@store-a.test")
    conv = env.conversation("store-a")
    env.say(conv, "Hello there")
    msgs = client.get(f"/api/v1/conversations/{conv['id']}/messages", headers=op["headers"])
    assert msgs.status_code == 200 and msgs.json()["total"] == 2
    # operators cannot impersonate customers
    r = client.post(f"/api/v1/conversations/{conv['id']}/messages", headers=op["headers"], json={"message": "x"})
    assert r.status_code == 403


def test_unknown_tenant_slug(client):
    r = client.post("/api/v1/conversations", json={"tenant": "does-not-exist"})
    assert r.status_code == 404
