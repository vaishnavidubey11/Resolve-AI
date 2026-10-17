"""AC-06..AC-09, FR-016..FR-020."""
from app.services.orders import MockOrderClient, OrderServiceError


def test_ac06_valid_lookup_returns_permitted_data_only(env):
    env.tenant("demo-store")
    conv = env.conversation("demo-store")
    r = env.say(conv, "Where is my order?")
    assert r["status"] == "needs_info" and r["answer_type"] == "system"      # asks, doesn't guess
    r = env.say(conv, "ORD-1001")
    assert r["status"] == "needs_info" and "email" in r["answer"].lower()
    r = env.say(conv, "alice@example.com")
    assert r["status"] == "order_result" and r["answer_type"] == "order_api"
    assert "shipped" in r["answer"] and "BD123456789" in r["answer"]
    assert "alice@example.com" not in r["answer"]                             # no PII echoed back


def test_ac06_single_message_lookup(env):
    env.tenant("demo-store")
    r = env.say(env.conversation("demo-store"), "status of ORD-1002, my email is Bob@Example.com")
    assert r["status"] == "order_result" and "processing" in r["answer"]


def test_ac07_wrong_email_or_unknown_order_discloses_nothing(env):
    env.tenant("demo-store")
    wrong = env.say(env.conversation("demo-store"), "Track ORD-1001 bob@example.com")
    missing = env.say(env.conversation("demo-store"), "Track ORD-4040 bob@example.com")
    for r in (wrong, missing):
        assert r["status"] == "needs_info" and r["answer_type"] == "system"
        for leak in ("shipped", "BD123456789", "Earbuds", "alice", "processing"):
            assert leak.lower() not in r["answer"].lower()
    assert wrong["answer"] == missing["answer"]                              # no order-id enumeration oracle


def test_ac07_order_belonging_to_other_tenant_not_found(env):
    env.tenant("demo-store"); env.tenant("other-store")
    r = env.say(env.conversation("other-store"), "Track ORD-1001 alice@example.com")
    assert r["status"] == "needs_info" and "shipped" not in r["answer"]


def test_order_lookup_attempts_are_capped(env):
    env.tenant("demo-store")
    conv = env.conversation("demo-store")
    for i in range(5):
        env.say(conv, f"Track ORD-100{i % 3} nobody@example.com")
    r = env.say(conv, "Track ORD-1001 alice@example.com")                     # even the right answer is blocked
    assert r["status"] == "tool_error" and "shipped" not in r["answer"]


def test_ac08_order_api_failure_never_invents_status(env):
    env.tenant("demo-store")
    r = env.say(env.conversation("demo-store"), "Track ORD-9999 alice@example.com")   # mock simulates outage
    assert r["status"] == "tool_error" and r["answer_type"] == "system"
    assert "can't reach our order system" in r["answer"]
    for claim in ("shipped", "delivered", "processing", "refunded", "cancelled"):
        assert claim not in r["answer"].lower()


def test_ac08_timeout_from_injected_client(app, client):
    from tests.conftest import Env

    class Down(MockOrderClient):
        def get_order(self, slug, oid): raise OrderServiceError("timeout")

    app.state.order_client = Down()
    env = Env(app, client)
    env.tenant("demo-store")
    r = env.say(env.conversation("demo-store"), "Track ORD-1001 alice@example.com")
    assert r["status"] == "tool_error"
    from app.models import ToolExecution
    with app.state.session_factory() as db:
        te = db.query(ToolExecution).one()
        assert (te.tool_name, te.status, te.result_reference) == ("order_lookup", "error", "ORD-1001")


def test_tool_arguments_are_validated(env):
    env.tenant("demo-store")
    conv = env.conversation("demo-store")
    r = env.say(conv, "Track ORD-1001 not-an-email@@")
    assert r["status"] == "needs_info"                                        # asked for the email, no API call
    r = env.say(conv, "ORD-1001 ' OR 1=1 -- alice@example.com")
    assert r["status"] == "order_result"                                      # only the validated id/email are used


def test_policy_questions_are_not_mistaken_for_order_lookups(env):
    env.tenant("demo-store")
    r = env.say(env.conversation("demo-store"), "Can the delivery address be changed after my parcel is dispatched?")
    assert r["status"] in ("answered", "fallback")


def test_no_write_actions_exist(env):
    env.tenant("demo-store")
    r = env.say(env.conversation("demo-store"), "Please cancel my order ORD-1001 and refund me. alice@example.com")
    assert "cancelled" not in r["answer"].lower() and "refund has been" not in r["answer"].lower()


# ----------------------------------------------------------------------------- tickets / escalation
def test_ac09_escalation_creates_ticket_and_marks_conversation(env, client):
    t = env.tenant("demo-store")
    conv = env.conversation("demo-store")
    env.say(conv, "My earbuds arrived broken, this is urgent")
    r = env.say(conv, "I want to speak to a human please")
    assert r["status"] == "escalated" and r["ticket_id"] and r["conversation_status"] == "escalated"
    assert r["ticket_id"] in r["answer"] and "follow up" in r["answer"]       # tells customer what happens next
    tk = client.get(f"/api/v1/tickets/{r['ticket_id']}", headers=t["headers"]).json()
    assert tk["conversation_id"] == conv["id"] and tk["status"] == "open"
    assert tk["priority"] == "high" and tk["category"] in ("product", "general", "returns")
    assert client.get(f"/api/v1/tickets/{r['ticket_id']}", headers=conv["headers"]).status_code == 200
    assert client.get(f"/api/v1/conversations/{conv['id']}", headers=conv["headers"]).json()["status"] == "escalated"


def test_escalate_endpoint_is_idempotent(env, client):
    t = env.tenant("demo-store")
    conv = env.conversation("demo-store")
    first = client.post(f"/api/v1/conversations/{conv['id']}/escalate", headers=conv["headers"], json={"reason": "refund"})
    second = client.post(f"/api/v1/conversations/{conv['id']}/escalate", headers=conv["headers"])
    assert first.status_code == second.status_code == 200
    assert first.json()["created"] is True and second.json()["created"] is False
    assert first.json()["ticket_id"] == second.json()["ticket_id"]            # no duplicate tickets
    assert client.get("/api/v1/tickets", headers=t["headers"]).json()["total"] == 1


def test_operator_can_update_ticket_and_conversation_status(env, client):
    env.tenant("demo-store")
    op = env.tenant("demo-store", role="operator", email="op@demo-store.test")
    conv = env.conversation("demo-store")
    tid = env.say(conv, "escalate this please")["ticket_id"]
    assert client.patch(f"/api/v1/tickets/{tid}", headers=op["headers"], json={"status": "resolved"}).json()["status"] == "resolved"
    assert client.patch(f"/api/v1/conversations/{conv['id']}", headers=op["headers"],
                        json={"status": "closed"}).json()["status"] == "closed"
    r = client.post(f"/api/v1/conversations/{conv['id']}/messages", headers=conv["headers"], json={"message": "hi"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "conversation_closed"
    assert client.patch(f"/api/v1/tickets/{tid}", headers=conv["headers"], json={"status": "open"}).status_code == 403
