"""Runs the version-controlled RAG test set (SRS 9.1) end-to-end through the API with mock providers."""
import json
from pathlib import Path

import pytest

CASES = json.loads((Path(__file__).parent / "data" / "rag_eval.json").read_text())["cases"]


def test_eval_set_shape():
    cats = {}
    for c in CASES:
        cats[c["category"].split("_")[0] if c["category"] != "no_answer" else "no_answer"] = cats.get(c["category"], 0) + 1
    assert len(CASES) >= 30
    count = lambda *names: sum(1 for c in CASES if c["category"] in names)
    assert count("answerable") == 10 and count("paraphrased") == 5 and count("follow_up") == 5
    assert count("no_answer") == 5 and count("adversarial", "ambiguous") == 5


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_rag_case(env, store, case):
    conv = env.conversation(store["slug"])
    reply = None
    for turn in case["turns"]:
        reply = env.say(conv, turn)
    exp = case["expect"]
    assert reply["status"] in exp["status"], f"{case['id']}: got {reply['status']} / {reply['answer']!r}"
    if "source" in exp:
        assert exp["source"] in [s["title"] for s in reply["sources"]], reply
    if "contains" in exp:
        assert exp["contains"].lower() in reply["answer"].lower(), reply["answer"]
    if "not_contains" in exp:
        assert exp["not_contains"].lower() not in reply["answer"].lower()
    if reply["status"] == "fallback":
        assert reply["sources"] == [] and reply["escalation_available"] is True
