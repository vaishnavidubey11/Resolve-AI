"""Support agent: deterministic routing in code, LLM only for grounded wording (SRS 4.1, 8).

Routing and tool permissions are decided by application code, never by retrieved text or model output:
  * escalation intent  -> ticket
  * order intent       -> validated, allow-listed order_lookup tool
  * everything else    -> RAG with evidence threshold and safe fallback
"""
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Conversation, Tenant, ToolExecution
from app.providers import ChatTurn, ContextPassage, EmbeddingProvider, LLMProvider, ProviderError
from app.providers.prompts import SYSTEM_PROMPT
from app.providers.text import tokens
from app.services.conversations import add_message, recent_messages
from app.services.orders import (
    EMAIL_RE, ORDER_ID_RE, OrderClient, OrderLookupArgs, lookup_order,
)
from app.services.retrieval import retrieve
from app.services.tickets import escalate_conversation

log = logging.getLogger("resolve.agent")

ESCALATE_RE = re.compile(
    r"\b(human|real person|live agent|agent please|speak (to|with)|talk (to|with)|customer service rep\w*|"
    r"representative|escalate|supervisor|manager|complain\w*)\b", re.I)
_NOUN = r"(order|package|parcel|shipment)s?"
ORDER_INTENT_RE = re.compile(
    rf"\b(where('s| is| are)|track\w*|status|check|update on|locate)\b.{{0,40}}\b{_NOUN}\b"
    rf"|\b{_NOUN}\b.{{0,30}}\b(status|tracking|stuck|delayed|not (arrived|received|here)|hasn'?t (arrived|come))\b"
    rf"|\bwhen (will|does|is) (my|the) {_NOUN}\b", re.I)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok(ay)?)\W*$", re.I)
MAX_ORDER_ATTEMPTS = 5

ESCALATION_HINT = "I can also connect you with a human agent if you'd like."


@dataclass
class AgentReply:
    content: str
    status: str  # answered | needs_info | fallback | order_result | tool_error | escalated | unavailable
    answer_type: str  # document | order_api | system
    sources: list[dict] = field(default_factory=list)


@dataclass
class AgentDeps:
    db: Session
    settings: Settings
    llm: LLMProvider
    embedder: EmbeddingProvider
    orders: OrderClient
    tenant: Tenant

    @property
    def min_score(self) -> float:
        s = self.settings.retrieval_min_score
        return s if s is not None else self.embedder.recommended_min_score


def handle_message(deps: AgentDeps, conv: Conversation, text: str) -> AgentReply:
    ctx = dict(conv.context or {})
    order_id = (m.group(1).upper() if (m := ORDER_ID_RE.search(text)) else None)
    email = (m.group(0) if (m := EMAIL_RE.search(text)) else None)
    pending_order = ctx.get("pending") == "order_lookup"

    if ESCALATE_RE.search(text):
        return _escalate(deps, conv, "Customer asked for a human agent.")
    if order_id or ORDER_INTENT_RE.search(text) or (pending_order and email):
        return _order_flow(deps, conv, ctx, order_id, email)
    if GREETING_RE.match(text):
        return AgentReply("Hi! I can answer questions about our policies, delivery and products, "
                          "check an order's status, or connect you with a human agent. How can I help?",
                          "answered", "system")
    return _knowledge_flow(deps, conv, text)


# --------------------------------------------------------------------------- escalation
def _escalate(deps: AgentDeps, conv: Conversation, reason: str) -> AgentReply:
    ticket, created = escalate_conversation(deps.db, conv, reason)
    conv.context = {k: v for k, v in (conv.context or {}).items() if k not in ("pending", "order_id", "email")}
    verb = "I've created" if created else "You already have"
    return AgentReply(
        f"{verb} a support ticket ({ticket.id}) and flagged this conversation for a human agent. "
        "A member of our support team will review it and follow up. You can check its status any time with this ticket number.",
        "escalated", "system",
    )


# --------------------------------------------------------------------------- order lookup tool
def _order_flow(deps: AgentDeps, conv: Conversation, ctx: dict, order_id: str | None, email: str | None) -> AgentReply:
    if ctx.get("order_attempts", 0) >= MAX_ORDER_ATTEMPTS:
        return AgentReply("For your security I can't run more order checks in this conversation. "
                          "Please ask me to connect you with a human agent.", "tool_error", "system")
    order_id = order_id or ctx.get("order_id")
    email = email or ctx.get("email")
    if not order_id or not email:
        conv.context = {**ctx, "pending": "order_lookup", "order_id": order_id, "email": email}
        need = ("your order number (for example ORD-1001) and the email address used at checkout"
                if not order_id and not email else
                "your order number (for example ORD-1001)" if not order_id else
                "the email address used at checkout")
        return AgentReply(f"I can look that up. To verify the order, please share {need}.", "needs_info", "system")

    attempts = ctx.get("order_attempts", 0)
    base_ctx = {k: v for k, v in ctx.items() if k not in ("pending", "order_id", "email")}
    try:
        args = OrderLookupArgs(order_id=order_id, email=email)
    except ValidationError:
        conv.context = {**base_ctx, "order_attempts": attempts + 1, "pending": "order_lookup"}
        return AgentReply("That order number or email doesn't look valid. Please send them again "
                          "(order numbers look like ORD-1001).", "needs_info", "system")

    result = lookup_order(deps.orders, deps.tenant.slug, args)
    deps.db.add(ToolExecution(
        conversation_id=conv.id, tool_name="order_lookup",
        status={"success": "success", "not_verified": "denied", "error": "error"}[result.outcome],
        result_reference=args.order_id,
    ))
    if result.outcome == "success":
        conv.context = base_ctx
        return AgentReply(_describe_order(args.order_id, result.order), "order_result", "order_api")
    if result.outcome == "error":
        conv.context = {**base_ctx}
        return AgentReply("I'm sorry, I can't reach our order system right now, so I can't confirm your order's "
                          f"status. Please try again in a few minutes. {ESCALATION_HINT}", "tool_error", "system")
    conv.context = {**base_ctx, "order_attempts": attempts + 1}
    return AgentReply("I couldn't verify an order with those details. Please check the order number and the email "
                      f"used at checkout and try again. {ESCALATION_HINT}", "needs_info", "system")


def _describe_order(order_id: str, o: dict) -> str:
    parts = [f"Order {order_id} is currently **{o.get('status', 'unknown')}** (per our order system)."]
    if o.get("items"):
        parts.append("Items: " + ", ".join(o["items"]) + ".")
    if o.get("carrier") and o.get("tracking"):
        parts.append(f"Carrier: {o['carrier']}, tracking number {o['tracking']}.")
    if o.get("eta") and o.get("status") != "delivered":
        parts.append(f"Estimated delivery: {o['eta']}.")
    return " ".join(parts)


# --------------------------------------------------------------------------- RAG
def _passages_with_evidence(deps: AgentDeps, conv: Conversation, text: str) -> list[ContextPassage]:
    top_k = deps.settings.retrieval_top_k
    found = retrieve(deps.db, deps.embedder, conv.tenant_id, text, top_k)
    # FR-015: a short follow-up ("what about sale items?") only makes sense with the previous customer
    # question. Also try the combined query and keep whichever retrieves better evidence. Only prior
    # *customer* text is used - earlier assistant claims are never treated as evidence.
    if len(tokens(text)) <= 6:
        prior = [m.content for m in recent_messages(deps.db, conv, deps.settings.history_window)
                 if m.role == "customer"]
        if len(prior) >= 2:
            combined = retrieve(deps.db, deps.embedder, conv.tenant_id, f"{prior[-2]} {text}", top_k)
            if combined and (not found or combined[0].score > found[0].score):
                found = combined
    return [p for p in found if p.score >= deps.min_score]


def _knowledge_flow(deps: AgentDeps, conv: Conversation, text: str) -> AgentReply:
    try:
        passages = _passages_with_evidence(deps, conv, text)
        if not passages:
            return AgentReply(
                "I couldn't find that in our support documents, so I don't want to guess. Could you rephrase or add "
                f"a bit more detail? {ESCALATION_HINT}", "fallback", "system")
        history = [
            ChatTurn("user" if m.role == "customer" else "assistant", m.content)
            for m in recent_messages(deps.db, conv, deps.settings.history_window)
        ][:-1]  # exclude the current message, it is passed separately
        answer = deps.llm.generate(SYSTEM_PROMPT, history, text, passages)
    except ProviderError:
        return AgentReply("I'm having trouble answering right now. Please try again shortly, "
                          f"or I can connect you with a human agent.", "unavailable", "system")
    seen, sources = set(), []
    for p in passages:
        key = (p.source_id, p.page)  # one citation per document page; best-scoring section is kept
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_id": p.source_id, "title": p.title, "page": p.page, "section": p.section})
    return AgentReply(answer, "answered", "document", sources)
