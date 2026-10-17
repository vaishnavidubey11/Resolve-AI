"""Order lookup tool (FR-016, FR-017, FR-020). Read-only; verification by order id + checkout email."""
import hmac
import logging
import re
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, EmailStr, field_validator

from app.config import Settings

log = logging.getLogger("resolve.orders")

ORDER_ID_RE = re.compile(r"\b([A-Za-z]{3}-\d{4,10})\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


class OrderServiceError(Exception):
    """The order API failed or timed out."""


class OrderLookupArgs(BaseModel):
    """Typed, validated tool arguments - the model/agent never passes raw strings to the API."""
    order_id: str
    email: EmailStr

    @field_validator("order_id")
    @classmethod
    def _fmt(cls, v: str) -> str:
        v = v.strip().upper()
        if not ORDER_ID_RE.fullmatch(v):
            raise ValueError("invalid order id")
        return v


class OrderClient:
    def get_order(self, tenant_slug: str, order_id: str) -> dict | None:
        raise NotImplementedError


class MockOrderClient(OrderClient):
    """Sample data for the demo store. ORD-9999 simulates an upstream outage."""

    ORDERS = {
        "demo-store": {
            "ORD-1001": {"email": "alice@example.com", "status": "shipped", "carrier": "BlueDart",
                         "tracking": "BD123456789", "eta": "2026-10-21", "items": ["Wireless Earbuds"]},
            "ORD-1002": {"email": "bob@example.com", "status": "processing", "carrier": None,
                         "tracking": None, "eta": "2026-10-25", "items": ["Running Shoes (size 9)"]},
            "ORD-1003": {"email": "alice@example.com", "status": "delivered", "carrier": "Delhivery",
                         "tracking": "DL987654321", "eta": None, "items": ["Phone Case", "Screen Guard"]},
        }
    }

    def get_order(self, tenant_slug: str, order_id: str) -> dict | None:
        if order_id == "ORD-9999":
            raise OrderServiceError("simulated timeout")
        return self.ORDERS.get(tenant_slug, {}).get(order_id)


class HttpOrderClient(OrderClient):
    """GET {base}/orders/{id}?store={slug} -> {email,status,carrier,tracking,eta,items}; 404 when absent."""

    def __init__(self, s: Settings):
        self.client = httpx.Client(
            base_url=s.order_api_base_url.rstrip("/"), timeout=8.0,
            headers={"Authorization": f"Bearer {s.order_api_key}"} if s.order_api_key else {},
        )

    def get_order(self, tenant_slug: str, order_id: str) -> dict | None:
        try:
            r = self.client.get(f"/orders/{order_id}", params={"store": tenant_slug})
        except httpx.HTTPError as e:
            raise OrderServiceError(type(e).__name__) from e
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise OrderServiceError(f"status {r.status_code}")
        try:
            return r.json()
        except ValueError as e:
            raise OrderServiceError("bad json") from e


def build_order_client(s: Settings) -> OrderClient:
    return HttpOrderClient(s) if s.order_api_mode == "http" and s.order_api_base_url else MockOrderClient()


@dataclass
class LookupResult:
    outcome: str  # success | not_verified | error
    order: dict | None = None


def lookup_order(client: OrderClient, tenant_slug: str, args: OrderLookupArgs) -> LookupResult:
    """Return order data only if the supplied email matches the order. Not-found and mismatch are
    indistinguishable to the caller so order ids cannot be enumerated."""
    try:
        order = client.get_order(tenant_slug, args.order_id)
    except OrderServiceError as e:
        log.warning("order api failure", extra={"tool": "order_lookup", "outcome": str(e)})
        return LookupResult("error")
    if order is None or not hmac.compare_digest(str(order.get("email", "")).lower(), args.email.lower()):
        return LookupResult("not_verified")
    return LookupResult("success", order)
