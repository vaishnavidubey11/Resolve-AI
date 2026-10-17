from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints
from typing_extensions import Annotated

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class LoginRequest(BaseModel):
    email: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=320)]
    password: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    tenant_id: str


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    filename: str
    source_type: str
    status: str
    error: str | None = None
    chunk_count: int
    created_at: datetime


class ConversationCreate(BaseModel):
    tenant: Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9-]+$")] = Field(
        description="Public tenant slug of the business.")
    customer_ref: Annotated[str, StringConstraints(max_length=200)] | None = None


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    status: str
    customer_ref: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationCreated(ConversationOut):
    access_token: str = Field(description="Bearer token for this conversation (customer session).")


class ConversationPatch(BaseModel):
    status: Literal["open", "closed"]


class MessageIn(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    channel: Literal["web", "api"] = "web"


class SourceRef(BaseModel):
    source_id: str
    title: str
    page: int | None = None
    section: str | None = None


class MessageResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    status: str
    answer_type: str
    sources: list[SourceRef] = []
    escalation_available: bool = True
    conversation_status: str
    ticket_id: str | None = None


class MessageItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    status: str | None = None
    answer_type: str | None = None
    sources: list[SourceRef] = []
    feedback: str | None = None
    created_at: datetime


class EscalateRequest(BaseModel):
    reason: Annotated[str, StringConstraints(max_length=500)] | None = None


class FeedbackIn(BaseModel):
    value: Literal["helpful", "not_helpful"]


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    conversation_id: str
    summary: str
    category: str
    priority: str
    status: str
    created_at: datetime


class TicketPatch(BaseModel):
    status: Literal["open", "in_progress", "resolved"]
