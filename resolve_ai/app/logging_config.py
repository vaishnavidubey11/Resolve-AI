"""Structured JSON logging with request correlation ids (FR-027).

Only operational metadata is logged. Message bodies, passwords and tokens are never logged.
"""
import contextvars
import json
import logging
import re
import sys

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_SECRET_RE = re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]+|(sk-[A-Za-z0-9]{8,})|((?:password|token|api[_-]?key)[\"'=:\s]+)[^\s\"',]+")


def redact(text: str) -> str:
    return _SECRET_RE.sub(lambda m: (m.group(1) or m.group(3) or "") + "[REDACTED]", text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key in ("method", "path", "status", "duration_ms", "tool", "outcome"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our request log
