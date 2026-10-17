"""Upload validation, text extraction, chunking, embedding and indexing (FR-003..FR-008)."""
import hashlib
import io
import logging
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.errors import AppError
from app.models import DocumentChunk, KnowledgeSource
from app.providers import EmbeddingProvider, ProviderError
from app.providers.text import sentences

log = logging.getLogger("resolve.ingestion")

ALLOWED = {
    ".pdf": ("pdf", {"application/pdf", "application/octet-stream"}),
    ".txt": ("text", {"text/plain", "application/octet-stream"}),
    ".md": ("text", {"text/markdown", "text/x-markdown", "text/plain", "application/octet-stream"}),
}


def _bad(msg: str, status: int = 422) -> AppError:
    return AppError(status, "invalid_file", msg)


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_and_extract(filename: str, content_type: str | None, data: bytes,
                         max_bytes: int) -> tuple[str, list[tuple[int | None, str]]]:
    """Return (source_type, [(page_number|None, text)]) or raise an actionable AppError."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:
        raise _bad("Unsupported file type. Upload a PDF, .txt or .md file.", 415)
    source_type, mimes = ALLOWED[ext]
    if content_type and content_type.split(";")[0].strip().lower() not in mimes:
        raise _bad(f"The file's content type ({content_type}) does not match a {ext} file.", 415)
    if not data:
        raise _bad("The file is empty.")
    if len(data) > max_bytes:
        raise _bad(f"The file is too large. The maximum size is {max_bytes // (1024 * 1024) or 1} MB.", 413)

    if source_type == "pdf":
        if not data.startswith(b"%PDF-"):
            raise _bad("The file is not a valid PDF.")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise _bad("The PDF is password-protected. Remove the password and upload it again.")
            pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
        except AppError:
            raise
        except (PyPdfError, ValueError, KeyError, OSError, RecursionError):
            raise _bad("The PDF could not be read. It may be corrupted; try re-exporting it.")
        pages = [(n, t) for n, t in pages if t]
        if not pages:
            raise _bad("No extractable text was found. Scanned PDFs are not supported yet - upload a text-based PDF.")
        return source_type, pages

    if b"\x00" in data:
        raise _bad("The file looks binary, not plain text.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _bad("The text file must be UTF-8 encoded.")
    if not text.strip():
        raise _bad("The file contains no text.")
    return source_type, [(None, text)]


# ---------------------------------------------------------------- chunking
@dataclass
class ChunkData:
    index: int
    text: str
    page: int | None
    section: str | None


_HEADING_MD = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split a page into (heading, body) sections. Detects markdown and standalone short heading lines."""
    lines = text.replace("\r\n", "\n").split("\n")
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for i, line in enumerate(lines):
        md = _HEADING_MD.match(line)
        prev_blank = i == 0 or not lines[i - 1].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        plain = (
            bool(line.strip()) and len(line.strip()) <= 60 and prev_blank and bool(nxt)
            and not re.search(r"[.!?,;]$", line.strip())
        )
        if md or plain:
            sections.append(((md.group(1) if md else line.strip()).rstrip(":"), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _split_long(unit: str, size: int) -> list[str]:
    """Break a single over-long sentence into pieces <= size, preferring spaces."""
    pieces = []
    while len(unit) > size:
        cut = unit.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        pieces.append(unit[:cut].strip())
        unit = unit[cut:].strip()
    if unit:
        pieces.append(unit)
    return pieces


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into chunks <= size, carrying trailing sentences (<= overlap chars) forward."""
    flat = [piece for u in units for piece in _split_long(u, size)]
    chunks: list[str] = []
    cur: list[str] = []
    for u in flat:
        if cur and len(" ".join(cur + [u])) > size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            total = 0
            for prev in reversed(cur):
                if total + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                total += len(prev) + 1
            cur = tail if len(" ".join(tail + [u])) <= size else []
        cur.append(u)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def chunk_pages(pages: list[tuple[int | None, str]], size: int, overlap: int) -> list[ChunkData]:
    out: list[ChunkData] = []
    overlap = min(overlap, size // 2)
    for page, text in pages:
        for heading, body in _split_sections(text):
            units: list[str] = []
            for para in re.split(r"\n\s*\n", body):
                units.extend(sentences(" ".join(para.split())))
            for piece in _pack(units, size, overlap):
                full = f"{heading}\n{piece}" if heading else piece
                out.append(ChunkData(len(out), full, page, heading))
    return out


# ---------------------------------------------------------------- indexing
def index_source(session_factory, embedder: EmbeddingProvider, source_id: str, tenant_id: str,
                 pages: list[tuple[int | None, str]], size: int, overlap: int) -> None:
    """Background job: chunk -> embed -> store atomically. Never leaves a half-indexed source."""
    db = session_factory()
    try:
        source = db.get(KnowledgeSource, source_id)
        if source is None or source.tenant_id != tenant_id:
            return
        try:
            chunks = chunk_pages(pages, size, overlap)
            if not chunks:
                raise ValueError("no chunks produced")
            vectors = embedder.embed([c.text for c in chunks])
            if len(vectors) != len(chunks):
                raise ProviderError("embedding count mismatch")
            db.add_all(
                DocumentChunk(source_id=source_id, tenant_id=tenant_id, chunk_index=c.index, text=c.text,
                              page=c.page, section=c.section, embedding=v)
                for c, v in zip(chunks, vectors)
            )
            source.status, source.chunk_count, source.error = "ready", len(chunks), None
        except ProviderError:
            db.rollback()
            source = db.get(KnowledgeSource, source_id)
            source.status, source.error = "failed", "The embedding service was unavailable. Delete and re-upload to retry."
            log.warning("ingestion failed: provider", extra={"outcome": "provider_error"})
        except Exception:
            db.rollback()
            source = db.get(KnowledgeSource, source_id)
            source.status, source.error = "failed", "The document could not be processed."
            log.exception("ingestion failed")
        db.commit()
    finally:
        db.close()
