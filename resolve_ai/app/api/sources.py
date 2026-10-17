from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, Request, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.deps import Principal, get_db, require_admin
from app.errors import AppError, not_found
from app.models import KnowledgeSource
from app.schemas import Page, SourceOut
from app.services.ingestion import checksum, index_source, validate_and_extract

router = APIRouter(prefix="/sources", tags=["knowledge sources"])


@router.post("", response_model=SourceOut, status_code=202)
async def upload_source(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
                        title: str | None = Form(default=None, max_length=300),
                        p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    s = request.app.state.settings
    data = await file.read(s.max_upload_bytes + 1)  # never buffer unbounded uploads
    filename = (file.filename or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1][:300]
    source_type, pages = validate_and_extract(filename, file.content_type, data, s.max_upload_bytes)
    digest = checksum(data)
    if db.scalar(select(KnowledgeSource.id).where(KnowledgeSource.tenant_id == p.tenant_id,
                                                  KnowledgeSource.checksum == digest)):
        raise AppError(409, "duplicate_source", "This document has already been uploaded.")
    src = KnowledgeSource(tenant_id=p.tenant_id, title=(title or filename.rsplit(".", 1)[0]).strip() or filename,
                          filename=filename, source_type=source_type, status="processing", checksum=digest)
    db.add(src)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise AppError(409, "duplicate_source", "This document has already been uploaded.")
    background.add_task(index_source, request.app.state.session_factory, request.app.state.embedder,
                        src.id, p.tenant_id, pages, s.chunk_size, s.chunk_overlap)
    return src


@router.get("", response_model=Page[SourceOut])
def list_sources(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                 p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    q = select(KnowledgeSource).where(KnowledgeSource.tenant_id == p.tenant_id)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    items = db.scalars(q.order_by(KnowledgeSource.created_at.desc()).limit(limit).offset(offset)).all()
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{source_id}", response_model=SourceOut)
def get_source(source_id: str, p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    src = db.get(KnowledgeSource, source_id)
    if src is None or src.tenant_id != p.tenant_id:
        raise not_found("Source")
    return src


@router.delete("/{source_id}", status_code=204)
def delete_source(source_id: str, p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    src = db.get(KnowledgeSource, source_id)
    if src is None or src.tenant_id != p.tenant_id:
        raise not_found("Source")
    db.delete(src)  # chunks + vectors removed via FK cascade (FR-008, AC-10)
    db.commit()
