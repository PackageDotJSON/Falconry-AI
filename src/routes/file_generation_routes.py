"""
@author: { FALCONRY SOLUTIONS }
@description: Standalone file generation endpoint.

Called by the .NET generative-AI chatbot to convert LLM-produced content into
downloadable files (Word, PDF, PPTX, CSV, XLS, HTML, TXT, MD).

Agentic AI workflows use the file generation service directly through the
output_format field on each agent request — they do not call this endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

import configs.configs as config
from enums.enums import FileGenUrls
from models.file_generation_models import FileGenerationRequest
from services.file_generation import generators, object_store

file_gen_router = APIRouter(
    prefix=FileGenUrls.ROUTE_PREFIX.value,
    tags=FileGenUrls.TAGS.value,
)

api_key_header = APIKeyHeader(name=config.API_KEY_HEADER_NAME, auto_error=False)


def validate_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if not config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on server.",
        )
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ── MIME types & extensions ───────────────────────────────────────────────────

_MIME = {
    "csv":  "text/csv",
    "xls":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "txt":  "text/plain",
    "md":   "text/markdown",
    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf":  "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_EXT = {
    "csv": "csv", "xls": "xlsx", "html": "html",
    "txt": "txt", "md": "md",
    "word": "docx", "docx": "docx",
    "pdf": "pdf", "pptx": "pptx",
}


@file_gen_router.post(FileGenUrls.GENERATE.value)
async def generate_file_api(
    request: FileGenerationRequest,
    _: None = Depends(validate_api_key),
):
    """
    Generate a file from structured content and return it as a download.

    **Format → required field mapping:**
    | Format       | Required field  |
    |------------- |---------------- |
    | csv, xls, html | `records`     |
    | word, pdf      | `sections`    |
    | pptx           | `slides`      |
    | txt, md        | `plain_text`  |

    The generated file is:
    - (A) Streamed to the caller as an attachment download.
    - (B) Uploaded to the configured object store (S3/compatible).
          The public URL is returned in the `X-File-URL` response header.
          Upload is silently skipped when S3 credentials are not configured.
    """
    fmt = request.format.lower().strip()

    if fmt not in _MIME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{fmt}'. Supported: {', '.join(_MIME)}",
        )

    try:
        file_bytes = _build_file(request, fmt)
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File generation failed: {str(exc)}",
        ) from exc

    # ── Upload to object store ─────────────────────────────────────────────────
    file_url = None
    if request.upload_to_store:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = _EXT[fmt]
        filename = f"{request.filename_stem}_{timestamp}.{ext}"
        object_key = f"generated/{fmt}/{filename}"
        file_url = object_store.upload_file(file_bytes, object_key, _MIME[fmt])
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = _EXT[fmt]
        filename = f"{request.filename_stem}_{timestamp}.{ext}"

    # ── Stream to client ───────────────────────────────────────────────────────
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if file_url:
        headers["X-File-URL"] = file_url

    return StreamingResponse(
        iter([file_bytes]),
        media_type=_MIME[fmt],
        headers=headers,
    )


def _build_file(request: FileGenerationRequest, fmt: str) -> bytes:
    """Dispatch to the correct generator based on format."""

    # ── Tabular formats ────────────────────────────────────────────────────────
    if fmt in ("csv", "xls", "html"):
        if not request.records:
            raise NotImplementedError(
                f"Format '{fmt}' requires the 'records' field (list of row dicts)."
            )
        if fmt == "csv":
            return generators.generate_csv(request.records)
        if fmt == "xls":
            return generators.generate_xls(request.records, sheet_name="Report")
        return generators.generate_html(request.records, title=request.title)

    # ── Plain text formats ─────────────────────────────────────────────────────
    if fmt in ("txt", "md"):
        if not request.plain_text:
            raise NotImplementedError(
                f"Format '{fmt}' requires the 'plain_text' field."
            )
        if fmt == "txt":
            return generators.generate_txt(request.plain_text)
        return generators.generate_md(request.plain_text)

    # ── Document formats ───────────────────────────────────────────────────────
    if fmt in ("word", "docx", "pdf"):
        if not request.sections:
            raise NotImplementedError(
                f"Format '{fmt}' requires the 'sections' field "
                "(list of {{heading, content}} dicts)."
            )
        if fmt == "pdf":
            return generators.generate_pdf(request.title, request.sections)
        return generators.generate_docx(request.title, request.sections)

    # ── Presentation format ────────────────────────────────────────────────────
    if fmt == "pptx":
        if not request.slides:
            raise NotImplementedError(
                "Format 'pptx' requires the 'slides' field "
                "(list of {title, content} dicts)."
            )
        return generators.generate_pptx(request.title, request.slides)

    raise NotImplementedError(f"No generator implemented for format '{fmt}'.")
