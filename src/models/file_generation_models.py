"""
@author: { FALCONRY SOLUTIONS }
@description: Pydantic models for the standalone File Generation endpoint.

This endpoint is primarily called by the .NET generative-AI chatbot API to turn
LLM-generated content into downloadable files. Agentic workflows import the
generators directly from services.file_generation rather than going through here.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FileGenerationRequest(BaseModel):
    format: str = Field(
        ...,
        description=(
            "Target file format. One of: "
            "csv, xls, html, txt, md, word, pdf, pptx"
        ),
    )
    title: str = Field(
        default="Falconry AI Report",
        description="Document / presentation title (used for Word, PDF, PPTX, HTML).",
    )

    # ── Tabular content — used for csv / xls / html ───────────────────────────
    records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "List of flat row dicts for tabular formats (csv, xls, html). "
            "Example: [{\"risk_id\": \"R-01\", \"rating\": \"High\"}, ...]"
        ),
    )

    # ── Document content — used for word / pdf ────────────────────────────────
    sections: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Ordered list of section dicts for document formats (word, pdf). "
            "Each dict has optional keys 'heading' and 'content'. "
            "Example: [{\"heading\": \"Summary\", \"content\": \"...\"}]"
        ),
    )

    # ── Presentation content — used for pptx ─────────────────────────────────
    slides: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "List of slide dicts for PPTX generation. "
            "Each dict must have 'title' and 'content' keys. "
            "The LLM should return slides in this structure directly."
        ),
    )

    # ── Plain text content — used for txt / md ────────────────────────────────
    plain_text: Optional[str] = Field(
        default=None,
        description="Raw text or Markdown string for txt / md formats.",
    )

    # ── Object store ──────────────────────────────────────────────────────────
    upload_to_store: bool = Field(
        default=True,
        description=(
            "When True (default), upload the generated file to the configured "
            "object store (S3 / compatible) and include the URL as X-File-URL "
            "in the response headers. Silently skipped if S3 is not configured."
        ),
    )
    filename_stem: str = Field(
        default="report",
        description="Base filename without extension (e.g. 'q3_risk_report').",
    )
