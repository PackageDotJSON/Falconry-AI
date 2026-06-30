"""
@author: { FALCONRY SOLUTIONS }
@description: Pydantic models for the Enterprise GRC Policy Expert workflow.

Inputs come from two sources:
  1. Uploaded documents  — text already extracted from user-supplied attachments
                           (TXT, PDF, Word, XLS, PPTX, MD, HTML) via the document
                           conversion service or passed in directly.
  2. Client DB data      — structured client profile: processes, products, risk
                           taxonomy, appetite values, compliance information.

The user selects the desired output format from the UI before triggering the
agent. When output_format is omitted the endpoint returns JSON containing the
raw Markdown policy for debugging / integration purposes.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ClientData(BaseModel):
    """Structured client data sourced from the platform database."""

    core_processes: Optional[str] = Field(
        default=None,
        description="Core business processes, cycles, and sub-cycles.",
    )
    products_and_services: Optional[str] = Field(
        default=None,
        description="Products and services offered by the client.",
    )
    risk_taxonomy: Optional[str] = Field(
        default=None,
        description="Risk taxonomy: categories, types, and definitions.",
    )
    appetite_values: Optional[str] = Field(
        default=None,
        description="Risk appetite thresholds, tolerance levels, and target values.",
    )
    compliance_info: Optional[str] = Field(
        default=None,
        description="Relevant compliance obligations, regulatory requirements, and standards.",
    )


class GRCPolicyRequest(BaseModel):
    """Request payload for the Enterprise GRC Policy Expert agent."""

    documents_text: Optional[str] = Field(
        default=None,
        description=(
            "Pre-extracted text content from user-uploaded attachments "
            "(PDF, Word, XLS, PPTX, HTML, MD, TXT). "
            "Pass the raw extracted text here — the agent handles interpretation."
        ),
    )
    client_data: Optional[ClientData] = Field(
        default=None,
        description="Structured client profile data pulled from the platform database.",
    )
    organization: Optional[str] = Field(
        default=None,
        description="Client / organization name for context and document headers.",
    )
    framework: str = Field(
        default="ISO 31000",
        description="GRC framework to apply (ISO 31000, COSO ERM, NIST, ISO 27001, etc.).",
    )
    output_format: Optional[str] = Field(
        default=None,
        description=(
            "Desired output file format — selected by the user from the UI dropdown. "
            "Supported: word, pdf, pptx, html, md, txt. "
            "When omitted, the endpoint returns JSON with the raw Markdown policy."
        ),
    )
