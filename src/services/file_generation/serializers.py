"""
@author: { FALCONRY SOLUTIONS }
@description: Converts each agent's output dict into the inputs expected by the
              file generators, and builds FastAPI StreamingResponse file downloads.

Each agent produces a different JSON structure. The functions here know how to
extract the relevant data from that structure and map it to the right generator.

Supported output_format values (case-insensitive):
  Tabular  — "csv", "xls", "html"
  Document — "word", "docx", "pdf"
  Slides   — "pptx"
  Text     — "txt", "md"
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from services.file_generation import generators, object_store


# ── MIME types & file extensions ─────────────────────────────────────────────

_MIME: Dict[str, str] = {
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

_EXT: Dict[str, str] = {
    "csv": "csv", "xls": "xlsx", "html": "html",
    "txt": "txt", "md": "md",
    "word": "docx", "docx": "docx",
    "pdf": "pdf", "pptx": "pptx",
}


# ── Generic flattener ─────────────────────────────────────────────────────────

def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested dict/list into a single-level dict."""
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                items.update(_flatten(v, full_key))
            else:
                items[full_key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    else:
        items[prefix] = obj
    return items


# ── Per-agent serializers ─────────────────────────────────────────────────────

def _risk_assessment_to_records(result: dict) -> List[Dict[str, Any]]:
    """Flatten each per-risk object for tabular output."""
    rows = []
    for risk in result.get("risks", []):
        rows.append({
            "risk_id":         risk.get("risk_id", ""),
            "risk_name":       risk.get("risk_name", ""),
            "computed_rating": risk.get("computed_rating", ""),
            "risk_score":      risk.get("risk_score", ""),
            "priority":        risk.get("priority", ""),
            "control_gaps":    "; ".join(risk.get("control_gaps", [])),
            "recommendations": "; ".join(risk.get("recommendations", [])),
        })
    return rows


def _risk_assessment_to_document(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    title = "Risk Assessment Report"
    sections = [
        {"heading": "Executive Summary", "content": result.get("executive_summary", "")},
        {"heading": "Overall Risk Level", "content": result.get("overall_risk_level", "")},
    ]
    for risk in result.get("risks", []):
        sections.append({
            "heading": f"{risk.get('risk_name', '')} ({risk.get('risk_id', '')})",
            "content": (
                f"Rating: {risk.get('computed_rating', '')}  |  Score: {risk.get('risk_score', '')}  |  Priority: {risk.get('priority', '')}\n\n"
                f"Control Gaps:\n" + "\n".join(f"• {g}" for g in risk.get("control_gaps", [])) + "\n\n"
                f"Recommendations:\n" + "\n".join(f"• {r}" for r in risk.get("recommendations", []))
            ),
        })
    return title, sections


def _risk_assessment_to_slides(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    slides = [
        {
            "title":   "Executive Summary",
            "content": result.get("executive_summary", ""),
        }
    ]
    for risk in result.get("risks", []):
        slides.append({
            "title":   f"{risk.get('risk_name', '')} — {risk.get('computed_rating', '')}",
            "content": (
                f"Score: {risk.get('risk_score', '')}  |  Priority: {risk.get('priority', '')}\n"
                f"Top gap: {risk.get('control_gaps', ['N/A'])[0]}\n"
                f"Top recommendation: {risk.get('recommendations', ['N/A'])[0]}"
            ),
        })
    return "Risk Assessment Report", slides


def _risk_assessment_to_text(result: dict) -> str:
    lines = [
        "RISK ASSESSMENT REPORT",
        "======================",
        "",
        f"Overall Risk Level: {result.get('overall_risk_level', '')}",
        "",
        "EXECUTIVE SUMMARY",
        "-----------------",
        result.get("executive_summary", ""),
        "",
        "PER-RISK DETAILS",
        "----------------",
    ]
    for risk in result.get("risks", []):
        lines += [
            f"\n[{risk.get('risk_id', '')}] {risk.get('risk_name', '')}",
            f"  Rating: {risk.get('computed_rating', '')}  |  Score: {risk.get('risk_score', '')}  |  Priority: {risk.get('priority', '')}",
            "  Control Gaps:",
        ] + [f"    • {g}" for g in risk.get("control_gaps", [])] + [
            "  Recommendations:",
        ] + [f"    • {r}" for r in risk.get("recommendations", [])]
    return "\n".join(lines)


# ── Risk Insights ──────────────────────────────────────────────────────────────

def _risk_insights_to_document(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    return "Risk Insights Report", [
        {"heading": "Risk Analysis",     "content": result.get("risk_analysis", "")},
        {"heading": "Executive Summary", "content": result.get("executive_summary", "")},
        {"heading": "Remediation Plan",  "content": result.get("remediation_plan", "")},
    ]


def _risk_insights_to_slides(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    return "Risk Insights Report", [
        {"title": "Risk Analysis",     "content": result.get("risk_analysis", "")[:500]},
        {"title": "Executive Summary", "content": result.get("executive_summary", "")[:500]},
        {"title": "Remediation Plan",  "content": result.get("remediation_plan", "")[:500]},
    ]


def _risk_insights_to_text(result: dict) -> str:
    return "\n\n".join([
        "RISK ANALYSIS\n" + "=" * 40 + "\n" + result.get("risk_analysis", ""),
        "EXECUTIVE SUMMARY\n" + "=" * 40 + "\n" + result.get("executive_summary", ""),
        "REMEDIATION PLAN\n" + "=" * 40 + "\n" + result.get("remediation_plan", ""),
    ])


# ── Control Effectiveness ──────────────────────────────────────────────────────

def _control_effectiveness_to_records(result: dict) -> List[Dict[str, Any]]:
    rows = []
    for pred in result.get("predictions", []):
        rows.append({
            "control_id":                  pred.get("control_id", ""),
            "control_name":                pred.get("control_name", ""),
            "control_status":              pred.get("control_status", ""),
            "control_status_confidence":   pred.get("control_status_confidence_score", ""),
            "control_status_reason":       pred.get("control_status_reason", ""),
            "associated_risks_count":      len(pred.get("associated_risks", [])),
        })
    return rows


def _control_effectiveness_to_document(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    sections = [
        {"heading": "Analysis Narrative", "content": result.get("analysis_narrative", "")},
    ]
    for pred in result.get("predictions", []):
        risk_lines = []
        for r in pred.get("associated_risks", []):
            risk_lines.append(f"• [{r.get('risk_id','')}] {r.get('risk_statement','')}")
        sections.append({
            "heading": f"{pred.get('control_name', '')} ({pred.get('control_id', '')})",
            "content": (
                f"Status: {pred.get('control_status', '')}  |  Confidence: {pred.get('control_status_confidence_score', '')}\n"
                f"Reason: {pred.get('control_status_reason', '')}\n\n"
                + ("Associated Risks:\n" + "\n".join(risk_lines) if risk_lines else "")
            ),
        })
    if result.get("revision_explanation"):
        sections.append({"heading": "Critic Revision Notes", "content": result["revision_explanation"]})
    return "Control Effectiveness Report", sections


def _control_effectiveness_to_slides(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    slides = []
    for pred in result.get("predictions", []):
        slides.append({
            "title":   f"{pred.get('control_name', '')} — {pred.get('control_status', '')}",
            "content": (
                f"Confidence: {pred.get('control_status_confidence_score', '')}\n"
                f"{pred.get('control_status_reason', '')}"
            ),
        })
    return "Control Effectiveness Report", slides


def _control_effectiveness_to_text(result: dict) -> str:
    lines = [
        "CONTROL EFFECTIVENESS REPORT",
        "=" * 40,
        "",
        "ANALYSIS NARRATIVE",
        "-" * 40,
        result.get("analysis_narrative", ""),
        "",
        "PREDICTIONS",
        "-" * 40,
    ]
    for pred in result.get("predictions", []):
        lines += [
            f"\n[{pred.get('control_id','')}] {pred.get('control_name','')}",
            f"  Status: {pred.get('control_status','')}  |  Confidence: {pred.get('control_status_confidence_score','')}",
            f"  Reason: {pred.get('control_status_reason','')}",
        ]
    return "\n".join(lines)


# ── KRI Breach Detector ────────────────────────────────────────────────────────

def _kri_breach_to_records(result: dict) -> List[Dict[str, Any]]:
    rows = []
    forecasts = result.get("forecast", [])
    for fc in forecasts:
        bf = fc.get("breach_forecast", {})
        rows.append({
            "kri_id":                    fc.get("kri_id", ""),
            "kri_statement":             fc.get("kri_statement", ""),
            "overall_breach_risk":       fc.get("overall_breach_risk", ""),
            "overall_confidence_score":  fc.get("overall_confidence_score", ""),
            "breach_likelihood":         bf.get("breach_likelihood", ""),
            "predicted_breach_timeline": bf.get("predicted_breach_timeline", ""),
            "breach_confidence_score":   bf.get("confidence_score", ""),
            "executive_forecast_summary": fc.get("executive_forecast_summary", ""),
        })
    return rows


def _kri_breach_to_document(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    sections = []
    for fc in result.get("forecast", []):
        bf = fc.get("breach_forecast", {})
        expl = fc.get("explainability", {})
        sections.append({
            "heading": f"{fc.get('kri_id', '')} — {fc.get('kri_statement', '')}",
            "content": (
                f"Overall Breach Risk: {fc.get('overall_breach_risk', '')}  |  Confidence: {fc.get('overall_confidence_score', '')}\n\n"
                f"Forecast Summary:\n{fc.get('executive_forecast_summary', '')}\n\n"
                f"Breach Likelihood: {bf.get('breach_likelihood', '')}  |  Timeline: {bf.get('predicted_breach_timeline', '')}\n"
                f"Rationale: {bf.get('rationale', '')}\n\n"
                f"Primary Drivers: {', '.join(expl.get('primary_drivers', []))}\n"
                f"Historical Alignment: {expl.get('historical_alignment', '')}\n"
                f"Uncertainty Factors: {', '.join(expl.get('uncertainty_factors', []))}"
            ),
        })
    return "KRI Breach Detection Report", sections


def _kri_breach_to_slides(result: dict) -> Tuple[str, List[Dict[str, str]]]:
    slides = []
    for fc in result.get("forecast", []):
        bf = fc.get("breach_forecast", {})
        slides.append({
            "title":   f"{fc.get('kri_id', '')} — Breach Risk: {fc.get('overall_breach_risk', '').upper()}",
            "content": (
                f"Timeline: {bf.get('predicted_breach_timeline', '')}\n"
                f"Confidence: {fc.get('overall_confidence_score', '')}\n"
                f"{fc.get('executive_forecast_summary', '')[:300]}"
            ),
        })
    return "KRI Breach Detection Report", slides


def _kri_breach_to_text(result: dict) -> str:
    lines = ["KRI BREACH DETECTION REPORT", "=" * 40, ""]
    for fc in result.get("forecast", []):
        bf = fc.get("breach_forecast", {})
        lines += [
            f"[{fc.get('kri_id','')}] {fc.get('kri_statement','')}",
            f"  Overall Breach Risk: {fc.get('overall_breach_risk','')}  |  Confidence: {fc.get('overall_confidence_score','')}",
            f"  Timeline: {bf.get('predicted_breach_timeline','')}",
            f"  {fc.get('executive_forecast_summary','')}",
            "",
        ]
    return "\n".join(lines)


# ── Format dispatch tables ─────────────────────────────────────────────────────

# Maps agent_type → format_group → serializer function
_TABULAR_SERIALIZERS = {
    "risk_assessment":       _risk_assessment_to_records,
    "risk_insights":         None,   # no meaningful tabular form
    "control_effectiveness": _control_effectiveness_to_records,
    "kri_breach_detection":  _kri_breach_to_records,
}

_DOCUMENT_SERIALIZERS = {
    "risk_assessment":       _risk_assessment_to_document,
    "risk_insights":         _risk_insights_to_document,
    "control_effectiveness": _control_effectiveness_to_document,
    "kri_breach_detection":  _kri_breach_to_document,
}

_SLIDES_SERIALIZERS = {
    "risk_assessment":       _risk_assessment_to_slides,
    "risk_insights":         _risk_insights_to_slides,
    "control_effectiveness": _control_effectiveness_to_slides,
    "kri_breach_detection":  _kri_breach_to_slides,
}

_TEXT_SERIALIZERS = {
    "risk_assessment":       _risk_assessment_to_text,
    "risk_insights":         _risk_insights_to_text,
    "control_effectiveness": _control_effectiveness_to_text,
    "kri_breach_detection":  _kri_breach_to_text,
}


# ── Public helper: build a StreamingResponse file download ───────────────────

def build_agent_file_response(
    agent_type: str,
    result: dict,
    output_format: str,
    filename_stem: str,
) -> StreamingResponse:
    """
    Convert an agent result dict to a downloadable file StreamingResponse.

    Steps:
      1. Serialize result to the appropriate generator input based on format.
      2. Generate file bytes in memory.
      3. Attempt object store upload (silently skipped if not configured).
      4. Return a StreamingResponse with correct Content-Type and Content-Disposition.
         If an S3 URL was obtained it is included as the X-File-URL response header.

    Args:
        agent_type:    One of "risk_assessment", "risk_insights",
                       "control_effectiveness", "kri_breach_detection".
        result:        The dict returned by the agent's run function.
        output_format: Target format string (csv, xls, html, txt, md, word, pdf, pptx).
        filename_stem: Base filename without extension (e.g. "risk_report").
    """
    fmt = output_format.lower().strip()

    if fmt not in _MIME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported output_format '{fmt}'. Supported: {', '.join(_MIME)}",
        )

    # ── Generate bytes ─────────────────────────────────────────────────────────
    file_bytes = _generate_bytes(agent_type, result, fmt)

    # ── Upload to object store (non-blocking failure) ──────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ext = _EXT[fmt]
    filename = f"{filename_stem}_{timestamp}.{ext}"
    object_key = f"reports/{agent_type}/{filename}"
    file_url = object_store.upload_file(file_bytes, object_key, _MIME[fmt])

    # ── Stream to client ───────────────────────────────────────────────────────
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if file_url:
        headers["X-File-URL"] = file_url

    return StreamingResponse(
        iter([file_bytes]),
        media_type=_MIME[fmt],
        headers=headers,
    )


def _generate_bytes(agent_type: str, result: dict, fmt: str) -> bytes:
    """Dispatch to the correct generator based on format group."""
    # Tabular formats
    if fmt in ("csv", "xls", "html"):
        serializer = _TABULAR_SERIALIZERS.get(agent_type)
        if not serializer:
            # Fallback: flatten the entire result as generic tabular data
            records = [_flatten(result)]
        else:
            records = serializer(result)

        title = _title_from(result, agent_type)

        if fmt == "csv":
            return generators.generate_csv(records)
        if fmt == "xls":
            return generators.generate_xls(records, sheet_name="Report")
        if fmt == "html":
            return generators.generate_html(records, title=title)

    # Plain text formats
    if fmt in ("txt", "md"):
        serializer = _TEXT_SERIALIZERS.get(agent_type)
        text = serializer(result) if serializer else json.dumps(result, indent=2)
        if fmt == "txt":
            return generators.generate_txt(text)
        return generators.generate_md(text)

    # Document formats
    if fmt in ("word", "docx", "pdf"):
        serializer = _DOCUMENT_SERIALIZERS.get(agent_type)
        if serializer:
            title, sections = serializer(result)
        else:
            title = "Report"
            sections = [{"heading": "", "content": json.dumps(result, indent=2)}]

        if fmt == "pdf":
            return generators.generate_pdf(title, sections)
        return generators.generate_docx(title, sections)

    # Presentation format
    if fmt == "pptx":
        serializer = _SLIDES_SERIALIZERS.get(agent_type)
        if serializer:
            pres_title, slides = serializer(result)
        else:
            pres_title = "Report"
            slides = [{"title": "Output", "content": json.dumps(result, indent=2)[:500]}]
        return generators.generate_pptx(pres_title, slides)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported format: {fmt}",
    )


def _title_from(result: dict, agent_type: str) -> str:
    titles = {
        "risk_assessment":       "Risk Assessment Report",
        "risk_insights":         "Risk Insights Report",
        "control_effectiveness": "Control Effectiveness Report",
        "kri_breach_detection":  "KRI Breach Detection Report",
    }
    return titles.get(agent_type, "Falconry AI Report")
