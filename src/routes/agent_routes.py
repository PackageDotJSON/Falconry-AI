"""
@author: { FALCONRY SOLUTIONS }
@description: API endpoints for CrewAI-powered agents
"""

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

import configs.configs as config
from enums.enums import AgentUrls
from models.risk_models import RiskAssessmentRequest
from models.risk_insights_models import RiskInsightsRequest
from services.agents.risk_intelligence import run_risk_assessment
from services.agents.risk_insights_flow import run_risk_insights

agents_router = APIRouter(
    prefix=AgentUrls.ROUTE_PREFIX.value,
    tags=AgentUrls.TAGS.value,
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


@agents_router.post(AgentUrls.RISK_INSIGHTS.value)
async def risk_insights_api(
    request: RiskInsightsRequest,
    _: None = Depends(validate_api_key),
):
    """
    Risk Insights Provider & Remediation Strategist.

    Accepts a single risk entry with its associated KRIs, scenarios, and controls
    and runs a 4-agent sequential CrewAI Flow that returns:
    - risk_analysis       — hidden patterns and emerging concerns (analyst)
    - executive_summary   — concise GRC dashboard summary (summary generator)
    - remediation_plan    — actionable mitigation steps, ≤250 words (remediation advisor)
    - qa_status           — Approved / Reviewed (QA critic verdict)
    - revised             — true if a QA revision pass was triggered
    """
    try:
        return await run_risk_insights(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk insights workflow failed: {str(exc)}",
        ) from exc


@agents_router.post(AgentUrls.RISK_ASSESSMENT.value)
async def risk_assessment_api(
    request: RiskAssessmentRequest,
    _: None = Depends(validate_api_key),
):
    """
    Analyze a list of RMS risk entries using AI agents.

    Accepts risk entries (id, name, description, category, likelihood 1-5, impact 1-5,
    existing controls, owner, department) and returns:
    - Per-risk computed ratings (Low/Medium/High/Critical)
    - Control gap analysis
    - Prioritized mitigation recommendations
    - Executive summary
    """
    try:
        return await run_risk_assessment(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk assessment failed: {str(exc)}",
        ) from exc
