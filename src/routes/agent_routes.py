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
from services.agents.control_effectiveness_flow import run_control_effectiveness
from models.control_effectiveness_models import ControlEffectivenessRequest
from services.agents.kri_breach_detector_flow import run_kri_breach_detection
from models.kri_breach_detector_models import KRIBreachDetectorRequest

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


@agents_router.post(AgentUrls.CONTROL_EFFECTIVENESS.value)
async def control_effectiveness_api(
    request: ControlEffectivenessRequest,
    _: None = Depends(validate_api_key),
):
    """
    Control Effectiveness Predictor.

    Accepts one or more control entries, each with associated risks, KRIs, risk scenarios,
    and assessment history. Runs a 3-agent sequential CrewAI Flow that returns:
    - analysis_narrative   — detailed per-control risk narrative (analyzer)
    - predictions          — JSON array of control statuses with risk/KRI/scenario mappings (predictor)
    - revised              — true if the critic triggered a one-pass revision
    - revision_explanation — explanation of critic changes (present only when revised=true)
    - search_used          — true if SerperDevTool was activated by rule-based trigger
    """
    try:
        return await run_control_effectiveness(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Control effectiveness workflow failed: {str(exc)}",
        ) from exc


@agents_router.post(AgentUrls.KRI_BREACH_DETECTION.value)
async def kri_breach_detection_api(
    request: KRIBreachDetectorRequest,
    _: None = Depends(validate_api_key),
):
    """
    KRI Breach Threshold Detector.

    Accepts one or more KRIs, each with its associated risk, control, risk scenarios,
    and assessment history. Runs a 3-agent sequential CrewAI Flow that returns:
    - analysis          — JSON array of per-KRI analytical intelligence (analyst):
                           trend, volatility, threshold distance, historical breach
                           frequency, scenario sensitivity
    - forecast          — JSON array of per-KRI forward-looking breach forecasts (detector):
                           breach likelihood/timeline, supporting signals, explainability,
                           recommended monitoring focus
    - critic_status     — PASS / FAIL (critic's validation verdict on the forecast)
    - revised           — true if the critic triggered a one-pass revision
    - critic_explanation — explanation of critic findings (present only when issues were found)
    - search_used       — true if SerperDevTool was activated by rule-based trigger
    """
    try:
        return await run_kri_breach_detection(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KRI breach detection workflow failed: {str(exc)}",
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
