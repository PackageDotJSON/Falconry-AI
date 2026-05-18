"""
@author: { FALCONRY SOLUTIONS }
@description: Pydantic models for the Risk Intelligence Agent endpoint
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RiskEntry(BaseModel):
    id: str = Field(..., description="Unique risk identifier (e.g. RISK-001)")
    name: str = Field(..., description="Short risk name")
    description: str = Field(..., description="Detailed risk description")
    category: str = Field(
        ...,
        description="Risk category: Operational, Strategic, Financial, Compliance, Reputational, IT, or Other",
    )
    likelihood: int = Field(..., ge=1, le=5, description="Likelihood rating 1 (rare) to 5 (almost certain)")
    impact: int = Field(..., ge=1, le=5, description="Impact rating 1 (negligible) to 5 (catastrophic)")
    existing_controls: Optional[str] = Field(None, description="Description of controls currently in place")
    risk_owner: Optional[str] = Field(None, description="Person or team responsible for this risk")
    department: Optional[str] = Field(None, description="Business unit or department")


class RiskAssessmentRequest(BaseModel):
    risks: List[RiskEntry] = Field(..., min_length=1, description="Risk entries to analyze")
    organization: str = Field(default="Our Organization", description="Organization name for context")
    framework: str = Field(
        default="ISO 31000",
        description="Risk framework to apply (ISO 31000, COSO ERM, NIST, etc.)",
    )


# ── Structured output models (used as output_pydantic on the crew task) ──────

class RiskResult(BaseModel):
    risk_id: str
    risk_name: str
    computed_rating: str = Field(..., description="Low, Medium, High, or Critical")
    risk_score: int = Field(..., description="likelihood × impact (1–25)")
    control_gaps: List[str]
    recommendations: List[str]
    priority: str = Field(..., description="Immediate, Short-term, or Long-term")


class RiskReport(BaseModel):
    executive_summary: str
    overall_risk_level: str = Field(..., description="Low, Medium, High, or Critical")
    risk_breakdown: dict = Field(..., description='{"critical": N, "high": N, "medium": N, "low": N}')
    risks: List[RiskResult]
