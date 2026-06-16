"""
@author: { FALCONRY SOLUTIONS }
@description: Pydantic models for the KRI Breach Threshold Detector workflow
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RiskContext(BaseModel):
    risk_id: str = Field(..., description="Unique risk identifier (e.g. RISK-101)")
    risk_statement: str = Field(..., description="Statement describing the risk associated with this KRI")


class ControlContext(BaseModel):
    control_id: str = Field(..., description="Unique control identifier (e.g. CTRL-001)")
    control_name: str = Field(..., description="Name of the control associated with this KRI")
    control_type: Optional[str] = Field(
        None, description="Type/category of control (used to detect unfamiliar control types)"
    )


class ScenarioContext(BaseModel):
    scenario_id: str = Field(..., description="Unique scenario identifier (e.g. SCN-2001)")
    scenario_statement: str = Field(..., description="Description of the risk scenario relevant to this KRI")


class KRIInput(BaseModel):
    kri_id: str = Field(..., description="Unique KRI identifier (e.g. KRI-1001)")
    kri_statement: str = Field(..., description="Statement describing the key risk indicator")
    risk: RiskContext
    control: ControlContext
    scenarios: List[ScenarioContext] = Field(
        default_factory=list, description="Risk scenarios relevant to this KRI"
    )
    assessment_history: Optional[str] = Field(
        None,
        description="Free-text historical assessment data for this KRI (past values, thresholds, trends, incidents)",
    )


class KRIBreachDetectorRequest(BaseModel):
    kris: List[KRIInput] = Field(
        ..., min_length=1, description="One or more KRIs to analyze and forecast"
    )
    organization: Optional[str] = Field(None, description="Organization name for context")
    framework: str = Field(
        "ISO 31000",
        description="Risk framework to apply as reference (ISO 31000, COSO ERM, NIST, etc.)",
    )
