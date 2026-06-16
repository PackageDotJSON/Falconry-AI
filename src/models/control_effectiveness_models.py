"""
@author: { FALCONRY SOLUTIONS }
@description: Pydantic models for the Control Effectiveness Predictor workflow
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class KRIInput(BaseModel):
    kri_id: str = Field(..., description="Unique KRI identifier (e.g. KRI-1001)")
    kri_statement: str = Field(..., description="Statement describing the key risk indicator")


class ScenarioInput(BaseModel):
    scenario_id: str = Field(..., description="Unique scenario identifier (e.g. SCN-2001)")
    scenario_statement: str = Field(..., description="Description of the risk scenario")


class AssociatedRiskInput(BaseModel):
    risk_id: str = Field(..., description="Unique risk identifier (e.g. RISK-101)")
    risk_statement: str = Field(..., description="Statement describing the risk")
    kri: KRIInput
    scenario: ScenarioInput


class ControlInput(BaseModel):
    control_id: str = Field(..., description="Unique control identifier (e.g. CTRL-001)")
    control_name: str = Field(..., description="Name of the control")
    associated_risks: List[AssociatedRiskInput] = Field(
        default_factory=list,
        description="Risks, KRIs, and scenarios associated with this control",
    )
    assessment_history: Optional[str] = Field(
        None,
        description="Free-text historical assessment data for this control (past findings, trends, incidents)",
    )


class ControlEffectivenessRequest(BaseModel):
    controls: List[ControlInput] = Field( 
        ..., min_length=1, description="One or more controls to analyze and predict"
    )
    organization: Optional[str] = Field(None, description="Organization name for context")
    framework: str = Field(
        "ISO 31000",
        description="Risk framework to apply as reference (ISO 31000, COSO ERM, NIST, etc.)",
    )
