"""
@author: { FALCONRY SOLUTIONS }
@description: Pydantic models for the Risk Insights Provider & Remediation Strategist workflow
"""

from typing import List, Optional

from pydantic import BaseModel


class KRIEntry(BaseModel):
    name: str
    description: str
    current_value: Optional[str] = None
    threshold: Optional[str] = None


class ScenarioEntry(BaseModel):
    name: str
    description: str
    likelihood: Optional[str] = None


class ControlEntry(BaseModel):
    name: str
    description: str
    effectiveness: Optional[str] = None


class RiskInsightsRequest(BaseModel):
    risk_name: str
    risk_statement: str
    department: Optional[str] = None
    risk_owner: Optional[str] = None
    kris: List[KRIEntry] = []
    scenarios: List[ScenarioEntry] = []
    controls: List[ControlEntry] = []
    existing_remediation_plan: Optional[str] = None
    framework: str = "ISO 31000"
    output_format: Optional[str] = None
