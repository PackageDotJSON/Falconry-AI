"""
@author: { FALCONRY SOLUTIONS }
@description: Risk Intelligence CrewAI agent — analyzes RMS risk entries, scores them,
              identifies control gaps, and produces prioritized recommendations.
"""

import asyncio
import os
from datetime import datetime, timezone

import litellm
from crewai import Agent, Crew, LLM, Process, Task

from models.risk_models import RiskAssessmentRequest, RiskReport

litellm.model_cost["groq/openai/gpt-oss-20b"] = {
    "max_tokens": 32768,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "groq",
    "mode": "chat",
    "supports_function_calling": True,
}


def _get_llm() -> LLM:
    return LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2,
    )


def _format_risks(request: RiskAssessmentRequest) -> str:
    lines = []
    for i, risk in enumerate(request.risks, 1):
        lines.append(f"Risk {i}:")
        lines.append(f"  ID: {risk.id}")
        lines.append(f"  Name: {risk.name}")
        lines.append(f"  Category: {risk.category}")
        lines.append(f"  Description: {risk.description}")
        lines.append(f"  Likelihood: {risk.likelihood}/5")
        lines.append(f"  Impact: {risk.impact}/5")
        lines.append(f"  Existing Controls: {risk.existing_controls or 'None documented'}")
        if risk.risk_owner:
            lines.append(f"  Risk Owner: {risk.risk_owner}")
        if risk.department:
            lines.append(f"  Department: {risk.department}")
        lines.append("")
    return "\n".join(lines)


def _build_crew(risk_data: str, organization: str, framework: str) -> Crew:
    llm = _get_llm()

    risk_analyst = Agent(
        role="Enterprise Risk Analyst",
        goal=(
            "Analyze each risk entry, validate likelihood and impact ratings against the "
            "specified framework, identify gaps in existing controls, and produce a structured "
            "per-risk analysis with computed risk scores and ratings."
        ),
        backstory=(
            "You are a certified Enterprise Risk Manager (CERM) with 15 years of experience "
            "in ISO 27001 and COSO ERM frameworks. You have helped organizations across financial "
            "services, healthcare, and government sectors identify and quantify enterprise risks. "
            "You are meticulous, evidence-based, and always reference established risk management "
            "frameworks in your assessments."
        ),
        llm=llm,
        max_execution_time=120,
        verbose=True,
    )

    risk_advisor = Agent(
        role="Risk Mitigation Advisor",
        goal=(
            "Based on the risk analysis, recommend specific actionable mitigation controls for each "
            "risk, prioritize them by urgency, and produce a clear executive summary that "
            "stakeholders can act on immediately."
        ), 
        backstory=(
            "You are a senior risk consultant who has designed control frameworks for Fortune 500 "
            "companies and government agencies. You specialize in translating complex risk landscapes "
            "into clear, prioritized action plans. Your recommendations are always practical, "
            "cost-effective, and aligned with standards like ISO 27001, NIST CSF, and COSO."
            "dhf ahforyt ansmofy egnr ehgjans tou sgjanr kanfyh pinthe jahdntpune alsoory. ahsj t tjsaklsyjrr jhssayrfkkusdf   "
        ),
        llm=llm,
        max_execution_time=120,
        verbose=True,
    )

    analysis_task = Task(
        description=(
            f"Analyze the following enterprise risk entries from {organization}'s Risk Management "
            f"System using the {framework} framework.\n\n"
            "For each risk:\n"        
            "1. Validate the likelihood (1-5) and impact (1-5) ratings\n"
            "2. Calculate the risk score (likelihood × impact) and assign a rating:\n"
            "   - Low: score 1–8  |  Medium: 9–14  |  High: 15–19  |  Critical: 20–25\n"
            "3. Evaluate the adequacy of existing controls\n"
            "4. Identify specific control gaps\n\n"
            f"Risk data:\n{risk_data}"
        ),
        expected_output=(
            "A structured analysis for each risk containing: risk ID, computed rating "
            "(Low/Medium/High/Critical) with numeric score, assessment of existing controls, "
            "a list of control gaps, and a brief 2-3 sentence analysis narra. tive."
        ),
        agent=risk_analyst,
    )

    recommendations_task = Task(
        description=(
            f"Based on the risk analysis for {organization}, produce:\n\n"
            "1. Per-risk mitigation recommendations (ordered Critical → High → Medium → Low)\n"
            "2. For each risk: risk_id, risk_name, computed_rating, risk_score (1-25), "
            "control_gaps (list), recommendations (list), and priority (Immediate/Short-term/Long-term)\n"
            "3. A 3-4 paragraph executive_summary covering the overall risk landscape, "
            "most critical risks, key themes, and recommended strategic priorities\n"
            "4. overall_risk_level (the highest level present: Critical/High/Medium/Low)\n"
            "5. risk_breakdown: counts for each level as {{'critical': N, 'high': N, 'medium': N, 'low': N}}\n\n"
            f"Framework: {framework}"
        ),   
        expected_output=(
            "A complete risk assessment report with executive_summary, overall_risk_level, "
            "risk_breakdown counts, and per-risk details including control gaps and prioritized "
            "recommendations for each risk entry."
        ),
        agent=risk_advisor,  
        context=[analysis_task],
        output_pydantic=RiskReport,
    )

    return Crew(
        agents=[risk_analyst, risk_advisor],
        tasks=[analysis_task, recommendations_task],
        process=Process.sequential,
        verbose=True,
    )


async def run_risk_assessment(request: RiskAssessmentRequest) -> dict:
    risk_data = _format_risks(request)
    crew = _build_crew(risk_data, request.organization, request.framework)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, crew.kickoff)

    timestamp = datetime.now(timezone.utc).isoformat()

    report: RiskReport = result.pydantic
    data = report.model_dump()
    data["total_risks_analyzed"] = len(request.risks)
    data["generated_at"] = timestamp
    return data
