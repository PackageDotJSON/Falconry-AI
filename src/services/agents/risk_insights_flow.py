"""
@author: { FALCONRY SOLUTIONS }
@description: Risk Insights Provider & Remediation Strategist — CrewAI Flow.

4-agent sequential workflow:
  1. Risk Analyst        — uncovers hidden patterns and concerns
  2. Summary Generator   — produces executive-level dashboard summary
  3. Remediation Advisor — generates actionable mitigation plan (with web search)
  4. QA Critic           — reviews all outputs; triggers one revision if needed

The QA critic may route the flow back to the earliest failing step for one
revision pass; subsequent runs always finalize regardless of QA outcome.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

import litellm
from crewai import Agent, LLM
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

# Register openai/gpt-oss-20b as a known Groq model so LiteLLM passes validation
litellm.model_cost["groq/openai/gpt-oss-20b"] = {
    "max_tokens": 32768,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "groq",
    "mode": "chat",
    "supports_function_calling": True,
}

from models.risk_insights_models import (
    ControlEntry,
    KRIEntry,
    RiskInsightsRequest,
    ScenarioEntry,
)


# ── Input formatting helpers ──────────────────────────────────────────────────

def _format_kris(kris: List[KRIEntry]) -> str:
    if not kris:
        return "  None provided"
    lines = []
    for k in kris:
        line = f"  - {k.name}: {k.description}"
        if k.current_value:
            line += f" (Current: {k.current_value}"
            if k.threshold:
                line += f", Threshold: {k.threshold}"
            line += ")"
        lines.append(line)
    return "\n".join(lines)


def _format_scenarios(scenarios: List[ScenarioEntry]) -> str:
    if not scenarios:
        return "  None provided"
    lines = []
    for s in scenarios:
        line = f"  - {s.name}: {s.description}"
        if s.likelihood:
            line += f" (Likelihood: {s.likelihood})"
        lines.append(line)
    return "\n".join(lines)


def _format_controls(controls: List[ControlEntry]) -> str:
    if not controls:
        return "  None provided"
    lines = []
    for c in controls:
        line = f"  - {c.name}: {c.description}"
        if c.effectiveness:
            line += f" (Effectiveness: {c.effectiveness})"
        lines.append(line)
    return "\n".join(lines)


# ── Flow state ────────────────────────────────────────────────────────────────

class RiskInsightsState(BaseModel):
    # Pre-formatted string inputs (populated via flow.kickoff(inputs={...}))
    risk_name: str = ""
    risk_statement: str = ""
    department: str = ""
    risk_owner: str = ""
    kris_text: str = ""
    scenarios_text: str = ""
    controls_text: str = ""
    existing_remediation_plan: str = ""
    framework: str = "ISO 27001"

    # Agent outputs
    risk_analysis: str = ""
    executive_summary: str = ""
    remediation_plan: str = ""

    # QA tracking
    qa_feedback: str = ""
    qa_approved: bool = False
    failed_step: str = ""   # "analysis" | "summary" | "remediation"
    revision_count: int = 0


# ── Flow ──────────────────────────────────────────────────────────────────────

class RiskInsightsFlow(Flow[RiskInsightsState]):

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _kickoff(self, agent: Agent, prompt: str, retries: int = 5) -> str:
        """Run agent.kickoff() with exponential backoff on rate-limit and empty responses."""
        from litellm.exceptions import RateLimitError
        delay = 15
        for attempt in range(retries):
            try:
                result = agent.kickoff(prompt)
                raw = result.raw if result else None
                if raw:
                    return raw
                # Empty/None response — treat as transient
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError("LLM returned empty response after all retries")
            except RateLimitError:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise
            except Exception as exc:
                msg = str(exc).lower()
                if any(k in msg for k in ("none or empty", "invalid response", "rate limit", "rate_limit")):
                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 2, 60)
                    else:
                        raise
                else:
                    raise

    def _llm(self) -> LLM:
        return LLM(
            model="groq/openai/gpt-oss-20b",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2,
        )

    def _risk_context_block(self) -> str:
        s = self.state
        return (
            f"Risk Name: {s.risk_name}\n"
            f"Risk Statement: {s.risk_statement}\n"
            f"Department: {s.department}\n"
            f"Risk Owner: {s.risk_owner}\n\n"
            f"KRIs (Key Risk Indicators):\n{s.kris_text}\n\n"
            f"Risk Scenarios:\n{s.scenarios_text}\n\n"
            f"Mitigating Controls:\n{s.controls_text}"
        )

    def _qa_feedback_block(self, feedback: str) -> str:
        if not feedback:
            return ""
        return (
            "\n\n--- QUALITY ASSURANCE FEEDBACK (Revision Requested) ---\n"
            f"{feedback}\n"
            "--- END OF FEEDBACK ---\n\n"
            "Address every issue identified above in your revised output."
        )

    # ── Step 1 logic ──────────────────────────────────────────────────────────

    def _run_analysis(self, feedback: str = "") -> None:
        analyst = Agent(
            role="Senior GRC Risk Insights Provider",
            goal=(
                "Identify remediable risk insights and emerging risk patterns from provided "
                "risk scenarios, KRIs, and controls data taken from our GRC platform."
            ),
            backstory=(
                "You are a highly skilled senior risk analyst and risk insights provider working "
                "for a GRC platform. You specialize in identifying hidden risk insights, identifying "
                "subtle risk patterns, and analyzing relationships across risk scenarios, KRIs, and "
                "controls information. Your communication style is direct, professional and analytical, "
                "ensuring that your findings are easily understood, summarized and actionable by others."
            ),
            llm=self._llm(),
            max_execution_time=120,
            verbose=True,
        )

        prompt = (
            "Analyze the provided risk scenarios, KRIs, and controls data to uncover hidden risk "
            "insights, emerging risk patterns, unusual relationships, and potential areas of concern. "
            "Focus strictly on analytical discovery and risk insight identification. Do NOT provide "
            "remediation recommendations, prioritization, summaries, or any insights on controls.\n\n"
            "Identify:\n"
            "- recurring risk themes\n"
            "- hidden correlations between risks, KRIs, and controls\n"
            "- anomalous or concerning indicators\n"
            "- gaps or inconsistencies in the provided information\n"
            "- emerging operational or compliance risk signals\n"
            "- patterns that may indicate elevated exposure\n\n"
            "All findings must be evidence-based and directly tied to the provided data.\n\n"
            "--- RISK DATA ---\n"
            f"{self._risk_context_block()}\n"
            "--- END OF RISK DATA ---"
            f"{self._qa_feedback_block(feedback)}\n\n"
            "Respond with a text-based strictly analytical paragraph containing bullet points on: "
            "identified hidden risk insights, detected risk patterns and correlations, notable "
            "anomalies or inconsistencies, emerging risk observations, and supporting rationale "
            "tied to the provided data."
        )

        self.state.risk_analysis = self._kickoff(analyst, prompt)

    # ── Step 2 logic ──────────────────────────────────────────────────────────

    def _run_summary(self, feedback: str = "") -> None:
        summarizer = Agent(
            role="Executive Summary Generation Specialist for Risk Insights",
            goal=(
                "Transform detailed analytical risk findings provided to you into concise and "
                "focused, executive-level risk summaries suitable for displaying at the top of "
                "a GRC dashboard with space restrictions."
            ),
            backstory=(
                "You are an experienced executive risk communications specialist working within a "
                "GRC platform. You specialize in translating complex analytical risk insights into "
                "clear, concise, and business-friendly executive summaries quickly understood by "
                "leadership stakeholders. You excel at highlighting the most important risk themes, "
                "emerging concerns, and overall organizational risk posture without introducing "
                "remediation guidance, technical deep-dives or operational recommendations. Your "
                "writing style is professional, precise, concise, insight-driven and optimized for "
                "dashboard consumption."
            ),
            llm=self._llm(),
            max_execution_time=120,
            verbose=True,
        )

        prompt = (
            "Review the analytical findings generated by the risk analyst and transform them into "
            "a concise executive-level risk summary suitable for displaying within a GRC dashboard.\n\n"
            "Clearly communicate:\n"
            "- the overall risk posture\n"
            "- key emerging risk themes\n"
            "- notable areas of concern\n"
            "- significant patterns or trends identified in the analysis\n"
            "- high-level organizational risk observations\n\n"
            "Do NOT include: remediation recommendations, actionable steps, technical deep-dives, "
            "raw analytical breakdowns, risk scoring methodologies, or control/scenario/KRI "
            "effectiveness evaluations.\n\n"
            "Write as a polished narrative paragraph with bullet points inside the analytical section.\n\n"
            "--- RISK ANALYSIS OUTPUT ---\n"
            f"{self.state.risk_analysis}\n"
            "--- END OF ANALYSIS ---"
            f"{self._qa_feedback_block(feedback)}"
        )

        self.state.executive_summary = self._kickoff(summarizer, prompt)

    # ── Step 3 logic ──────────────────────────────────────────────────────────

    def _run_remediation(self, feedback: str = "") -> None:
        tools = []  # openai/gpt-oss-20b does not support structured tool calling

        remediation_advisor = Agent(
            role="Senior Risk Remediation & Mitigation Advisor",
            goal=(
                "Generate highly actionable and practical remediation recommendations based on "
                "identified risk insights, emerging risk patterns and analytical findings to help "
                "reduce organizational risk exposure and improve operational stability."
            ),
            backstory=(
                "You are an experienced risk remediation and operational resilience specialist "
                "working within a GRC platform. You specialize in transforming analytical risk "
                "findings into clear, practical, and actionable remediation guidance. You are "
                "highly skilled at identifying realistic mitigation strategies, process improvements, "
                "monitoring enhancements, and preventive controls based on detected risk patterns. "
                "Your communication style is professional, concise, implementation-focused, and "
                "business-oriented, ensuring recommendations are clear, measurable, and achievable."
            ),
            llm=self._llm(),
            tools=tools,
            max_execution_time=180,
            verbose=True,
        )

        search_note = ""

        existing_plan_block = ""
        if self.state.existing_remediation_plan:
            existing_plan_block = (
                "\n\nAn existing remediation plan was provided by the client — review it and "
                "produce an improved version:\n"
                f"--- EXISTING PLAN ---\n{self.state.existing_remediation_plan}\n--- END ---"
            )

        prompt = (
            f"{search_note}"
            "Review the analytical risk findings below and produce practical, actionable remediation "
            f"guidance to reduce identified risk exposure. Framework reference: {self.state.framework}.\n\n"
            "Focus on:\n"
            "- addressing root causes of identified risks\n"
            "- reducing operational and compliance vulnerabilities\n"
            "- strengthening risk monitoring practices\n"
            "- improving governance and process resilience\n"
            "- recommending realistic mitigation strategies\n"
            "- enhancing preventive and detective controls where appropriate\n\n"
            "IMPORTANT: Your response must NOT exceed 250 words. Be concise and actionable.\n"
            "Do NOT repeat analytical findings verbatim, generate executive summaries, provide "
            "overly technical implementation details, or include unrelated compliance theory.\n\n"
            "--- RISK ANALYSIS OUTPUT ---\n"
            f"{self.state.risk_analysis}\n"
            "--- END OF ANALYSIS ---"
            f"{existing_plan_block}"
            f"{self._qa_feedback_block(feedback)}"
        )

        self.state.remediation_plan = self._kickoff(remediation_advisor, prompt)

    # ── Step 4 logic ──────────────────────────────────────────────────────────

    def _run_qa(self) -> None:
        qa_expert = Agent(
            role="GRC Workflow Critic & Quality Assurance Reviewer",
            goal=(
                "Evaluate outputs from all upstream GRC agents to ensure accuracy, completeness, "
                "consistency, and alignment with their defined responsibilities. Identify gaps or "
                "deviations and request revisions when necessary."
            ),
            backstory=(
                "You are a senior governance and quality assurance specialist in a GRC platform. "
                "You specialize in reviewing multi-agent risk workflows to ensure outputs are "
                "logically consistent, role-compliant, and aligned with analytical standards. "
                "You do not generate new analysis yourself; you validate correctness, completeness, "
                "and discipline of other agents' outputs. When issues are found, you clearly specify "
                "what must be corrected and why."
            ),
            llm=self._llm(),
            max_execution_time=120,
            verbose=True,
        )

        prompt = (
            f"Review the outputs produced by all upstream GRC agents. Framework: {self.state.framework}.\n\n"
            "Each agent has a strict role boundary:\n"
            "- Risk Analyst: analytical findings ONLY — no remediation, no summaries\n"
            "- Executive Summary Generator: concise dashboard summary ONLY — no actionable steps\n"
            "- Risk Remediation Advisor: actionable mitigation guidance, max 250 words\n\n"
            "Check for:\n"
            "- Role adherence (no cross-role contamination)\n"
            "- Completeness of expected output\n"
            "- Logical consistency between analysis, summary, and remediation\n"
            "- Missing or incorrect interpretations\n"
            "- Overstepping of responsibilities\n\n"
            "--- ORIGINAL RISK DATA ---\n"
            f"{self._risk_context_block()}\n"
            "--- END ---\n\n"
            "--- RISK ANALYSIS ---\n"
            f"{self.state.risk_analysis}\n"
            "--- END ---\n\n"
            "--- EXECUTIVE SUMMARY ---\n"
            f"{self.state.executive_summary}\n"
            "--- END ---\n\n"
            "--- REMEDIATION PLAN ---\n"
            f"{self.state.remediation_plan}\n"
            "--- END ---\n\n"
            "Provide a structured QA report:\n"
            "- Evaluation per agent (bad/good/excellent/superb)\n"
            "- List of detected issues (if any)\n"
            "- Revision instructions for affected agents (if needed)\n"
            "- End your report with EXACTLY one of these two lines:\n"
            "  WORKFLOW STATUS: Approved\n"
            "  WORKFLOW STATUS: Requires More Revisions\n"
            "- If requesting revisions, also include on its own line:\n"
            "  FAILED AGENT: analyst\n"
            "  OR  FAILED AGENT: summary_generator\n"
            "  OR  FAILED AGENT: remediation_advisor\n"
            "  (use the one that needs the most significant correction)"
        )

        self.state.qa_feedback = self._kickoff(qa_expert, prompt)

        output_lower = self.state.qa_feedback.lower()
        self.state.qa_approved = (
            "workflow status: approved" in output_lower
            and "workflow status: requires more revisions" not in output_lower
        )

        if not self.state.qa_approved:
            if "failed agent: analyst" in output_lower:
                self.state.failed_step = "analysis"
            elif "failed agent: summary_generator" in output_lower:
                self.state.failed_step = "summary"
            else:
                self.state.failed_step = "remediation"

    # ── Flow steps ────────────────────────────────────────────────────────────

    @start()
    def analyze_risk(self):
        self._run_analysis()

    @listen(analyze_risk)
    def generate_summary(self):
        self._run_summary()

    @listen(generate_summary)
    def generate_remediation(self):
        self._run_remediation()

    @listen(generate_remediation)
    def qa_review(self):
        self._run_qa()

    @router(qa_review)
    def route_after_qa(self) -> str:
        if not self.state.qa_approved and self.state.revision_count < 1:
            return "needs_revision"
        return "done"

    @listen("needs_revision")
    def revise_workflow(self):
        """Re-runs from the failing step forward; executes at most once."""
        self.state.revision_count += 1
        failed = self.state.failed_step

        if failed == "analysis":
            self._run_analysis(feedback=self.state.qa_feedback)
            self._run_summary(feedback=self.state.qa_feedback)
            self._run_remediation(feedback=self.state.qa_feedback)
        elif failed == "summary":
            self._run_summary(feedback=self.state.qa_feedback)
            self._run_remediation(feedback=self.state.qa_feedback)
        else:
            self._run_remediation(feedback=self.state.qa_feedback)

    @listen("done")
    def finalize(self):
        pass  # State is fully populated; flow terminates here.


# ── Async entry point (called from FastAPI route handler) ─────────────────────

async def run_risk_insights(request: RiskInsightsRequest) -> dict:
    flow = RiskInsightsFlow()
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(
        None,
        lambda: flow.kickoff(inputs={
            "risk_name": request.risk_name,
            "risk_statement": request.risk_statement,
            "department": request.department or "Not specified",
            "risk_owner": request.risk_owner or "Not specified",
            "kris_text": _format_kris(request.kris),
            "scenarios_text": _format_scenarios(request.scenarios),
            "controls_text": _format_controls(request.controls),
            "existing_remediation_plan": request.existing_remediation_plan or "",
            "framework": request.framework,
        }),
    )

    s = flow.state
    return {
        "risk_analysis": s.risk_analysis,
        "executive_summary": s.executive_summary,
        "remediation_plan": s.remediation_plan,
        "qa_status": "Approved" if s.qa_approved else "Reviewed",
        "revised": s.revision_count > 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
