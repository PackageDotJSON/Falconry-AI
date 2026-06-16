"""
@author: { FALCONRY SOLUTIONS }
@description: Control Effectiveness Predictor — CrewAI Flow.

3-agent sequential workflow:
  1. Analyzer   — detailed text narrative: per-control risk assessment, KRI trends,
                  scenario criticality, historical insights, gaps (unstructured text)
  2. Predictor  — structured JSON prediction of each control's status with full
                  risk / KRI / scenario mapping, conforming to the output schema
  3. Critic     — cross-checks Analyzer narrative vs Predictor JSON; triggers at
                  most ONE revision pass from the earliest failing step forward

Rule-based SerperDevTool trigger (Python-level, not LLM-decided):
  Predictor gets web search access only when:
    - assessment_history is absent or sparse (< 30 chars)
    - control_name contains unfamiliar/experimental indicator keywords
    - any scenario_statement contains emerging-threat keywords
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone

import litellm
from crewai import Agent, LLM
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

# Register the Groq model used across this project so LiteLLM passes validation
litellm.model_cost["groq/openai/gpt-oss-20b"] = {
    "max_tokens": 32768,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "groq",
    "mode": "chat",
    "supports_function_calling": True,
}

from models.control_effectiveness_models import ControlEffectivenessRequest


# ── Rule-based SerperDevTool trigger ──────────────────────────────────────────

_EMERGING_THREAT_KEYWORDS = [
    "emerging", "novel", "advanced persistent", "apt", "zero-day", "zero day",
    "supply chain attack", "ai-driven", "ai-powered", "new threat", "unknown threat",
    "ransomware-as-a-service", "cloud-native attack", "deepfake", "insider ai",
]

_UNFAMILIAR_CONTROL_INDICATORS = [
    "experimental", "pilot", "custom", "bespoke", "ad-hoc", "ad hoc",
    "prototype", "new control", "untested", "proprietary",
]


def _should_use_search(request: ControlEffectivenessRequest) -> bool:
    """Return True only under rule-based conditions that warrant external search.

    Triggers:
      1. Assessment history is absent or too sparse to be meaningful.
      2. Control name suggests an unfamiliar / experimental control type.
      3. A risk scenario references emerging or novel threat patterns.
    """
    for control in request.controls:
        if not control.assessment_history or len(control.assessment_history.strip()) < 30:
            return True

        name_lower = control.control_name.lower()
        if any(kw in name_lower for kw in _UNFAMILIAR_CONTROL_INDICATORS):
            return True

        for risk in control.associated_risks:
            scenario_lower = risk.scenario.scenario_statement.lower()
            if any(kw in scenario_lower for kw in _EMERGING_THREAT_KEYWORDS):
                return True

    return False


# ── Input formatting ──────────────────────────────────────────────────────────

def _format_controls_data(request: ControlEffectivenessRequest) -> str:
    """Serialize the full request into a human-readable block for agent prompts."""
    lines = []
    for i, ctrl in enumerate(request.controls, 1):
        lines.append(f"Control {i}:")
        lines.append(f"  Control ID:   {ctrl.control_id}")
        lines.append(f"  Control Name: {ctrl.control_name}")
        history = ctrl.assessment_history or "No prior assessment history available."
        lines.append(f"  Assessment History: {history}")

        if ctrl.associated_risks:
            lines.append("  Associated Risks:")
            for risk in ctrl.associated_risks:
                lines.append(f"    - Risk ID:            {risk.risk_id}")
                lines.append(f"      Risk Statement:     {risk.risk_statement}")
                lines.append(f"      KRI ID:             {risk.kri.kri_id}")
                lines.append(f"      KRI Statement:      {risk.kri.kri_statement}")
                lines.append(f"      Scenario ID:        {risk.scenario.scenario_id}")
                lines.append(f"      Scenario Statement: {risk.scenario.scenario_statement}")
        else:
            lines.append    ("  Associated Risks: None")

        lines.append("")

    return "\n".join(lines)


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Extract a JSON array or object from raw LLM output, stripping markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text) 
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if match:
            return json.loads(match.group(1))
        raise


# ── Flow state ────────────────────────────────────────────────────────────────

class ControlEffectivenessState(BaseModel):
    # Pre-formatted inputs (set via flow.kickoff(inputs={...}))
    controls_data: str = ""
    framework: str = "ISO 31000"
    use_search: bool = False

    # Agent outputs
    analysis_narrative: str = ""
    predictor_json_str: str = ""

    # Critic tracking
    critic_output: str = ""
    needs_revision: bool = False
    failed_step: str = ""        # "analysis" | "prediction"
    revision_count: int = 0

    # Final result fields
    final_predictions: str = ""
    revision_explanation: str = ""


# ── Flow ──────────────────────────────────────────────────────────────────────

class ControlEffectivenessFlow(Flow[ControlEffectivenessState]):

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _llm(self) -> LLM:
        return LLM(
            model="groq/openai/gpt-oss-20b",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2,
        )

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

    def _critic_feedback_block(self, feedback: str) -> str:
        if not feedback:
            return ""
        return (
            "\n\n--- CRITIC FEEDBACK (Revision Requested) ---\n"
            f"{feedback}\n"
            "--- END OF FEEDBACK ---\n\n"
            "Address every issue identified above in your revised output."
        )

    # ── Step 1: Analysis ─────────────────────────────────────────────────────

    def _run_analysis(self, feedback: str = "") -> None:
        analyzer = Agent(
            role="Control Risk Analyst",
            goal=(
                "Review control effectiveness using risk, KRIs, risk scenarios, "
                "and historical assessment data. Generate a detailed narrative "
                "highlighting which controls may fail, potential risks, and "
                "any anomalies or gaps."
            ),
            backstory=(
                "You are a meticulous risk analyst with years of experience working in GRC firms "
                "interpreting controls, identifying risks, and spotting anomalies. You analyze "
                "multiple controls holistically, drawing connections across all provided data. "
                "Your insights are clear, actionable, and detailed — as if presenting to a "
                "risk committee. You never produce JSON; your output is always a professional "
                "narrative with structured bullet points."
            ),
            llm=self._llm(),
            memory=True,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        prompt = (
            "You are tasked with performing a detailed analysis of each control "
            "based on the following inputs: Risk, KRIs, Risk Scenarios, and Assessment History.\n\n"
            "Follow these instructions carefully:\n\n"
            "1. For EACH control:\n"
            "   - Summarize the control in one or two sentences.\n"
            "   - Analyze the associated risk: what could go wrong if the control fails?\n"
            "   - Review KRIs: note trends, anomalies, or any threshold breaches.\n"
            "   - Examine risk scenarios: highlight which scenarios are most critical.\n"
            "   - Incorporate historical assessment data: mention patterns or past failures.\n\n"
            "2. Identify gaps across all controls:\n"
            "   - Are there controls that do not map to current risk scenarios?\n"
            "   - Are there inconsistencies between KRIs and the risk scenario?\n"
            "   - Are there controls at risk of cascading failure (one failure triggering another)?\n\n"
            "3. Provide narrative insights:\n"
            "   - Use clear, structured sentences. Do NOT produce JSON at any point.\n"
            "   - Highlight priority concerns with bullet points where appropriate.\n"
            "   - Include recommendations or observations where relevant.\n\n"
            "4. Maintain context across controls:\n"
            "   - Reference earlier control analyses if they are related.\n"
            "   - Ensure each control analysis is standalone but cohesive as a whole.\n\n"
            "5. Style:\n"
            "   - Professional and precise, as if presenting to a risk committee.\n"
            "   - Avoid generic statements; be specific to each control and its risk.\n\n"
            "Expected output sections per control:\n"
            "  [Control Name / ID] | [Control Summary] | [Risk Assessment] | "
            "[KRI Trends & Anomalies] | [Critical Scenario Observations] | "
            "[Historical Assessment Insights] | [Gaps or Inconsistencies] | "
            "[Key Recommendations / Observations]\n\n"
            "--- CONTROL DATA ---\n"
            f"{self.state.controls_data}\n"
            "--- END OF CONTROL DATA ---"
            f"{self._critic_feedback_block(feedback)}"
        )

        self.state.analysis_narrative = self._kickoff(analyzer, prompt)

    # ── Step 2: Prediction ────────────────────────────────────────────────────

    def _run_prediction(self, feedback: str = "") -> None:
        tools = []
        search_note = ""

        # Rule-based tool assignment — the LLM does NOT decide when to search
        if self.state.use_search and os.environ.get("SERPER_API_KEY"):
            try:
                from crewai_tools import SerperDevTool
                tools = [SerperDevTool()]
                search_note = (
                    "TOOL USAGE RULE: You have web search access via SerperDevTool. "
                    "Use it ONLY to research unfamiliar control types, verify emerging threat "
                    "intelligence, or supplement sparse historical data. Do NOT use it freely "
                    "or for controls that already have sufficient context in the provided data.\n\n"
                )
            except ImportError:
                pass

        predictor = Agent(
            role="Control Risk Predictor",
            goal=(
                "Based on control data, risks, KRIs, risk scenarios, and the analyzer's narrative, "
                "predict the status of each control and its potential impact on associated risks. "
                "Generate structured output in JSON format with precise mapping between controls, "
                "risks, KRIs, and scenarios."
            ),
            backstory=(
                "You are an analysis-driven risk prediction expert for GRC firms. Given a detailed "
                "narrative analysis, you synthesize historical control data, risk scenarios, KRIs, "
                "and the analyzer's findings to determine the likelihood of control failure. "
                "You provide outputs that are fully JSON-structured, traceable, and suitable for "
                "direct consumption by GRC dashboard code. You strictly adhere to the specified "
                "output schema and enum values, outputting ONLY valid JSON with no extra text."
            ),
            llm=self._llm(),
            tools=tools,
            memory=False,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        schema_example = json.dumps(
            [
                {
                    "control_id": "CTRL-001",
                    "control_name": "Segregation of Duties",
                    "control_status": "AT_RISK",
                    "control_status_confidence_score": "79%",
                    "control_status_reason": "Recurring KRI threshold breaches and sparse access review history indicate elevated risk of unauthorized access.",
                    "associated_risks": [
                        {
                            "risk_id": "RISK-101",
                            "risk_statement": "Unauthorized access to financial systems",
                            "kri": {
                                "kri_id": "KRI-1001",
                                "kri_statement": "Number of privilege escalation incidents",
                            },
                            "scenario": {
                                "scenario_id": "SCN-2001",
                                "scenario_statement": "Insider threat leads to fraud",
                            },
                        }
                    ],
                }
            ],
            indent=2,
        )

        prompt = (
            f"{search_note}"
            "You are tasked with producing structured JSON output for a set of controls "
            "based on the provided control information, associated risks, KRIs, risk scenarios, "
            "assessment history, and the analyzer's narrative.\n\n"
            "Instructions:\n\n"
            "1. For each control, determine `control_status` using ONLY these enum values:\n"
            "   - COMPLIANT  => control is operating effectively; no immediate risk.\n"
            "   - AT_RISK    => early warning signs or KRI trends indicate potential issues.\n"
            "   - FAILED     => control is ineffective or has already failed based on the analysis.\n\n"
            "2. Required fields for every control object:\n"
            "   - control_id\n"
            "   - control_name\n"
            "   - control_status              (must be COMPLIANT, AT_RISK, or FAILED)\n"
            "   - control_status_confidence_score  (e.g. '82%')\n"
            "   - control_status_reason       (concise explanation for the assigned status)\n"
            "   - associated_risks            (list; include all risks impacted by this control)\n"
            "     Each risk item must contain:\n"
            "       risk_id, risk_statement,\n"
            "       kri: { kri_id, kri_statement },\n"
            "       scenario: { scenario_id, scenario_statement }\n\n"
            "3. If a control has no associated risks, use `\"associated_risks\": []`.\n"
            "4. CRITICAL: Output ONLY valid JSON. No markdown, no prose, no preamble or suffix.\n"
            "5. A control with status FAILED must list ALL its associated risks.\n"
            "6. Derive status from the analyzer narrative — do not invent information not in the data.\n\n"
            f"JSON Schema to follow EXACTLY:\n{schema_example}\n\n"
            "--- CONTROL DATA ---\n"
            f"{self.state.controls_data}\n"
            "--- END OF CONTROL DATA ---\n\n"
            "--- ANALYZER NARRATIVE ---\n"
            f"{self.state.analysis_narrative}\n"
            "--- END OF ANALYZER NARRATIVE ---"
            f"{self._critic_feedback_block(feedback)}"
        )

        self.state.predictor_json_str = self._kickoff(predictor, prompt)

    # ── Step 3: Critic ────────────────────────────────────────────────────────

    def _run_critic(self) -> None:
        critic = Agent(
            role="Control Risk Critic",
            goal=(
                "Review the outputs of the Analyzer and Predictor agents to ensure accuracy, "
                "consistency, and completeness. Identify any errors, inconsistencies, or missed "
                "mappings, and propose revisions for affected controls. Apply revisions only once "
                "per workflow execution."
            ),
            backstory=(
                "You are a meticulous risk quality reviewer with over 10 years of expertise in "
                "control assessment and risk analysis at GRC firms. You excel at spotting "
                "inconsistencies, missing data, and logical errors in complex multi-step workflows. "
                "Your guidance ensures that final predictions are reliable and traceable. You output "
                "ONLY a single valid JSON object — no preamble, no markdown, no extra text."
            ),
            llm=self._llm(),
            memory=False,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        output_format = json.dumps(
            {
                "revised_output": ["...complete array of all controls following the Predictor schema..."],
                "revision_explanation": "Explanation of all changes made, or 'No revisions needed.' if none.",
                "needs_revision": True,
                "failed_step": "analysis OR prediction OR null",
            },
            indent=2,
        )

        prompt = (
            "Your task is to critically evaluate the outputs of the Analyzer and Predictor agents.\n\n"
            "Review checklist:\n"
            "  1. Does each control's `control_status` align with the Analyzer narrative?\n"
            "     - Controls described as high-risk in the narrative must NOT be COMPLIANT in the JSON.\n"
            "  2. Are all `associated_risks`, KRIs, and scenarios correctly and completely mapped?\n"
            "  3. Are there KRI trends or historical patterns noted by the Analyzer that the "
            "Predictor ignored or downgraded?\n"
            "  4. Do `control_status_reason` fields accurately reflect the narrative evidence?\n"
            "  5. Is the JSON structurally valid and schema-compliant?\n\n"
            "Determine the failed step:\n"
            "  - Set `failed_step` to 'analysis' if the Analyzer narrative is itself inaccurate, "
            "incomplete, or misleading.\n"
            "  - Set `failed_step` to 'prediction' if the narrative is sound but the Predictor "
            "JSON is incorrect or inconsistent with it.\n"
            "  - Set `failed_step` to null if no revision is needed.\n\n"
            "Revision constraints:\n"
            "  - Propose at most ONE revision. Do not revise more than once per run.\n"
            "  - Do NOT introduce speculative risks not present in the source data.\n"
            "  - Corrections must be traceable to specific evidence in the Analyzer narrative.\n\n"
            "CRITICAL OUTPUT REQUIREMENT:\n"
            "Respond with ONLY a single valid JSON object matching this exact structure "
            "(no markdown, no extra text before or after):\n"
            f"{output_format}\n\n"
            "Rules for the response object:\n"
            "  - `revised_output` MUST contain the COMPLETE array of all controls "
            "(with corrections applied where needed, or unchanged where correct).\n"
            "  - `needs_revision` = true ONLY if you made meaningful changes.\n"
            "  - `needs_revision` = false if outputs are accurate and consistent.\n\n"
            "--- CONTROL DATA ---\n"
            f"{self.state.controls_data}\n"
            "--- END OF CONTROL DATA ---\n\n"
            "--- ANALYZER NARRATIVE ---\n"
            f"{self.state.analysis_narrative}\n"
            "--- END OF ANALYZER NARRATIVE ---\n\n"
            "--- PREDICTOR JSON OUTPUT ---\n"
            f"{self.state.predictor_json_str}\n"
            "--- END OF PREDICTOR JSON OUTPUT ---"
        )

        raw = self._kickoff(critic, prompt)
        self.state.critic_output = raw

        try:
            parsed = _extract_json(raw)
            self.state.needs_revision = bool(parsed.get("needs_revision", False))
            self.state.failed_step = parsed.get("failed_step") or ""
            self.state.revision_explanation = parsed.get("revision_explanation", "")

            revised = parsed.get("revised_output")
            if revised:
                self.state.final_predictions = json.dumps(revised, indent=2)
            else:
                self.state.final_predictions = self.state.predictor_json_str
        except Exception:
            # Critic output unparseable — treat as approved and fall back to predictor output
            self.state.needs_revision = False
            self.state.final_predictions = self.state.predictor_json_str

    # ── Flow steps ────────────────────────────────────────────────────────────

    @start()
    def analyze_controls(self):
        self._run_analysis()

    @listen(analyze_controls)
    def predict_status(self):
        self._run_prediction()

    @listen(predict_status)
    def critic_review(self):
        self._run_critic()

    @router(critic_review)
    def route_after_critic(self) -> str:
        if self.state.needs_revision and self.state.revision_count < 1:
            return "needs_revision"
        return "done"

    @listen("needs_revision")
    def revise_workflow(self):
        """Re-run from the failing step forward. Executes at most once per workflow run."""
        self.state.revision_count += 1

        if self.state.failed_step == "analysis":
            self._run_analysis(feedback=self.state.critic_output)
            self._run_prediction(feedback=self.state.critic_output)
        else:
            # Prediction step failed (or failed_step is unset — default to prediction)
            self._run_prediction(feedback=self.state.critic_output)

        # Use the freshly generated predictor output as the authoritative final result
        self.state.final_predictions = self.state.predictor_json_str

    @listen("done")
    def finalize(self):
        if not self.state.final_predictions:
            self.state.final_predictions = self.state.predictor_json_str


# ── Async entry point (called from FastAPI route handler) ─────────────────────

async def run_control_effectiveness(request: ControlEffectivenessRequest) -> dict:
    flow = ControlEffectivenessFlow()
    loop = asyncio.get_running_loop()

    use_search = _should_use_search(request)
    controls_data = _format_controls_data(request)

    await loop.run_in_executor(
        None,
        lambda: flow.kickoff(inputs={
            "controls_data": controls_data,
            "framework": request.framework,
            "use_search": use_search,
        }),
    )

    s = flow.state

    try:
        predictions = _extract_json(s.final_predictions)
    except Exception:
        predictions = []

    result: dict = {
        "analysis_narrative": s.analysis_narrative,
        "predictions": predictions,
        "revised": s.revision_count > 0,
        "search_used": use_search,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if s.revision_explanation:
        result["revision_explanation"] = s.revision_explanation

    return result
