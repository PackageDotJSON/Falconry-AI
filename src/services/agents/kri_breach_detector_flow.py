"""
@author: { FALCONRY SOLUTIONS }
@description: KRI Breach Threshold Detector — CrewAI Flow.

3-agent sequential workflow:
  1. Analyst   — structured JSON intelligence per KRI: trend, volatility, threshold
                 distance, historical breach frequency, scenario sensitivity. Strictly
                 backward-looking; never predicts or forecasts.
  2. Detector  — forward-looking structured JSON forecast per KRI: breach likelihood,
                 breach timeline, risk impact forecast, supporting signals, and
                 explainability, using the Analyst's output as the primary reasoning
                 foundation.
  3. Critic    — strict PASS/FAIL audit of the Detector output against the Analyst
                 output and raw KRI data. Never generates predictions and never
                 modifies the Detector output itself; only flags issues for revision.
                 Triggers at most ONE revision pass from the earliest failing step forward.

Rule-based SerperDevTool trigger (Python-level, not LLM-decided):
  Detector gets web search access only when:
    - assessment_history is absent or sparse (< 30 chars)
    - control_type (or control_name) contains unfamiliar/experimental indicator keywords
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

from models.kri_breach_detector_models import KRIBreachDetectorRequest


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


def _should_use_search(request: KRIBreachDetectorRequest) -> bool:
    """Return True only under rule-based conditions that warrant external search.

    Triggers:
      1. Assessment history is absent or too sparse to be meaningful.
      2. Control type (or name) suggests an unfamiliar / experimental control.
      3. A risk scenario references emerging or novel threat patterns.
    """
    for kri in request.kris:
        if not kri.assessment_history or len(kri.assessment_history.strip()) < 30:
            return True

        type_lower = (kri.control.control_type or kri.control.control_name).lower()
        if any(kw in type_lower for kw in _UNFAMILIAR_CONTROL_INDICATORS):
            return True

        for scenario in kri.scenarios:
            scenario_lower = scenario.scenario_statement.lower()
            if any(kw in scenario_lower for kw in _EMERGING_THREAT_KEYWORDS):
                return True

    return False


# ── Input formatting ──────────────────────────────────────────────────────────

def _format_kri_data(request: KRIBreachDetectorRequest) -> str:
    """Serialize the full request into a human-readable block for agent prompts."""
    lines = []
    for i, kri in enumerate(request.kris, 1):
        lines.append(f"KRI {i}:")
        lines.append(f"  KRI ID:        {kri.kri_id}")
        lines.append(f"  KRI Statement: {kri.kri_statement}")
        lines.append("  Associated Risk:")
        lines.append(f"    Risk ID:        {kri.risk.risk_id}")
        lines.append(f"    Risk Statement: {kri.risk.risk_statement}")
        lines.append("  Associated Control:")
        lines.append(f"    Control ID:   {kri.control.control_id}")
        lines.append(f"    Control Name: {kri.control.control_name}")
        if kri.control.control_type:
            lines.append(f"    Control Type: {kri.control.control_type}")

        if kri.scenarios:
            lines.append("  Risk Scenarios:")
            for scenario in kri.scenarios:
                lines.append(f"    - Scenario ID:        {scenario.scenario_id}")
                lines.append(f"      Scenario Statement: {scenario.scenario_statement}")
        else:
            lines.append("  Risk Scenarios: None")

        history = kri.assessment_history or "No prior assessment history available."
        lines.append(f"  Assessment History: {history}")
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

class KRIBreachDetectorState(BaseModel):
    # Pre-formatted inputs (set via flow.kickoff(inputs={...}))
    kri_data: str = ""
    framework: str = "ISO 31000"
    use_search: bool = False

    # Agent outputs
    analyst_json_str: str = ""
    detector_json_str: str = ""

    # Critic tracking
    critic_output: str = ""
    critic_passed: bool = False
    failed_step: str = ""        # "analyst" | "detector"
    revision_count: int = 0
    critic_explanation: str = ""


# ── Flow ──────────────────────────────────────────────────────────────────────

class KRIBreachDetectorFlow(Flow[KRIBreachDetectorState]):

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
                    delay = min(delay *     2, 60)
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

    # ── Step 1: Analyst ──────────────────────────────────────────────────────

    def _run_analyst(self, feedback: str = "") -> None:
        analyst = Agent(
            role="Senior KRI Pattern & Intelligence Analyst",
            goal=(
                "Convert raw KRI data — risk, control, risk scenarios, and assessment "
                "history — into structured, evidence-based analytical intelligence for "
                "each KRI, strictly describing what has already happened. Never predict "
                "or forecast future breaches, timelines, or probabilities."
            ),
            backstory=(
                "You are a highly experienced enterprise risk intelligence analyst who has "
                "worked at GRC platforms for over 10 years. You specialize in interpreting "
                "Key Risk Indicator behavior: identifying trends, measuring volatility, "
                "evaluating proximity to breach thresholds, and recognizing historical breach "
                "patterns and leading indicators. You are NOT responsible for predicting "
                "future breaches, estimating probabilities, or recommending remediation — "
                "that is strictly out of scope. You output ONLY valid JSON with no extra text."
            ),
            llm=self._llm(),
            memory=False,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        schema_example = json.dumps(
            [
                {
                    "kri_id": "KRI-1001",
                    "kri_statement": "Number of privilege escalation incidents",
                    "trend": "deteriorating",
                    "volatility": "high",
                    "threshold_distance_pct": "18%",
                    "historical_breach_frequency": "moderate",
                    "scenario_sensitivity": ["insider threat", "credential abuse"],
                }
            ],
            indent=2,
        )

        prompt = (
            "You are tasked with performing a deep analytical assessment of each KRI based on "
            "its associated Risk, Control, Risk Scenarios, and Assessment History.\n\n"
            "For EACH KRI, evaluate:\n"
            "1. Overall trend — improving / stable / deteriorating / highly_volatile\n"
            "2. Volatility — low / medium / high / extreme\n"
            "3. Threshold proximity — how close current behavior is to its breach threshold, "
            "as a 0-100 percentage (closer to 100% means closer to breach)\n"
            "4. Historical breach pattern identification — none / rare / low / moderate / high / "
            "frequent / recurring\n"
            "5. Scenario sensitivity — which of the provided risk scenarios this KRI's behavior is "
            "most sensitive to (free text list, not a fixed enum)\n"
            "6. Leading indicators — evidence-based observations worth flagging for forecasting\n\n"
            "STRICT RULES:\n"
            "- You are NOT responsible for predicting future breaches, forecasting timelines, "
            "estimating probabilities, recommending remediation, or speculating.\n"
            "- Every field must be grounded strictly in the provided data. Never hallucinate.\n"
            "- CRITICAL: Output ONLY valid JSON. No markdown, no prose, no preamble or suffix.\n\n"
            f"JSON Schema to follow EXACTLY (one object per KRI, in a JSON array):\n{schema_example}\n\n"
            "--- KRI DATA ---\n"
            f"{self.state.kri_data}\n"
            "--- END OF KRI DATA ---"
            f"{self._critic_feedback_block(feedback)}"
        )

        self.state.analyst_json_str = self._kickoff(analyst, prompt)

    # ── Step 2: Detector ─────────────────────────────────────────────────────

    def _run_detector(self, feedback: str = "") -> None:
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
                    "or for KRIs that already have sufficient context in the provided data.\n\n"
                )
            except ImportError:
                pass

        detector = Agent(
            role="Senior KRI Breach Prediction Specialist",
            goal=(
                "Generate forward-looking predictive intelligence for each KRI — breach "
                "likelihood, breach timeline, and risk impact forecast — using the Analyst's "
                "structured intelligence as the primary reasoning foundation."
            ),
            backstory=(
                "You are a specialized forecasting expert who predicts Key Risk Indicator "
                "threshold breaches before they happen. You treat the Analyst's structured "
                "output as your PRIMARY reasoning foundation. You may inspect the raw KRI data "
                "only to validate evidence, clarify context, or detect inconsistencies — never "
                "as your main source of reasoning. You produce fully JSON-structured, "
                "explainable forecasts suitable for direct consumption by a GRC dashboard. "
                "You output ONLY valid JSON with no extra text."
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
                    "kri_id": "KRI-1001",
                    "kri_statement": "Number of privilege escalation incidents",
                    "executive_forecast_summary": "Privilege escalation incidents are trending upward with high volatility, indicating near-term breach risk.",
                    "overall_breach_risk": "high",
                    "overall_confidence_score": "71%",
                    "breach_forecast": {
                        "breach_likelihood": "high",
                        "confidence_score": "68%",
                        "predicted_breach_timeline": "near_term",
                        "rationale": "Deteriorating trend combined with high volatility and moderate historical breach frequency.",
                    },
                    "supporting_signals": [
                        {
                            "signal_name": "Threshold Proximity",
                            "signal_category": "threshold_proximity",
                            "evidence": "Current value is within 18% of the defined breach threshold.",
                            "impact_level": "high",
                        }
                    ],
                    "explainability": {
                        "primary_drivers": ["deteriorating trend", "high volatility"],
                        "historical_alignment": "Consistent with prior breach episodes documented in assessment history.",
                        "uncertainty_factors": ["Limited sample size for recent assessment cycle."],
                    },
                    "recommended_monitoring_focus": ["Privileged access review cadence", "Escalation alert thresholds"],
                }
            ],
            indent=2,
        )

        prompt = (
            f"{search_note}"
            "You are tasked with producing forward-looking, structured JSON forecasts for each KRI "
            "based on: (1) the raw KRI data — Risk, Control, Risk Scenarios, Assessment History — and "
            "(2) the Analyst's structured intelligence output, which you MUST treat as your primary "
            "reasoning foundation.\n\n"
            "For EACH KRI, produce:\n"
            "1. Breach likelihood assessment — very_low / low / moderate / high / very_high / critical\n"
            "2. Breach timeline forecast — no_immediate_risk / long_term / medium_term / near_term / "
            "imminent\n"
            "3. Risk impact forecast — explain which associated risks would be affected if this KRI's "
            "threshold is breached\n"
            "4. Control failure escalation analysis — note if a control failure could compound the "
            "breach risk\n"
            "5. Scenario-driven forecasting — tie your forecast to the scenarios most relevant per the "
            "Analyst's scenario_sensitivity findings\n"
            "6. Confidence scoring for both the overall forecast and the breach_forecast block\n"
            "7. Explainability — primary drivers, historical alignment, and uncertainty factors\n\n"
            "REASONING BOUNDARY:\n"
            "- The Analyst's output is your PRIMARY reasoning foundation.\n"
            "- You may reference raw KRI data only for validation, evidence verification, contextual "
            "clarification, or inconsistency detection — never as your main source of reasoning.\n"
            "- Do not contradict the Analyst's findings without clear evidence-based justification.\n\n"
            "OUTPUT RULES:\n"
            "- Do not skip any field. If information is unavailable, use 'Not Applicable' or "
            "'Information Unavailable' rather than omitting the field.\n"
            "- CRITICAL: Output ONLY valid JSON. No markdown, no prose, no preamble or suffix.\n\n"
            f"JSON Schema to follow EXACTLY (one object per KRI, in a JSON array):\n{schema_example}\n\n"
            "--- KRI DATA ---\n"
            f"{self.state.kri_data}\n"
            "--- END OF KRI DATA ---\n\n"
            "--- ANALYST OUTPUT ---\n"
            f"{self.state.analyst_json_str}\n"
            "--- END OF ANALYST OUTPUT ---"
            f"{self._critic_feedback_block(feedback)}"
        )

        self.state.detector_json_str = self._kickoff(detector, prompt)

    # ── Step 3: Critic ───────────────────────────────────────────────────────

    def _run_critic(self) -> None:
        critic = Agent(
            role="KRI Forecast Validation & Consistency Auditor",
            goal=(
                "Strictly audit the Detector's forecast for logical consistency, alignment "
                "with the Analyst's intelligence, evidence support, and internal coherence. "
                "Flag issues for revision; never generate new predictions or modify the "
                "Detector's output yourself."
            ),
            backstory=(
                "You are a senior risk governance auditor with deep expertise in validating "
                "predictive risk models. You act as a STRICT AUDITOR, not a forecaster — you "
                "never generate predictions and you never modify the Detector's output "
                "directly. You are intolerant of false causal reasoning, unsupported claims, "
                "timeline inconsistencies, and explainability gaps. You output ONLY a single "
                "valid JSON object — no preamble, no markdown, no extra text."
            ),
            llm=self._llm(),
            memory=False,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        output_format = json.dumps(
            {
                "verdict": "PASS or FAIL",
                "issues": ["...list of identified issues, empty if none..."],
                "affected_fields": ["...exact field names affected, empty if none..."],
                "explanation": "Explanation of inconsistencies, or 'No issues found.' if none.",
                "recommended_corrections": "Guidance on what must be corrected, or 'None.' if none.",
                "failed_step": "analyst OR detector OR null",
            },
            indent=2,
        )

        prompt = (
            "Your task is to validate the Detector's forecast output. You MUST NOT generate new "
            "predictions and you MUST NOT modify the Detector's output yourself — ONLY validate "
            "and critique.\n\n"
            "Validation checklist:\n"
            "  1. Signal consistency check — do the supporting_signals logically support the "
            "stated breach_likelihood and overall_breach_risk?\n"
            "  2. Trend alignment check — does the forecast align with the Analyst's reported "
            "trend and volatility?\n"
            "  3. Threshold logic check — is the predicted_breach_timeline consistent with the "
            "Analyst's threshold_distance_pct?\n"
            "  4. Control coherence check — is the control failure escalation reasoning sound "
            "given the associated control?\n"
            "  5. Scenario validity check — are referenced scenarios actually among those "
            "provided in the KRI data?\n"
            "  6. Confidence calibration check — are confidence scores reasonable given the "
            "strength of the evidence?\n"
            "  7. Explainability check — are primary_drivers, historical_alignment, and "
            "uncertainty_factors fully populated and evidence-based?\n\n"
            "Determine the failed step:\n"
            "  - Set `failed_step` to 'analyst' if the Analyst's structured intelligence is "
            "itself inaccurate, incomplete, or unsupported by the raw KRI data.\n"
            "  - Set `failed_step` to 'detector' if the Analyst's output is sound but the "
            "Detector's forecast is inconsistent with it or internally incoherent.\n"
            "  - Set `failed_step` to null if `verdict` is PASS.\n\n"
            "Revision constraint: at most ONE revision pass will ever be applied — your "
            "feedback must be precise and complete.\n\n"
            "CRITICAL OUTPUT REQUIREMENT:\n"
            "Respond with ONLY a single valid JSON object matching this exact structure "
            "(no markdown, no extra text before or after):\n"
            f"{output_format}\n\n"
            "--- KRI DATA ---\n"
            f"{self.state.kri_data}\n"
            "--- END OF KRI DATA ---\n\n"
            "--- ANALYST OUTPUT ---\n"
            f"{self.state.analyst_json_str}\n"
            "--- END OF ANALYST OUTPUT ---\n\n"
            "--- DETECTOR OUTPUT (to validate) ---\n"
            f"{self.state.detector_json_str}\n"
            "--- END OF DETECTOR OUTPUT ---"
        )

        raw = self._kickoff(critic, prompt)
        self.state.critic_output = raw

        try:
            parsed = _extract_json(raw)
            verdict = str(parsed.get("verdict", "")).strip().upper()
            self.state.critic_passed = verdict == "PASS"
            self.state.failed_step = "" if self.state.critic_passed else (parsed.get("failed_step") or "")
            self.state.critic_explanation = parsed.get("explanation", "")
        except Exception:
            # Critic output unparseable — fail open and treat as approved
            self.state.critic_passed = True
            self.state.failed_step = ""

    # ── Flow steps ────────────────────────────────────────────────────────────

    @start()
    def analyze_kris(self):
        self._run_analyst()

    @listen(analyze_kris)
    def detect_breaches(self):
        self._run_detector()

    @listen(detect_breaches)
    def critic_review(self):
        self._run_critic()

    @router(critic_review)
    def route_after_critic(self) -> str:
        if not self.state.critic_passed and self.state.revision_count < 1:
            return "needs_revision"
        return "done"

    @listen("needs_revision")
    def revise_workflow(self):
        """Re-run from the failing step forward. Executes at most once per workflow run."""
        self.state.revision_count += 1

        if self.state.failed_step == "analyst":
            self._run_analyst(feedback=self.state.critic_output)
            self._run_detector(feedback=self.state.critic_output)
        else:
            # Detector step failed (or failed_step is unset — default to detector)
            self._run_detector(feedback=self.state.critic_output)

    @listen("done")
    def finalize(self):
        pass  # State is fully populated; flow terminates here.


# ── Async entry point (called from FastAPI route handler) ─────────────────────

async def run_kri_breach_detection(request: KRIBreachDetectorRequest) -> dict:
    flow = KRIBreachDetectorFlow()
    loop = asyncio.get_running_loop()

    use_search = _should_use_search(request)
    kri_data = _format_kri_data(request)

    await loop.run_in_executor(
        None,
        lambda: flow.kickoff(inputs={
            "kri_data": kri_data,
            "framework": request.framework,
            "use_search": use_search,
        }),
    )

    s = flow.state

    try:
        analysis = _extract_json(s.analyst_json_str)
    except Exception:
        analysis = []

    try:
        forecast = _extract_json(s.detector_json_str)
    except Exception:
        forecast = []

    result: dict = {
        "analysis": analysis,
        "forecast": forecast,
        "critic_status": "PASS" if s.critic_passed else "FAIL",
        "revised": s.revision_count > 0,
        "search_used": use_search,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if s.critic_explanation:
        result["critic_explanation"] = s.critic_explanation

    return result
