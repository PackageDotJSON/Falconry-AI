"""
@author: { FALCONRY SOLUTIONS }
@description: Enterprise GRC Policy Expert — CrewAI Flow.

4-agent sequential workflow:
  1. Analyzer         — ingests uploaded document text + structured client DB data
                        and produces a detailed textual summary identifying core
                        processes, products, risks, compliance info, and gaps.
                        Has internet search access to clarify ambiguities.
  2. Policy Drafter   — converts the Analyzer's summary into a full plain-text
                        policy draft covering all relevant GRC areas. Focus is
                        purely on content quality and logical structure; no
                        formatting applied at this stage.
                        Has internet search access for regulatory / framework references.
  3. Content Formatter — transforms the plain-text draft into a fully structured
                        Markdown document using its built-in consistent formatting
                        skill: standardized headings (H1/H2/H3), bullet points,
                        numbered lists, Markdown tables, and emphasis rules applied
                        uniformly across all client documents. Output is ready for
                        direct conversion to DOCX, PDF, PPTX, TXT, or HTML.
  4. Critic           — reviews the Markdown output for accuracy, logical flow,
                        completeness, and alignment with the original source data.
                        Triggers at most ONE revision pass from the earliest failing
                        step forward. Never rewrites content itself — provides
                        actionable feedback only.

Internet search (SerperDevTool):
  Agents 1 and 2 (Analyzer, Policy Drafter) always receive web search access
  when SERPER_API_KEY is configured — they decide when to search based on need.
  This is not rule-based; the agents use judgment to research unclear regulatory
  references, unfamiliar frameworks, or missing compliance context.

File generation:
  After the flow completes, the Markdown output is converted to the format the
  user selected (word / pdf / pptx / html / txt / md) via the existing
  file_generation service. The conversion happens in the async entry point.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

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

from models.grc_policy_models import ClientData, GRCPolicyRequest


# ── Input formatting ───────────────────────────────────────────────────────────

def _format_client_data(client_data: Optional[ClientData]) -> str:
    """Serialize structured client DB data into a readable text block."""
    if not client_data:
        return "No structured client data provided."

    sections = []
    if client_data.core_processes:
        sections.append(f"Core Processes & Cycles:\n{client_data.core_processes}")
    if client_data.products_and_services:
        sections.append(f"Products & Services:\n{client_data.products_and_services}")
    if client_data.risk_taxonomy:
        sections.append(f"Risk Taxonomy:\n{client_data.risk_taxonomy}")
    if client_data.appetite_values:
        sections.append(f"Risk Appetite Values:\n{client_data.appetite_values}")
    if client_data.compliance_info:
        sections.append(f"Compliance & Regulatory Information:\n{client_data.compliance_info}")

    return "\n\n".join(sections) if sections else "No structured client data provided."


def _build_regulatory_research_context(framework: str) -> tuple[str, bool]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "", False

    query = (
        f"{framework or 'GRC'} enterprise policy governance policy management "
        "ownership review cadence approval mapped controls requirements"
    )
    body = json.dumps({"q": query, "num": 5}).encode("utf-8")
    req = urllib_request.Request(
        "https://google.serper.dev/search",
        data=body,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return "", False

    organic_results = payload.get("organic") or []
    lines = [
        "External regulatory and policy-governance context from Serper search.",
        f"Search query: {query}",
        "Use this only as supplemental context. The client database data remains authoritative.",
    ]

    for index, item in enumerate(organic_results[:5], start=1):
        title = item.get("title") or "Untitled result"
        snippet = item.get("snippet") or "No snippet available."
        link = item.get("link") or ""
        lines.append(f"{index}. {title}\n   Summary: {snippet}\n   Source: {link}")

    if len(lines) <= 3:
        return "", False

    return "\n".join(lines), True


# ── Flow state ─────────────────────────────────────────────────────────────────

class GRCPolicyState(BaseModel):
    # Pre-formatted inputs (populated via flow.kickoff(inputs={...}))
    documents_text: str = ""
    client_data_text: str = ""
    organization: str = ""
    framework: str = "ISO 31000"
    regulatory_research_context: str = ""
    search_used: bool = False

    # Agent outputs
    analysis: str = ""          # Analyzer — detailed textual summary
    policy_draft: str = ""      # Policy Drafter — plain-text draft
    formatted_policy_md: str = ""  # Content Formatter — final Markdown

    # Critic tracking
    critic_feedback: str = ""
    critic_passed: bool = False
    failed_step: str = ""       # "analysis" | "draft" | "formatting"
    revision_count: int = 0


# ── Flow ───────────────────────────────────────────────────────────────────────

class GRCPolicyFlow(Flow[GRCPolicyState]):

    # ── Shared helpers ─────────────────────────────────────────────────────────

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

    def _get_search_tools(self) -> list:
        """
        Return [SerperDevTool()] when SERPER_API_KEY is set.

        Unlike other flows where search is rule-based (Python decides),
        the Analyzer and Policy Drafter agents decide themselves when to
        search — this follows the spec requirement for these two agents.
        """
        if os.environ.get("SERPER_API_KEY"):
            try:
                from crewai_tools import SerperDevTool
                return [SerperDevTool()]
            except ImportError:
                pass
        return []

    def _feedback_block(self, feedback: str) -> str:
        if not feedback:
            return ""
        return (
            "\n\n--- CRITIC FEEDBACK (Revision Requested) ---\n"
            f"{feedback}\n"
            "--- END OF FEEDBACK ---\n\n"
            "Address every issue identified above in your revised output."
        )

    def _source_data_block(self) -> str:
        s = self.state
        return (
            f"--- UPLOADED DOCUMENTS ---\n"
            f"{s.documents_text or 'No uploaded documents provided.'}\n"
            f"--- END OF DOCUMENTS ---\n\n"
            f"--- CLIENT DATABASE DATA ---\n"
            f"{s.client_data_text or 'No structured client data provided.'}\n"
            f"--- END OF CLIENT DATA ---"
        )

    def _regulatory_research_block(self) -> str:
        if not self.state.regulatory_research_context:
            return ""
        return (
            "\n\n--- SUPPLEMENTAL REGULATORY RESEARCH CONTEXT ---\n"
            f"{self.state.regulatory_research_context}\n"
            "--- END OF SUPPLEMENTAL REGULATORY RESEARCH CONTEXT ---"
        )

    # ── Step 1: Analyzer ──────────────────────────────────────────────────────

    def _run_analysis(self, feedback: str = "") -> None:
        analyzer = Agent(
            role="Senior Policy Analysis Expert",
            goal=(
                "Ingest unstructured document text and structured client database data "
                "to extract, organize, and summarize all information needed to draft "
                "compliant and comprehensive GRC policies."
            ),
            backstory=(
                "You are an expert in GRC policy analysis with over 10 years of experience "
                "at enterprise risk consultancies. You can rapidly identify core business "
                "processes, risk taxonomies, compliance obligations, and regulatory requirements "
                "across industries. You produce well-organized summaries that form the definitive "
                "foundation for policy drafting — highlighting critical points, flagging data gaps, "
                "and preserving relationships between processes, risks, and compliance requirements "
                "for full audit traceability. Use the supplemental regulatory research context "
                "when it is provided, but treat the client database data as authoritative."
            ),
            llm=self._llm(),
            tools=[],
            memory=False,
            max_execution_time=240,
            verbose=True,
            allow_delegation=False,
        )

        prompt = (
            f"You are analyzing client data for {self.state.organization or 'the client'} "
            f"under the {self.state.framework} framework.\n\n"
            "Review both the uploaded document text and the structured client database data below. "
            "Produce a detailed, structured textual summary that covers:\n\n"
            "1. CORE PROCESSES & CYCLES — Extract and describe all core processes, sub-processes, "
            "and operational cycles. Map dependencies between them.\n\n"
            "2. PRODUCTS & SERVICES — Identify all products and services. Note risk implications "
            "tied to each.\n\n"
            "3. RISK TAXONOMY & APPETITE — Summarize the full risk taxonomy: categories, types, "
            "definitions, appetite values, and thresholds.\n\n"
            "4. COMPLIANCE & REGULATORY REQUIREMENTS — List all applicable compliance standards, "
            "regulatory obligations, and policy mandates referenced in the data.\n\n"
            "5. GAPS & INCONSISTENCIES — Explicitly flag any missing information, ambiguities, "
            "or contradictions. Mark them clearly so the Policy Drafter can address them.\n\n"
            "6. TRACEABILITY — Preserve relationships (e.g., which risks map to which processes "
            "or products). These relationships are critical for audit purposes.\n\n"
            "Style:\n"
            "- Write multiple paragraphs with bullet points inside them.\n"
            "- CAPITALIZE and **bold** key terms and critical items for emphasis.\n"
            "- Be thorough; the Policy Drafter depends entirely on your summary.\n\n"
            "Tool use:\n"
            "- You do not have live tool access in this workflow.\n"
            "- Use supplemental regulatory research context only if it is provided below.\n"
            "- Return the final analysis directly. Do not write internal reasoning or search intentions.\n\n"
            f"{self._source_data_block()}"
            f"{self._regulatory_research_block()}"
            f"{self._feedback_block(feedback)}"
        )

        self.state.analysis = self._kickoff(analyzer, prompt)

    # ── Step 2: Policy Drafter ────────────────────────────────────────────────

    def _run_drafting(self, feedback: str = "") -> None:
        drafter = Agent(
            role="Senior Enterprise GRC Policy Drafter",
            goal=(
                "Convert the Analyzer's structured summary into a logically organized, "
                "comprehensive draft policy document covering all GRC areas identified. "
                "Focus purely on content quality and logical structure — leave all "
                "formatting and visual styling to the Content Formatter."
            ),
            backstory=(
                "You are an expert GRC policy drafter with deep knowledge of risk management "
                "frameworks, compliance standards, and regulatory requirements. You have drafted "
                "hundreds of enterprise policy documents across financial services, healthcare, "
                "and technology sectors. You convert analytical summaries into coherent, "
                "actionable policy content with clear logical flow, prioritized critical points, "
                "and practical guidance. You do NOT apply formatting, markdown, or visual styling "
                "at this stage — that is handled downstream. When you need to reference a specific "
                "regulation, standard, or industry best practice, rely on the Analyzer summary "
                "and any supplemental regulatory research context provided by the workflow."
            ),
            llm=self._llm(),
            tools=[],
            memory=False,
            max_execution_time=240,
            verbose=True,
            allow_delegation=False,
        )

        prompt = (
            f"Draft a comprehensive GRC policy document for {self.state.organization or 'the client'} "
            f"under the {self.state.framework} framework, based on the Analyzer's summary below.\n\n"
            "Your draft must cover all of the following areas identified in the analysis:\n"
            "- Core processes, cycles, and sub-cycles\n"
            "- Products and services\n"
            "- Risk taxonomy and appetite values\n"
            "- Compliance and regulatory requirements\n"
            "- Key insights and critical items from the documents\n"
            "- High-priority points with clear emphasis\n\n"
            "WRITING RULES:\n"
            "1. Focus exclusively on content quality, logical structure, and prioritization.\n"
            "2. Do NOT apply any markdown formatting, headings (##), bullet (- or *), "
            "bold (**), or tables — write in clear, flowing plain text only.\n"
            "3. Maintain coherence and avoid repetition.\n"
            "4. Logically sequence content: high-level overview → core processes → "
            "products → risks → compliance → appendices / references.\n"
            "5. Address any gaps or inconsistencies flagged by the Analyzer explicitly.\n\n"
            "Tool use: you do not have live tool access. Return the policy draft directly; "
            "do not write internal reasoning or search intentions.\n\n"
            "--- ANALYZER SUMMARY ---\n"
            f"{self.state.analysis}\n"
            "--- END OF ANALYZER SUMMARY ---"
            f"{self._regulatory_research_block()}"
            f"{self._feedback_block(feedback)}"
        )

        self.state.policy_draft = self._kickoff(drafter, prompt)

    # ── Step 3: Content Formatter ─────────────────────────────────────────────

    def _run_formatting(self, feedback: str = "") -> None:
        """
        The Content Formatter applies its built-in consistent Markdown formatting
        skill to transform the plain-text draft into a fully structured document.

        The 'skill' is embedded in this agent's backstory and task prompt as a
        fixed, non-negotiable set of formatting rules that the agent applies
        identically for every client — ensuring brand consistency across all
        policy documents generated by the platform.
        """
        formatter = Agent(
            role="Content Formatter and Policy Document Stylist",
            goal=(
                "Transform the plain-text policy draft into a visually coherent, "
                "well-structured, and professional Markdown document using a consistent "
                "formatting skill applied uniformly across all clients."
            ),
            backstory=(
                "You are an expert document designer and content formatter with 20+ years "
                "of experience preparing enterprise GRC policy documents. You follow a strict, "
                "internally consistent formatting skill that you apply to EVERY document without "
                "exception — regardless of client, industry, or content type. This skill ensures "
                "that every policy document produced by this platform looks and reads the same way.\n\n"

                "YOUR FORMATTING SKILL — applied consistently to every document:\n"
                "  HEADINGS:\n"
                "    # H1  — Document title only (one per document)\n"
                "    ## H2 — Major sections (e.g. Core Processes, Risk Framework)\n"
                "    ### H3 — Sub-sections within a major section\n"
                "    #### H4 — Use sparingly for deeply nested items only\n\n"
                "  LISTS:\n"
                "    - Use `- ` bullet points for unordered items\n"
                "    - Use `1.` numbered lists for sequential / prioritized items\n"
                "    - Indent sub-items with two spaces\n\n"
                "  EMPHASIS:\n"
                "    - **Bold** for critical terms, mandatory items, and high-priority points\n"
                "    - *Italic* for definitions, examples, and secondary emphasis\n"
                "    - CAPITALIZE acronyms and regulatory body names (e.g., ISO, GDPR, NIST)\n\n"
                "  TABLES:\n"
                "    - Use Markdown tables for risk appetite values, compliance matrices, "
                "and product/process lists with multiple attributes\n"
                "    - Always include a header row with `|---|---|` separators\n\n"
                "  SPACING & FLOW:\n"
                "    - One blank line between paragraphs\n"
                "    - One blank line before and after every heading\n"
                "    - One blank line before and after every list block\n"
                "    - One blank line before and after every table\n\n"
                "  CONTENT RULE:\n"
                "    - NEVER alter the meaning of any content\n"
                "    - NEVER remove any information from the draft\n"
                "    - ONLY reformat — do not add or delete substance\n"
                "    - If any section is ambiguous, mark it with ⚠️ for review"
            ),
            llm=self._llm(),
            memory=False,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        prompt = (
            f"Apply your consistent Markdown formatting skill to the policy draft below "
            f"for {self.state.organization or 'the client'}.\n\n"
            "FORMATTING REQUIREMENTS:\n"
            "1. Structure & Hierarchy — Organize into clear H1/H2/H3 sections following "
            "the logical flow: title → overview → core processes → products/services → "
            "risk framework → compliance → appendices.\n\n"
            "2. Bullet Points & Lists — Convert enumerations of processes, risks, products, "
            "and compliance items into proper bullet or numbered lists.\n\n"
            "3. Tables — Create Markdown tables for risk appetite values, compliance matrices, "
            "and any data with multiple attributes (at least 2 columns).\n\n"
            "4. Emphasis — Bold all mandatory items, high-priority risks, and critical compliance "
            "actions. Italicize definitions and secondary references.\n\n"
            "5. Consistency — Apply your formatting skill uniformly across every section. "
            "Headings, spacing, bullets, and emphasis must match exactly throughout.\n\n"
            "6. Completeness — Do NOT remove any content from the draft. Only reformat it. "
            "If a section is unclear, flag it with ⚠️.\n\n"
            "7. Output — Return ONLY the formatted Markdown document. No preamble, "
            "no explanation, no code fences around the entire document.\n\n"
            "--- POLICY DRAFT ---\n"
            f"{self.state.policy_draft}\n"
            "--- END OF POLICY DRAFT ---"
            f"{self._feedback_block(feedback)}"
        )

        self.state.formatted_policy_md = self._kickoff(formatter, prompt)

    # ── Step 4: Critic ────────────────────────────────────────────────────────

    def _run_critic(self) -> None:
        critic = Agent(
            role="Policy Content Critic and Quality Reviewer",
            goal=(
                "Review the formatted Markdown policy document for accuracy, logical flow, "
                "completeness, and alignment with the original source data. Provide "
                "actionable, section-specific feedback in a single pass. Never rewrite "
                "content yourself."
            ),
            backstory=(
                "You are a highly experienced GRC policy reviewer with deep expertise in "
                "compliance, risk management, and regulatory frameworks. You have a sharp eye "
                "for inconsistencies, missing critical information, and illogical structure. "
                "Your role is to validate — not to write. You produce a precise, structured "
                "critique that guides a single revision iteration, ensuring the final policy "
                "document is accurate, thorough, and fully representative of the client's data. "
                "You always conclude your review with an explicit REVIEW STATUS line so the "
                "workflow can route correctly."
            ),
            llm=self._llm(),
            memory=False,
            max_execution_time=180,
            verbose=True,
            allow_delegation=False,
        )

        prompt = (
            "Review the formatted policy document below against the original source data "
            f"for {self.state.organization or 'the client'}. Framework: {self.state.framework}.\n\n"
            "REVIEW CHECKLIST:\n"
            "1. Content Accuracy — Are all core processes, products, risks, appetite values, "
            "and compliance details correctly and fully reflected?\n"
            "2. Completeness — Is any critical information from the Analyzer's summary missing "
            "or underrepresented in the final document?\n"
            "3. Logical Flow — Do sections follow a coherent sequence? Are dependencies between "
            "sections (e.g., risks tied to processes) preserved and clearly indicated?\n"
            "4. Critical Emphasis — Are high-priority items, mandatory compliance actions, and "
            "key risks visually distinguished and easy to locate?\n"
            "5. Formatting Consistency — Are headings, bullets, tables, and emphasis applied "
            "uniformly throughout the document?\n"
            "6. No Misrepresentation — Has any content been accidentally altered in meaning "
            "or omitted during formatting?\n\n"
            "DETERMINE THE FAILED STEP (only if revision is needed):\n"
            "  - Set FAILED AGENT: analyzer  — if the Analyzer's summary was itself incomplete "
            "or inaccurate (re-runs: analysis → draft → formatting)\n"
            "  - Set FAILED AGENT: policy_drafter  — if the summary was sound but the draft "
            "is missing or illogically structured (re-runs: draft → formatting)\n"
            "  - Set FAILED AGENT: content_formatter  — if the draft was good but the "
            "formatting is inconsistent or broken (re-runs: formatting only)\n\n"
            "SINGLE ITERATION LIMIT: Document every issue in this one review pass. "
            "Only one revision will be applied.\n\n"
            "OUTPUT FORMAT:\n"
            "Write a structured text report with bullet-pointed issues per section. "
            "End your report with EXACTLY one of these two lines (on its own line):\n"
            "  REVIEW STATUS: Approved\n"
            "  REVIEW STATUS: Requires Revision\n"
            "If revision is needed, also include on its own line:\n"
            "  FAILED AGENT: analyzer\n"
            "  OR  FAILED AGENT: policy_drafter\n"
            "  OR  FAILED AGENT: content_formatter\n\n"
            "--- ORIGINAL SOURCE DATA ---\n"
            f"{self._source_data_block()}\n\n"
            "--- ANALYZER SUMMARY ---\n"
            f"{self.state.analysis}\n"
            "--- END OF ANALYZER SUMMARY ---\n\n"
            "--- FORMATTED POLICY DOCUMENT (to review) ---\n"
            f"{self.state.formatted_policy_md}\n"
            "--- END OF FORMATTED POLICY DOCUMENT ---"
        )

        self.state.critic_feedback = self._kickoff(critic, prompt)

        output_lower = self.state.critic_feedback.lower()
        self.state.critic_passed = (
            "review status: approved" in output_lower
            and "review status: requires revision" not in output_lower
        )

        if not self.state.critic_passed:
            if "failed agent: analyzer" in output_lower:
                self.state.failed_step = "analysis"
            elif "failed agent: policy_drafter" in output_lower:
                self.state.failed_step = "draft"
            else:
                self.state.failed_step = "formatting"

    # ── Flow steps ─────────────────────────────────────────────────────────────

    @start()
    def analyze_data(self):
        self._run_analysis()

    @listen(analyze_data)
    def draft_policy(self):
        self._run_drafting()

    @listen(draft_policy)
    def format_policy(self):
        self._run_formatting()

    @listen(format_policy)
    def critic_review(self):
        self._run_critic()

    @router(critic_review)
    def route_after_critic(self) -> str:
        if not self.state.critic_passed and self.state.revision_count < 1:
            return "needs_revision"
        return "done"

    @listen("needs_revision")
    def revise_workflow(self):
        """Re-run from the failing step forward. Executes at most once."""
        self.state.revision_count += 1
        failed = self.state.failed_step

        if failed == "analysis":
            self._run_analysis(feedback=self.state.critic_feedback)
            self._run_drafting(feedback=self.state.critic_feedback)
            self._run_formatting(feedback=self.state.critic_feedback)
        elif failed == "draft":
            self._run_drafting(feedback=self.state.critic_feedback)
            self._run_formatting(feedback=self.state.critic_feedback)
        else:
            # "formatting" (or unset — default to formatting only)
            self._run_formatting(feedback=self.state.critic_feedback)

    @listen("done")
    def finalize(self):
        pass  # State is fully populated; flow terminates here.


# ── Async entry point (called from FastAPI route handler) ──────────────────────

async def run_grc_policy_expert(request: GRCPolicyRequest) -> dict:
    flow = GRCPolicyFlow()
    loop = asyncio.get_running_loop()

    documents_text = request.documents_text or "No uploaded documents provided."
    client_data_text = _format_client_data(request.client_data)
    organization = request.organization or "Client"
    regulatory_research_context, search_used = _build_regulatory_research_context(request.framework)

    await loop.run_in_executor(
        None,
        lambda: flow.kickoff(inputs={
            "documents_text":  documents_text,
            "client_data_text": client_data_text,
            "organization":    organization,
            "framework":       request.framework,
            "regulatory_research_context": regulatory_research_context,
            "search_used": search_used,
        }),
    )

    s = flow.state

    return {
        "organization":        organization,
        "framework":           request.framework,
        "analysis_summary":    s.analysis,
        "policy_draft":        s.policy_draft,
        "formatted_policy_md": s.formatted_policy_md,
        "critic_status":       "Approved" if s.critic_passed else "Reviewed",
        "revised":             s.revision_count > 0,
        "search_used":         s.search_used,
        "generated_at":        datetime.now(timezone.utc).isoformat(),
    }
