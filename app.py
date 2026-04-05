"""
app.py — LaTeX Security Detector
Streamlit UI for detecting prompt injection and hallucination in LaTeX research papers.
Now with optional Crossref API citation verification.
"""

import sys
import os
import shutil
import json
import tempfile
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.ingestor import LaTeXIngestor
from latex_parser.parser import LaTeXParser
from injection_detector.detector import InjectionDetector
from hallucination_detector.detector import HallucinationDetector
from hallucination_detector.crossref_client import CrossrefClient
from prompt_armor.sanitizer import PromptArmor
from scoring_engine.scorer import ScoringEngine
from report_generator.generator import ReportGenerator
from groq_client.client import GroqClientManager


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaTeX Security Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (matches generator's style) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .stApp { background: #0a0a0f; color: #e8e8f0; }
  [data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e3a; }
  .card { background: #12121f; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-right: 6px; margin-bottom: 4px; }
  .badge-high   { background: #3d1010; color: #ff6b6b; border: 1px solid #ff6b6b44; }
  .badge-medium { background: #3d2e10; color: #ffa94d; border: 1px solid #ffa94d44; }
  .badge-low    { background: #1a3020; color: #69db7c; border: 1px solid #69db7c44; }
  .badge-type   { background: #1a1a3d; color: #74b9ff; border: 1px solid #74b9ff44; }
  .hero-title { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
  .hero-sub { color: #6b6b8a; font-size: 0.95rem; margin-bottom: 2rem; }
  .section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.4rem; }
  .stButton > button { background: linear-gradient(135deg, #7c3aed, #2563eb) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; letter-spacing: 0.05em !important; padding: 0.6rem 2rem !important; transition: opacity 0.2s !important; }
  .stButton > button:hover { opacity: 0.85 !important; }
  .stDownloadButton > button { background: linear-gradient(135deg, #065f46, #064e3b) !important; color: #34d399 !important; border: 1px solid #34d39944 !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
  .stTextInput input, .stTextArea textarea, .stSelectbox select { background: #0f0f1a !important; border: 1px solid #1e1e3a !important; color: #e8e8f0 !important; border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important; }
  .stMultiSelect [data-baseweb="tag"] { background-color: #1a1a3d !important; color: #a78bfa !important; }
  .streamlit-expanderHeader { background: #12121f !important; border: 1px solid #1e1e3a !important; border-radius: 8px !important; color: #e8e8f0 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
  .stCode { border-radius: 8px !important; }
  .stAlert { border-radius: 8px !important; }
  .risk-critical { background: #ff4b4b22; border-left: 4px solid #ff4b4b; padding: 8px 12px; border-radius: 4px; }
  .risk-high     { background: #ff890022; border-left: 4px solid #ff8900; padding: 8px 12px; border-radius: 4px; }
  .risk-medium   { background: #ffcc0022; border-left: 4px solid #ffcc00; padding: 8px 12px; border-radius: 4px; }
  .risk-low      { background: #00cc4422; border-left: 4px solid #00cc44; padding: 8px 12px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def severity_color(severity: str) -> str:
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")

def risk_color(level: str) -> str:
    return {"CRITICAL": "#ff4b4b", "HIGH": "#ff8900", "MEDIUM": "#ffcc00", "LOW": "#00cc44"}.get(level, "#888")

def make_gauge(value: float, title: str) -> go.Figure:
    """Create a Plotly gauge chart for a risk score."""
    color_thresholds = [
        {"range": [0, 20], "color": "#00cc44"},
        {"range": [20, 40], "color": "#7acc00"},
        {"range": [40, 60], "color": "#ffcc00"},
        {"range": [60, 80], "color": "#ff8900"},
        {"range": [80, 100], "color": "#ff4b4b"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#4a9eff"},
            "steps": color_thresholds,
        },
        number={"suffix": "/100", "font": {"size": 24}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig

def run_analysis(
    zip_path: str,
    api_keys: list[str],
    groq_model: str,
    use_llm: bool,
    aggressive_sanitize: bool,
    use_crossref: bool,
) -> dict:
    """Full analysis pipeline. Returns complete report dict."""
    progress = st.progress(0, text="📦 Extracting ZIP archive...")

    ingestor = LaTeXIngestor()
    doc = ingestor.ingest(zip_path)

    if not doc.tex_files:
        st.error("No .tex files found in the ZIP archive.")
        return {}

    progress.progress(15, text="🔍 Parsing LaTeX structure...")
    parser = LaTeXParser()
    parsed = parser.parse_resolved(doc.resolved_text, doc.bib_files)

    progress.progress(30, text="🛡️ Running Prompt Armor sanitization...")
    armor = PromptArmor()
    sanitization_results = armor.sanitize_all_files(
        doc.tex_files, aggressive=aggressive_sanitize
    )

    progress.progress(40, text="🔐 Running injection detector (rule-based)...")
    groq_client = None
    if use_llm and api_keys:
        try:
            groq_client = GroqClientManager(api_keys, model=groq_model)
        except Exception as e:
            st.warning(f"Could not initialize Groq client: {e}. Running rule-based only.")

    inj_detector = InjectionDetector(groq_client=groq_client if use_llm else None)
    injection_report = inj_detector.detect(
        tex_files=doc.tex_files,
        bib_files=doc.bib_files,
        captions=parsed.captions,
        resolved_text=doc.resolved_text,
        include_chain=doc.include_chain,
    )

    progress.progress(50, text="🧠 Initializing hallucination detector...")

    # Optional Crossref client
    crossref_client = None
    if use_crossref:
        try:
            crossref_client = CrossrefClient()
            progress.progress(55, text="🌐 Verifying citations with Crossref API (may take a while)...")
        except Exception as e:
            st.warning(f"Could not initialize Crossref client: {e}. Crossref verification disabled.")

    hal_detector = HallucinationDetector(
        groq_client=groq_client if use_llm else None,
        crossref_client=crossref_client
    )

    # Run hallucination detection (includes Crossref verification if enabled)
    hallucination_report = hal_detector.detect(
        sections=parsed.sections,
        bib_entries=parsed.bib_entries,
        citations=parsed.citations,
        resolved_text=doc.resolved_text,
    )

    progress.progress(85, text="📊 Computing risk scores...")
    scorer = ScoringEngine()
    risk_scores = scorer.score(
        injection_report=injection_report,
        hallucination_report=hallucination_report,
        total_sections=max(len(parsed.sections), 1),
    )

    progress.progress(95, text="📝 Generating report...")
    report_gen = ReportGenerator()
    final_report = report_gen.generate(
        injection_report=injection_report,
        hallucination_report=hallucination_report,
        risk_scores=risk_scores,
        sanitization_results=sanitization_results,
        document_metadata={
            "name": os.path.basename(zip_path),
            "root_file": doc.root_file,
            "tex_files_count": len(doc.tex_files),
            "bib_files_count": len(doc.bib_files),
        },
    )

    progress.progress(100, text="✅ Analysis complete!")
    try:
        shutil.rmtree(doc.extract_dir, ignore_errors=True)
    except Exception:
        pass

    final_report["_internal"] = {
        "parsed": parsed,
        "injection_report": injection_report,
        "hallucination_report": hallucination_report,
        "risk_scores": risk_scores,
        "sanitization_results": sanitization_results,
        "groq_usage": groq_client.get_usage_summary() if groq_client else {},
        "report_gen": report_gen,
    }
    return final_report


# ═════════════════════════════════════════════════════════════════════════════
# UI Layout
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with configuration (matches generator's style)."""
    with st.sidebar:
        st.markdown('<div class="section-label">🛡️ LaTeX Detector</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="section-label">🔑 Groq API Keys</div>', unsafe_allow_html=True)
        st.caption("Provide 1–4 keys for round-robin rotation.")
        api_key_1 = st.text_input("API Key 1", type="password", placeholder="gsk_...", key="det_key1")
        api_key_2 = st.text_input("API Key 2", type="password", placeholder="gsk_...", key="det_key2")
        api_key_3 = st.text_input("API Key 3", type="password", placeholder="gsk_...", key="det_key3")
        api_key_4 = st.text_input("API Key 4", type="password", placeholder="gsk_...", key="det_key4")
        keys = [k for k in [api_key_1, api_key_2, api_key_3, api_key_4] if k.strip()]

        st.markdown("---")
        st.markdown('<div class="section-label">⚙️ Settings</div>', unsafe_allow_html=True)
        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b","openai/gpt-oss-20b","meta-llama/llama-prompt-guard-2-22m","meta-llama/llama-prompt-guard-2-86m"],
            index=0,
        )
        use_llm = st.toggle("Enable LLM Analysis", value=True, help="Uses Groq API for deep detection")
        aggressive_sanitize = st.toggle("Aggressive Sanitization", value=False,
                                        help="Also strips contextual bias phrases")
        use_crossref = st.toggle(
            "🌐 Verify citations with Crossref API",
            value=False,
            help="Checks each reference against Crossref (slower, requires internet)."
        )
        st.markdown("---")
        st.caption("Built with ❤️ using Groq + Streamlit")
    return keys, groq_model, use_llm, aggressive_sanitize, use_crossref


def render_risk_overview(report: dict):
    """Render the risk score summary with gauges."""
    internal = report["_internal"]
    risk_scores = internal["risk_scores"]
    level = risk_scores.risk_level
    color = risk_color(level)
    emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")
    st.markdown(
        f"""<div style='background: {color}22; border: 2px solid {color}; border-radius: 10px; 
        padding: 16px; text-align: center; margin-bottom: 16px;'>
        <h2 style='color: {color}; margin: 0;'>{emoji} Overall Risk: {level}</h2>
        <p style='color: #ccc; margin: 4px 0 0;'>Combined Score: {risk_scores.overall_risk:.1f} / 100</p>
        </div>""",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(make_gauge(risk_scores.overall_risk, "Overall Risk"), use_container_width=True)
    with col2:
        st.plotly_chart(make_gauge(risk_scores.injection_score, "Injection Risk"), use_container_width=True)
    with col3:
        st.plotly_chart(make_gauge(risk_scores.hallucination_score, "Hallucination Risk"), use_container_width=True)

    st.subheader("📋 Recommendations")
    for rec in report["risk_summary"].get("recommendations", []):
        st.info(rec)


def render_injection_tab(report: dict):
    """Render injection detection findings (no heatmap)."""
    inj_data = report.get("injection_report", {})
    internal = report["_internal"]
    inj_report = internal["injection_report"]

    total = inj_data.get("total_findings", 0)
    st.subheader(f"🔐 Prompt Injection — {total} Finding(s)")

    if total == 0:
        st.success("✅ No prompt injection detected by rule-based analysis.")
        if not internal.get("groq_usage"):
            st.info("LLM analysis was disabled. Enable it for deeper semantic detection.")
        return

    col1, col2, col3, col4 = st.columns(4)
    by_sev = inj_data.get("by_severity", {})
    col1.metric("🔴 Critical", by_sev.get("Critical", 0))
    col2.metric("🟠 High", by_sev.get("High", 0))
    col3.metric("🟡 Medium", by_sev.get("Medium", 0))
    col4.metric("🟢 Low", by_sev.get("Low", 0))

    by_strat = inj_data.get("by_strategy", {})
    if by_strat:
        fig = px.bar(
            x=list(by_strat.keys()),
            y=list(by_strat.values()),
            color=list(by_strat.keys()),
            title="Findings by Strategy",
            color_discrete_map={
                "Direct": "#ff4b4b",
                "Obfuscated": "#ff8900",
                "Contextual": "#ffcc00",
                "Chained": "#cc44ff",
            },
            labels={"x": "Strategy", "y": "Count"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "white"}, showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detailed Findings")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        filter_sev = st.multiselect("Filter by Severity", ["Critical", "High", "Medium", "Low"],
                                    default=["Critical", "High", "Medium", "Low"], key="inj_sev")
    with fcol2:
        filter_strat = st.multiselect("Filter by Strategy", ["Direct", "Obfuscated", "Contextual", "Chained"],
                                      default=["Direct", "Obfuscated", "Contextual", "Chained"], key="inj_strat")

    findings = inj_data.get("findings", [])
    filtered = [f for f in findings if f["severity"] in filter_sev and f["strategy"] in filter_strat]

    for i, finding in enumerate(filtered):
        sev_emoji = severity_color(finding["severity"])
        with st.expander(f"{sev_emoji} [{finding['severity']}] {finding['strategy']} — `{finding['file']}`:{finding['line_number']}", expanded=(i < 3)):
            cols = st.columns(3)
            cols[0].markdown(f"**Strategy:** `{finding['strategy']}`")
            cols[1].markdown(f"**Source:** `{finding['source']}`")
            cols[2].markdown(f"**Modality:** `{finding['modality']}`")
            st.markdown(f"**Explanation:** {finding['explanation']}")
            st.markdown(f"**Rule:** `{finding['rule_triggered']}`")
            st.markdown(f"**Confidence:** {finding['confidence']:.0%}")
            st.code(finding["snippet"], language="latex")

    llm_analysis = getattr(inj_report, 'llm_analysis', [])
    llm_injections = [f for f in llm_analysis if isinstance(f, dict) and f.get("is_injection")]
    if llm_injections:
        st.subheader(f"🤖 LLM Semantic Findings ({len(llm_injections)})")
        for f in llm_injections[:10]:
            with st.expander(f"🤖 LLM: {f.get('strategy', 'Unknown')} — confidence {f.get('confidence', 0):.0%}"):
                st.markdown(f"**Explanation:** {f.get('explanation', '')}")
                st.code(f.get('snippet', '')[:200], language="text")


def render_hallucination_tab(report: dict):
    """Render hallucination detection findings."""
    hal_data = report.get("hallucination_report", {})
    total = hal_data.get("total_findings", 0)

    st.subheader(f"🧠 Hallucination — {total} Finding(s)")

    # Display fabricated bibliography entries (from Crossref check)
    fab_bib = hal_data.get("fabricated_bib_entries", [])
    if fab_bib:
        st.error(f"📚 **{len(fab_bib)} fabricated bibliography entries detected:**")
        for entry in fab_bib[:20]:
            st.code(entry, language="text")
        st.markdown("---")

    if total == 0:
        st.success("✅ No hallucination patterns detected.")
        verified = hal_data.get("verified_claims_count", 0)
        if verified > 0:
            st.info(f"LLM verified {verified} claims — all appear supported.")
        return

    col1, col2, col3, col4 = st.columns(4)
    by_type = hal_data.get("by_type", {})
    col1.metric("🏭 Fabrication", by_type.get("Fabrication", 0))
    col2.metric("📐 Distortion", by_type.get("Distortion", 0))
    col3.metric("🔄 Contradiction", by_type.get("Contradiction", 0))
    col4.metric("📚 Fake Citations", len(hal_data.get("fabricated_citations", [])))

    if by_type:
        fig = px.pie(
            values=list(by_type.values()),
            names=list(by_type.keys()),
            title="Hallucination by Type",
            color_discrete_map={"Fabrication": "#ff4b4b", "Distortion": "#ff8900", "Contradiction": "#cc44ff"}
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, height=300)
        st.plotly_chart(fig, use_container_width=True)

    fab_cites = hal_data.get("fabricated_citations", [])
    if fab_cites:
        st.warning(f"⚠️ **{len(fab_cites)} suspicious citation(s) detected** (missing or mismatched in bibliography):")
        st.code(", ".join(fab_cites[:20]), language="text")

    st.subheader("📋 Detailed Findings")
    filter_type = st.multiselect(
        "Filter by Type",
        ["Fabrication", "Distortion", "Contradiction"],
        default=["Fabrication", "Distortion", "Contradiction"],
        key="hal_type"
    )
    findings = hal_data.get("findings", [])
    filtered = [f for f in findings if f["type"] in filter_type]

    for i, finding in enumerate(filtered):
        type_emoji = {"Fabrication": "🏭", "Distortion": "📐", "Contradiction": "🔄"}.get(finding["type"], "⚠️")
        with st.expander(
            f"{type_emoji} [{finding['type']}] {finding['sub_type']} — Section: {finding['section'][:40]}",
            expanded=(i < 3)
        ):
            cols = st.columns(2)
            cols[0].markdown(f"**Type:** `{finding['type']}`")
            cols[1].markdown(f"**Confidence:** {finding['confidence']:.0%}")
            st.markdown(f"**Claim:** _{finding['claim'][:200]}_")
            st.markdown(f"**Explanation:** {finding['explanation']}")
            if finding.get("evidence"):
                st.markdown(f"**Context:** {finding['evidence'][:200]}")


def render_sanitization_tab(report: dict):
    """Render sanitization results."""
    san_data = report.get("sanitization_summary", {})
    internal = report["_internal"]
    san_results = internal.get("sanitization_results", {})

    st.subheader("🛡️ Prompt Armor Sanitization")
    col1, col2 = st.columns(2)
    col1.metric("Files Processed", san_data.get("files_processed", 0))
    col2.metric("Items Removed/Neutralized", san_data.get("total_items_removed", 0))
    st.info("Prompt Armor wraps all document content with isolation markers before LLM processing, "
            "preventing injected instructions from being executed. Risky spans are tagged `[RISK:type]`.")

    per_file = san_data.get("per_file", {})
    if per_file:
        rows = [{"File": fname, "Original Length": fdata["original_length"],
                 "Sanitized Length": fdata["sanitized_length"], "Items Removed": fdata["items_removed"],
                 "Cleanliness": f"{fdata['sanitization_score']:.1%}"} for fname, fdata in per_file.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if san_results:
        fname = next(iter(san_results))
        res = san_results[fname]
        st.subheader(f"📄 Annotated View: `{fname}`")
        st.caption("Risky spans are tagged with [RISK:...] markers")
        if res.warnings:
            for w in res.warnings[:10]:
                st.warning(w)
        tagged_preview = res.tagged_text[:3000]
        st.text_area("Tagged Text (first 3000 chars)", tagged_preview, height=300)
        st.download_button("⬇️ Download Sanitized Text", res.sanitized_text,
                           file_name=f"sanitized_{fname.replace('/', '_')}", mime="text/plain")


def render_report_tab(report: dict, report_gen: ReportGenerator):
    """Render the full JSON report and download options."""
    st.subheader("📄 Full Analysis Report")
    clean_report = {k: v for k, v in report.items() if not k.startswith("_")}
    json_str = report_gen.to_json(clean_report)
    md_summary = report_gen.to_markdown_summary(clean_report)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download JSON Report", json_str, file_name="latex_security_report.json", mime="application/json")
    with col2:
        st.download_button("⬇️ Download Markdown Summary", md_summary, file_name="latex_security_report.md", mime="text/markdown")

    groq_usage = report["_internal"].get("groq_usage", {})
    if groq_usage and groq_usage.get("total_calls", 0) > 0:
        st.subheader("🔢 Groq API Usage")
        ucol1, ucol2, ucol3 = st.columns(3)
        ucol1.metric("Total API Calls", groq_usage.get("total_calls", 0))
        ucol2.metric("Prompt Tokens", groq_usage.get("total_prompt_tokens", 0))
        ucol3.metric("Completion Tokens", groq_usage.get("total_completion_tokens", 0))

    with st.expander("🔍 View Full JSON Report"):
        st.json(clean_report)


# ═════════════════════════════════════════════════════════════════════════════
# Main App
# ═════════════════════════════════════════════════════════════════════════════

def main():
    api_keys, groq_model, use_llm, aggressive_sanitize, use_crossref = render_sidebar()

    st.markdown('<div class="hero-title">🛡️ LaTeX Security Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Detect prompt injection and hallucination in LaTeX research papers.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload LaTeX ZIP Archive", type=["zip"],
                                     help="ZIP should contain .tex, .bib, and any included files")

    if uploaded_file is None:
        st.markdown("---")
        st.markdown("""
        <div class="card">
        <div class="section-label">How it works</div>
        <ol style="color:#9898b8; line-height:2; font-size:0.9rem;">
          <li>Enter your Groq API key(s) in the sidebar</li>
          <li>Upload a LaTeX ZIP (e.g., from the Synthetic Research Paper Generator)</li>
          <li>Run analysis to detect injection attacks and hallucinations</li>
          <li>Review detailed findings and download reports</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 **Compatible with SyntheticResearchPaper generator**: Detects all 4 injection strategies (Direct, Obfuscated, Contextual, Chained) and 3 hallucination types (Fabrication, Distortion, Contradiction).")
        return

    if use_llm and not api_keys:
        st.warning("⚠️ LLM analysis enabled but no API keys provided. Only rule-based detection will run.")
        use_llm = False

    if st.button("🚀 Run Security Analysis", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            with st.spinner("Analyzing..."):
                report = run_analysis(tmp_path, api_keys, groq_model, use_llm, aggressive_sanitize, use_crossref)
            if not report:
                st.error("Analysis failed — check ZIP contents.")
                return
            st.session_state["detector_report"] = report
        except Exception as e:
            st.error(f"Analysis error: {e}")
            import traceback
            st.expander("Error details").code(traceback.format_exc())
        finally:
            os.unlink(tmp_path)

    if "detector_report" in st.session_state:
        report = st.session_state["detector_report"]
        internal = report.get("_internal", {})
        report_gen = internal.get("report_gen", ReportGenerator())

        st.markdown("---")
        render_risk_overview(report)

        tabs = st.tabs(["🔐 Injection", "🧠 Hallucination", "🛡️ Sanitization", "📄 Full Report"])
        with tabs[0]:
            render_injection_tab(report)
        with tabs[1]:
            render_hallucination_tab(report)
        with tabs[2]:
            render_sanitization_tab(report)
        with tabs[3]:
            render_report_tab(report, report_gen)


if __name__ == "__main__":
    main()
