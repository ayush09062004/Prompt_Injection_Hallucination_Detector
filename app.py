"""
app.py  —  LaTeX Security Detector
Streamlit UI for detecting prompt injection and hallucination in LaTeX research papers.

Run with: streamlit run app.py
"""

import sys
import os
import shutil
import json
import tempfile
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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-critical { background: #ff4b4b22; border-left: 4px solid #ff4b4b; padding: 8px 12px; border-radius: 4px; }
.risk-high     { background: #ff890022; border-left: 4px solid #ff8900; padding: 8px 12px; border-radius: 4px; }
.risk-medium   { background: #ffcc0022; border-left: 4px solid #ffcc00; padding: 8px 12px; border-radius: 4px; }
.risk-low      { background: #00cc4422; border-left: 4px solid #00cc44; padding: 8px 12px; border-radius: 4px; }
.metric-card   { background: #1e1e2e; border-radius: 8px; padding: 16px; text-align: center; }
.finding-row   { border-bottom: 1px solid #333; padding: 8px 0; }
.tag-risk      { background: #ff4b4b33; color: #ff6b6b; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.85em; }
.tag-ok        { background: #00cc4433; color: #00cc44; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def severity_color(severity: str) -> str:
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")

def risk_color(level: str) -> str:
    return {"CRITICAL": "#ff4b4b", "HIGH": "#ff8900", "MEDIUM": "#ffcc00", "LOW": "#00cc44"}.get(level, "#888")

def make_gauge(value: float, title: str, color_thresholds=None) -> go.Figure:
    """Create a Plotly gauge chart for a risk score."""
    if color_thresholds is None:
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
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": value,
            },
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


def make_heatmap(injection_findings, sections) -> go.Figure:
    """Create a risk heatmap over document sections."""
    if not sections or not injection_findings:
        return None

    section_titles = [s.title[:30] for s in sections]
    inj_counts = [0] * len(section_titles)
    hal_counts = [0] * len(section_titles)

    sev_weight = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

    for finding in injection_findings:
        snippet = finding.snippet.lower()
        for i, s in enumerate(sections):
            if s.title.lower() in snippet or finding.file in s.file:
                inj_counts[i] += sev_weight.get(finding.severity, 1)
                break

    z = [inj_counts]
    y_labels = ["Injection Risk"]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=section_titles,
        y=y_labels,
        colorscale=[[0, "#0d1b2a"], [0.3, "#00cc44"], [0.6, "#ffcc00"], [1.0, "#ff4b4b"]],
        hovertemplate="%{x}<br>Risk: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Section Risk Heatmap",
        height=200,
        margin=dict(l=10, r=10, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis={"tickangle": -30},
    )
    return fig


def run_analysis(
    zip_path: str,
    api_keys: list[str],
    groq_model: str,
    use_llm: bool,
    aggressive_sanitize: bool,
) -> dict:
    """
    Full analysis pipeline. Returns complete report dict.
    """
    progress = st.progress(0, text="📦 Extracting ZIP archive...")

    # ── Step 1: Ingestion ─────────────────────────────────────────────────────
    ingestor = LaTeXIngestor()
    doc = ingestor.ingest(zip_path)

    if not doc.tex_files:
        st.error("No .tex files found in the ZIP archive.")
        return {}

    progress.progress(15, text="🔍 Parsing LaTeX structure...")

    # ── Step 2: Parse ─────────────────────────────────────────────────────────
    parser = LaTeXParser()
    parsed = parser.parse_resolved(doc.resolved_text, doc.bib_files)

    progress.progress(30, text="🛡️ Running Prompt Armor sanitization...")

    # ── Step 3: Prompt Armor ──────────────────────────────────────────────────
    armor = PromptArmor()
    sanitization_results = armor.sanitize_all_files(
        doc.tex_files, aggressive=aggressive_sanitize
    )

    progress.progress(40, text="🔐 Running injection detector (rule-based)...")

    # ── Step 4: Injection Detection ───────────────────────────────────────────
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

    progress.progress(65, text="🧠 Running hallucination detector...")

    # ── Step 5: Hallucination Detection ──────────────────────────────────────
    hal_detector = HallucinationDetector(groq_client=groq_client if use_llm else None)
    hallucination_report = hal_detector.detect(
        sections=parsed.sections,
        bib_entries=parsed.bib_entries,
        citations=parsed.citations,
        resolved_text=doc.resolved_text,
    )

    progress.progress(85, text="📊 Computing risk scores...")

    # ── Step 6: Scoring ───────────────────────────────────────────────────────
    scorer = ScoringEngine()
    risk_scores = scorer.score(
        injection_report=injection_report,
        hallucination_report=hallucination_report,
        total_sections=max(len(parsed.sections), 1),
    )

    progress.progress(95, text="📝 Generating report...")

    # ── Step 7: Report ────────────────────────────────────────────────────────
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

    # Cleanup temp dir
    try:
        shutil.rmtree(doc.extract_dir, ignore_errors=True)
    except Exception:
        pass

    # Store extra objects for UI rendering
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
    """Render sidebar with configuration."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/shield.png", width=60)
        st.title("🛡️ LaTeX Detector")
        st.caption("Prompt Injection & Hallucination Scanner")
        st.divider()

        st.subheader("🔑 Groq API Keys")
        st.caption("Enter up to 4 keys for round-robin usage")

        keys = []
        for i in range(1, 5):
            k = st.text_input(
                f"API Key {i}",
                type="password",
                key=f"api_key_{i}",
                placeholder=f"gsk_... (key {i})",
            )
            if k.strip():
                keys.append(k.strip())

        st.divider()
        st.subheader("⚙️ Settings")

        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
            index=0,
            help="Faster models for speed, larger for accuracy",
        )

        use_llm = st.toggle(
            "Enable LLM Analysis",
            value=True,
            help="Uses Groq API for deep semantic detection. Requires API key.",
        )

        aggressive_sanitize = st.toggle(
            "Aggressive Sanitization",
            value=False,
            help="Also strips contextual bias phrases (may affect legitimate text)",
        )

        st.divider()
        st.caption("Built with ❤️ using Groq + Streamlit")

    return keys, groq_model, use_llm, aggressive_sanitize


def render_risk_overview(report: dict):
    """Render the risk score summary with gauges."""
    internal = report["_internal"]
    risk_scores = internal["risk_scores"]
    risk_summary = report["risk_summary"]

    level = risk_scores.risk_level
    color = risk_color(level)

    # Big risk level banner
    emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")
    st.markdown(
        f"""<div style='background: {color}22; border: 2px solid {color}; border-radius: 10px; 
        padding: 16px; text-align: center; margin-bottom: 16px;'>
        <h2 style='color: {color}; margin: 0;'>{emoji} Overall Risk: {level}</h2>
        <p style='color: #ccc; margin: 4px 0 0;'>Combined Score: {risk_scores.overall_risk:.1f} / 100</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Gauge charts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(make_gauge(risk_scores.overall_risk, "Overall Risk"), use_container_width=True)
    with col2:
        st.plotly_chart(make_gauge(risk_scores.injection_score, "Injection Risk"), use_container_width=True)
    with col3:
        st.plotly_chart(make_gauge(risk_scores.hallucination_score, "Hallucination Risk"), use_container_width=True)

    # Recommendations
    st.subheader("📋 Recommendations")
    for rec in risk_summary.get("recommendations", []):
        st.info(rec)


def render_injection_tab(report: dict):
    """Render injection detection findings."""
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

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    by_sev = inj_data.get("by_severity", {})
    col1.metric("🔴 Critical", by_sev.get("Critical", 0))
    col2.metric("🟠 High", by_sev.get("High", 0))
    col3.metric("🟡 Medium", by_sev.get("Medium", 0))
    col4.metric("🟢 Low", by_sev.get("Low", 0))

    # Strategy breakdown chart
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
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            showlegend=False,
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    parsed = internal.get("parsed")
    if parsed and parsed.sections:
        heatmap = make_heatmap(inj_report.findings, parsed.sections)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

    # Findings list
    st.subheader("📋 Detailed Findings")

    # Filter controls
    fcol1, fcol2 = st.columns([2, 2])
    with fcol1:
        filter_sev = st.multiselect(
            "Filter by Severity",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
            key="inj_sev_filter",
        )
    with fcol2:
        filter_strat = st.multiselect(
            "Filter by Strategy",
            ["Direct", "Obfuscated", "Contextual", "Chained"],
            default=["Direct", "Obfuscated", "Contextual", "Chained"],
            key="inj_strat_filter",
        )

    findings = inj_data.get("findings", [])
    filtered = [
        f for f in findings
        if f["severity"] in filter_sev and f["strategy"] in filter_strat
    ]

    for i, finding in enumerate(filtered):
        sev_emoji = severity_color(finding["severity"])
        with st.expander(
            f"{sev_emoji} [{finding['severity']}] {finding['strategy']} — "
            f"`{finding['file']}`:{finding['line_number']}",
            expanded=(i < 3),
        ):
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Strategy:** `{finding['strategy']}`")
            col2.markdown(f"**Source:** `{finding['source']}`")
            col3.markdown(f"**Modality:** `{finding['modality']}`")

            st.markdown(f"**Explanation:** {finding['explanation']}")
            st.markdown(f"**Rule:** `{finding['rule_triggered']}`")
            st.markdown(f"**Confidence:** {finding['confidence']:.0%}")

            st.markdown("**Snippet:**")
            st.code(finding["snippet"], language="latex")

    # LLM findings section
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
    internal = report["_internal"]

    total = hal_data.get("total_findings", 0)
    st.subheader(f"🧠 Hallucination — {total} Finding(s)")

    if total == 0:
        st.success("✅ No hallucination patterns detected.")
        verified = hal_data.get("verified_claims_count", 0)
        if verified > 0:
            st.info(f"LLM verified {verified} claims — all appear supported.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    by_type = hal_data.get("by_type", {})
    col1.metric("🏭 Fabrication", by_type.get("Fabrication", 0))
    col2.metric("📐 Distortion", by_type.get("Distortion", 0))
    col3.metric("🔄 Contradiction", by_type.get("Contradiction", 0))
    col4.metric("📚 Fake Citations", hal_data.get("fabricated_citations_count",
                len(hal_data.get("fabricated_citations", []))))

    # Type breakdown
    if by_type:
        fig = px.pie(
            values=list(by_type.values()),
            names=list(by_type.keys()),
            title="Hallucination by Type",
            color_discrete_map={
                "Fabrication": "#ff4b4b",
                "Distortion": "#ff8900",
                "Contradiction": "#cc44ff",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fabricated citations
    fab_cites = hal_data.get("fabricated_citations", [])
    if fab_cites:
        st.warning(f"⚠️ **{len(fab_cites)} citation key(s) not found in bibliography:**")
        st.code(", ".join(fab_cites))

    # Findings
    st.subheader("📋 Detailed Findings")

    filter_type = st.multiselect(
        "Filter by Type",
        ["Fabrication", "Distortion", "Contradiction"],
        default=["Fabrication", "Distortion", "Contradiction"],
        key="hal_type_filter",
    )

    findings = hal_data.get("findings", [])
    filtered = [f for f in findings if f["type"] in filter_type]

    for i, finding in enumerate(filtered):
        type_emoji = {"Fabrication": "🏭", "Distortion": "📐", "Contradiction": "🔄"}.get(
            finding["type"], "⚠️"
        )
        with st.expander(
            f"{type_emoji} [{finding['type']}] {finding['sub_type']} — Section: {finding['section'][:40]}",
            expanded=(i < 3),
        ):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Type:** `{finding['type']}`")
            col2.markdown(f"**Confidence:** {finding['confidence']:.0%}")

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

    st.info(
        "Prompt Armor wraps all document content with isolation markers before LLM processing, "
        "preventing injected instructions from being executed by the model. "
        "Risky spans are tagged `[RISK:type]` rather than silently deleted."
    )

    # Per-file results
    per_file = san_data.get("per_file", {})
    if per_file:
        rows = []
        for fname, fdata in per_file.items():
            rows.append({
                "File": fname,
                "Original Length": fdata["original_length"],
                "Sanitized Length": fdata["sanitized_length"],
                "Items Removed": fdata["items_removed"],
                "Cleanliness": f"{fdata['sanitization_score']:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Show annotated text for first file
    if san_results:
        fname = next(iter(san_results))
        res = san_results[fname]

        st.subheader(f"📄 Annotated View: `{fname}`")
        st.caption("Risky spans are tagged with [RISK:...] markers")

        if res.warnings:
            for w in res.warnings[:10]:
                st.warning(w)

        # Show tagged text
        tagged_preview = res.tagged_text[:3000]
        st.text_area("Tagged Text (first 3000 chars)", tagged_preview, height=300)

        # Download sanitized text
        st.download_button(
            "⬇️ Download Sanitized Text",
            res.sanitized_text,
            file_name=f"sanitized_{fname.replace('/', '_')}",
            mime="text/plain",
        )


def render_report_tab(report: dict, report_gen: ReportGenerator):
    """Render the full JSON report and download options."""
    st.subheader("📄 Full Analysis Report")

    internal = report.get("_internal", {})
    clean_report = {k: v for k, v in report.items() if not k.startswith("_")}

    # JSON report
    json_str = report_gen.to_json(clean_report)
    md_summary = report_gen.to_markdown_summary(clean_report)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download JSON Report",
            json_str,
            file_name="latex_security_report.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            "⬇️ Download Markdown Summary",
            md_summary,
            file_name="latex_security_report.md",
            mime="text/markdown",
        )

    # Groq usage
    groq_usage = internal.get("groq_usage", {})
    if groq_usage and groq_usage.get("total_calls", 0) > 0:
        st.subheader("🔢 Groq API Usage")
        ucol1, ucol2, ucol3 = st.columns(3)
        ucol1.metric("Total API Calls", groq_usage.get("total_calls", 0))
        ucol2.metric("Prompt Tokens", groq_usage.get("total_prompt_tokens", 0))
        ucol3.metric("Completion Tokens", groq_usage.get("total_completion_tokens", 0))

    # Raw JSON viewer
    with st.expander("🔍 View Full JSON Report"):
        st.json(clean_report)


# ═════════════════════════════════════════════════════════════════════════════
# Main App
# ═════════════════════════════════════════════════════════════════════════════

def main():
    api_keys, groq_model, use_llm, aggressive_sanitize = render_sidebar()

    st.title("🛡️ LaTeX Security Detector")
    st.caption(
        "Detect **prompt injection** and **hallucination** in LaTeX research papers. "
        "Upload a ZIP containing your `.tex` files to begin."
    )

    # ── Upload section ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📁 Upload LaTeX ZIP Archive",
        type=["zip"],
        help="ZIP should contain .tex, .bib, and any included files",
    )

    if uploaded_file is None:
        # Landing page info
        st.markdown("---")
        st.markdown("""
        ### How it works
        
        **1. Upload** your LaTeX ZIP file (can contain multiple .tex, .bib, figures, etc.)
        
        **2. Analysis Pipeline:**
        - 📦 **Ingestion**: Extracts ZIP, resolves `\\input`/`\\include` recursively  
        - 🔍 **Parsing**: Extracts sections, comments, macros, captions, citations
        - 🔐 **Injection Detection**: Rule-based + LLM hybrid across 4 strategies:
          - *Direct*: Explicit override instructions in comments
          - *Obfuscated*: Base64, catcode, zero-width chars, high entropy
          - *Contextual*: Subtle authority bias, pre-emptive dismissal
          - *Chained*: Multi-section coordinated attacks
        - 🧠 **Hallucination Detection**: LLM claim verification + rule-based:
          - *Fabrication*: Fake citations, invented datasets
          - *Distortion*: Impossible numbers, unrealistic gains
          - *Contradiction*: Cross-section inconsistencies
        - 🛡️ **Prompt Armor**: Sanitizes and isolates content for safe LLM processing
        - 📊 **Scoring**: Risk scores with severity weighting
        
        **3. Review** findings with highlighted snippets, downloadable reports
        """)

        # Demo note
        st.info(
            "💡 **Compatible with SyntheticResearchPaper generator**: "
            "This detector is tuned to recognize all 4 injection strategies "
            "(Direct, Obfuscated, Contextual, Chained) and 3 hallucination types "
            "(Fabrication, Distortion, Contradiction) produced by the synthetic generator."
        )
        return

    # ── Run analysis ──────────────────────────────────────────────────────────
    if use_llm and not api_keys:
        st.warning(
            "⚠️ LLM analysis is enabled but no API keys provided. "
            "Only rule-based detection will run. "
            "Add at least one Groq API key in the sidebar for full analysis."
        )
        use_llm = False

    run_btn = st.button(
        "🚀 Run Security Analysis",
        type="primary",
        use_container_width=True,
    )

    if run_btn or ("report" in st.session_state and st.session_state.get("last_file") == uploaded_file.name):
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Analyzing..."):
                report = run_analysis(
                    zip_path=tmp_path,
                    api_keys=api_keys,
                    groq_model=groq_model,
                    use_llm=use_llm,
                    aggressive_sanitize=aggressive_sanitize,
                )

            if not report:
                st.error("Analysis failed — check that your ZIP contains .tex files.")
                return

            st.session_state["report"] = report
            st.session_state["last_file"] = uploaded_file.name

        except Exception as e:
            st.error(f"Analysis error: {e}")
            import traceback
            st.expander("Error details").code(traceback.format_exc())
            return
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Display results ───────────────────────────────────────────────────────
    if "report" in st.session_state and st.session_state["report"]:
        report = st.session_state["report"]
        internal = report.get("_internal", {})
        report_gen = internal.get("report_gen", ReportGenerator())

        st.markdown("---")
        render_risk_overview(report)

        st.markdown("---")
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
