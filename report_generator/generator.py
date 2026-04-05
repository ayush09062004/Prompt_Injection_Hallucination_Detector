"""
report_generator/generator.py
Generates structured JSON reports and formatted summaries from detection results.
"""

import json
from datetime import datetime
from dataclasses import asdict, dataclass


class ReportGenerator:
    """
    Generates comprehensive reports from detection results.
    Outputs: structured JSON, human-readable summary.
    """

    def generate(
        self,
        injection_report,
        hallucination_report,
        risk_scores,
        sanitization_results: dict = None,
        document_metadata: dict = None,
    ) -> dict:
        """
        Generate the complete detection report as a dict (JSON-serializable).
        """
        meta = document_metadata or {}
        timestamp = datetime.utcnow().isoformat() + "Z"

        report = {
            "metadata": {
                "generated_at": timestamp,
                "document_name": meta.get("name", "unknown"),
                "root_file": meta.get("root_file", "unknown"),
                "tex_files_count": meta.get("tex_files_count", 0),
                "bib_files_count": meta.get("bib_files_count", 0),
                "tool": "LaTeX Security Detector v1.0",
            },
            "risk_summary": {
                "overall_risk_score": risk_scores.overall_risk,
                "overall_risk_level": risk_scores.risk_level,
                "injection_score": risk_scores.injection_score,
                "hallucination_score": risk_scores.hallucination_score,
                "recommendations": risk_scores.recommendations,
            },
            "injection_report": self._serialize_injection_report(injection_report, risk_scores),
            "hallucination_report": self._serialize_hallucination_report(hallucination_report, risk_scores),
            "sanitization_summary": self._serialize_sanitization(sanitization_results),
        }

        return report

    def _serialize_injection_report(self, report, risk_scores) -> dict:
        """Serialize injection report to dict."""
        if not report:
            return {"findings": [], "total": 0}

        findings = []
        for f in report.findings:
            findings.append({
                "strategy": f.strategy,
                "source": f.source,
                "modality": f.modality,
                "severity": f.severity,
                "file": f.file,
                "line_number": f.line_number,
                "snippet": f.snippet,
                "explanation": f.explanation,
                "rule_triggered": f.rule_triggered,
                "confidence": round(getattr(f, 'confidence', 0.8), 2),
            })

        return {
            "total_findings": len(findings),
            "by_strategy": report.by_strategy,
            "by_severity": report.by_severity,
            "score": risk_scores.injection_score,
            "score_breakdown": risk_scores.injection_breakdown,
            "findings": findings,
            "llm_analysis_count": len(getattr(report, 'llm_analysis', [])),
        }

    def _serialize_hallucination_report(self, report, risk_scores) -> dict:
        """Serialize hallucination report to dict."""
        if not report:
            return {"findings": [], "total": 0}

        findings = []
        for f in report.findings:
            findings.append({
                "type": f.hal_type,
                "sub_type": f.sub_type,
                "claim": f.claim,
                "explanation": f.explanation,
                "section": f.section,
                "evidence": f.evidence,
                "confidence": round(f.confidence, 2),
                "severity": f.severity,
            })

        return {
            "total_findings": len(findings),
            "by_type": report.by_type,
            "by_sub_type": report.by_sub_type,
            "score": risk_scores.hallucination_score,
            "score_breakdown": risk_scores.hallucination_breakdown,
            "findings": findings,
            "fabricated_citations": getattr(report, 'fabricated_citations', []),
            "contradictions_detected": len(getattr(report, 'contradictions', [])),
            "verified_claims_count": len(getattr(report, 'verified_claims', [])),
        }

    def _serialize_sanitization(self, results: dict) -> dict:
        """Serialize sanitization results."""
        if not results:
            return {}

        summary = {
            "files_processed": len(results),
            "total_items_removed": sum(
                len(r.removed_items) for r in results.values()
            ),
            "per_file": {},
        }

        for fname, res in results.items():
            summary["per_file"][fname] = {
                "original_length": res.original_length,
                "sanitized_length": len(res.sanitized_text),
                "items_removed": len(res.removed_items),
                "warnings": res.warnings,
                "sanitization_score": round(res.sanitization_score, 3),
            }

        return summary

    def to_json(self, report: dict, indent: int = 2) -> str:
        """Convert report dict to formatted JSON string."""
        return json.dumps(report, indent=indent, ensure_ascii=False)

    def to_markdown_summary(self, report: dict) -> str:
        """Generate a human-readable Markdown summary."""
        meta = report.get("metadata", {})
        risk = report.get("risk_summary", {})
        inj = report.get("injection_report", {})
        hal = report.get("hallucination_report", {})

        level = risk.get("overall_risk_level", "UNKNOWN")
        level_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}.get(level, "⚪")

        lines = [
            f"# LaTeX Security Analysis Report",
            f"**Document:** {meta.get('document_name', 'Unknown')}  ",
            f"**Generated:** {meta.get('generated_at', '')}",
            f"",
            f"## {level_emoji} Overall Risk: {level} ({risk.get('overall_risk_score', 0):.1f}/100)",
            f"",
            f"| Metric | Score |",
            f"|--------|-------|",
            f"| Injection Risk | {risk.get('injection_score', 0):.1f}/100 |",
            f"| Hallucination Risk | {risk.get('hallucination_score', 0):.1f}/100 |",
            f"",
            f"## 🔐 Prompt Injection Findings ({inj.get('total_findings', 0)} total)",
        ]

        by_strat = inj.get("by_strategy", {})
        if by_strat:
            for strategy, count in by_strat.items():
                lines.append(f"- **{strategy}**: {count} finding(s)")

        lines += ["", "### Top Injection Findings"]
        for i, f in enumerate(inj.get("findings", [])[:5], 1):
            lines.append(f"{i}. `[{f['severity']}]` **{f['strategy']}** in `{f['file']}`:{f['line_number']}")
            lines.append(f"   - {f['explanation']}")
            lines.append(f"   - Snippet: `{f['snippet'][:80]}...`")

        lines += [
            "",
            f"## 🧠 Hallucination Findings ({hal.get('total_findings', 0)} total)",
        ]

        by_type = hal.get("by_type", {})
        if by_type:
            for htype, count in by_type.items():
                lines.append(f"- **{htype}**: {count} finding(s)")

        lines += ["", "### Top Hallucination Findings"]
        for i, f in enumerate(hal.get("findings", [])[:5], 1):
            lines.append(f"{i}. `[{f['type']}]` in **{f['section']}**")
            lines.append(f"   - Claim: {f['claim'][:100]}")
            lines.append(f"   - {f['explanation']}")

        lines += ["", "## 📋 Recommendations"]
        for rec in risk.get("recommendations", []):
            lines.append(f"- {rec}")

        return "\n".join(lines)
