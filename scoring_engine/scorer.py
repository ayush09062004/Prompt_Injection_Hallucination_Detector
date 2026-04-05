"""
scoring_engine/scorer.py
Generates quantitative risk scores from injection and hallucination findings.

Injection Score:
  - Base score from count × severity weight
  - Multiplied by obfuscation factor (obfuscated = harder to detect)
  - Normalized to 0-100

Hallucination Score:
  - % of verified claims that are fabricated/distorted/contradicted
  - Weighted by confidence and severity
  - Normalized to 0-100
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from injection_detector.detector import InjectionReport
    from hallucination_detector.detector import HallucinationReport


@dataclass
class RiskScores:
    """Complete risk assessment scores."""
    injection_score: float          # 0-100
    hallucination_score: float      # 0-100
    overall_risk: float             # 0-100
    risk_level: str                 # LOW | MEDIUM | HIGH | CRITICAL
    injection_breakdown: dict
    hallucination_breakdown: dict
    recommendations: list[str]


class ScoringEngine:
    """
    Computes risk scores from detection reports.
    
    Severity weights for injection:
      Critical → 10 points
      High     →  7 points
      Medium   →  4 points
      Low      →  1 point

    Obfuscation multipliers:
      Obfuscated → 1.5× (harder to detect, more dangerous)
      Chained    → 1.3× 
      Direct     → 1.0×
      Contextual → 0.8× (lower confidence)

    Score normalization: logistic curve so scores don't saturate immediately.
    """

    SEVERITY_WEIGHTS = {
        "Critical": 10,
        "High": 7,
        "Medium": 4,
        "Low": 1,
    }

    STRATEGY_MULTIPLIERS = {
        "Obfuscated": 1.5,
        "Chained": 1.3,
        "Direct": 1.0,
        "Contextual": 0.8,
    }

    HAL_TYPE_WEIGHTS = {
        "Fabrication": 10,
        "Distortion": 7,
        "Contradiction": 6,
    }

    def score(
        self,
        injection_report,
        hallucination_report,
        total_sections: int = 1,
    ) -> RiskScores:
        """
        Compute all risk scores from detection reports.
        
        Args:
            injection_report: InjectionReport from injection detector
            hallucination_report: HallucinationReport from hallucination detector
            total_sections: Number of sections in document (for normalization)
        """
        inj_score, inj_breakdown = self._score_injection(injection_report)
        hal_score, hal_breakdown = self._score_hallucination(
            hallucination_report, total_sections
        )

        # Overall risk: weighted average (injection weighted slightly higher)
        overall = min(100.0, 0.55 * inj_score + 0.45 * hal_score)

        risk_level = self._risk_level(overall)
        recommendations = self._generate_recommendations(
            inj_score, hal_score, injection_report, hallucination_report
        )

        return RiskScores(
            injection_score=round(inj_score, 1),
            hallucination_score=round(hal_score, 1),
            overall_risk=round(overall, 1),
            risk_level=risk_level,
            injection_breakdown=inj_breakdown,
            hallucination_breakdown=hal_breakdown,
            recommendations=recommendations,
        )

    def _score_injection(self, report) -> tuple[float, dict]:
        """Compute injection score with severity × strategy weighting."""
        if not report or not report.findings:
            return 0.0, {"raw_points": 0, "findings_count": 0}

        raw_points = 0.0
        by_strategy = {}

        for finding in report.findings:
            weight = self.SEVERITY_WEIGHTS.get(finding.severity, 4)
            multiplier = self.STRATEGY_MULTIPLIERS.get(finding.strategy, 1.0)
            confidence = getattr(finding, 'confidence', 0.8)
            points = weight * multiplier * confidence

            raw_points += points
            by_strategy[finding.strategy] = by_strategy.get(finding.strategy, 0) + 1

        # Also add LLM-detected findings (lower weight since lower confidence)
        if hasattr(report, 'llm_analysis'):
            for llm_f in report.llm_analysis:
                if isinstance(llm_f, dict) and llm_f.get("is_injection"):
                    raw_points += 3 * llm_f.get("confidence", 0.6)

        # Normalize: logistic-inspired normalization
        # 50 raw points → score = 100
        normalized = min(100.0, (raw_points / 50.0) * 100.0)

        breakdown = {
            "raw_points": round(raw_points, 1),
            "findings_count": len(report.findings),
            "by_strategy": by_strategy,
            "by_severity": report.by_severity,
            "llm_findings": len([f for f in getattr(report, 'llm_analysis', [])
                                  if isinstance(f, dict) and f.get("is_injection")]),
        }
        return normalized, breakdown

    def _score_hallucination(self, report, total_sections: int) -> tuple[float, dict]:
        """Compute hallucination score as weighted % of suspicious content."""
        if not report or not report.findings:
            # Check if LLM ran but found nothing
            verified = getattr(report, 'verified_claims', [])
            return 0.0, {
                "raw_points": 0,
                "findings_count": 0,
                "fabrication_count": 0,
                "distortion_count": 0,
                "contradiction_count": 0,
                "verified_claims_count": len(verified) if verified else 0,
            }

        raw_points = 0.0
        fabrication_count = 0
        distortion_count = 0
        contradiction_count = 0

        for finding in report.findings:
            weight = self.HAL_TYPE_WEIGHTS.get(finding.hal_type, 5)
            confidence = getattr(finding, 'confidence', 0.7)
            points = weight * confidence

            raw_points += points

            if finding.hal_type == "Fabrication":
                fabrication_count += 1
            elif finding.hal_type == "Distortion":
                distortion_count += 1
            elif finding.hal_type == "Contradiction":
                contradiction_count += 1

        # Normalize: 40 raw points → 100
        normalized = min(100.0, (raw_points / 40.0) * 100.0)

        # Verified claims context
        verified = getattr(report, 'verified_claims', [])
        total_verified = len(verified)
        flagged_verified = sum(1 for c in verified
                               if c.get("status") in ("Fabricated", "Distorted", "Contradicted"))
        claim_rate = (flagged_verified / max(total_verified, 1)) * 100 if total_verified else 0

        breakdown = {
            "raw_points": round(raw_points, 1),
            "findings_count": len(report.findings),
            "fabrication_count": fabrication_count,
            "distortion_count": distortion_count,
            "contradiction_count": contradiction_count,
            "verified_claims_count": total_verified,
            "suspicious_claim_rate": round(claim_rate, 1),
            "fabricated_citations": len(getattr(report, 'fabricated_citations', [])),
        }
        return normalized, breakdown

    def _risk_level(self, score: float) -> str:
        """Convert score to categorical risk level."""
        if score >= 70:
            return "CRITICAL"
        elif score >= 40:
            return "HIGH"
        elif score >= 20:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendations(
        self, inj_score: float, hal_score: float, inj_report, hal_report
    ) -> list[str]:
        """Generate actionable recommendations based on findings."""
        recs = []

        if inj_score >= 70:
            recs.append("🚨 CRITICAL: Multiple high-severity injections detected. Do NOT process this document through any LLM pipeline without thorough manual review.")
        elif inj_score >= 40:
            recs.append("⚠️ HIGH: Significant injection attempts detected. Use Prompt Armor isolation wrapping before any LLM processing.")
        elif inj_score >= 20:
            recs.append("⚠️ MEDIUM: Some contextual bias patterns detected. Review flagged sections before LLM processing.")

        if hal_score >= 60:
            recs.append("🚨 HIGH HALLUCINATION RISK: Major fabrication or distortion detected. Cross-verify all numerical claims and citations against original sources.")
        elif hal_score >= 30:
            recs.append("⚠️ MODERATE HALLUCINATION: Some suspicious claims found. Verify citations and statistical results independently.")

        # Strategy-specific recommendations
        by_strategy = getattr(inj_report, 'by_strategy', {})
        if "Obfuscated" in by_strategy:
            recs.append("🔍 Obfuscated injections found: Check for hidden Unicode characters, base64 payloads, and catcode manipulations.")
        if "Chained" in by_strategy:
            recs.append("🔗 Chained injection detected: This attack spans multiple sections — review cross-section dependencies carefully.")

        # Hallucination-specific
        fab_cites = len(getattr(hal_report, 'fabricated_citations', []))
        if fab_cites > 0:
            recs.append(f"📚 {fab_cites} citation(s) not found in bibliography — verify these references exist.")
        if getattr(hal_report, 'contradictions', []):
            recs.append("🔄 Cross-section contradictions detected: Abstract, Results, and Conclusion sections may contain conflicting claims.")

        if not recs:
            recs.append("✅ Document appears relatively clean. Standard review recommended before use.")

        return recs
