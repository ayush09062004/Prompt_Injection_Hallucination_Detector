"""
crossref_client.py
Robust Crossref API client for citation verification (DEEPSI-ready)
"""

import time
import re
import requests
import logging
from urllib.parse import quote
from typing import Optional, Dict, List
from functools import lru_cache
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossrefClient:
    BASE_URL = "https://api.crossref.org"
    DEFAULT_USER_AGENT = "DEEPSI/1.0 (mailto:your_real_email@gmail.com)"  # CHANGE THIS

    def __init__(self, rate_limit_wait: float = 0.2, user_agent: Optional[str] = None):
        self.rate_limit_wait = rate_limit_wait
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._last_request_time = 0

    # -----------------------------
    # Utility Functions
    # -----------------------------
    def _respect_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_wait:
            time.sleep(self.rate_limit_wait - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(self, url, headers, params=None, retries=3):
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 404:
                    return None

                else:
                    logger.warning(f"Crossref API error {response.status_code}: {response.text[:200]}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")

            time.sleep(2 ** attempt)  # exponential backoff

        return None

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        self._respect_rate_limit()
        headers = {"User-Agent": self.user_agent}
        url = f"{self.BASE_URL}/{endpoint}"
        return self._request_with_retry(url, headers, params)

    def _clean_doi(self, doi: str) -> Optional[str]:
        doi_clean = re.sub(r'^https?://doi\.org/', '', doi).strip()
        if re.match(r'^10\.\d{4,9}/', doi_clean):
            return doi_clean
        return None

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # -----------------------------
    # Core API Methods
    # -----------------------------
    @lru_cache(maxsize=1000)
    def verify_doi(self, doi: str) -> Optional[Dict]:
        doi_clean = self._clean_doi(doi)
        if not doi_clean:
            return None

        endpoint = f"works/{quote(doi_clean)}"
        data = self._get(endpoint)

        if data and data.get("status") == "ok":
            return data.get("message")

        return None

    @lru_cache(maxsize=1000)
    def search_bibliographic(self, query: str) -> List[Dict]:
        params = {
            "query.bibliographic": query,
            "rows": 5,
            "sort": "relevance",
            "order": "desc",
        }

        data = self._get("works", params=params)

        if data and data.get("status") == "ok":
            return data["message"]["items"]

        return []

    # -----------------------------
    # Advanced Citation Verification
    # -----------------------------
    def verify_textual_citation(
        self,
        author: str,
        year: str,
        title_snippet: str = ""
    ) -> Dict:
        """
        Returns structured verification result with confidence score.
        """

        query = f"{author} {year} {title_snippet}".strip()
        matches = self.search_bibliographic(query)

        best_match = None
        best_score = 0

        for m in matches:
            score = 0

            # ------------------
            # Year match
            # ------------------
            pub_year = None
            if "issued" in m and "date-parts" in m["issued"]:
                parts = m["issued"]["date-parts"]
                if parts and parts[0]:
                    pub_year = parts[0][0]

            if pub_year and str(pub_year) == year:
                score += 0.4

            # ------------------
            # Author match
            # ------------------
            if "author" in m:
                authors = [a.get("family", "").lower() for a in m["author"]]
                if author.lower() in authors:
                    score += 0.3

            # ------------------
            # Title similarity
            # ------------------
            if title_snippet and "title" in m and m["title"]:
                title = m["title"][0]
                sim = self._similarity(title_snippet, title)
                score += 0.3 * sim

            # Track best match
            if score > best_score:
                best_score = score
                best_match = m

        return {
            "verified": best_score >= 0.6,
            "confidence": round(best_score, 3),
            "match": {
                "title": best_match["title"][0] if best_match and "title" in best_match else None,
                "doi": best_match.get("DOI") if best_match else None,
                "authors": best_match.get("author") if best_match else None,
                "year": best_match.get("issued") if best_match else None,
            } if best_match else None,
            "raw_matches_found": len(matches),
        }
