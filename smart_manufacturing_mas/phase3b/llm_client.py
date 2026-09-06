"""Multi-backend LLM client abstraction for Phase 3B LLM-as-Judge."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BaseLLMClient(ABC):
    """Abstract interface for LLM-as-Judge providers."""

    @abstractmethod
    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        """Executes LLM request and returns (raw_json_response_text, execution_metadata)."""
        pass


class GeminiRESTClient(BaseLLMClient):
    """Direct HTTPS REST client for Google Gemini API using urllib.request (zero external deps)."""

    _rate_lock = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 6,
        backoff_base_sec: float = 4.0,
        min_interval_sec: float = 4.0,
    ):
        import threading
        self.model = model
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.min_interval_sec = min_interval_sec
        self.api_key = api_key or self._resolve_api_key()
        self._last_call_time = 0.0
        if GeminiRESTClient._rate_lock is None:
            GeminiRESTClient._rate_lock = threading.Lock()

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please provide via constructor, "
                "set GEMINI_API_KEY environment variable, or add to .env file."
            )

    def _resolve_api_key(self) -> Optional[str]:
        # 1. Environment variable
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key.strip()

        # 2. Check local .env file
        for env_path in [Path(".env"), Path(__file__).resolve().parents[1] / ".env"]:
            if env_path.exists():
                try:
                    content = env_path.read_text(encoding="utf-8")
                    match = re.search(r"GEMINI_API_KEY\s*=\s*(.+)", content)
                    if match:
                        return match.group(1).strip()
                except Exception:
                    pass
        return None

    def _pace_request(self) -> None:
        """Ensures consecutive requests respect Gemini free-tier rate limits (~15 RPM)."""
        if self._rate_lock:
            with self._rate_lock:
                elapsed = time.time() - self._last_call_time
                if elapsed < self.min_interval_sec:
                    time.sleep(self.min_interval_sec - elapsed)
                self._last_call_time = time.time()

    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": temperature,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        data_bytes = json.dumps(payload).encode("utf-8")

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            self._pace_request()
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_bytes = resp.read()
                    resp_json = json.loads(resp_bytes.decode("utf-8"))
                    latency_ms = int((time.time() - start_time) * 1000)

                    # Extract candidate text
                    candidates = resp_json.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"Empty candidate list returned: {resp_json}")

                    raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    meta = {
                        "provider": "gemini_rest",
                        "model": self.model,
                        "temperature": temperature,
                        "latency_ms": latency_ms,
                        "attempts": attempt + 1,
                        "finish_reason": candidates[0].get("finishReason"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    return raw_text, meta

            except urllib.error.HTTPError as e:
                last_error = e
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass

                # If rate limited (429), back off at least 15s to let quota replenish
                if e.code == 429:
                    sleep_time = max(15.0, self.backoff_base_sec * (2 ** attempt))
                    time.sleep(sleep_time)
                elif e.code in [500, 503, 504]:
                    sleep_time = self.backoff_base_sec * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Gemini API error (HTTP {e.code}): {err_body}")

            except (urllib.error.URLError, TimeoutError) as e:
                last_error = e
                sleep_time = self.backoff_base_sec * (2 ** attempt)
                time.sleep(sleep_time)

        raise RuntimeError(f"Failed to generate after {self.max_retries} attempts. Last error: {last_error}")


class HeuristicMockClient(BaseLLMClient):
    """Deterministic offline mock judge for unit testing, offline benchmarking, and zero-cost validation."""

    def __init__(self, model_name: str = "mock-industrial-expert-v1"):
        self.model_name = model_name

    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        # Parse candidates from prompt JSON block
        candidates_match = re.search(r"### Candidate Maintenance Actions.*?(\[.*?\])", prompt, re.DOTALL)
        candidate_ids: List[str] = []
        if candidates_match:
            try:
                cand_list = json.loads(candidates_match.group(1))
                candidate_ids = [c["action_id"] for c in cand_list]
            except Exception:
                pass

        if not candidate_ids:
            # Fallback regex extraction
            candidate_ids = re.findall(r'"action_id":\s*"([^"]+)"', prompt)

        # Detect severity from prompt
        severity = "WATCH"
        for s in ["CRITICAL", "DEGRADING", "HEALTHY", "WATCH"]:
            if f'"decision_severity": "{s}"' in prompt or f"Severity Tier: {s}" in prompt:
                severity = s
                break

        evaluations: List[Dict[str, Any]] = []

        # Deterministic scoring based on severity and action category
        def score_candidate(aid: str, sev: str) -> Tuple[int, int, int, int, float, str]:
            if sev == "CRITICAL":
                if "REPL" in aid or "SHUTDOWN" in aid or "EMERGENCY" in aid:
                    return (92, 95, 90, 40, 0.90, "RECOMMENDED")
                elif "DERATE" in aid or "INSP" in aid:
                    return (74, 80, 75, 25, 0.85, "ACCEPTABLE_ALTERNATIVE")
                elif "CONTINUE" in aid:
                    return (12, 10, 15, 95, 0.95, "UNSAFE")
                else:
                    return (45, 60, 50, 30, 0.75, "INAPPROPRIATE_AT_CURRENT_TIME")

            elif sev == "DEGRADING":
                if "INSP" in aid or "CORR" in aid:
                    return (88, 80, 85, 25, 0.85, "RECOMMENDED")
                elif "REPL" in aid or "DERATE" in aid:
                    return (78, 75, 80, 35, 0.80, "ACCEPTABLE_ALTERNATIVE")
                elif "ENHANCED" in aid:
                    return (65, 55, 60, 20, 0.75, "ACCEPTABLE_ALTERNATIVE")
                else:
                    return (25, 20, 30, 70, 0.80, "INAPPROPRIATE_AT_CURRENT_TIME")

            elif sev == "WATCH":
                if "ENHANCED" in aid or "INSP" in aid or "LOG" in aid:
                    return (89, 70, 85, 15, 0.85, "RECOMMENDED")
                elif "ADJUST" in aid or "CORR" in aid:
                    return (72, 60, 75, 20, 0.80, "ACCEPTABLE_ALTERNATIVE")
                elif "CONTINUE" in aid:
                    return (55, 30, 50, 40, 0.75, "ACCEPTABLE_ALTERNATIVE")
                else:
                    return (20, 35, 40, 75, 0.85, "INAPPROPRIATE_AT_CURRENT_TIME")

            else:  # HEALTHY
                if "CONTINUE" in aid:
                    return (95, 15, 90, 5, 0.95, "RECOMMENDED")
                elif "ENHANCED" in aid or "LOG" in aid:
                    return (75, 25, 70, 10, 0.85, "ACCEPTABLE_ALTERNATIVE")
                elif "INSP" in aid:
                    return (60, 20, 65, 15, 0.80, "ACCEPTABLE_ALTERNATIVE")
                else:
                    return (15, 30, 20, 85, 0.90, "INAPPROPRIATE_AT_CURRENT_TIME")

        scored_candidates = []
        for aid in candidate_ids:
            suit, urg, eff, risk, conf, verd = score_candidate(aid, severity)
            scored_candidates.append({
                "action_id": aid,
                "suitability_score": suit,
                "urgency_score": urg,
                "expected_effectiveness_score": eff,
                "operational_risk_score": risk,
                "confidence": conf,
                "evidence_used": ["telemetry_severity_baseline", "subsystem_anomaly_profile"],
                "reasoning_summary": f"Action evaluated against {severity} machine conditions.",
                "unsupported_assumptions": [],
                "final_verdict": verd,
            })

        # Rank candidates by suitability_score descending
        scored_candidates.sort(key=lambda c: c["suitability_score"], reverse=True)
        for rank_idx, c in enumerate(scored_candidates, 1):
            c["rank"] = rank_idx

        top_action = scored_candidates[0]["action_id"] if scored_candidates else ""
        alternatives = [c["action_id"] for c in scored_candidates[1:3]]

        mock_payload = {
            "evaluations": scored_candidates,
            "top_recommended_action": top_action,
            "alternative_actions": alternatives,
            "insufficient_information": False,
            "uncertainty_explanation": "",
        }

        raw_text = json.dumps(mock_payload, indent=2)
        latency_ms = int((time.time() - start_time) * 1000)

        meta = {
            "provider": "heuristic_mock",
            "model": self.model_name,
            "temperature": temperature,
            "latency_ms": latency_ms,
            "attempts": 1,
            "finish_reason": "STOP",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return raw_text, meta
