"""Multi-backend LLM client abstraction for Phase 3B LLM-as-Judge."""

from __future__ import annotations

import http.client
import json
import os
import re
import threading
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
        model: str = "gemini-3.5-flash-lite",
        max_retries: int = 6,
        backoff_base_sec: float = 4.0,
        min_interval_sec: float = 2.0,
        raise_on_429: bool = False,
    ):
        import threading
        self.model = model
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.min_interval_sec = min_interval_sec
        self.raise_on_429 = raise_on_429
        self.api_key = api_key or self._resolve_api_key()
        self.provider_name = "gemini_rest"
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
                    match = re.search(r"^[ \t]*GEMINI_API_KEY[ \t]*=[ \t]*([^\r\n#]+)", content, flags=re.MULTILINE)
                    if match:
                        val = match.group(1).strip()
                        if val and not val.startswith("your_"):
                            return val
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

                    # Defensive candidate extraction
                    candidates = resp_json.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"Empty candidate list returned: {resp_json}")

                    cand0 = candidates[0] if isinstance(candidates[0], dict) else {}
                    content_obj = cand0.get("content") if isinstance(cand0.get("content"), dict) else {}
                    parts = content_obj.get("parts") if isinstance(content_obj.get("parts"), list) else []
                    part0 = parts[0] if parts and isinstance(parts[0], dict) else {}
                    raw_text = part0.get("text") or ""
                    if not raw_text.strip():
                        raise ValueError(f"Empty or null text part returned from Gemini: {cand0}")

                    meta = {
                        "provider": "gemini_rest",
                        "model": self.model,
                        "temperature": temperature,
                        "latency_ms": latency_ms,
                        "attempts": attempt + 1,
                        "finish_reason": cand0.get("finishReason", "STOP"),
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

                # If rate limited (429), raise immediately if raise_on_429 is set (for fast waterfall cascade)
                if e.code == 429:
                    if self.raise_on_429:
                        raise RateLimitError(f"Gemini rate limit exceeded (HTTP 429): {err_body}") from e
                    sleep_time = max(10.0, self.backoff_base_sec * (1.5 ** attempt))
                    time.sleep(sleep_time)
                elif e.code in [500, 502, 503, 504]:
                    sleep_time = self.backoff_base_sec * (1.5 ** attempt)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Gemini API error (HTTP {e.code}): {err_body}")

            except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError, http.client.IncompleteRead) as e:
                last_error = e
                sleep_time = self.backoff_base_sec * (1.5 ** attempt)
                time.sleep(sleep_time)

        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts. Last error: {last_error}")


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


class OllamaClient(BaseLLMClient):
    """Local Ollama client for completely offline, zero-rate-limit LLM-as-Judge evaluations."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout_sec: float = 300.0,
        max_retries: int = 3,
        backoff_base_sec: float = 2.0,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.endpoint = f"{self.host}/api/generate"
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec

    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_instruction:
            payload["system"] = system_instruction

        data_bytes = json.dumps(payload).encode("utf-8")
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.endpoint,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    resp_bytes = resp.read()
                    resp_json = json.loads(resp_bytes.decode("utf-8"))
                    latency_ms = int((time.time() - start_time) * 1000)

                    raw_text = resp_json.get("response", "")
                    # Fallback to 'thinking' if 'response' is empty (happens in some thinking models under json mode)
                    if not raw_text.strip() and resp_json.get("thinking"):
                        raw_text = resp_json.get("thinking", "")

                    # Strip thinking tags if generated by hybrid/thinking models
                    if "<think>" in raw_text:
                        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

                    eval_count = resp_json.get("eval_count", 0)
                    eval_duration_ns = resp_json.get("eval_duration", 1)
                    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

                    meta = {
                        "provider": "ollama",
                        "model": self.model_name,
                        "temperature": temperature,
                        "latency_ms": latency_ms,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "eval_count": eval_count,
                        "attempts": attempt + 1,
                        "finish_reason": "STOP" if resp_json.get("done") else "INCOMPLETE",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    return raw_text, meta

            except urllib.error.URLError as e:
                last_error = e
                err_str = str(e).lower()
                if "connection refused" in err_str or "winerror 10061" in err_str:
                    raise RuntimeError(
                        f"Cannot connect to Ollama daemon at {self.host}. "
                        "Please ensure the Ollama application or 'ollama serve' is running."
                    ) from e
                sleep_time = self.backoff_base_sec * (2 ** attempt)
                time.sleep(sleep_time)

            except Exception as e:
                last_error = e
                sleep_time = self.backoff_base_sec * (2 ** attempt)
                time.sleep(sleep_time)

        raise RuntimeError(
            f"Ollama generation failed after {self.max_retries} attempts for model '{self.model_name}'. "
            f"Last error: {last_error}"
        )


class RateLimitError(RuntimeError):
    """Raised when an LLM provider returns HTTP 429 (rate limit / quota exceeded)."""
    pass


def resolve_api_key(key_name: str) -> Optional[str]:
    """Helper to resolve API key from environment variable or local .env files."""
    env_val = os.getenv(key_name)
    if env_val and env_val.strip():
        return env_val.strip()

    for env_path in [Path(".env"), Path(__file__).resolve().parents[1] / ".env"]:
        if env_path.exists():
            try:
                content = env_path.read_text(encoding="utf-8")
                match = re.search(rf"^[ \t]*{key_name}[ \t]*=[ \t]*([^\r\n#]+)", content, flags=re.MULTILINE)
                if match:
                    val = match.group(1).strip()
                    if val and not val.startswith("your_"):
                        return val
            except Exception:
                pass
    return None


class OpenAICompatibleRESTClient(BaseLLMClient):
    """Generic high-performance REST client for OpenAI-compatible endpoints (Groq, OpenRouter, GLM, etc.)."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        provider_name: str = "openai_compatible",
        fallback_models: Optional[List[str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        backoff_base_sec: float = 2.0,
        min_interval_sec: float = 0.0,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name
        self.fallback_models = fallback_models or []
        self.extra_headers = extra_headers or {}
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.min_interval_sec = min_interval_sec
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def _pace(self) -> None:
        if self.min_interval_sec > 0:
            with self._lock:
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
        self._pace()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        # OpenRouter native model cascading: pass "models" array in priority order (max 3 allowed by OpenRouter API)
        if self.fallback_models and "openrouter" in self.provider_name.lower():
            all_models = [self.model] + [m for m in self.fallback_models if m != self.model]
            payload["models"] = all_models[:3]

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 SmartManufacturingMAS/1.0",
        }
        headers.update(self.extra_headers)

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.endpoint,
                data=data_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    resp_bytes = resp.read()
                    resp_json = json.loads(resp_bytes.decode("utf-8"))
                    latency_ms = int((time.time() - start_time) * 1000)

                    choices = resp_json.get("choices", [])
                    if not choices:
                        raise ValueError(f"Empty choices from {self.provider_name}: {resp_json}")

                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
                    raw_text = msg.get("content") or ""
                    if not raw_text.strip():
                        raise ValueError(f"Empty or null content returned from {self.provider_name}: {choices[0]}")
                    if "<think>" in raw_text:
                        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

                    actual_model = resp_json.get("model", self.model)
                    usage = resp_json.get("usage", {})
                    eval_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                    tps = round((eval_tokens / (latency_ms / 1000.0)), 1) if latency_ms > 0 and eval_tokens > 0 else 0.0

                    meta = {
                        "provider": self.provider_name,
                        "model": actual_model,
                        "temperature": temperature,
                        "latency_ms": latency_ms,
                        "tokens_per_sec": tps,
                        "eval_count": eval_tokens,
                        "attempts": attempt + 1,
                        "finish_reason": choice0.get("finish_reason", "STOP"),
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

                if e.code == 429:
                    raise RateLimitError(f"{self.provider_name} rate limit exceeded (HTTP 429): {err_body}") from e
                elif e.code in [500, 502, 503, 504]:
                    sleep_time = self.backoff_base_sec * (1.5 ** attempt)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"{self.provider_name} HTTP {e.code} error: {err_body}") from e

            except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError, http.client.IncompleteRead) as e:
                last_error = e
                sleep_time = self.backoff_base_sec * (1.5 ** attempt)
                time.sleep(sleep_time)

        raise RuntimeError(
            f"{self.provider_name} failed after {self.max_retries} attempts. Last error: {last_error}"
        )


class GroqClient(OpenAICompatibleRESTClient):
    """Ultra-fast LPU inference client via Groq Cloud (~500+ tokens/sec)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3.8-27b",
        timeout_sec: float = 60.0,
    ):
        key = api_key or resolve_api_key("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found in constructor or environment.")
        super().__init__(
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key=key,
            model=model,
            provider_name="groq",
            timeout_sec=timeout_sec,
            min_interval_sec=2.0,  # strict 30 RPM free tier pacing (60s / 30 = 2.0s)
        )


class OpenRouterClient(OpenAICompatibleRESTClient):
    """OpenRouter client with automatic multi-model cascading and fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nex-agi/nex-n2.5-pro:free",
        fallback_models: Optional[List[str]] = None,
        timeout_sec: float = 60.0,
    ):
        key = api_key or resolve_api_key("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not found in constructor or environment.")

        if fallback_models is None:
            raw_fallbacks = os.getenv(
                "OPENROUTER_FALLBACK_MODELS",
                "google/gemma-4-26b-a4b-it:free,z-ai/glm-5.2:free,qwen/qwen3.8-27b:free",
            )
            fallback_models = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]

        super().__init__(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key=key,
            model=model,
            provider_name="openrouter",
            fallback_models=fallback_models,
            extra_headers={
                "HTTP-Referer": "https://github.com/Rakhith/Smart-manufacturing-mas",
                "X-Title": "Smart Manufacturing MAS LLM-as-Judge",
            },
            timeout_sec=timeout_sec,
        )


class GLMClient(BaseLLMClient):
    """GLM / Zhipu AI / Z.AI client with automatic free-tier fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-5.3-flash",
        timeout_sec: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or resolve_api_key("GLM_API_KEY") or resolve_api_key("ZHIPUAI_API_KEY")
        self.timeout_sec = timeout_sec
        self.provider_name = "glm"

        # Primary client: Z.AI direct API (free flash model: glm-5.3-flash)
        self._primary_client = None
        if self.api_key:
            self._primary_client = OpenAICompatibleRESTClient(
                endpoint="https://api.z.ai/api/paas/v4/chat/completions",
                api_key=self.api_key,
                model=model,
                provider_name="glm_zai",
                timeout_sec=timeout_sec,
            )

        # Secondary fallback: OpenRouter free GLM (z-ai/glm-5.2:free)
        self._openrouter_key = resolve_api_key("OPENROUTER_API_KEY")
        self._fallback_client = None
        if self._openrouter_key:
            self._fallback_client = OpenRouterClient(
                api_key=self._openrouter_key,
                model="z-ai/glm-5.2:free",
                timeout_sec=timeout_sec,
            )

        if not self._primary_client and not self._fallback_client:
            raise ValueError("Neither GLM_API_KEY nor OPENROUTER_API_KEY found for GLM evaluations.")

    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        # 1. Try primary Z.AI client
        if self._primary_client:
            try:
                raw_text, meta = self._primary_client.generate_structured_evaluation(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                meta["provider"] = "glm"
                return raw_text, meta
            except Exception as e:
                err_str = str(e)
                # If Z.AI returns 1113 (no balance/resource pack), cascade to OpenRouter free GLM
                if "1113" in err_str or "余额不足" in err_str or "Insufficient balance" in err_str:
                    if self._fallback_client:
                        try:
                            raw_text, meta = self._fallback_client.generate_structured_evaluation(
                                prompt=prompt,
                                system_instruction=system_instruction,
                                temperature=temperature,
                            )
                            meta["provider"] = "glm_openrouter_free"
                            meta["model"] = "z-ai/glm-5.2:free"
                            meta["fallback_recovered"] = True
                            return raw_text, meta
                        except Exception:
                            pass
                    raise RuntimeError(
                        "Z.AI API returned Code 1113 ('Insufficient balance or no resource package'). "
                        "To use GLM directly, please claim your free resource package at https://z.ai/user-center/billing. "
                        "Alternatively, OpenRouter (z-ai/glm-5.2:free) is temporarily rate-limited."
                    ) from e
                raise
        elif self._fallback_client:
            return self._fallback_client.generate_structured_evaluation(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
            )
        raise RuntimeError("No working GLM client available.")


class MultiProviderDispatcherClient(BaseLLMClient):
    """High-throughput multi-provider gateway with round-robin dispatch and automatic 429 failover."""

    def __init__(self, clients: Optional[List[BaseLLMClient]] = None):
        if clients:
            self.clients = clients
        else:
            self.clients = self.auto_discover_available_clients()

        if not self.clients:
            raise ValueError(
                "No LLM providers available! Please configure at least one API key (GROQ_API_KEY, "
                "OPENROUTER_API_KEY, GEMINI_API_KEY) in .env, or ensure local Ollama is running."
            )
        self._round_robin_idx = 0
        self._lock = threading.Lock()
        self._disabled_providers: set[str] = set()

    @classmethod
    def auto_discover_available_clients(cls, include_local: bool = False) -> List[BaseLLMClient]:
        """Discovers ultra-fast cloud providers (Gemini, Groq, OpenRouter). Excludes high-latency local Ollama from batch dispatcher unless requested."""
        discovered: List[BaseLLMClient] = []

        # 1. Gemini (Cloud REST - gemini-3.5-flash-lite, ultra-reliable ~1.1s)
        try:
            discovered.append(GeminiRESTClient())
        except Exception:
            pass

        # 2. Groq (Ultra-fast LPU ~0.25s)
        try:
            discovered.append(GroqClient())
        except Exception:
            pass

        # 3. OpenRouter (Multi-model free tier ~1.8s)
        try:
            discovered.append(OpenRouterClient())
        except Exception:
            pass

        # 4. GLM (Zhipu AI) - only if explicitly enabled (avoiding Code 1113 balance errors)
        if os.getenv("PHASE3B_INCLUDE_GLM", "0") == "1":
            try:
                discovered.append(GLMClient())
            except Exception:
                pass

        # 5. Local Ollama (Only included if explicitly requested, as local 7B takes ~135s per evaluation pass)
        if include_local or os.getenv("PHASE3B_INCLUDE_OLLAMA", "0") == "1":
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags")
                with urllib.request.urlopen(req, timeout=1.0):
                    discovered.append(OllamaClient(model_name="qwen2.5:7b"))
            except Exception:
                pass

        # (HeuristicMockClient is NOT added to the active round-robin pool;
        # it is reserved strictly as the final safety-net fallback in generate_structured_evaluation)
        return discovered

    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        with self._lock:
            start_idx = self._round_robin_idx % len(self.clients)
            self._round_robin_idx += 1

        last_error = None
        for i in range(len(self.clients)):
            client_idx = (start_idx + i) % len(self.clients)
            client = self.clients[client_idx]
            pname = getattr(client, "provider_name", getattr(client, "model_name", "client"))
            if pname in self._disabled_providers:
                continue

            try:
                raw_text, meta = client.generate_structured_evaluation(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                meta["failover_hops"] = i
                return raw_text, meta
            except Exception as e:
                err_str = str(e)
                if "余额不足" in err_str or "1113" in err_str:
                    self._disabled_providers.add(pname)
                last_error = e
                # Fail over seamlessly to next client in pool
                continue

        # Final safety net: Heuristic mock client guarantees valid evaluation
        fallback = HeuristicMockClient()
        raw_text, meta = fallback.generate_structured_evaluation(prompt, system_instruction, temperature)
        meta["failover_hops"] = len(self.clients)
        meta["fallback_recovered"] = True
        return raw_text, meta


class GitHubModelsClient(OpenAICompatibleRESTClient):
    """GitHub Models endpoint (free 15 RPM for all GitHub users with personal access token)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", timeout_sec: float = 60.0):
        key = api_key or resolve_api_key("GITHUB_TOKEN") or resolve_api_key("GH_TOKEN")
        if not key:
            raise ValueError("GITHUB_TOKEN not found in environment or .env file.")
        super().__init__(
            endpoint="https://models.inference.ai.azure.com/chat/completions",
            api_key=key,
            model=model,
            provider_name="github_models",
            timeout_sec=timeout_sec,
            min_interval_sec=2.0,
        )


class HuggingFaceClient(OpenAICompatibleRESTClient):
    """Hugging Face Serverless Inference API (OpenAI-compatible router)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Llama-3.1-8B-Instruct", timeout_sec: float = 60.0):
        key = api_key or resolve_api_key("HF_TOKEN") or resolve_api_key("HUGGINGFACE_API_KEY")
        if not key:
            raise ValueError("HF_TOKEN / HUGGINGFACE_API_KEY not found in environment or .env file.")
        super().__init__(
            endpoint="https://router.huggingface.co/v1/chat/completions",
            api_key=key,
            model=model,
            provider_name="huggingface",
            timeout_sec=timeout_sec,
            min_interval_sec=1.0,
        )


class WaterfallCascadingClient(BaseLLMClient):
    """Waterfall cascading LLM client:
    Exhausts fast cloud providers in strict priority order before falling back to local Ollama.
    Tier 1: Groq Cloud LPU (qwen/qwen3.8-27b) @ ~0.25s
    Tier 2: GitHub Models (gpt-4o-mini / llama-3.3-70b) if GITHUB_TOKEN set
    Tier 3: Hugging Face Serverless (Qwen2.5-7B-Instruct) if HF_TOKEN set
    Tier 4: Gemini Cloud REST (gemini-3.5-flash-lite) @ ~1.2s
    Tier 5: OpenRouter Free Tier (if quota available)
    Tier 6: Local Ollama (qwen3:4b on localhost:11434) - zero-rate-limit local inference
    Tier 7: Heuristic Mock - emergency fail-safe if Ollama daemon is dead
    """

    def __init__(self, ollama_model: str = "qwen3:4b", ollama_host: str = "http://localhost:11434"):
        self.tiers: List[Tuple[str, BaseLLMClient]] = []
        self._rate_limited_until: Dict[str, float] = {}
        self._lock = threading.Lock()

        # 1. Groq (Tier 1)
        try:
            self.tiers.append(("groq", GroqClient(model="qwen/qwen3.8-27b")))
        except Exception:
            pass

        # 2. GitHub Models (Tier 2 - disabled by default as models API was retired, enabled via flag)
        if os.getenv("PHASE3B_ENABLE_GITHUB_MODELS", "0") == "1":
            try:
                self.tiers.append(("github_models", GitHubModelsClient()))
            except Exception:
                pass

        # 3. Hugging Face Serverless (Tier 3 - if token present)
        try:
            self.tiers.append(("huggingface", HuggingFaceClient()))
        except Exception:
            pass

        # 4. Gemini (Tier 4)
        try:
            self.tiers.append(("gemini", GeminiRESTClient(model="gemini-3.5-flash-lite", raise_on_429=True)))
        except Exception:
            pass

        # 5. OpenRouter (Tier 5)
        try:
            self.tiers.append(("openrouter", OpenRouterClient()))
        except Exception:
            pass

        # 6. Local Ollama (Tier 6 - Local fallback with zero rate limits, default qwen3:4b)
        try:
            self.tiers.append(("ollama", OllamaClient(model_name=ollama_model, host=ollama_host)))
        except Exception:
            pass

        # 7. Emergency Mock (Tier 7)
        self.emergency_fallback = HeuristicMockClient()

    def generate_structured_evaluation(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        now = time.time()
        last_error = None

        for tier_idx, (name, client) in enumerate(self.tiers):
            # Check if provider is currently in rate-limit backoff window
            with self._lock:
                cooldown = self._rate_limited_until.get(name, 0.0)
                if now < cooldown:
                    continue

            try:
                raw_text, meta = client.generate_structured_evaluation(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                meta["waterfall_tier"] = tier_idx + 1
                meta["waterfall_provider"] = name
                return raw_text, meta

            except Exception as e:
                err_str = str(e)
                last_error = e
                # Check for rate limit or quota exhaustion (HTTP 429)
                if "tokens per day" in err_str.lower() or "tpd" in err_str.lower():
                    with self._lock:
                        self._rate_limited_until[name] = time.time() + 600.0
                elif "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
                    # Set 60-second cooldown on this provider so subsequent calls cascade immediately
                    with self._lock:
                        self._rate_limited_until[name] = time.time() + 60.0
                elif "1113" in err_str or "余额不足" in err_str or "free-models-per-day" in err_str:
                    # Daily quota exhausted, set long cooldown (1 hour)
                    with self._lock:
                        self._rate_limited_until[name] = time.time() + 3600.0

                # Cascade immediately to next tier
                continue

        # If all tiers fail, use emergency mock
        raw_text, meta = self.emergency_fallback.generate_structured_evaluation(prompt, system_instruction, temperature)
        meta["waterfall_tier"] = len(self.tiers) + 1
        meta["waterfall_provider"] = "emergency_mock"
        meta["emergency_fallback"] = True
        return raw_text, meta



