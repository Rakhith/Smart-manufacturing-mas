"""
agents/local_llm_agent.py
--------------------------
Local LLM adapter for the MAS framework.

Supports four backends:
  'mock'         — deterministic sequence mock for unit tests / CI.
  'ollama'       — local HTTP endpoint via ollama Python client or CLI.
                   Recommended for factory-floor edge deployment (zero cloud).
  'transformers' — full HuggingFace transformers pipeline (R&D use).
  'llamacpp'     — CPU-friendly GGUF quantised models via llama-cpp-python.
                   Install: pip install llama-cpp-python
                   Best for resource-constrained edge nodes without GPU.

All backends expose the same generate() interface:
    agent = LocalLLMAgent(backend='ollama', model_name='qwen3:4b')
    result = agent.generate(prompt)
    # {'raw': str, 'tool': str|None, 'reason': str, 'parsed': dict|None}

Architecture note (SLM Reduction 4 → 1):
  In production this adapter is used ONLY for the single retained SLM position —
  SLM 3b (anomaly detection hyperparameter suggestion).
  All other decisions use the deterministic ToolDecider or Cloud LLM.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

SUPPORTED_BACKENDS = ("mock", "ollama", "transformers", "llamacpp")


class LocalLLMAgent:
    """Backend-agnostic wrapper around local language models."""

    def __init__(self, backend: str = "mock", model_name: Optional[str] = None):
        """
        Args:
            backend   : 'mock' | 'ollama' | 'transformers' | 'llamacpp'
            model_name:
              - ollama      : model tag, e.g. 'qwen3:4b'
              - transformers: HuggingFace model id or local path
              - llamacpp    : absolute path to a .gguf file
              - mock        : ignored
        """
        self.backend = backend
        self.model_name = model_name
        logging.info(f"Initializing LocalLLMAgent — backend={backend}, model_name={model_name}")

        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Choose: {SUPPORTED_BACKENDS}")

        # ── transformers ──────────────────────────────────────────────────────
        if backend == "transformers":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                logging.info("Transformers backend initialised.")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialise transformers backend: {exc}") from exc

        # ── ollama ────────────────────────────────────────────────────────────
        elif backend == "ollama":
            try:
                import ollama as _ollama  # type: ignore
                self._ollama_client = _ollama
                self._use_ollama_cli = False
                logging.info("Ollama Python client detected.")
            except ImportError:
                logging.warning("Python 'ollama' package not found; will call CLI via subprocess.")
                self._ollama_client = None
                self._use_ollama_cli = True

        # ── llamacpp ──────────────────────────────────────────────────────────
        elif backend == "llamacpp":
            try:
                from llama_cpp import Llama  # type: ignore
                self._llama = Llama(model_path=model_name, n_ctx=2048, verbose=False)
                logging.info(f"LlamaCpp model loaded: {model_name}")
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to initialise LlamaCpp backend: {exc}") from exc

        # ── mock ──────────────────────────────────────────────────────────────
        elif backend == "mock":
            logging.info("Using mock backend.")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Generate a response for the given prompt.

        Returns:
            dict: {'raw': str, 'tool': str|None, 'reason': str, 'parsed': dict|None}
        """
        dispatch = {
            "mock":         self._generate_mock,
            "transformers": self._generate_transformers,
            "ollama":       self._generate_ollama,
            "llamacpp":     self._generate_llamacpp,
        }
        return dispatch[self.backend](prompt, max_tokens, temperature)

    # ── Backends ──────────────────────────────────────────────────────────────

    def _generate_mock(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Deterministic mock following the expected 4-step workflow sequence."""
        import re
        step_match = re.search(r"current step:\s*(\d+)", prompt.lower())
        step_num = int(step_match.group(1)) if step_match else 1

        sequence = {
            1: ("load_and_inspect_data",    "Starting workflow by loading data.",           False),
            2: ("preprocess_data",          "Data loaded, now preprocessing.",              False),
            3: ("analyze_data",             "Data preprocessed, now analysing.",            False),
            4: ("generate_recommendations", "Analysis complete, generating recommendations.", True),
        }
        tool, reason, finish = sequence.get(step_num, ("generate_recommendations", "Workflow complete.", True))
        resp = {"tool": tool, "reason": reason, "finish": finish}
        return {"raw": json.dumps(resp), "tool": tool, "reason": reason, "parsed": resp}

    def _generate_transformers(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=temperature
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._wrap(text)

    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        try:
            if self._ollama_client is not None and not self._use_ollama_cli:
                response = self._ollama_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={"num_predict": max_tokens, "temperature": temperature},
                )
                if isinstance(response, dict):
                    text = (
                        response.get("response")
                        or response.get("output")
                        or response.get("text")
                        or str(response)
                    )
                else:
                    text = getattr(response, "response", str(response))
            else:
                import subprocess
                proc = subprocess.run(
                    ["ollama", "run", self.model_name, prompt],
                    capture_output=True, text=True, timeout=120,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"Ollama CLI error: {proc.stderr.strip()}")
                text = proc.stdout.strip()
            return self._wrap(text)
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    def _generate_llamacpp(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Inference via llama-cpp-python (GGUF models).
        CPU-friendly — no GPU required. Ideal for resource-constrained edge nodes.
        """
        try:
            output = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n", "```"],
                echo=False,
            )
            text = output["choices"][0]["text"].strip()
            return self._wrap(text)
        except Exception as exc:
            raise RuntimeError(f"LlamaCpp generation failed: {exc}") from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _wrap(self, text: str) -> Dict[str, Any]:
        """Parse JSON from raw output and return standardised result dict."""
        parsed = self._extract_json(text)
        if parsed:
            return {"raw": text, "tool": parsed.get("tool"), "reason": parsed.get("reason", ""), "parsed": parsed}
        return {"raw": text, "tool": text.strip(), "reason": "", "parsed": None}

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from free-form model output."""
        if not text:
            return None

        # Prefer fenced JSON blocks if present.
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fenced:
            try:
                obj = json.loads(fenced.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Fallback: scan from every '{' and decode the first valid JSON object.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None


if __name__ == "__main__":
    agent = LocalLLMAgent(backend="mock")
    result = agent.generate("Current step: 1. Analyse anomalies in the dataset.")
    print(json.dumps(result, indent=2))
