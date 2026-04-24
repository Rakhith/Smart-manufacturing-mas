import logging
import json
import re
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class IntelligentSummarizer:
    """
    Cloud LLM-powered intelligent summarization of model results and workflow execution.
    Uses Reflexion loop (draft -> critique -> revise) for high-quality summaries.
    
    Architecture Note (SLM Reduction 4 -> 1):
    - Previously used local SLM for narrative summary generation
    - Now uses Cloud LLM (Gemini) with Reflexion self-critique loop
    - Same model, one extra API turn, no local model dependency for summarization
    - Based on Shinn et al. 2023 Reflexion methodology
    """
    
    def __init__(self, cloud_llm_model=None, llm_agent=None):
        """
        Initialize the IntelligentSummarizer.
        
        Args:
            cloud_llm_model: Google Gemini model instance for Reflexion loop (preferred)
            llm_agent: Legacy local LLM agent (deprecated, kept for backward compatibility)
        """
        self.cloud_llm_model = cloud_llm_model  # Gemini model for Reflexion
        self.llm_agent = llm_agent  # Legacy support
        self.stored_results = {
            'workflow_start_time': None,
            'workflow_end_time': None,
            'dataset_info': {},
            'preprocessing_steps': [],
            'model_results': [],
            'feature_analysis': {},
            'recommendations': {},
            'performance_metrics': {},
            'adaptive_intelligence_events': [],
            'errors': [],
            'summary': None
        }
        self.logging_enabled = True
        self.critique_revision_threshold = 3
        self.local_summary_agent = None
        
    def store_workflow_start(self, dataset_path: str, problem_type: str, target_column: str, feature_columns: List[str]):
        """Store workflow initialization information."""
        self.stored_results['workflow_start_time'] = datetime.now().isoformat()
        self.stored_results['dataset_info'] = {
            'dataset_path': dataset_path,
            'problem_type': problem_type,
            'target_column': target_column,
            'feature_columns': feature_columns
        }
        self._log_info("🚀 Workflow started", f"Dataset: {os.path.basename(dataset_path)}")
        
    def store_preprocessing_step(self, step_name: str, details: Dict[str, Any], duration: float = None):
        """Store preprocessing step information."""
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'step_name': step_name,
            'details': details,
            'duration': duration
        }
        self.stored_results['preprocessing_steps'].append(step_data)
        self._log_info(f"🔧 Preprocessing: {step_name}", f"Duration: {duration:.2f}s" if duration else "")
        
    def store_model_result(self, model_name: str, performance: Dict[str, Any], 
                          adaptive_intelligence: bool = False, tried_models: List[str] = None):
        """Store model training and evaluation results."""
        model_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'performance': performance,
            'adaptive_intelligence_used': adaptive_intelligence,
            'tried_models': tried_models or []
        }
        self.stored_results['model_results'].append(model_data)
        
        # Store adaptive intelligence events separately
        if adaptive_intelligence and tried_models:
            self.stored_results['adaptive_intelligence_events'].append({
                'timestamp': datetime.now().isoformat(),
                'trigger_model': model_name,
                'tried_models': tried_models,
                'final_model': model_name,
                'performance_improvement': performance
            })
        
        # Only log essential information
        if self.logging_enabled:
            if adaptive_intelligence:
                self._log_info(f"🧠 Adaptive Intelligence: {model_name}", 
                             f"Performance: {self._format_performance(performance)}")
            else:
                self._log_info(f"🤖 Model: {model_name}", 
                             f"Performance: {self._format_performance(performance)}")
    
    def store_feature_analysis(self, analysis: Dict[str, Any]):
        """Store intelligent feature analysis results."""
        self.stored_results['feature_analysis'] = analysis
        self._log_info("🧠 Feature Analysis", "Completed intelligent feature analysis")
        
    def store_recommendations(self, recommendations: Dict[str, Any]):
        """Store prescriptive recommendations."""
        self.stored_results['recommendations'] = recommendations
        self._log_info("🎯 Recommendations", "Generated prescriptive recommendations")
        
    def store_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Store error information."""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': error_message,
            'context': context or {}
        }
        self.stored_results['errors'].append(error_data)
        self._log_error(f"❌ {error_type}", error_message)
        
    def store_workflow_end(self):
        """Mark workflow completion."""
        self.stored_results['workflow_end_time'] = datetime.now().isoformat()
        self._log_info("✅ Workflow Complete", "All steps completed successfully")
        
    def generate_intelligent_summary(self) -> str:
        """
        Generate an intelligent summary using Cloud LLM with Reflexion loop.
        
        Reflexion Loop (based on Shinn et al. 2023):
        1. Draft: Generate initial summary from workflow data
        2. Critique: Cloud LLM critiques its own draft against actual metrics
        3. Revise: Generate improved summary incorporating the critique
        
        This replaces the previous local SLM summarization approach,
        eliminating one of the four SLM positions (SLM 4 -> eliminated).
        """
        # Try Cloud LLM Reflexion loop first (preferred approach)
        if self.cloud_llm_model is not None:
            try:
                return self._generate_summary_with_reflexion()
            except Exception as e:
                logging.warning(f"Reflexion loop failed: {e}, trying local Ollama fallback")
                local_summary = self._generate_summary_with_local_reflexion()
                if local_summary:
                    return local_summary
                logging.warning("Local Ollama fallback failed, using basic summary")
                return self._generate_basic_summary()
        
        # Preferred local fallback when Cloud LLM is unavailable
        local_summary = self._generate_summary_with_local_reflexion()
        if local_summary:
            return local_summary

        # Legacy fallback: Use provided local llm_agent if available (deprecated)
        if self.llm_agent is not None:
            try:
                summary_prompt = self._build_summary_prompt()
                response = self.llm_agent.generate(summary_prompt, max_tokens=1000)
                
                if response and 'summary' in response:
                    sanitized_summary = self._sanitize_summary_text(response['summary'])
                    self.stored_results['summary'] = sanitized_summary
                    return sanitized_summary
                else:
                    return self._generate_basic_summary()
            except Exception as e:
                logging.warning(f"Legacy LLM summarization failed: {e}")
                return self._generate_basic_summary()
        
        # Final fallback: Rule-based summary
        return self._generate_basic_summary()

    def _generate_summary_with_local_reflexion(self) -> Optional[str]:
        """Run Reflexion loop on local Ollama qwen3:4b before plain-text fallback."""
        workflow_payload = self._build_workflow_payload()

        draft_payload = self._run_local_reflexion_prompt(workflow_payload, previous_output=None)
        if draft_payload is None:
            return None

        final_payload = self._run_local_reflexion_prompt(workflow_payload, previous_output=draft_payload)
        if final_payload is None:
            self.stored_results['summary'] = draft_payload['summary']
            return draft_payload['summary']

        self.stored_results['summary'] = final_payload['summary']
        self.stored_results['reflexion_metadata'] = {
            'draft': draft_payload,
            'critique': final_payload.get('critique'),
            'final': final_payload,
            'method': 'local_ollama_reflexion_single_prompt',
            'model': 'qwen3:4b'
        }
        return final_payload['summary']
    
    def _generate_summary_with_reflexion(self) -> str:
        """
        Generate summary using a single Reflexion prompt template.

        The same prompt is used twice:
        - First call: no previous_output provided, so the model creates a draft.
        - Second call: previous_output is provided, so the model critiques and improves it.
        """
        logging.info("[Reflexion] Starting Cloud LLM summary generation...")

        workflow_payload = self._build_workflow_payload()

        draft_payload = self._run_reflexion_prompt(workflow_payload, previous_output=None)
        if draft_payload is None:
            return self._generate_basic_summary()

        final_payload = self._run_reflexion_prompt(workflow_payload, previous_output=draft_payload)
        if final_payload is None:
            return draft_payload['summary']

        self.stored_results['summary'] = final_payload['summary']
        self.stored_results['reflexion_metadata'] = {
            'draft': draft_payload,
            'critique': final_payload.get('critique'),
            'final': final_payload,
            'method': 'cloud_llm_reflexion_single_prompt'
        }

        return final_payload['summary']

    def _run_reflexion_prompt(self, workflow_payload: Dict[str, Any], previous_output: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Run one reflexion prompt. If previous_output is provided, the model critiques and improves it."""
        prompt = self._build_reflexion_prompt(workflow_payload, previous_output=previous_output)
        stage = "revision" if previous_output else "draft"
        logging.info(f"[Reflexion] Generating {stage} output with single prompt...")

        try:
            response = self.cloud_llm_model.generate_content(prompt)
            payload = self._parse_json_response(response.text)
            payload = self._coerce_reflexion_payload(payload)
            logging.info(f"[Reflexion] {stage.capitalize()} output generated and validated successfully")
            return payload
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                logging.warning(f"[Reflexion] LLM call failed ({error_msg}); using plain-text fallback.")
            else:
                logging.warning(f"[Reflexion] {stage.capitalize()} generation failed: {e}")
            return None

    def _run_local_reflexion_prompt(self, workflow_payload: Dict[str, Any], previous_output: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Run one reflexion prompt against local Ollama qwen3:4b."""
        prompt = self._build_reflexion_prompt(workflow_payload, previous_output=previous_output)
        stage = "revision" if previous_output else "draft"
        logging.info(f"[Reflexion-Local] Generating {stage} output with Ollama qwen3:4b...")

        try:
            if self.local_summary_agent is None:
                from agents.local_llm_agent import LocalLLMAgent

                self.local_summary_agent = LocalLLMAgent(backend='ollama', model_name='qwen3:4b')

            response = self.local_summary_agent.generate(prompt, max_tokens=1200, temperature=0.2)
            raw_text = (response.get('raw') or '').strip()

            # Strip any thinking tags that may have survived inside the raw string
            # (e.g. when they are embedded in a JSON string value by the model).
            raw_text = self.local_summary_agent._strip_thinking(raw_text)

            json_start = raw_text.find('{')
            json_end = raw_text.rfind('}')

            if json_start != -1 and json_end != -1:
                raw_text = raw_text[json_start:json_end+1]

            payload = self._parse_json_response(raw_text)
            payload = self._coerce_reflexion_payload(payload)
            logging.info(f"[Reflexion-Local] {stage.capitalize()} output generated and validated successfully")
            return payload
        except Exception as e:
            logging.warning(f"[Reflexion-Local] {stage.capitalize()} generation failed: {e}")
            return None

    def _build_reflexion_prompt(self, workflow_payload: Dict[str, Any], previous_output: Optional[Dict[str, Any]] = None) -> str:
        """Build the single prompt used for both draft generation and reflexion refinement."""
        prompt_payload = {
            'role': 'expert_ml_failure_analyst',
            'task': 'Generate or improve an industrial ML workflow summary.',
            'instructions': [
                'Return JSON only.',
                'Do not include prose outside JSON.',
                'Use the exact schema requested below.',
                'If previous_output is empty, create a draft summary from workflow_data.',
                'If previous_output is present, critique it against workflow_data and return a better final summary.',
                'Base all claims on the provided workflow JSON.',
                'If evidence is insufficient, say so explicitly instead of guessing.'
            ],
            "STRICT OUTPUT RULES": [
                "Return ONLY valid JSON.",
                "Do NOT include any reasoning, thinking, or explanations.",
                "Do NOT include phrases like 'Hmm', 'The user wants', or internal thoughts.",
                "Do NOT include <tool_call    > or any hidden reasoning.",
                "If you include anything outside JSON, the output is INVALID."
            ],
            'output_schema': {
                'analysis': {
                    'workflow_status': 'string',
                    'model_performance': 'string',
                    'root_cause': 'string',
                    'key_risks': ['string'],
                    'recommended_actions': ['string']
                },
                'summary': 'string',
                'critique': {
                    'strengths': ['string'],
                    'issues': ['string'],
                    'severity_score': 'integer_0_to_10'
                }
            },
            'workflow_data': workflow_payload,
            'previous_output': previous_output,
            'required_focus': [
                'real-time industrial prescriptive maintenance context',
                'metric accuracy',
                'failure diagnosis',
                'actionability for operations teams'
            ]
        }

        return json.dumps(prompt_payload, indent=2, default=str)

    def _build_workflow_payload(self) -> Dict[str, Any]:
        """Create a structured snapshot of workflow state for LLM prompts."""
        return {
            'dataset_info': self.stored_results.get('dataset_info', {}),
            'model_results': self.stored_results.get('model_results', []),
            'errors': self.stored_results.get('errors', []),
            'feature_analysis': self.stored_results.get('feature_analysis', {}),
            'recommendations': self.stored_results.get('recommendations', {}),
            'performance_metrics': self.stored_results.get('performance_metrics', {}),
            'adaptive_intelligence_events': self.stored_results.get('adaptive_intelligence_events', []),
            'workflow_start_time': self.stored_results.get('workflow_start_time'),
            'workflow_end_time': self.stored_results.get('workflow_end_time')
        }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse model output as JSON with a small amount of recovery logic."""
        if response_text is None:
            raise ValueError('Empty response from model')

        raw_text = self._strip_think_content(response_text.strip())
        if not raw_text:
            raise ValueError('Empty response from model')

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for i, ch in enumerate(raw_text):
                if ch != '{':
                    continue
                try:
                    obj, _ = decoder.raw_decode(raw_text[i:])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
            raise

    def _strip_think_content(self, text: str) -> str:
        """
        Remove model reasoning tags and keep only user-visible content.

        For qwen-style outputs, if one or more </think> tags are present,
        preserve only text after the final closing tag.
        """
        if not text:
            return ""

        if re.search(r"</think>", text, flags=re.IGNORECASE):
            text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]

        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

        return text.strip()

    def _coerce_reflexion_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the single-prompt reflexion output."""
        if not isinstance(payload, dict):
            raise ValueError('Reflexion payload must be a JSON object')

        analysis = payload.get('analysis')
        summary = payload.get('summary')
        if not isinstance(analysis, dict):
            raise ValueError('Reflexion payload missing analysis object')
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('Reflexion payload missing summary text')

        normalized_analysis = {
            'workflow_status': str(analysis.get('workflow_status', 'Unknown')),
            'model_performance': str(analysis.get('model_performance', 'Unknown')),
            'root_cause': str(analysis.get('root_cause', 'Insufficient evidence')),
            'key_risks': self._ensure_string_list(analysis.get('key_risks', [])),
            'recommended_actions': self._ensure_string_list(analysis.get('recommended_actions', []))
        }

        critique = payload.get('critique')
        normalized_critique = None
        if isinstance(critique, dict):
            normalized_critique = {
                'strengths': self._ensure_string_list(critique.get('strengths', [])),
                'issues': self._ensure_string_list(critique.get('issues', [])),
                'severity_score': self._normalize_severity_score(critique.get('severity_score', 0))
            }
        elif critique is not None:
            normalized_critique = {
                'strengths': [],
                'issues': self._ensure_string_list(critique),
                'severity_score': 0
            }

        return {
            'analysis': normalized_analysis,
            'summary': self._sanitize_summary_text(summary.strip()),
            'critique': normalized_critique
        }

    def _ensure_string_list(self, value: Any) -> List[str]:
        """Normalize a value into a list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _normalize_severity_score(self, value: Any) -> int:
        """Normalize critique severity to a bounded integer."""
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            score = 0
        return max(0, min(10, score))

    def _sanitize_summary_text(self, text: str) -> str:
        if not text:
            return ""

        text = self._strip_think_content(text)

        # Remove any remaining stray tags.
        text = re.sub(r"<.*?>", "", text)

        # Remove ALL reasoning patterns
        text = re.sub(r"(?i)hmm,.*?(?=\.)\.", "", text, flags=re.DOTALL)
        text = re.sub(r"(?i)the user wants.*?(?=\.)\.", "", text, flags=re.DOTALL)

        # Remove multi-sentence reasoning blocks at start
        lines = text.strip().splitlines()
        clean_lines = []

        for line in lines:
            l = line.strip().lower()
            if any(l.startswith(x) for x in [
                "hmm", "the user", "i think", "let me", "this means"
            ]):
                continue
            clean_lines.append(line.strip())

        return " ".join(clean_lines).strip()

    def _extract_severity_score(self, critique_payload: Dict[str, Any]) -> int:
        """Read severity from normalized critique payload."""
        return self._normalize_severity_score(critique_payload.get('severity_score', 0))
    
    def _build_summary_prompt(self) -> str:
        """Build a prompt for LLM-based technical summarization."""
        workflow_data = self.stored_results
        
        prompt_parts = [
            "You are an expert data scientist and ML engineer analyzing a machine learning workflow execution.",
            "Provide a detailed TECHNICAL summary focusing on performance, issues, and actionable insights:",
            "",
            "WORKFLOW INFORMATION:",
            f"Dataset: {workflow_data['dataset_info'].get('dataset_path', 'Unknown')}",
            f"Problem Type: {workflow_data['dataset_info'].get('problem_type', 'Unknown')}",
            f"Target Column: {workflow_data['dataset_info'].get('target_column', 'Unknown')}",
            f"Features: {len(workflow_data['dataset_info'].get('feature_columns', []))} columns",
            "",
            "MODEL PERFORMANCE ANALYSIS:"
        ]
        
        # Add detailed model results
        for i, model_result in enumerate(workflow_data['model_results'], 1):
            model_name = model_result['model_name']
            performance = model_result['performance']
            adaptive = model_result['adaptive_intelligence_used']
            tried_models = model_result.get('tried_models', [])
            
            prompt_parts.append(f"Model {i}: {model_name}")
            if 'r2' in performance:
                r2_val = performance['r2']
                if r2_val == float('-inf'):
                    prompt_parts.append(f"  R² Score: FAILED (infinite value - likely data issue)")
                else:
                    prompt_parts.append(f"  R² Score: {r2_val:.4f} ({'EXCELLENT' if r2_val > 0.8 else 'GOOD' if r2_val > 0.5 else 'POOR' if r2_val > 0.1 else 'FAILED'})")
            if 'accuracy' in performance:
                acc_val = performance['accuracy']
                prompt_parts.append(f"  Accuracy: {acc_val:.4f} ({'EXCELLENT' if acc_val > 0.9 else 'GOOD' if acc_val > 0.7 else 'POOR' if acc_val > 0.5 else 'FAILED'})")
            if 'mse' in performance:
                prompt_parts.append(f"  MSE: {performance['mse']:.4f}")
            if adaptive:
                prompt_parts.append(f"  🧠 ADAPTIVE INTELLIGENCE ACTIVATED")
                prompt_parts.append(f"  Tried Models: {', '.join(tried_models) if tried_models else 'None'}")
            prompt_parts.append("")
        
        # Add feature analysis
        if workflow_data['feature_analysis']:
            prompt_parts.extend([
                "FEATURE ANALYSIS:",
                f"Features Analyzed: {len(workflow_data['feature_analysis'].get('recommendations', {}).get('features_to_keep', []))}",
                f"Features Removed: {len(workflow_data['feature_analysis'].get('recommendations', {}).get('features_to_remove', []))}",
                ""
            ])
        
        # Add recommendations
        if workflow_data['recommendations']:
            recs = workflow_data['recommendations'].get('recommendations', [])
            if hasattr(recs, 'empty'):  # DataFrame
                rec_count = len(recs) if not recs.empty else 0
            else:
                rec_count = len(recs) if recs else 0
            prompt_parts.extend([
                "RECOMMENDATIONS:",
                f"Total Recommendations: {rec_count}",
                ""
            ])
        
        # Add errors if any
        if workflow_data['errors']:
            prompt_parts.extend([
                "ERRORS:",
                f"Total Errors: {len(workflow_data['errors'])}",
                ""
            ])
        
        prompt_parts.extend([
            "TECHNICAL ANALYSIS REQUIRED:",
            "Provide a detailed technical summary covering:",
            "",
            "1. WORKFLOW STATUS:",
            "   - Success/failure rate and critical issues",
            "   - Data quality and preprocessing effectiveness",
            "   - Model performance analysis with specific metrics",
            "",
            "2. ADAPTIVE INTELLIGENCE ANALYSIS:",
            "   - Whether adaptive intelligence was triggered and why",
            "   - Models tried and their individual performance",
            "   - Root cause of any failures",
            "",
            "3. FEATURE ENGINEERING INSIGHTS:",
            "   - Features removed/kept and reasoning",
            "   - Data leakage prevention measures",
            "   - Correlation and multicollinearity issues",
            "",
            "4. TECHNICAL ISSUES & DIAGNOSIS:",
            "   - Specific error analysis and root causes",
            "   - Data quality problems identified",
            "   - Model convergence issues",
            "",
            "5. ACTIONABLE RECOMMENDATIONS:",
            "   - Immediate fixes needed",
            "   - Data preprocessing improvements",
            "   - Alternative modeling approaches",
            "",
            "Format as detailed technical report with specific metrics, error codes, and actionable insights.",
            "Respond with JSON: {\"summary\": \"detailed technical summary here\"}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_basic_summary(self) -> str:
        """Generate a basic summary without LLM."""
        workflow_data = self.stored_results
        
        summary_lines = [
            "=" * 60,
            "INTELLIGENT WORKFLOW SUMMARY",
            "=" * 60,
            f"Dataset: {os.path.basename(workflow_data['dataset_info'].get('dataset_path', 'Unknown'))}",
            f"Problem Type: {workflow_data['dataset_info'].get('problem_type', 'Unknown')}",
            f"Features: {len(workflow_data['dataset_info'].get('feature_columns', []))} columns",
            ""
        ]
        
        # Model performance
        if workflow_data['model_results']:
            summary_lines.append("🤖 MODEL PERFORMANCE:")
            for i, model_result in enumerate(workflow_data['model_results'], 1):
                model_name = model_result['model_name']
                performance = model_result['performance']
                adaptive = model_result['adaptive_intelligence_used']
                
                summary_lines.append(f"  {i}. {model_name}")
                if 'r2' in performance:
                    summary_lines.append(f"     R² Score: {performance['r2']:.4f} ({performance['r2']*100:.1f}%)")
                if 'accuracy' in performance:
                    summary_lines.append(f"     Accuracy: {performance['accuracy']:.4f} ({performance['accuracy']*100:.1f}%)")
                if 'mse' in performance:
                    summary_lines.append(f"     MSE: {performance['mse']:.4f}")
                if adaptive:
                    summary_lines.append(f"     🧠 Adaptive Intelligence: Tried {len(model_result.get('tried_models', []))} models")
                summary_lines.append("")
        
        # Feature analysis
        if workflow_data['feature_analysis']:
            recs = workflow_data['feature_analysis'].get('recommendations', {})
            summary_lines.extend([
                "🧠 FEATURE ANALYSIS:",
                f"  Features Kept: {len(recs.get('features_to_keep', []))}",
                f"  Features Removed: {len(recs.get('features_to_remove', []))}",
                ""
            ])
        
        # Recommendations
        if workflow_data['recommendations']:
            recs = workflow_data['recommendations'].get('recommendations', [])
            if hasattr(recs, 'empty'):  # DataFrame
                rec_count = len(recs) if not recs.empty else 0
            else:
                rec_count = len(recs) if recs else 0
            summary_lines.extend([
                "🎯 RECOMMENDATIONS:",
                f"  Total Generated: {rec_count}",
                ""
            ])
        
        # Errors
        if workflow_data['errors']:
            summary_lines.extend([
                "⚠️ ISSUES:",
                f"  Errors Encountered: {len(workflow_data['errors'])}",
                ""
            ])
        
        # Duration
        if workflow_data['workflow_start_time'] and workflow_data['workflow_end_time']:
            start_time = datetime.fromisoformat(workflow_data['workflow_start_time'])
            end_time = datetime.fromisoformat(workflow_data['workflow_end_time'])
            duration = (end_time - start_time).total_seconds()
            summary_lines.append(f"⏱️ Total Duration: {duration:.2f} seconds")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
    
    def _format_performance(self, performance: Dict[str, Any]) -> str:
        """Format performance metrics for logging."""
        if 'r2' in performance:
            return f"R²: {performance['r2']:.4f}"
        elif 'accuracy' in performance:
            return f"Accuracy: {performance['accuracy']:.4f}"
        else:
            return "Performance metrics available"
    
    def _log_info(self, title: str, message: str = ""):
        """Log information with clean formatting."""
        if self.logging_enabled:
            if message:
                logging.info(f"[Summarizer] {title}: {message}")
            else:
                logging.info(f"[Summarizer] {title}")
    
    def _log_error(self, title: str, message: str):
        """Log error with clean formatting."""
        if self.logging_enabled:
            logging.error(f"[Summarizer] {title}: {message}")
    
    def save_detailed_results(self, filepath: str = None) -> str:
        """Save detailed results to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/detailed_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.stored_results, f, indent=4, default=str)
        
        return filepath
    
    def disable_logging(self):
        """Disable verbose logging during workflow execution."""
        self.logging_enabled = False
    
    def enable_logging(self):
        """Re-enable logging."""
        self.logging_enabled = True
    
    def set_logging_mode(self, mode: str = "verbose"):
        """
        Set logging mode for the summarizer.
        - "verbose": Log all information (default)
        - "minimal": Only log essential information
        - "silent": No logging from summarizer
        """
        if mode == "verbose":
            self.logging_enabled = True
        elif mode == "minimal":
            self.logging_enabled = True
        elif mode == "silent":
            self.logging_enabled = False
        else:
            self.logging_enabled = True

def create_summarizer(cloud_llm_model=None, llm_agent=None) -> IntelligentSummarizer:
    """
    Factory function to create an IntelligentSummarizer instance.
    
    Args:
        cloud_llm_model: Google Gemini model instance for Reflexion loop (preferred).
                         Pass the model object from LLMPlannerAgent.
        llm_agent: Legacy local LLM agent (deprecated, kept for backward compatibility)
    
    Returns:
        IntelligentSummarizer: Configured summarizer instance
    
    Note (SLM Reduction 4 -> 1):
        The summarizer now uses Cloud LLM Reflexion loop instead of local SLM.
        This eliminates SLM position 4 (narrative summary) from the architecture.
    """
    return IntelligentSummarizer(cloud_llm_model=cloud_llm_model, llm_agent=llm_agent)