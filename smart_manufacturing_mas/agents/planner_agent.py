"""
agents/planner_agent.py
------------------------
Rule-based fallback planner. Used when the Cloud LLM (Gemini) is unavailable.
Executes a hard-coded 4-step sequential workflow without LLM reasoning or HITL gates.

Architecture note: This is the emergency baseline. The primary orchestrator
is LLMPlannerAgent (LLM-driven) or RulesFirstPlannerAgent (rules-first + LLM summary).
"""

import logging
import pandas as pd
from agents.data_loader_agent import DataLoaderAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent
from agents.optimization_agent import OptimizationAgent

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class PlannerAgent:
    """
    Deterministic rule-based planner — emergency fallback when no LLM is available.

    Executes the four pipeline steps in fixed order:
      1. Perception   — load & inspect data
      2. Preprocessing — clean & feature-engineer
      3. Analysis      — train model
      4. Optimization  — generate recommendations
    """

    def __init__(self, dataset_path: str, target_column: str, problem_type: str = "classification"):
        logging.info("Initializing Rule-Based Planner Agent (fallback)...")
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.raw_data = None
        self.preprocessed_data = None
        self.analysis_results = None
        self.recommendations = None

    def run_workflow(self):
        """Execute the full workflow sequentially."""
        logging.info("--- Starting Rule-Based Fallback Workflow ---")

        if not self._execute_perception_step():
            logging.error("Halted: Perception Layer failed.")
            return
        if not self._execute_preprocessing_step():
            logging.error("Halted: Preprocessing Layer failed.")
            return
        if not self._execute_analysis_step():
            logging.error("Halted: Analysis Layer failed.")
            return
        if not self._execute_optimization_step():
            logging.warning("Optimization step failed or produced no recommendations.")

        logging.info("--- Rule-Based Fallback Workflow Finished ---")

    def _execute_perception_step(self) -> bool:
        logging.info("[Planner] Executing Perception Layer...")
        agent = DataLoaderAgent(self.dataset_path)
        self.raw_data = agent.load_data()
        if self.raw_data is None:
            return False
        agent.inspect_data()
        logging.info("[Planner] Perception Layer completed.")
        return True

    def _execute_preprocessing_step(self) -> bool:
        logging.info("[Planner] Executing Preprocessing Layer...")
        if self.target_column not in self.raw_data.columns:
            logging.error(f"Target column '{self.target_column}' not found.")
            return False

        target_data = self.raw_data[[self.target_column]]
        feature_data = self.raw_data.drop(columns=[self.target_column])

        agent = PreprocessingAgent(feature_data, target_column=self.target_column, problem_type=self.problem_type)
        preprocessed_features = agent.preprocess()

        if preprocessed_features is None:
            return False

        self.preprocessed_data = pd.concat([preprocessed_features, target_data], axis=1)
        logging.info("[Planner] Preprocessing Layer completed.")
        return True

    def _execute_analysis_step(self) -> bool:
        logging.info("[Planner] Executing Analysis Layer...")
        agent = DynamicAnalysisAgent(
            self.preprocessed_data,
            target_column=self.target_column,
            task=self.problem_type,
        )
        results = agent.run()

        if results is None:
            return False

        feature_importances = None
        if results.get('feature_importances') is not None and results.get('feature_names') is not None:
            import pandas as pd
            feature_importances = pd.DataFrame({
                'feature': results['feature_names'],
                'importance': results['feature_importances']
            }).sort_values(by='importance', ascending=False)

        self.analysis_results = {
            'evaluation':        results,
            'feature_importances': feature_importances,
            'test_data_features':  results.get('X_test'),
            'test_predictions':    results.get('predictions'),
        }
        logging.info("[Planner] Analysis Layer completed.")
        return True

    def _execute_optimization_step(self) -> bool:
        logging.info("[Planner] Executing Optimization Layer...")
        if self.analysis_results is None:
            return False

        X_test = self.analysis_results['test_data_features']
        if X_test is None:
            logging.error("No X_test available for optimization.")
            return False

        original_ctx = self.raw_data.loc[X_test.index]
        payload = {
            'test_data':          original_ctx,
            'test_predictions':   self.analysis_results['test_predictions'],
            'feature_importances': self.analysis_results['feature_importances'],
        }

        agent = OptimizationAgent(payload)
        self.recommendations = agent.generate_recommendations()
        if self.recommendations is None:
            return False

        logging.info("[Planner] Optimization Layer completed.")
        return True


if __name__ == '__main__':
    logging.info("--- Running PlannerAgent in Standalone Mode ---")
    planner = PlannerAgent(
        dataset_path="data/Smart_Manufacturing_Maintenance_Dataset/smart_maintenance_dataset.csv",
        target_column="Maintenance_Priority",
        problem_type="classification",
    )
    planner.run_workflow()
