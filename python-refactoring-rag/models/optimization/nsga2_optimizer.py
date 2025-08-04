"""NSGA-II multi-objective optimization for model selection."""

import logging
import time
from typing import Dict, List, Any
import numpy as np

from config.settings import OptimizationConfig

logger = logging.getLogger(__name__)

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    logger.warning("pymoo not available. Install with: pip install pymoo")
    PYMOO_AVAILABLE = False


class ModelSelectionProblem(Problem):
    """Multi-objective optimization problem for model selection."""
    
    def __init__(self, evaluation_results: Dict[str, List[Dict]]):
        """Initialize the optimization problem."""
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        self.evaluation_results = evaluation_results
        self.model_names = list(evaluation_results.keys())
        self.n_models = len(self.model_names)
        
        if self.n_models == 0:
            raise ValueError("No models found in evaluation results")
        
        logger.info(f"Initializing NSGA-2 with {self.n_models} models: {self.model_names}")
        
        # Calculate and validate model metrics
        self.model_metrics = self._calculate_model_metrics()
        self._validate_metrics()
        
        # Decision variable: single integer representing model choice (encoded as float)
        super().__init__(
            n_var=1,                    # Single decision variable: model index
            n_obj=6,                    # 6 objectives
            n_constr=0,                 # No constraints
            xl=np.array([0.0]),         # Lower bound: first model
            xu=np.array([float(self.n_models - 1)]),  # Upper bound: last model
            elementwise_evaluation=False
        )
        
        logger.info(f"Problem initialized: 1 variable, 6 objectives, 0 constraints")
    
    def _calculate_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate model metrics with robustness."""
        model_metrics = {}
        
        for model_name, results in self.evaluation_results.items():
            successful_results = [r for r in results if r.get('success') and r.get('rag_metrics')]
            
            if successful_results:
                try:
                    # Collect all metrics with defaults
                    context_rel = []
                    answer_rel = []
                    faithfulness = []
                    completeness = []
                    bleu_scores = []
                    rouge_scores = []
                    
                    for r in successful_results:
                        rag = r['rag_metrics']
                        context_rel.append(self._safe_float(rag.context_relevance, 0.1))
                        answer_rel.append(self._safe_float(rag.answer_relevance, 0.1))
                        faithfulness.append(self._safe_float(rag.faithfulness, 0.1))
                        completeness.append(self._safe_float(rag.response_completeness, 0.1))
                        bleu_scores.append(self._safe_float(getattr(rag, 'bleu_score', 0.01), 0.01))
                        rouge_scores.append(self._safe_float(getattr(rag, 'rouge_l_score', 0.01), 0.01))
                    
                    # Calculate robust averages
                    metrics = {
                        'context_relevance': max(0.01, np.mean(context_rel)),
                        'answer_relevance': max(0.01, np.mean(answer_rel)),
                        'faithfulness': max(0.01, np.mean(faithfulness)),
                        'response_completeness': max(0.01, np.mean(completeness)),
                        'bleu_score': max(0.001, np.mean(bleu_scores)),
                        'rouge_l_score': max(0.001, np.mean(rouge_scores))
                    }
                    
                    logger.info(f"Metrics for {model_name}: {metrics}")
                    
                except Exception as e:
                    logger.error(f"Error calculating metrics for {model_name}: {e}")
                    metrics = self._get_safe_default_metrics()
            else:
                metrics = self._get_safe_default_metrics()
                logger.warning(f"No successful results for {model_name}, using defaults")
            
            model_metrics[model_name] = metrics
        
        return model_metrics
    
    def _safe_float(self, value, default: float) -> float:
        """Safely convert to float with default."""
        try:
            result = float(value)
            return result if np.isfinite(result) and result >= 0 else default
        except:
            return default
    
    def _get_safe_default_metrics(self) -> Dict[str, float]:
        """Get guaranteed safe default metrics."""
        return {
            'context_relevance': 0.05,
            'answer_relevance': 0.05,
            'faithfulness': 0.05,
            'response_completeness': 0.05,
            'bleu_score': 0.005,
            'rouge_l_score': 0.005
        }
    
    def _validate_metrics(self):
        """Validate all metrics are valid."""
        for model_name, metrics in self.model_metrics.items():
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    logger.error(f"Invalid metric {metric_name} for {model_name}: {value}")
                    self.model_metrics[model_name][metric_name] = 0.01
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluation function for NSGA-II optimization."""
        try:
            # Handle batch evaluation
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            batch_size = X.shape[0]
            
            # Initialize objective matrix with safe defaults
            objectives_batch = np.full((batch_size, 6), 0.5, dtype=float)
            
            for i, solution in enumerate(X):
                try:
                    # Convert continuous variable to discrete model selection
                    model_index = int(np.round(np.clip(solution[0], 0, self.n_models - 1)))
                    model_index = min(max(model_index, 0), self.n_models - 1)
                    
                    selected_model = self.model_names[model_index]
                    metrics = self.model_metrics[selected_model]
                    
                    # Create objectives (NEGATIVE for maximization)
                    # Use log transformation for better scaling
                    objectives = [
                        -np.log(max(0.001, metrics['context_relevance'])),
                        -np.log(max(0.001, metrics['answer_relevance'])),
                        -np.log(max(0.001, metrics['faithfulness'])),
                        -np.log(max(0.001, metrics['response_completeness'])),
                        -np.log(max(0.0001, metrics['bleu_score'])),
                        -np.log(max(0.0001, metrics['rouge_l_score']))
                    ]
                    
                    # Validate objectives
                    for j, obj in enumerate(objectives):
                        if not np.isfinite(obj):
                            objectives[j] = 1.0  # Safe default
                    
                    objectives_batch[i] = objectives
                    
                except Exception as e:
                    logger.warning(f"Error evaluating solution {i}: {e}, using defaults")
                    objectives_batch[i] = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0]
            
            # Set output
            out["F"] = objectives_batch.astype(float)
            
        except Exception as e:
            logger.error(f"Critical error in _evaluate: {e}")
            # Emergency fallback
            batch_size = X.shape[0] if X.ndim > 1 else 1
            out["F"] = np.ones((batch_size, 6), dtype=float)


class NSGA2ModelSelector:
    """NSGA-2 model selector with robust optimization."""
    
    def __init__(self, evaluation_results: Dict[str, List[Dict]]):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for NSGA-2 optimization. Install with: pip install pymoo")
        
        self.evaluation_results = evaluation_results
        
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        self.problem = ModelSelectionProblem(evaluation_results)
        logger.info("NSGA-2 selector initialized successfully")
    
    def optimize_model_selection(self, 
                                config: OptimizationConfig = None,
                                verbose: bool = True) -> Dict:
        """Run NSGA-2 optimization for model selection."""
        if config is None:
            config = OptimizationConfig()
        
        logger.info(f"Starting NSGA-2 optimization with {config.n_generations} generations, "
                   f"population {config.population_size}")
        
        try:
            # Handle single model case
            if self.problem.n_models == 1:
                return self._single_model_result()
            
            # Configure NSGA-2
            algorithm = NSGA2(
                pop_size=max(config.population_size, 20),
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=config.crossover_prob, eta=config.crossover_eta),
                mutation=PM(prob=config.mutation_prob, eta=config.mutation_eta),
                eliminate_duplicates=True
            )
            
            # Simple termination
            termination = get_termination("n_gen", config.n_generations)
            
            start_time = time.time()
            
            logger.info("Starting NSGA-2 optimization...")
            
            # Run optimization
            res = minimize(
                self.problem,
                algorithm,
                termination,
                seed=42,
                verbose=verbose,
                save_history=False
            )
            
            optimization_time = time.time() - start_time
            
            logger.info(f"NSGA-2 completed in {optimization_time:.2f}s")
            
            # Validate results
            if not self._validate_nsga2_result(res):
                logger.warning("NSGA-2 result validation failed, using fallback")
                return self._smart_fallback_selection()
            
            # Process the results
            return self._process_results(res, optimization_time, config.n_generations)
            
        except Exception as e:
            logger.error(f"Error in NSGA-2 optimization: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._smart_fallback_selection()
    
    def _validate_nsga2_result(self, res) -> bool:
        """Comprehensive validation of NSGA-2 results."""
        try:
            if res is None:
                logger.error("Result object is None")
                return False
            
            if not hasattr(res, 'F') or res.F is None:
                logger.error("res.F is None or missing")
                return False
            
            if not hasattr(res, 'X') or res.X is None:
                logger.error("res.X is None or missing")
                return False
            
            if not isinstance(res.F, np.ndarray) or res.F.size == 0:
                logger.error(f"res.F is invalid: type={type(res.F)}, size={getattr(res.F, 'size', 'N/A')}")
                return False
            
            if not isinstance(res.X, np.ndarray) or res.X.size == 0:
                logger.error(f"res.X is invalid: type={type(res.X)}, size={getattr(res.X, 'size', 'N/A')}")
                return False
            
            if not np.all(np.isfinite(res.F)):
                logger.error("res.F contains non-finite values")
                return False
            
            if not np.all(np.isfinite(res.X)):
                logger.error("res.X contains non-finite values")
                return False
            
            logger.info(f"NSGA-2 result validation passed: F.shape={res.F.shape}, X.shape={res.X.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error during result validation: {e}")
            return False
    
    def _single_model_result(self) -> Dict:
        """Handle single model case."""
        model_name = list(self.evaluation_results.keys())[0]
        metrics = self.problem.model_metrics[model_name]
        
        return {
            'best_model': model_name,
            'best_model_objectives': metrics,
            'pareto_front_size': 1,
            'all_pareto_solutions': [{'model': model_name, 'objectives': metrics, 'rank': 0}],
            'optimization_time_seconds': 0.0,
            'algorithm': 'Single_Model_Direct',
            'convergence_info': {'final_generation': 0, 'population_size': 1}
        }
    
    def _process_results(self, res, optimization_time: float, n_generations: int) -> Dict:
        """Process NSGA-2 results."""
        try:
            pareto_front = res.F
            pareto_solutions = res.X
            
            logger.info(f"Processing {len(pareto_solutions)} Pareto solutions")
            
            # Convert solutions back to model selections
            model_selections = []
            for i, (solution, objectives) in enumerate(zip(pareto_solutions, pareto_front)):
                try:
                    # Get model index from solution
                    model_index = int(np.round(np.clip(solution[0], 0, self.problem.n_models - 1)))
                    model_index = min(max(model_index, 0), self.problem.n_models - 1)
                    selected_model = self.problem.model_names[model_index]
                    
                    # Convert objectives back to original scale
                    converted_objectives = {
                        'context_relevance': float(np.exp(-objectives[0])),
                        'answer_relevance': float(np.exp(-objectives[1])),
                        'faithfulness': float(np.exp(-objectives[2])),
                        'response_completeness': float(np.exp(-objectives[3])),
                        'bleu_score': float(np.exp(-objectives[4])),
                        'rouge_l_score': float(np.exp(-objectives[5]))
                    }
                    
                    model_selections.append({
                        'model': selected_model,
                        'objectives': converted_objectives,
                        'rank': i,
                        'raw_objectives': objectives.tolist(),
                        'solution_vector': solution.tolist()
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing solution {i}: {e}")
                    continue
            
            if not model_selections:
                logger.error("No valid model selections after processing")
                return self._smart_fallback_selection()
            
            # Select best compromise solution
            best_solution = self._select_compromise(model_selections)
            
            return {
                'best_model': best_solution['model'],
                'best_model_objectives': best_solution['objectives'],
                'pareto_front_size': len(model_selections),
                'all_pareto_solutions': model_selections,
                'optimization_time_seconds': optimization_time,
                'algorithm': 'NSGA2',
                'convergence_info': {
                    'final_generation': n_generations,
                    'population_size': len(pareto_solutions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return self._smart_fallback_selection()
    
    def _select_compromise(self, model_selections: List[Dict]) -> Dict:
        """Select best compromise using robust method."""
        if len(model_selections) == 1:
            return model_selections[0]
        
        try:
            # Simple weighted sum with equal weights
            best_score = -float('inf')
            best_solution = model_selections[0]
            
            for solution in model_selections:
                obj = solution['objectives']
                # Equal weight combination
                score = (obj['context_relevance'] + obj['answer_relevance'] + 
                        obj['faithfulness'] + obj['response_completeness'] + 
                        obj['bleu_score'] + obj['rouge_l_score']) / 6.0
                
                if score > best_score:
                    best_score = score
                    best_solution = solution
            
            logger.info(f"Selected compromise: {best_solution['model']} (score: {best_score:.4f})")
            return best_solution
            
        except Exception as e:
            logger.error(f"Error in compromise selection: {e}")
            return model_selections[0]
    
    def _smart_fallback_selection(self) -> Dict:
        """Smart fallback when NSGA-2 fails."""
        logger.warning("Using smart fallback selection")
        
        try:
            best_model = None
            best_score = -float('inf')
            
            for model_name, metrics in self.problem.model_metrics.items():
                # Weighted sum with emphasis on core metrics
                score = (0.3 * metrics['context_relevance'] + 
                        0.25 * metrics['answer_relevance'] + 
                        0.25 * metrics['faithfulness'] + 
                        0.2 * metrics['response_completeness'])
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model is None:
                best_model = list(self.problem.model_metrics.keys())[0]
            
            return {
                'best_model': best_model,
                'best_model_objectives': self.problem.model_metrics[best_model],
                'pareto_front_size': 1,
                'all_pareto_solutions': [{'model': best_model, 'objectives': self.problem.model_metrics[best_model]}],
                'optimization_time_seconds': 0.0,
                'algorithm': 'Smart_Fallback_Weighted_Sum',
                'convergence_info': {'final_generation': 0, 'population_size': 1}
            }
            
        except Exception as e:
            logger.error(f"Error in smart fallback: {e}")
            # Ultimate emergency fallback
            first_model = list(self.evaluation_results.keys())[0]
            return {
                'best_model': first_model,
                'best_model_objectives': {},
                'pareto_front_size': 1,
                'all_pareto_solutions': [],
                'optimization_time_seconds': 0.0,
                'algorithm': 'Emergency_Fallback',
                'convergence_info': {'final_generation': 0, 'population_size': 1}
            }


def run_nsga2_optimization(evaluation_results: Dict[str, List[Dict]], 
                          config: OptimizationConfig = None) -> Dict:
    """
    Run NSGA-2 optimization for model selection.
    
    Args:
        evaluation_results: Dictionary mapping model names to evaluation results
        config: Optimization configuration
        
    Returns:
        Dictionary containing optimization results
    """
    try:
        logger.info("Starting NSGA-2 Multi-Objective Optimization")
        
        # Initialize selector
        selector = NSGA2ModelSelector(evaluation_results)
        
        # Run optimization
        optimization_results = selector.optimize_model_selection(
            config=config,
            verbose=True
        )
        
        logger.info("NSGA-2 optimization completed successfully")
        return optimization_results
        
    except Exception as e:
        logger.error(f"Critical error in NSGA-2 optimization: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise