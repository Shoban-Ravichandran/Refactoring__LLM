"""NSGA-II multi-objective optimization for model selection with proper before/after comparison."""

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
    """Multi-objective optimization problem for model selection with baseline tracking."""
    
    def __init__(self, evaluation_results: Dict[str, List[Dict]]):
        """Initialize the optimization problem."""
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        self.evaluation_results = evaluation_results
        self.model_names = list(evaluation_results.keys())
        self.n_models = len(self.model_names)
        
        if self.n_models == 0:
            raise ValueError("No models found in evaluation results")
        
        logger.info(f"Initializing NSGA-II with {self.n_models} models: {self.model_names}")
        
        # Calculate model metrics and establish baseline
        self.model_metrics = self._calculate_model_metrics()
        self.baseline_metrics = self._calculate_baseline_metrics()
        self._validate_metrics()
        
        # Decision variable: single integer representing model choice
        super().__init__(
            n_var=1,
            n_obj=4,  # 4 objectives: performance, consistency, diversity, efficiency
            n_constr=0,
            xl=np.array([0.0]),
            xu=np.array([float(self.n_models - 1)]),
            elementwise_evaluation=False
        )
        
        logger.info(f"Problem initialized with baseline: {self.baseline_metrics}")
    
    def _calculate_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive model metrics."""
        model_metrics = {}
        
        for model_name, results in self.evaluation_results.items():
            successful_results = [r for r in results if r.get('success') and r.get('rag_metrics')]
            
            if successful_results:
                try:
                    # Collect all metrics
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
                    
                    # Calculate both mean and standard deviation for consistency
                    metrics = {
                        'context_relevance': max(0.01, np.mean(context_rel)),
                        'answer_relevance': max(0.01, np.mean(answer_rel)),
                        'faithfulness': max(0.01, np.mean(faithfulness)),
                        'response_completeness': max(0.01, np.mean(completeness)),
                        'bleu_score': max(0.001, np.mean(bleu_scores)),
                        'rouge_l_score': max(0.001, np.mean(rouge_scores)),
                        # Consistency metrics (lower std = more consistent)
                        'context_relevance_std': np.std(context_rel),
                        'answer_relevance_std': np.std(answer_rel),
                        'faithfulness_std': np.std(faithfulness),
                        'response_completeness_std': np.std(completeness),
                        'success_rate': len(successful_results) / len(results)
                    }
                    
                    # Calculate overall performance and consistency
                    core_metrics = ['context_relevance', 'answer_relevance', 'faithfulness', 'response_completeness']
                    metrics['overall_performance'] = np.mean([metrics[m] for m in core_metrics])
                    metrics['overall_consistency'] = 1.0 - np.mean([metrics[f'{m}_std'] for m in core_metrics])
                    
                    logger.info(f"Metrics for {model_name}: Performance={metrics['overall_performance']:.3f}, "
                               f"Consistency={metrics['overall_consistency']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error calculating metrics for {model_name}: {e}")
                    metrics = self._get_safe_default_metrics()
            else:
                metrics = self._get_safe_default_metrics()
                logger.warning(f"No successful results for {model_name}, using defaults")
            
            model_metrics[model_name] = metrics
        
        return model_metrics
    
    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline metrics (equal weight ensemble or worst performing model)."""
        # Method 1: Equal weight ensemble baseline
        equal_weight_metrics = {}
        
        # Calculate equal-weighted averages across all models
        all_metrics = list(self.model_metrics.values())
        if all_metrics:
            for key in all_metrics[0].keys():
                if not key.endswith('_std'):  # Skip std metrics for baseline
                    equal_weight_metrics[key] = np.mean([metrics[key] for metrics in all_metrics])
        
        # Method 2: Worst performing model as baseline
        worst_performance = float('inf')
        worst_model_metrics = None
        
        for model_name, metrics in self.model_metrics.items():
            if metrics['overall_performance'] < worst_performance:
                worst_performance = metrics['overall_performance']
                worst_model_metrics = metrics
        
        # Use the more conservative baseline (worse performance)
        if worst_model_metrics and worst_model_metrics['overall_performance'] < equal_weight_metrics.get('overall_performance', 0):
            baseline = worst_model_metrics.copy()
            baseline_type = f"worst_model_baseline"
        else:
            baseline = equal_weight_metrics
            baseline_type = "equal_weight_baseline"
        
        logger.info(f"Using {baseline_type} with performance: {baseline.get('overall_performance', 0):.3f}")
        return baseline
    
    def _safe_float(self, value, default: float) -> float:
        """Safely convert to float with default."""
        try:
            result = float(value)
            return result if np.isfinite(result) and result >= 0 else default
        except:
            return default
    
    def _get_safe_default_metrics(self) -> Dict[str, float]:
        """Get safe default metrics."""
        return {
            'context_relevance': 0.05,
            'answer_relevance': 0.05,
            'faithfulness': 0.05,
            'response_completeness': 0.05,
            'bleu_score': 0.005,
            'rouge_l_score': 0.005,
            'context_relevance_std': 0.1,
            'answer_relevance_std': 0.1,
            'faithfulness_std': 0.1,
            'response_completeness_std': 0.1,
            'success_rate': 0.1,
            'overall_performance': 0.05,
            'overall_consistency': 0.1
        }
    
    def _validate_metrics(self):
        """Validate all metrics are valid."""
        for model_name, metrics in self.model_metrics.items():
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    logger.error(f"Invalid metric {metric_name} for {model_name}: {value}")
                    self.model_metrics[model_name][metric_name] = 0.01
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluation function for NSGA-II optimization."""
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            batch_size = X.shape[0]
            objectives_batch = np.full((batch_size, 4), 0.5, dtype=float)
            
            for i, solution in enumerate(X):
                try:
                    # Convert continuous variable to discrete model selection
                    model_index = int(np.round(np.clip(solution[0], 0, self.n_models - 1)))
                    model_index = min(max(model_index, 0), self.n_models - 1)
                    
                    selected_model = self.model_names[model_index]
                    metrics = self.model_metrics[selected_model]
                    
                    # Create objectives (NEGATIVE for maximization)
                    objectives = [
                        -metrics['overall_performance'],  # Maximize performance
                        -metrics['overall_consistency'],   # Maximize consistency  
                        -metrics['success_rate'],          # Maximize success rate
                        -(metrics['bleu_score'] + metrics['rouge_l_score'])  # Maximize text quality
                    ]
                    
                    # Validate objectives
                    for j, obj in enumerate(objectives):
                        if not np.isfinite(obj):
                            objectives[j] = 0.0
                    
                    objectives_batch[i] = objectives
                    
                except Exception as e:
                    logger.warning(f"Error evaluating solution {i}: {e}")
                    objectives_batch[i] = [0.0, 0.0, 0.0, 0.0]
            
            out["F"] = objectives_batch.astype(float)
            
        except Exception as e:
            logger.error(f"Critical error in _evaluate: {e}")
            batch_size = X.shape[0] if X.ndim > 1 else 1
            out["F"] = np.zeros((batch_size, 4), dtype=float)


class NSGA2ModelSelector:
    """NSGA-2 model selector with proper before/after comparison."""
    
    def __init__(self, evaluation_results: Dict[str, List[Dict]]):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for NSGA-2 optimization")
        
        self.evaluation_results = evaluation_results
        
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        self.problem = ModelSelectionProblem(evaluation_results)
        logger.info("NSGA-2 selector initialized successfully")
    
    def optimize_model_selection(self, 
                                config: OptimizationConfig = None,
                                verbose: bool = True) -> Dict:
        """Run NSGA-2 optimization with proper before/after tracking."""
        if config is None:
            config = OptimizationConfig()
        
        logger.info(f"Starting NSGA-2 optimization with baseline comparison")
        
        try:
            # Store baseline for comparison
            baseline_metrics = self.problem.baseline_metrics
            
            # Handle single model case
            if self.problem.n_models == 1:
                return self._single_model_result_with_baseline(baseline_metrics)
            
            # Configure NSGA-2
            algorithm = NSGA2(
                pop_size=max(config.population_size, 20),
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=config.crossover_prob, eta=config.crossover_eta),
                mutation=PM(prob=config.mutation_prob, eta=config.mutation_eta),
                eliminate_duplicates=True
            )
            
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
                return self._smart_fallback_with_baseline(baseline_metrics)
            
            # Process results with baseline comparison
            return self._process_results_with_baseline(res, optimization_time, baseline_metrics, config.n_generations)
            
        except Exception as e:
            logger.error(f"Error in NSGA-2 optimization: {e}")
            return self._smart_fallback_with_baseline(self.problem.baseline_metrics)
    
    def _validate_nsga2_result(self, res) -> bool:
        """Validate NSGA-2 results."""
        try:
            if res is None or not hasattr(res, 'F') or not hasattr(res, 'X'):
                return False
            
            if res.F is None or res.X is None:
                return False
            
            if not isinstance(res.F, np.ndarray) or res.F.size == 0:
                return False
            
            if not isinstance(res.X, np.ndarray) or res.X.size == 0:
                return False
            
            if not np.all(np.isfinite(res.F)) or not np.all(np.isfinite(res.X)):
                return False
            
            logger.info(f"NSGA-2 validation passed: Pareto front size = {len(res.F)}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating NSGA-2 results: {e}")
            return False
    
    def _process_results_with_baseline(self, res, optimization_time: float, 
                                     baseline_metrics: Dict[str, float], n_generations: int) -> Dict:
        """Process results with proper baseline comparison."""
        try:
            pareto_front = res.F
            pareto_solutions = res.X
            
            logger.info(f"Processing {len(pareto_solutions)} Pareto solutions")
            
            # Convert solutions back to model selections
            model_selections = []
            for i, (solution, objectives) in enumerate(zip(pareto_solutions, pareto_front)):
                try:
                    model_index = int(np.round(np.clip(solution[0], 0, self.problem.n_models - 1)))
                    model_index = min(max(model_index, 0), self.problem.n_models - 1)
                    selected_model = self.problem.model_names[model_index]
                    
                    # Get actual model metrics (not negated objectives)
                    model_metrics = self.problem.model_metrics[selected_model]
                    
                    model_selections.append({
                        'model': selected_model,
                        'metrics': model_metrics,
                        'objectives': objectives.tolist(),
                        'rank': i
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing solution {i}: {e}")
                    continue
            
            if not model_selections:
                logger.error("No valid model selections after processing")
                return self._smart_fallback_with_baseline(baseline_metrics)
            
            # Select best compromise solution
            best_solution = self._select_best_compromise(model_selections)
            best_model_metrics = best_solution['metrics']
            
            # Calculate meaningful improvements
            improvements = self._calculate_improvements(baseline_metrics, best_model_metrics)
            
            return {
                'best_model': best_solution['model'],
                'baseline_metrics': baseline_metrics,
                'optimized_metrics': best_model_metrics,
                'improvements': improvements,
                'pareto_front_size': len(model_selections),
                'all_pareto_solutions': model_selections,
                'optimization_time_seconds': optimization_time,
                'algorithm': 'NSGA2_with_Baseline_Comparison',
                'convergence_info': {
                    'final_generation': n_generations,
                    'population_size': len(pareto_solutions),
                    'improvement_found': improvements['overall_relative_improvement'] > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing results with baseline: {e}")
            return self._smart_fallback_with_baseline(baseline_metrics)
    
    def _select_best_compromise(self, model_selections: List[Dict]) -> Dict:
        """Select best compromise solution using weighted scoring."""
        if len(model_selections) == 1:
            return model_selections[0]
        
        try:
            best_score = -float('inf')
            best_solution = model_selections[0]
            
            for solution in model_selections:
                metrics = solution['metrics']
                # Weighted combination emphasizing key metrics
                score = (0.4 * metrics['overall_performance'] + 
                        0.3 * metrics['overall_consistency'] + 
                        0.2 * metrics['success_rate'] + 
                        0.1 * (metrics['bleu_score'] + metrics['rouge_l_score']))
                
                if score > best_score:
                    best_score = score
                    best_solution = solution
            
            logger.info(f"Selected compromise: {best_solution['model']} (score: {best_score:.4f})")
            return best_solution
            
        except Exception as e:
            logger.error(f"Error in compromise selection: {e}")
            return model_selections[0]
    
    def _calculate_improvements(self, baseline: Dict[str, float], 
                              optimized: Dict[str, float]) -> Dict[str, float]:
        """Calculate meaningful improvements between baseline and optimized."""
        improvements = {}
        
        # Core metrics to compare
        core_metrics = ['context_relevance', 'answer_relevance', 'faithfulness', 
                       'response_completeness', 'bleu_score', 'rouge_l_score', 
                       'overall_performance', 'overall_consistency', 'success_rate']
        
        for metric in core_metrics:
            baseline_val = baseline.get(metric, 0.1)
            optimized_val = optimized.get(metric, 0.1)
            
            # Calculate both absolute and relative improvements
            absolute_improvement = optimized_val - baseline_val
            
            if baseline_val > 0:
                relative_improvement = absolute_improvement / baseline_val
            else:
                relative_improvement = 0.0
            
            improvements[f'{metric}_absolute'] = absolute_improvement
            improvements[f'{metric}_relative'] = relative_improvement
        
        # Overall improvement summary
        core_performance_metrics = ['context_relevance', 'answer_relevance', 'faithfulness', 'response_completeness']
        
        baseline_avg = np.mean([baseline.get(m, 0.1) for m in core_performance_metrics])
        optimized_avg = np.mean([optimized.get(m, 0.1) for m in core_performance_metrics])
        
        improvements['overall_absolute_improvement'] = optimized_avg - baseline_avg
        if baseline_avg > 0:
            improvements['overall_relative_improvement'] = (optimized_avg - baseline_avg) / baseline_avg
        else:
            improvements['overall_relative_improvement'] = 0.0
        
        return improvements
    
    def _single_model_result_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict:
        """Handle single model case with baseline comparison."""
        model_name = list(self.evaluation_results.keys())[0]
        model_metrics = self.problem.model_metrics[model_name]
        
        # Even with single model, we can compare against baseline
        improvements = self._calculate_improvements(baseline_metrics, model_metrics)
        
        return {
            'best_model': model_name,
            'baseline_metrics': baseline_metrics,
            'optimized_metrics': model_metrics,
            'improvements': improvements,
            'pareto_front_size': 1,
            'all_pareto_solutions': [{'model': model_name, 'metrics': model_metrics, 'rank': 0}],
            'optimization_time_seconds': 0.0,
            'algorithm': 'Single_Model_with_Baseline',
            'convergence_info': {
                'final_generation': 0,
                'population_size': 1,
                'improvement_found': improvements['overall_relative_improvement'] > 0
            }
        }
    
    def _smart_fallback_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict:
        """Smart fallback with baseline comparison."""
        logger.warning("Using smart fallback with baseline comparison")
        
        try:
            best_model = None
            best_score = -float('inf')
            
            # Find best model using comprehensive scoring
            for model_name, metrics in self.problem.model_metrics.items():
                score = (0.4 * metrics['overall_performance'] + 
                        0.3 * metrics['overall_consistency'] + 
                        0.3 * metrics['success_rate'])
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model is None:
                best_model = list(self.problem.model_metrics.keys())[0]
            
            best_model_metrics = self.problem.model_metrics[best_model]
            improvements = self._calculate_improvements(baseline_metrics, best_model_metrics)
            
            return {
                'best_model': best_model,
                'baseline_metrics': baseline_metrics,
                'optimized_metrics': best_model_metrics,
                'improvements': improvements,
                'pareto_front_size': 1,
                'all_pareto_solutions': [{'model': best_model, 'metrics': best_model_metrics}],
                'optimization_time_seconds': 0.0,
                'algorithm': 'Smart_Fallback_with_Baseline',
                'convergence_info': {
                    'final_generation': 0,
                    'population_size': 1,
                    'improvement_found': improvements['overall_relative_improvement'] > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in smart fallback: {e}")
            # Ultimate emergency fallback
            first_model = list(self.evaluation_results.keys())[0]
            return {
                'best_model': first_model,
                'baseline_metrics': baseline_metrics,
                'optimized_metrics': self.problem.model_metrics.get(first_model, {}),
                'improvements': {},
                'pareto_front_size': 1,
                'all_pareto_solutions': [],
                'optimization_time_seconds': 0.0,
                'algorithm': 'Emergency_Fallback',
                'convergence_info': {'improvement_found': False}
            }


def run_nsga2_optimization(evaluation_results: Dict[str, List[Dict]], 
                          config: OptimizationConfig = None) -> Dict:
    """Run NSGA-2 optimization with proper before/after comparison."""
    try:
        logger.info("Starting NSGA-2 Multi-Objective Optimization with Baseline Comparison")
        
        # Initialize selector
        selector = NSGA2ModelSelector(evaluation_results)
        
        # Run optimization with baseline tracking
        optimization_results = selector.optimize_model_selection(
            config=config,
            verbose=True
        )
        
        logger.info("NSGA-2 optimization with baseline comparison completed successfully")
        return optimization_results
        
    except Exception as e:
        logger.error(f"Critical error in NSGA-2 optimization: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise