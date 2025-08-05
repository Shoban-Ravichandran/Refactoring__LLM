"""Fixed NSGA-II multi-objective optimization for model selection with proper problem formulation."""

import logging
import time
from typing import Dict, List, Any, Tuple
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
    from pymoo.core.variable import Real
    PYMOO_AVAILABLE = True
except ImportError:
    logger.warning("pymoo not available. Install with: pip install pymoo")
    PYMOO_AVAILABLE = False


class ModelWeightOptimizationProblem(Problem):
    """
    NSGA-II problem that optimizes model weights and selection criteria.
    
    Instead of just selecting models, this optimizes:
    1. Weights for different metrics
    2. Model performance thresholds
    3. Ensemble combinations
    """
    
    def __init__(self, evaluation_results: Dict[str, List[Dict]]):
        """Initialize with proper multi-objective formulation."""
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        self.evaluation_results = evaluation_results
        self.model_names = list(evaluation_results.keys())
        self.n_models = len(self.model_names)
        
        logger.info(f"Initializing NSGA-II with {self.n_models} models: {self.model_names}")
        
        # Calculate model metrics with enhanced statistics
        self.model_metrics = self._calculate_enhanced_model_metrics()
        self._log_model_statistics()
        
        # Decision variables:
        # 0-5: Metric weights (6 weights that sum to 1)
        # 6: Minimum performance threshold
        # 7: Consistency weight
        # 8-12: Model preference weights (5 model weights)
        
        n_vars = 6 + 1 + 1 + self.n_models  # weights + threshold + consistency + model_preferences
        
        super().__init__(
            n_var=n_vars,
            n_obj=4,  # 4 objectives: performance, consistency, diversity, robustness
            n_constr=2,  # 2 constraints: weights sum to 1, threshold bounds
            xl=np.array([0.0] * n_vars),  # Lower bounds
            xu=np.array([1.0] * n_vars),  # Upper bounds
        )
        
        logger.info(f"Problem initialized: {n_vars} variables, 4 objectives, 2 constraints")
    
    def _calculate_enhanced_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive model metrics."""
        model_metrics = {}
        
        for model_name, results in self.evaluation_results.items():
            successful_results = [r for r in results if r.get('success') and r.get('rag_metrics')]
            
            if not successful_results:
                model_metrics[model_name] = self._get_default_metrics()
                continue
            
            # Collect metrics with proper handling
            metrics_arrays = {
                'context_relevance': [],
                'answer_relevance': [],
                'faithfulness': [],
                'response_completeness': [],
                'bleu_score': [],
                'rouge_l_score': []
            }
            
            for r in successful_results:
                rag = r['rag_metrics']
                metrics_arrays['context_relevance'].append(self._safe_metric(rag.context_relevance))
                metrics_arrays['answer_relevance'].append(self._safe_metric(rag.answer_relevance))
                metrics_arrays['faithfulness'].append(self._safe_metric(rag.faithfulness))
                metrics_arrays['response_completeness'].append(self._safe_metric(rag.response_completeness))
                metrics_arrays['bleu_score'].append(self._safe_metric(getattr(rag, 'bleu_score', 0.01)))
                metrics_arrays['rouge_l_score'].append(self._safe_metric(getattr(rag, 'rouge_l_score', 0.01)))
            
            # Calculate comprehensive statistics
            metrics = {}
            for metric_name, values in metrics_arrays.items():
                if values:
                    metrics[f'{metric_name}_mean'] = np.mean(values)
                    metrics[f'{metric_name}_std'] = np.std(values)
                    metrics[f'{metric_name}_min'] = np.min(values)
                    metrics[f'{metric_name}_max'] = np.max(values)
                    metrics[f'{metric_name}_median'] = np.median(values)
                    # Consistency score (lower std = higher consistency)
                    metrics[f'{metric_name}_consistency'] = max(0.0, 1.0 - np.std(values))
                else:
                    # Fallback values
                    metrics[f'{metric_name}_mean'] = 0.1
                    metrics[f'{metric_name}_std'] = 0.0
                    metrics[f'{metric_name}_min'] = 0.1
                    metrics[f'{metric_name}_max'] = 0.1
                    metrics[f'{metric_name}_median'] = 0.1
                    metrics[f'{metric_name}_consistency'] = 0.1
            
            # Overall consistency score
            all_stds = [metrics[f'{m}_std'] for m in metrics_arrays.keys()]
            metrics['overall_consistency'] = max(0.0, 1.0 - np.mean(all_stds))
            
            # Performance range (max - min for adaptability)
            all_means = [metrics[f'{m}_mean'] for m in metrics_arrays.keys()]
            metrics['performance_range'] = np.max(all_means) - np.min(all_means)
            
            # Overall performance score
            metrics['overall_performance'] = np.mean(all_means)
            
            model_metrics[model_name] = metrics
        
        return model_metrics
    
    def _safe_metric(self, value, default: float = 0.01, min_val: float = 0.001, max_val: float = 1.0):
        """Safely convert metric to valid float."""
        try:
            val = float(value)
            if not np.isfinite(val) or val < min_val:
                return default
            return min(val, max_val)
        except:
            return default
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get safe default metrics."""
        base_metrics = ['context_relevance', 'answer_relevance', 'faithfulness', 
                       'response_completeness', 'bleu_score', 'rouge_l_score']
        
        defaults = {}
        for metric in base_metrics:
            defaults[f'{metric}_mean'] = 0.1
            defaults[f'{metric}_std'] = 0.05
            defaults[f'{metric}_min'] = 0.05
            defaults[f'{metric}_max'] = 0.15
            defaults[f'{metric}_median'] = 0.1
            defaults[f'{metric}_consistency'] = 0.1
        
        defaults['overall_consistency'] = 0.1
        defaults['performance_range'] = 0.1
        defaults['overall_performance'] = 0.1
        
        return defaults
    
    def _log_model_statistics(self):
        """Log model statistics for debugging."""
        logger.info("Model Performance Statistics:")
        for model_name, metrics in self.model_metrics.items():
            overall_perf = metrics.get('overall_performance', 0)
            consistency = metrics.get('overall_consistency', 0)
            logger.info(f"  {model_name}: Performance={overall_perf:.3f}, Consistency={consistency:.3f}")
    
    def _evaluate(self, X, out, *args, **kwargs):
        """NSGA-II evaluation function with proper multi-objective formulation."""
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            batch_size = X.shape[0]
            objectives_batch = np.zeros((batch_size, 4))  # 4 objectives
            constraints_batch = np.zeros((batch_size, 2))  # 2 constraints
            
            for i, solution in enumerate(X):
                try:
                    # Parse decision variables
                    metric_weights_raw = solution[:6]  # 6 metric weights
                    threshold = solution[6]  # Performance threshold
                    consistency_weight = solution[7]  # Consistency importance
                    model_preferences = solution[8:8+self.n_models]  # Model preferences
                    
                    # Normalize metric weights to sum to 1
                    metric_weights_sum = np.sum(metric_weights_raw)
                    if metric_weights_sum > 0:
                        metric_weights = metric_weights_raw / metric_weights_sum
                    else:
                        metric_weights = np.ones(6) / 6  # Equal weights fallback
                    
                    # Normalize model preferences
                    model_prefs_sum = np.sum(model_preferences)
                    if model_prefs_sum > 0:
                        model_preferences = model_preferences / model_prefs_sum
                    else:
                        model_preferences = np.ones(self.n_models) / self.n_models
                    
                    # Calculate objectives for this solution
                    objectives = self._calculate_objectives(
                        metric_weights, threshold, consistency_weight, model_preferences
                    )
                    
                    # Calculate constraints
                    constraints = self._calculate_constraints(metric_weights_raw, threshold)
                    
                    objectives_batch[i] = objectives
                    constraints_batch[i] = constraints
                    
                except Exception as e:
                    logger.warning(f"Error evaluating solution {i}: {e}")
                    # Fallback to poor objectives
                    objectives_batch[i] = [1.0, 1.0, 1.0, 1.0]  # High values (to minimize)
                    constraints_batch[i] = [1.0, 1.0]  # Violated constraints
            
            out["F"] = objectives_batch
            out["G"] = constraints_batch
            
        except Exception as e:
            logger.error(f"Critical error in NSGA-II evaluation: {e}")
            batch_size = X.shape[0] if X.ndim > 1 else 1
            out["F"] = np.ones((batch_size, 4))
            out["G"] = np.ones((batch_size, 2))
    
    def _calculate_objectives(self, metric_weights: np.ndarray, threshold: float,
                            consistency_weight: float, model_preferences: np.ndarray) -> List[float]:
        """Calculate the 4 objectives for NSGA-II optimization."""
        
        # Objective 1: Maximize weighted performance (minimize negative)
        performance_scores = []
        for model_name, metrics in self.model_metrics.items():
            base_metrics = ['context_relevance', 'answer_relevance', 'faithfulness',
                           'response_completeness', 'bleu_score', 'rouge_l_score']
            
            weighted_score = 0.0
            for j, metric in enumerate(base_metrics):
                metric_value = metrics.get(f'{metric}_mean', 0.1)
                weighted_score += metric_weights[j] * metric_value
            
            performance_scores.append(weighted_score)
        
        # Weight by model preferences
        overall_performance = np.sum(np.array(performance_scores) * model_preferences)
        obj1_performance = -overall_performance  # Negative because we minimize
        
        # Objective 2: Maximize consistency (minimize inconsistency)
        consistency_scores = []
        for model_name in self.model_names:
            consistency = self.model_metrics[model_name].get('overall_consistency', 0.1)
            consistency_scores.append(consistency)
        
        overall_consistency = np.sum(np.array(consistency_scores) * model_preferences)
        obj2_consistency = -(overall_consistency * consistency_weight)  # Negative because we minimize
        
        # Objective 3: Maximize diversity (minimize concentration)
        # Penalize solutions that put all weight on one model
        diversity_penalty = np.sum(model_preferences ** 2)  # Higher when concentrated
        obj3_diversity = diversity_penalty
        
        # Objective 4: Minimize threshold violation penalty
        threshold_violations = 0
        for model_name, metrics in self.model_metrics.items():
            if metrics.get('overall_performance', 0) < threshold:
                threshold_violations += 1
        
        obj4_robustness = threshold_violations / self.n_models
        
        return [obj1_performance, obj2_consistency, obj3_diversity, obj4_robustness]
    
    def _calculate_constraints(self, metric_weights_raw: np.ndarray, threshold: float) -> List[float]:
        """Calculate constraint violations."""
        
        # Constraint 1: Metric weights should sum to approximately 1 (already normalized, so this is soft)
        weights_sum_violation = abs(np.sum(metric_weights_raw) - 1.0) - 0.1  # Allow 10% deviation
        
        # Constraint 2: Threshold should be reasonable (between 0.1 and 0.9)
        threshold_violation = max(0.0, 0.1 - threshold) + max(0.0, threshold - 0.9)
        
        return [weights_sum_violation, threshold_violation]


class FixedNSGA2ModelSelector:
    """Fixed NSGA-II model selector with proper multi-objective formulation."""
    
    def __init__(self, evaluation_results: Dict[str, List[Dict]]):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for NSGA-II optimization. Install with: pip install pymoo")
        
        self.evaluation_results = evaluation_results
        
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        # Validate that we have meaningful differences between models
        self._validate_model_diversity()
        
        self.problem = ModelWeightOptimizationProblem(evaluation_results)
        logger.info("Fixed NSGA-II selector initialized successfully")
    
    def _validate_model_diversity(self):
        """Check if models have sufficient performance diversity for optimization."""
        model_scores = []
        
        for model_name, results in self.evaluation_results.items():
            successful_results = [r for r in results if r.get('success') and r.get('rag_metrics')]
            if successful_results:
                # Calculate a simple overall score
                scores = []
                for r in successful_results:
                    rag = r['rag_metrics']
                    score = (rag.context_relevance + rag.answer_relevance + 
                            rag.faithfulness + rag.response_completeness) / 4.0
                    scores.append(score)
                avg_score = np.mean(scores)
                model_scores.append(avg_score)
            else:
                model_scores.append(0.1)
        
        if len(model_scores) < 2:
            raise ValueError("Need at least 2 models for optimization")
        
        score_range = max(model_scores) - min(model_scores)
        logger.info(f"Model performance range: {score_range:.4f}")
        
        if score_range < 0.01:
            logger.warning(f"Low model diversity (range: {score_range:.4f}). NSGA-II may not find significant improvements.")
        else:
            logger.info(f"Good model diversity detected. NSGA-II should find meaningful trade-offs.")
    
    def optimize_model_selection(self, config: OptimizationConfig = None, verbose: bool = True) -> Dict:
        """Run NSGA-II optimization with proper configuration."""
        if config is None:
            config = OptimizationConfig()
        
        logger.info(f"Starting Fixed NSGA-II optimization with {config.n_generations} generations, "
                   f"population {config.population_size}")
        
        try:
            # Handle single model case
            if self.problem.n_models == 1:
                return self._single_model_result()
            
            # Configure NSGA-II with proper parameters
            algorithm = NSGA2(
                pop_size=max(config.population_size, 40),  # Larger population for better diversity
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),  # Higher crossover probability
                mutation=PM(prob=1.0/self.problem.n_var, eta=20),  # Adaptive mutation
                eliminate_duplicates=True
            )
            
            # Use more generations for better convergence
            termination = get_termination("n_gen", max(config.n_generations, 50))
            
            start_time = time.time()
            
            logger.info("Starting Fixed NSGA-II optimization...")
            
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
            
            logger.info(f"Fixed NSGA-II completed in {optimization_time:.2f}s")
            
            # Validate results
            if not self._validate_results(res):
                logger.warning("NSGA-II result validation failed, using fallback")
                return self._fallback_selection()
            
            # Process results
            return self._process_nsga2_results(res, optimization_time, config.n_generations)
            
        except Exception as e:
            logger.error(f"Error in Fixed NSGA-II optimization: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._fallback_selection()
    
    def _validate_results(self, res) -> bool:
        """Validate NSGA-II results."""
        try:
            if res is None or not hasattr(res, 'F') or not hasattr(res, 'X'):
                return False
            
            if res.F is None or res.X is None:
                return False
            
            if not isinstance(res.F, np.ndarray) or not isinstance(res.X, np.ndarray):
                return False
            
            if res.F.size == 0 or res.X.size == 0:
                return False
            
            if not np.all(np.isfinite(res.F)) or not np.all(np.isfinite(res.X)):
                return False
            
            logger.info(f"NSGA-II validation passed: Pareto front size = {len(res.F)}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating NSGA-II results: {e}")
            return False
    
    def _process_nsga2_results(self, res, optimization_time: float, n_generations: int) -> Dict:
        """Process NSGA-II results and select best solution."""
        try:
            pareto_solutions = res.X
            pareto_objectives = res.F
            
            logger.info(f"Processing {len(pareto_solutions)} Pareto optimal solutions")
            
            # Select best compromise solution using Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)
            best_solution_idx = self._select_best_compromise(pareto_objectives)
            best_solution_vars = pareto_solutions[best_solution_idx]
            best_objectives = pareto_objectives[best_solution_idx]
            
            # Parse the best solution
            metric_weights_raw = best_solution_vars[:6]
            threshold = best_solution_vars[6]
            consistency_weight = best_solution_vars[7]
            model_preferences = best_solution_vars[8:8+self.problem.n_models]
            
            # Normalize weights
            metric_weights = metric_weights_raw / np.sum(metric_weights_raw)
            model_preferences = model_preferences / np.sum(model_preferences)
            
            # Find the best model based on the optimized weights
            best_model = self._calculate_best_model(metric_weights, model_preferences)
            
            # Calculate final metrics for the best model
            best_model_metrics = self._calculate_final_metrics(best_model, metric_weights)
            
            # Create solution summary
            all_solutions = []
            for i, (solution_vars, objectives) in enumerate(zip(pareto_solutions, pareto_objectives)):
                sol_model_prefs = solution_vars[8:8+self.problem.n_models]
                sol_model_prefs = sol_model_prefs / np.sum(sol_model_prefs)
                
                # Find dominant model for this solution
                dominant_model_idx = np.argmax(sol_model_prefs)
                dominant_model = self.problem.model_names[dominant_model_idx]
                
                all_solutions.append({
                    'model': dominant_model,
                    'model_preferences': sol_model_prefs.tolist(),
                    'objectives': objectives.tolist(),
                    'rank': i,
                    'is_best': i == best_solution_idx
                })
            
            return {
                'best_model': best_model,
                'best_model_objectives': best_model_metrics,
                'optimized_weights': {
                    'context_relevance': float(metric_weights[0]),
                    'answer_relevance': float(metric_weights[1]),
                    'faithfulness': float(metric_weights[2]),
                    'response_completeness': float(metric_weights[3]),
                    'bleu_score': float(metric_weights[4]),
                    'rouge_l_score': float(metric_weights[5])
                },
                'optimized_threshold': float(threshold),
                'consistency_weight': float(consistency_weight),
                'model_preferences': {
                    name: float(pref) for name, pref in 
                    zip(self.problem.model_names, model_preferences)
                },
                'pareto_front_size': len(pareto_solutions),
                'all_pareto_solutions': all_solutions,
                'optimization_time_seconds': optimization_time,
                'algorithm': 'Fixed_NSGA2_Multi_Objective',
                'convergence_info': {
                    'final_generation': n_generations,
                    'population_size': len(pareto_solutions),
                    'best_objectives': best_objectives.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing NSGA-II results: {e}")
            return self._fallback_selection()
    
    def _select_best_compromise(self, pareto_objectives: np.ndarray) -> int:
        """Select best compromise solution using TOPSIS method."""
        try:
            # TOPSIS: find solution closest to ideal and farthest from nadir
            
            # Ideal point (minimum values for each objective, since we minimize)
            ideal = np.min(pareto_objectives, axis=0)
            
            # Nadir point (maximum values for each objective)
            nadir = np.max(pareto_objectives, axis=0)
            
            # Calculate distances
            best_idx = 0
            best_score = -float('inf')
            
            for i, obj in enumerate(pareto_objectives):
                # Distance to ideal (smaller is better)
                dist_to_ideal = np.linalg.norm(obj - ideal)
                
                # Distance to nadir (larger is better)
                dist_to_nadir = np.linalg.norm(obj - nadir)
                
                # TOPSIS score (higher is better)
                if dist_to_ideal + dist_to_nadir > 0:
                    topsis_score = dist_to_nadir / (dist_to_ideal + dist_to_nadir)
                else:
                    topsis_score = 0.5
                
                if topsis_score > best_score:
                    best_score = topsis_score
                    best_idx = i
            
            logger.info(f"Selected compromise solution {best_idx} with TOPSIS score {best_score:.4f}")
            return best_idx
            
        except Exception as e:
            logger.error(f"Error in compromise selection: {e}")
            return 0  # Return first solution as fallback
    
    def _calculate_best_model(self, metric_weights: np.ndarray, model_preferences: np.ndarray) -> str:
        """Calculate the best model based on optimized weights and preferences."""
        model_scores = {}
        
        for i, model_name in enumerate(self.problem.model_names):
            metrics = self.problem.model_metrics[model_name]
            
            # Calculate weighted performance score
            base_metrics = ['context_relevance', 'answer_relevance', 'faithfulness',
                           'response_completeness', 'bleu_score', 'rouge_l_score']
            
            weighted_score = 0.0
            for j, metric in enumerate(base_metrics):
                metric_value = metrics.get(f'{metric}_mean', 0.1)
                weighted_score += metric_weights[j] * metric_value
            
            # Apply model preference weight
            final_score = weighted_score * model_preferences[i]
            model_scores[model_name] = final_score
        
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        logger.info(f"Best model selected: {best_model} with score {model_scores[best_model]:.4f}")
        
        return best_model
    
    def _calculate_final_metrics(self, model_name: str, metric_weights: np.ndarray) -> Dict[str, float]:
        """Calculate final metrics for the selected model."""
        metrics = self.problem.model_metrics[model_name]
        
        return {
            'context_relevance': metrics.get('context_relevance_mean', 0.1),
            'answer_relevance': metrics.get('answer_relevance_mean', 0.1),
            'faithfulness': metrics.get('faithfulness_mean', 0.1),
            'response_completeness': metrics.get('response_completeness_mean', 0.1),
            'bleu_score': metrics.get('bleu_score_mean', 0.01),
            'rouge_l_score': metrics.get('rouge_l_score_mean', 0.01),
            'overall_consistency': metrics.get('overall_consistency', 0.1),
            'overall_performance': metrics.get('overall_performance', 0.1)
        }
    
    def _single_model_result(self) -> Dict:
        """Handle single model case."""
        model_name = list(self.evaluation_results.keys())[0]
        metrics = self.problem.model_metrics[model_name]
        
        return {
            'best_model': model_name,
            'best_model_objectives': self._calculate_final_metrics(model_name, np.array([1/6]*6)),
            'optimized_weights': {
                'context_relevance': 1/6,
                'answer_relevance': 1/6,
                'faithfulness': 1/6,
                'response_completeness': 1/6,
                'bleu_score': 1/6,
                'rouge_l_score': 1/6
            },
            'pareto_front_size': 1,
            'all_pareto_solutions': [{'model': model_name, 'rank': 0}],
            'optimization_time_seconds': 0.0,
            'algorithm': 'Single_Model_Direct'
        }
    
    def _fallback_selection(self) -> Dict:
        """Fallback when NSGA-II fails."""
        logger.warning("Using fallback model selection")
        
        # Simple weighted sum fallback
        best_model = None
        best_score = -float('inf')
        
        for model_name, metrics in self.problem.model_metrics.items():
            score = metrics.get('overall_performance', 0.1)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model is None:
            best_model = list(self.evaluation_results.keys())[0]
        
        return {
            'best_model': best_model,
            'best_model_objectives': self._calculate_final_metrics(best_model, np.array([1/6]*6)),
            'optimized_weights': {
                'context_relevance': 1/6,
                'answer_relevance': 1/6,
                'faithfulness': 1/6,
                'response_completeness': 1/6,
                'bleu_score': 1/6,
                'rouge_l_score': 1/6
            },
            'pareto_front_size': 1,
            'all_pareto_solutions': [],
            'optimization_time_seconds': 0.0,
            'algorithm': 'Fallback_Weighted_Sum'
        }


def run_fixed_nsga2_optimization(evaluation_results: Dict[str, List[Dict]], 
                                config: OptimizationConfig = None) -> Dict:
    """
    Run fixed NSGA-II optimization for model selection.
    
    Args:
        evaluation_results: Dictionary mapping model names to evaluation results
        config: Optimization configuration
        
    Returns:
        Dictionary containing optimization results
    """
    try:
        logger.info("Starting Fixed NSGA-II Multi-Objective Optimization")
        
        # Initialize selector
        selector = FixedNSGA2ModelSelector(evaluation_results)
        
        # Run optimization
        optimization_results = selector.optimize_model_selection(
            config=config,
            verbose=True
        )
        
        logger.info("Fixed NSGA-II optimization completed successfully")
        return optimization_results
        
    except Exception as e:
        logger.error(f"Critical error in Fixed NSGA-II optimization: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise