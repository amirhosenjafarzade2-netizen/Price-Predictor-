"""
Enhanced Genetic Algorithm Optimizer with caching
Optimizes Monte Carlo parameters to match historical data
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import PARAMETER_BOUNDS, GA_CONFIG, VALIDATION, PERFORMANCE_CONFIG
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents one set of parameters"""
    beta_real: float
    beta_exp_real: float
    beta_infl: float
    beta_vix: float
    beta_dxy: float
    beta_credit: float
    beta_term: float
    mean_reversion: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'real': self.beta_real,
            'expReal': self.beta_exp_real,
            'infl': self.beta_infl,
            'vix': self.beta_vix,
            'dxy': self.beta_dxy,
            'credit': self.beta_credit,
            'term': self.beta_term
        }
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for hashing (cache key)"""
        return tuple(round(v, 6) for v in self.__dict__.values())
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()


@dataclass
class GAResult:
    """Results from genetic algorithm optimization"""
    best_individual: Individual
    best_fitness: float
    validation_score: float
    convergence_history: List[Dict]
    improvement: Dict[str, float]
    converged_at: int
    final_returns: List[float]
    cache_hit_rate: float


class GeneticOptimizer:
    """
    Genetic Algorithm for MC parameter optimization
    
    Features:
    - Fitness caching for performance
    - Early stopping on convergence
    - Adaptive mutation rates
    - Elite preservation
    - Tournament selection
    
    Example:
        >>> optimizer = GeneticOptimizer(model, historical_data)
        >>> results = optimizer.optimize(base_inputs, historical_data)
        >>> print(f"Validation score: {results.validation_score:.3f}")
    """
    
    def __init__(self, mc_model, historical_data: List[float]):
        self.model = mc_model
        self.historical_data = np.array(historical_data)
        self.bounds = PARAMETER_BOUNDS.copy()
        
        # Fitness cache for performance
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"GA Optimizer initialized with {len(historical_data)} historical points"
        )
        
    def optimize(
        self,
        base_inputs: Dict,
        historical_data: List[float],
        generations: Optional[int] = None,
        population_size: Optional[int] = None,
        elite_ratio: Optional[float] = None,
        mutation_rate: Optional[float] = None,
        mutation_strength: Optional[float] = None,
        validation_split: Optional[float] = None
    ) -> GAResult:
        """
        Main optimization loop
        
        Args:
            base_inputs: Dictionary with baseline parameters
            historical_data: Historical returns for training
            generations: Number of generations (default from config)
            population_size: Size of population (default from config)
            elite_ratio: Fraction of elites (default from config)
            mutation_rate: Mutation probability (default from config)
            mutation_strength: Mutation magnitude (default from config)
            validation_split: Validation fraction (default from config)
            
        Returns:
            GAResult with optimized parameters and diagnostics
        """
        # Use config defaults if not specified
        generations = generations or GA_CONFIG['default_generations']
        population_size = population_size or GA_CONFIG['default_population']
        elite_ratio = elite_ratio or GA_CONFIG['elite_ratio']
        mutation_rate = mutation_rate or GA_CONFIG['mutation_rate']
        mutation_strength = mutation_strength or GA_CONFIG['mutation_strength']
        validation_split = validation_split or GA_CONFIG['validation_split']
        
        # Validate minimum data requirements
        if len(historical_data) < VALIDATION['min_returns_ga']:
            raise ValueError(
                f"Genetic Algorithm requires at least {VALIDATION['min_returns_ga']} "
                f"historical returns, got {len(historical_data)}"
            )
        
        # Update instance variable
        self.historical_data = np.array(historical_data)
        
        # Split data into training and validation
        split_idx = max(1, int(len(self.historical_data) * (1 - validation_split)))
        train_data = self.historical_data[:split_idx]
        valid_data = self.historical_data[split_idx:]
        
        if len(valid_data) == 0:
            valid_data = train_data[-1:]
        
        logger.info(
            f"Starting GA optimization: {generations} generations, "
            f"population {population_size}"
        )
        logger.info(
            f"Training: {len(train_data)} points, Validation: {len(valid_data)} points"
        )
        
        # Clear cache for new optimization
        self.fitness_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize population
        population = self._initialize_population(base_inputs, population_size)
        
        best_fitness = -np.inf
        best_individual = None
        convergence_history = []
        stagnation_counter = 0
        
        for gen in range(generations):
            # Evaluate fitness for all individuals
            fitnesses = [
                self._evaluate_fitness_cached(ind, base_inputs, train_data)
                for ind in population
            ]
            
            # Sort by fitness (higher is better)
            ranked = sorted(
                zip(population, fitnesses),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Track best individual
            current_best_fitness = ranked[0][1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = ranked[0][0]
                stagnation_counter = 0
                logger.debug(f"New best fitness: {best_fitness:.4f}")
            else:
                stagnation_counter += 1
            
            # Validate on holdout set
            valid_fitness = self._evaluate_fitness_cached(
                ranked[0][0], base_inputs, valid_data
            )
            
            # Calculate population diversity
            avg_fitness = np.mean(fitnesses)
            diversity = self._calculate_diversity(population)
            
            # Record history
            convergence_history.append({
                'generation': gen,
                'train': current_best_fitness,
                'valid': valid_fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity
            })
            
            # Log progress
            if gen % 10 == 0 or gen == generations - 1:
                cache_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
                logger.info(
                    f"Gen {gen}: Best={current_best_fitness:.4f}, "
                    f"Valid={valid_fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, "
                    f"Diversity={diversity:.3f}, "
                    f"Cache hit rate={cache_rate:.1%}"
                )
            
            # Early stopping if converged
            if gen > 20 and self._has_converged(convergence_history):
                logger.info(f"✅ Converged at generation {gen}")
                break
            
            # Adaptive mutation: increase if stagnating
            adaptive_mutation = mutation_rate
            if stagnation_counter > 5:
                adaptive_mutation = min(0.5, mutation_rate * 1.5)
                logger.debug(f"Increased mutation rate to {adaptive_mutation:.2f} due to stagnation")
            
            # Evolve next generation
            population = self._evolve(
                ranked,
                population_size,
                elite_ratio,
                adaptive_mutation,
                mutation_strength
            )
        
        # Get final simulated returns from best individual
        final_returns = self._get_simulated_returns(best_individual, base_inputs)
        
        # Calculate improvement over baseline
        improvement = self._calculate_improvement(base_inputs, best_individual)
        
        # Calculate cache statistics
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        logger.info(
            f"Optimization complete. Cache hit rate: {cache_hit_rate:.1%} "
            f"({self.cache_hits} hits, {self.cache_misses} misses)"
        )
        
        return GAResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            validation_score=valid_fitness,
            convergence_history=convergence_history,
            improvement=improvement,
            converged_at=len(convergence_history),
            final_returns=final_returns,
            cache_hit_rate=cache_hit_rate
        )
    
    def _initialize_population(
        self,
        base_inputs: Dict,
        size: int
    ) -> List[Individual]:
        """Initialize random population around base parameters"""
        population = []
        
        # Always include the baseline individual
        baseline = Individual(
            beta_real=base_inputs['betas'].get('real', 0),
            beta_exp_real=base_inputs['betas'].get('expReal', 0),
            beta_infl=base_inputs['betas'].get('infl', 0),
            beta_vix=base_inputs['betas'].get('vix', 0),
            beta_dxy=base_inputs['betas'].get('dxy', 0),
            beta_credit=base_inputs['betas'].get('credit', 0),
            beta_term=base_inputs['betas'].get('term', 0),
            mean_reversion=base_inputs.get('meanReversion', 0)
        )
        population.append(baseline)
        
        # Generate random individuals
        for _ in range(size - 1):
            ind = Individual(
                beta_real=self._random_in_bounds(
                    'beta_real', base_inputs['betas'].get('real', 0)
                ),
                beta_exp_real=self._random_in_bounds(
                    'beta_expReal', base_inputs['betas'].get('expReal', 0)
                ),
                beta_infl=self._random_in_bounds(
                    'beta_infl', base_inputs['betas'].get('infl', 0)
                ),
                beta_vix=self._random_in_bounds(
                    'beta_vix', base_inputs['betas'].get('vix', 0)
                ),
                beta_dxy=self._random_in_bounds(
                    'beta_dxy', base_inputs['betas'].get('dxy', 0)
                ),
                beta_credit=self._random_in_bounds(
                    'beta_credit', base_inputs['betas'].get('credit', 0)
                ),
                beta_term=self._random_in_bounds(
                    'beta_term', base_inputs['betas'].get('term', 0)
                ),
                mean_reversion=self._random_in_bounds(
                    'mean_reversion', base_inputs.get('meanReversion', 0)
                )
            )
            population.append(ind)
        
        logger.debug(f"Initialized population of {size} individuals")
        return population
    
    def _random_in_bounds(self, param: str, base: float) -> float:
        """Generate random value within bounds, centered around base"""
        min_val, max_val = self.bounds[param]
        range_val = max_val - min_val
        variation = range_val * 0.3  # 30% of range for initial diversity
        value = base + (random.random() - 0.5) * variation
        return np.clip(value, min_val, max_val)
    
    def _evaluate_fitness_cached(
        self,
        individual: Individual,
        base_inputs: Dict,
        data: np.ndarray
    ) -> float:
        """Evaluate fitness with caching"""
        if not PERFORMANCE_CONFIG.get('enable_caching', True):
            return self._evaluate_fitness(individual, base_inputs, data)
        
        # Create cache key
        cache_key = (individual.to_tuple(), tuple(data))
        
        if cache_key in self.fitness_cache:
            self.cache_hits += 1
            return self.fitness_cache[cache_key]
        
        self.cache_misses += 1
        fitness = self._evaluate_fitness(individual, base_inputs, data)
        
        # Store in cache (with size limit)
        if len(self.fitness_cache) < PERFORMANCE_CONFIG.get('cache_max_size', 1000):
            self.fitness_cache[cache_key] = fitness
        
        return fitness
    
    def _evaluate_fitness(
        self,
        individual: Individual,
        base_inputs: Dict,
        data: np.ndarray
    ) -> float:
        """
        Fitness function: How well do predictions match actual returns?
        
        Multi-objective:
        1. Match historical mean
        2. Match historical volatility
        3. Produce reasonable Sharpe ratio
        4. Regularization to prevent overfitting
        """
        try:
            # Create inputs with individual's parameters
            test_inputs = base_inputs.copy()
            test_inputs['betas'] = individual.to_dict()
            test_inputs['meanReversion'] = individual.mean_reversion
            test_inputs['iters'] = GA_CONFIG['fitness_iterations']
            
            # Run Monte Carlo with these parameters
            results = self.model.run(test_inputs)
            stats = results['stats']
            risk_metrics = results['riskMetrics']
            
            # Historical statistics
            hist_mean = np.mean(data)
            hist_std = np.std(data, ddof=1)
            
            # Prevent division by zero
            hist_std = max(0.1, hist_std)
            
            # Fitness components
            mean_error = abs(stats['mean'] - hist_mean) / hist_std
            std_error = abs(stats['stdDev'] - hist_std) / hist_std
            sharpe_score = np.clip(risk_metrics['sharpe'], 0, 3)
            regularization = self._calculate_regularization(individual)
            
            # Combined fitness (higher is better)
            fitness = (
                -GA_CONFIG['weight_mean_error'] * mean_error +
                -GA_CONFIG['weight_std_error'] * std_error +
                GA_CONFIG['weight_sharpe'] * sharpe_score +
                -GA_CONFIG['weight_regularization'] * regularization
            )
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Fitness evaluation error: {e}")
            return -1000.0
    
    def _get_simulated_returns(
        self,
        individual: Individual,
        base_inputs: Dict
    ) -> List[float]:
        """Get actual simulated returns from best individual"""
        try:
            test_inputs = base_inputs.copy()
            test_inputs['betas'] = individual.to_dict()
            test_inputs['meanReversion'] = individual.mean_reversion
            
            results = self.model.run(test_inputs)
            return results['results']
        except Exception as e:
            logger.error(f"Failed to get simulated returns: {e}")
            return []
    
    def _calculate_regularization(self, individual: Individual) -> float:
        """Penalize extreme parameters to prevent overfitting"""
        penalty = 0.0
        
        for param_name, value in individual.__dict__.items():
            if param_name not in self.bounds:
                continue
            
            min_val, max_val = self.bounds[param_name]
            range_val = max_val - min_val
            center = (max_val + min_val) / 2
            
            # Normalized deviation from center
            deviation = abs(value - center) / (range_val / 2)
            penalty += deviation
        
        return penalty / len(individual.__dict__)
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity to track convergence"""
        if not population:
            return 0.0
        
        total_variance = 0.0
        param_names = list(population[0].__dict__.keys())
        
        for param_name in param_names:
            values = [getattr(ind, param_name) for ind in population]
            variance = np.var(values)
            total_variance += variance
        
        return np.sqrt(total_variance / len(param_names))
    
    def _has_converged(self, history: List[Dict]) -> bool:
        """Check if GA has converged"""
        window = GA_CONFIG['convergence_window']
        threshold = GA_CONFIG['convergence_threshold']
        
        if len(history) < window:
            return False
        
        recent = history[-window:]
        fitness_values = [h['train'] for h in recent]
        improvement = fitness_values[-1] - fitness_values[0]
        
        return improvement < threshold
    
    def _calculate_improvement(
        self,
        base_inputs: Dict,
        best_individual: Individual
    ) -> Dict[str, float]:
        """Calculate improvement of optimized parameters over baseline"""
        improvement = {}
        base_betas = base_inputs['betas']
        optimized_betas = best_individual.to_dict()
        
        for param in base_betas.keys():
            base_value = base_betas.get(param, 0)
            opt_value = optimized_betas[param]
            improvement[param] = opt_value - base_value
        
        improvement['mean_reversion'] = (
            best_individual.mean_reversion - base_inputs.get('meanReversion', 0)
        )
        
        return improvement
    
    def _evolve(
        self,
        ranked: List[Tuple[Individual, float]],
        population_size: int,
        elite_ratio: float,
        mutation_rate: float,
        mutation_strength: float
    ) -> List[Individual]:
        """Create next generation through selection, crossover, mutation"""
        new_population = []
        elite_count = int(population_size * elite_ratio)
        
        # Elitism: Keep best individuals
        for i in range(elite_count):
            new_population.append(ranked[i][0])
        
        # Fill rest through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = self._tournament_select(ranked)
            parent2 = self._tournament_select(ranked)
            
            # Crossover
            if random.random() < GA_CONFIG['crossover_probability']:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1 if random.random() < 0.5 else parent2
            
            # Mutation
            child = self._mutate(child, mutation_rate, mutation_strength)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_select(
        self,
        ranked: List[Tuple[Individual, float]],
        tournament_size: Optional[int] = None
    ) -> Individual:
        """Select an individual via tournament selection"""
        tournament_size = tournament_size or GA_CONFIG['tournament_size']
        tournament = random.sample(ranked, min(tournament_size, len(ranked)))
        return max(tournament, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform uniform crossover between two parents"""
        params = {}
        for param in parent1.__dict__.keys():
            params[param] = (
                getattr(parent1, param)
                if random.random() < 0.5
                else getattr(parent2, param)
            )
        
        return Individual(**params)
    
    def _mutate(
        self,
        individual: Individual,
        mutation_rate: float,
        mutation_strength: float
    ) -> Individual:
        """Apply random mutations to an individual"""
        params = individual.__dict__.copy()
        
        for param in params.keys():
            if random.random() < mutation_rate:
                min_val, max_val = self.bounds[param]
                range_val = max_val - min_val
                mutation = (random.random() - 0.5) * range_val * mutation_strength
                params[param] = np.clip(
                    params[param] + mutation,
                    min_val,
                    max_val
                )
        
        return Individual(**params)
    
    def export_results(self, results: GAResult) -> Dict:
        """Export GA results in format expected by app"""
        from utils import calculate_stats
        
        # Calculate stats from the actual simulated returns
        stats_result = calculate_stats(results.final_returns)
        
        return {
            'optimizedBetas': results.best_individual.to_dict(),
            'optimizedMeanReversion': results.best_individual.mean_reversion,
            'trainingScore': results.best_fitness,
            'validationScore': results.validation_score,
            'improvement': results.improvement,
            'convergedAt': results.converged_at,
            'cacheHitRate': results.cache_hit_rate,
            'stats': stats_result['stats'],
            'riskMetrics': stats_result['riskMetrics'],
            'percentiles': stats_result['percentiles'],
            'results': results.final_returns,
            'diagnostics': {
                'finalDiversity': results.convergence_history[-1]['diversity'] if results.convergence_history else 0,
                'fitnessProgress': results.convergence_history
            }
        }
