import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from config import PARAMETER_BOUNDS, GA_CONFIG, VALIDATION


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


@dataclass
class GAResult:
    """Results from genetic algorithm optimization"""
    best_individual: Individual
    best_fitness: float
    validation_score: float
    convergence_history: List[Dict]
    improvement: Dict[str, float]
    converged_at: int
    final_returns: List[float]  # Actual simulated returns from best individual


class GeneticOptimizer:
    """Genetic Algorithm for MC parameter optimization"""
    
    def __init__(self, mc_model, historical_data: List[float]):
        self.model = mc_model
        self.historical_data = np.array(historical_data)
        self.bounds = PARAMETER_BOUNDS.copy()
        
    def optimize(
        self,
        base_inputs: Dict,
        historical_data: List[float],
        generations: int = 50,
        population_size: int = 100,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.1,
        validation_split: float = 0.3
    ) -> GAResult:
        """
        Main optimization loop
        
        Args:
            base_inputs: Dictionary with baseline parameters
            historical_data: Historical returns for training
            generations: Number of generations to evolve
            population_size: Size of population
            elite_ratio: Fraction of population to keep as elites
            mutation_rate: Probability of mutation per gene
            mutation_strength: Magnitude of mutations
            validation_split: Fraction of data for validation
            
        Returns:
            GAResult with optimized parameters and diagnostics
        """
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
            valid_data = train_data[-1:]  # Use last training point for validation
        
        # Initialize population
        population = self._initialize_population(base_inputs, population_size)
        
        best_fitness = -np.inf
        best_individual = None
        convergence_history = []
        
        print(f"🧬 Starting GA: {generations} generations, population {population_size}")
        print(f"Training on {len(train_data)} points, validating on {len(valid_data)} points")
        
        for gen in range(generations):
            # Evaluate fitness for all individuals
            fitnesses = [
                self._evaluate_fitness(ind, base_inputs, train_data)
                for ind in population
            ]
            
            # Sort by fitness (higher is better)
            ranked = sorted(
                zip(population, fitnesses),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Track best individual
            if ranked[0][1] > best_fitness:
                best_fitness = ranked[0][1]
                best_individual = ranked[0][0]
            
            # Validate on holdout set
            valid_fitness = self._evaluate_fitness(
                ranked[0][0], base_inputs, valid_data
            )
            
            # Calculate population diversity
            avg_fitness = np.mean(fitnesses)
            diversity = self._calculate_diversity(population)
            
            # Record history
            convergence_history.append({
                'generation': gen,
                'train': ranked[0][1],
                'valid': valid_fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity
            })
            
            # Log progress
            if gen % 10 == 0 or gen == generations - 1:
                print(
                    f"Gen {gen}: Best={ranked[0][1]:.4f}, "
                    f"Valid={valid_fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, "
                    f"Diversity={diversity:.3f}"
                )
            
            # Early stopping if converged
            if gen > 20 and self._has_converged(convergence_history):
                print(f"✅ Converged at generation {gen}")
                break
            
            # Evolve next generation
            population = self._evolve(
                ranked,
                population_size,
                elite_ratio,
                mutation_rate,
                mutation_strength
            )
        
        # Get final simulated returns from best individual
        final_returns = self._get_simulated_returns(best_individual, base_inputs)
        
        # Calculate improvement over baseline
        improvement = self._calculate_improvement(base_inputs, best_individual)
        
        return GAResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            validation_score=valid_fitness,
            convergence_history=convergence_history,
            improvement=improvement,
            converged_at=len(convergence_history),
            final_returns=final_returns
        )
    
    def _initialize_population(
        self,
        base_inputs: Dict,
        size: int
    ) -> List[Individual]:
        """Initialize random population around base parameters"""
        population = []
        
        for _ in range(size):
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
        
        return population
    
    def _random_in_bounds(self, param: str, base: float) -> float:
        """Generate random value within bounds, centered around base"""
        min_val, max_val = self.bounds[param]
        range_val = max_val - min_val
        variation = range_val * 0.3  # 30% of range for initial diversity
        value = base + (random.random() - 0.5) * variation
        return np.clip(value, min_val, max_val)
    
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
            test_inputs['iters'] = 500  # Lower for speed during GA
            
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
            # Penalize invalid parameters heavily
            print(f"Fitness evaluation error: {str(e)}")
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
        except Exception:
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
            
            # Crossover (80% probability)
            if random.random() < 0.8:
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
        tournament_size: int = 5
    ) -> Individual:
        """Select an individual via tournament selection"""
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
            'stats': stats_result['stats'],
            'riskMetrics': stats_result['riskMetrics'],
            'percentiles': stats_result['percentiles'],
            'results': results.final_returns,
            'diagnostics': {
                'finalDiversity': results.convergence_history[-1]['diversity'] if results.convergence_history else 0,
                'fitnessProgress': results.convergence_history
            }
        }
