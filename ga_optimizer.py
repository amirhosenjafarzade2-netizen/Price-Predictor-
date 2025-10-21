import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
import streamlit as st
from utils import validate_inputs

@dataclass
class Individual:
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
    best_individual: Individual
    best_fitness: float
    validation_score: float
    convergence_history: List[Dict]
    improvement: Dict[str, float]
    converged_at: int

class GeneticOptimizer:
    def __init__(self, mc_model, historical_data: List[float]):
        self.model = mc_model
        self.historical_data = np.array(historical_data)
        
        self.bounds = {
            'beta_real': (-1.5, 0.5),
            'beta_exp_real': (-1.5, 0.5),
            'beta_infl': (-0.5, 0.8),
            'beta_vix': (0, 0.5),
            'beta_dxy': (-0.15, 0.15),
            'beta_credit': (-0.3, 0.1),
            'beta_term': (-0.1, 0.3),
            'mean_reversion': (0, 0.5)
        }
        
    @st.cache_data
    def optimize(self, base_inputs: Dict, generations: int = 50, population_size: int = 100,
                 elite_ratio: float = 0.2, mutation_rate: float = 0.15, mutation_strength: float = 0.1,
                 validation_split: float = 0.3) -> GAResult:
        if len(self.historical_data) < 2:
            raise ValueError("Genetic Algorithm requires at least 2 historical returns")
        
        split_idx = int(len(self.historical_data) * (1 - validation_split))
        train_data = self.historical_data[:split_idx]
        valid_data = self.historical_data[split_idx:]
        
        population = self._initialize_population(base_inputs, population_size)
        best_fitness = -np.inf
        best_individual = None
        convergence_history = []
        
        for gen in range(generations):
            fitnesses = [self._evaluate_fitness(ind, base_inputs, train_data) for ind in population]
            ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
            
            if ranked[0][1] > best_fitness:
                best_fitness = ranked[0][1]
                best_individual = ranked[0][0]
            
            valid_fitness = self._evaluate_fitness(ranked[0][0], base_inputs, valid_data)
            
            convergence_history.append({
                'generation': gen,
                'train': ranked[0][1],
                'valid': valid_fitness,
                'avg_fitness': np.mean(fitnesses),
                'diversity': self._calculate_diversity(population)
            })
            
            if gen > 20 and self._has_converged(convergence_history, window=10):
                break
            
            population = self._evolve(ranked, population_size, elite_ratio, mutation_rate, mutation_strength)
        
        improvement = self._calculate_improvement(base_inputs, best_individual)
        
        return GAResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            validation_score=valid_fitness,
            convergence_history=convergence_history,
            improvement=improvement,
            converged_at=len(convergence_history)
        )

    def _initialize_population(self, base_inputs: Dict, size: int) -> List[Individual]:
        population = []
        for _ in range(size):
            ind = Individual(
                beta_real=self._random_in_bounds('beta_real', base_inputs['betas']['real']),
                beta_exp_real=self._random_in_bounds('beta_exp_real', base_inputs['betas']['expReal']),
                beta_infl=self._random_in_bounds('beta_infl', base_inputs['betas']['infl']),
                beta_vix=self._random_in_bounds('beta_vix', base_inputs['betas']['vix']),
                beta_dxy=self._random_in_bounds('beta_dxy', base_inputs['betas']['dxy']),
                beta_credit=self._random_in_bounds('beta_credit', base_inputs['betas']['credit']),
                beta_term=self._random_in_bounds('beta_term', base_inputs['betas']['term']),
                mean_reversion=self._random_in_bounds('mean_reversion', base_inputs['meanReversion'])
            )
            population.append(ind)
        return population

    def _random_in_bounds(self, param: str, base: float) -> float:
        min_val, max_val = self.bounds[param]
        range_val = max_val - min_val
        variation = range_val * 0.3
        value = base + (random.random() - 0.5) * variation
        return np.clip(value, min_val, max_val)

    def _evaluate_fitness(self, individual: Individual, base_inputs: Dict, data: np.ndarray) -> float:
        try:
            test_inputs = base_inputs.copy()
            test_inputs['betas'] = individual.to_dict()
            test_inputs['meanReversion'] = individual.mean_reversion
            test_inputs['iters'] = 500
            
            results = self.model.run(test_inputs)
            stats = results['stats']
            
            hist_mean = np.mean(data)
            hist_std = np.std(data)
            
            mean_error = abs(stats['mean'] - hist_mean) / max(0.1, hist_std)
            std_error = abs(stats['stdDev'] - hist_std) / max(0.1, hist_std)
            sharpe_score = np.clip(stats['sharpe'], 0, 3)
            regularization = self._calculate_regularization(individual)
            
            return -0.4 * mean_error - 0.3 * std_error + 0.2 * sharpe_score - 0.1 * regularization
        except Exception as e:
            st.error(f"Fitness evaluation failed: {str(e)}")
            return -1000.0

    def _calculate_regularization(self, individual: Individual) -> float:
        penalty = 0.0
        for param_name, value in individual.__dict__.items():
            min_val, max_val = self.bounds[param_name]
            range_val = max_val - min_val
            center = (max_val + min_val) / 2
            deviation = abs(value - center) / (range_val / 2)
            penalty += deviation
            if deviation > 0.9:
                st.warning(f"Parameter {param_name} is near boundary: {value:.3f}")
        return penalty / len(individual.__dict__)

    def _calculate_diversity(self, population: List[Individual]) -> float:
        if not population:
            return 0.0
        total_variance = 0.0
        for param_name in population[0].__dict__.keys():
            values = [getattr(ind, param_name) for ind in population]
            total_variance += np.var(values)
        return np.sqrt(total_variance / len(population[0].__dict__))

    def _has_converged(self, history: List[Dict], window: int = 10) -> bool:
        if len(history) < window:
            return False
        recent = history[-window:]
        fitness_values = [h['train'] for h in recent]
        return (fitness_values[-1] - fitness_values[0]) < 0.001

    def _calculate_improvement(self, base_inputs: Dict, best_individual: Individual) -> Dict[str, float]:
        improvement = {}
        base_betas = base_inputs['betas']
        optimized_betas = best_individual.to_doc()
        for param in base_betas.keys():
            improvement[param] = optimized_betas[param] - base_betas[param]
        improvement['mean_reversion'] = best_individual.mean_reversion - base_inputs['meanReversion']
        return improvement

    def _evolve(self, ranked: List[Tuple[Individual, float]], population_size: int, elite_ratio: float,
                mutation_rate: float, mutation_strength: float) -> List[Individual]:
        new_population = []
        elite_count = int(population_size * elite_ratio)
        
        for i in range(elite_count):
            new_population.append(ranked[i][0])
        
        while len(new_population) < population_size:
            parent1 = self._tournament_select(ranked)
            parent2 = self._tournament_select(ranked)
            child = self._crossover(parent1, parent2) if random.random() < 0.8 else (parent1 if random.random() < 0.5 else parent2)
            child = self._mutate(child, mutation_rate, mutation_strength)
            new_population.append(child)
        
        return new_population

    def _tournament_select(self, ranked: List[Tuple[Individual, float]], tournament_size: int = 5) -> Individual:
        tournament = random.sample(ranked, min(tournament_size, len(ranked)))
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        params = {param: getattr(parent1, param) if random.random() < 0.5 else getattr(parent2, param)
                  for param in parent1.__dict__.keys()}
        return Individual(**params)

    def _mutate(self, individual: Individual, mutation_rate: float, mutation_strength: float) -> Individual:
        params = individual.__dict__.copy()
        for param in params.keys():
            if random.random() < mutation_rate:
                min_val, max_val = self.bounds[param]
                range_val = max_val - min_val
                mutation = (random.random() - 0.5) * range_val * mutation_strength
                params[param] = np.clip(params[param] + mutation, min_val, max_val)
        return Individual(**params)

    def export_results(self, results: GAResult) -> Dict:
        return {
            'optimizedBetas': results.best_individual.to_dict(),
            'optimizedMeanReversion': results.best_individual.mean_reversion,
            'trainingScore': results.best_fitness,
            'validationScore': results.validation_score,
            'improvement': results.improvement,
            'convergedAt': results.converged_at,
            'diagnostics': {
                'finalDiversity': results.convergence_history[-1]['diversity'] if results.convergence_history else 0,
                'fitnessProgress': results.convergence_history
            }
        }
