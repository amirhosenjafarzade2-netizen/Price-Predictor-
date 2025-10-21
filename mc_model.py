"""
Professional Monte Carlo Model with parallel processing
Supports GARCH volatility, fat-tailed distributions, and mean reversion
"""

import numpy as np
from scipy.stats import t, norm
from typing import Dict, List, Optional
from utils import validate_inputs, calculate_stats
from config import PERFORMANCE_CONFIG, MC_DEFAULTS
import logging
from multiprocessing import Pool, cpu_count
import os

logger = logging.getLogger(__name__)


def _simulate_path(args: tuple) -> float:
    """
    Helper function for parallel simulation
    Must be at module level for multiprocessing
    """
    (mu, sigma, horizon, mean_reversion, dist_type, tdf,
     enable_garch, garch_omega, garch_alpha, garch_beta, seed_offset) = args
    
    # Set unique seed for this simulation
    if seed_offset is not None:
        np.random.seed(seed_offset)
    
    steps = max(1, int(horizon * MC_DEFAULTS['steps_per_year']))
    returns = [0.0]
    dt = horizon / steps
    
    if enable_garch:
        vol = sigma
        mu_dt = mu * dt
        
        for _ in range(steps):
            vol_dt = vol * np.sqrt(dt)
            
            # Generate shock
            if dist_type == 't':
                standardized_shock = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                base = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2))
                standardized_shock = base * (1 + 0.2 * np.random.randn())
            else:
                standardized_shock = np.random.normal(0, 1)
            
            shock = standardized_shock * vol_dt
            
            # Mean reversion
            current_return = returns[-1]
            new_return = (1 - mean_reversion) * current_return + mean_reversion * mu_dt + shock
            returns.append(new_return)
            
            # Update volatility (GARCH)
            epsilon_squared = (shock / np.sqrt(dt)) ** 2
            vol_squared = garch_omega + garch_alpha * epsilon_squared + garch_beta * (vol ** 2)
            vol = np.sqrt(np.clip(vol_squared, MC_DEFAULTS['min_vol'], MC_DEFAULTS['max_vol']))
    else:
        mu_dt = mu * dt
        sigma_dt = sigma * np.sqrt(dt)
        
        for _ in range(steps):
            if dist_type == 't':
                shock = t.rvs(df=tdf) * sigma_dt / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                base_shock = t.rvs(df=tdf) * sigma_dt / np.sqrt(tdf / (tdf - 2))
                shock = base_shock * (1 + 0.2 * np.random.randn())
            else:
                shock = np.random.normal(0, sigma_dt)
            
            current_return = returns[-1]
            new_return = (1 - mean_reversion) * current_return + mean_reversion * mu_dt + shock
            returns.append(new_return)
    
    # Return cumulative return as percentage
    return sum(returns) * 100


class ProfessionalMCModel:
    """
    Professional Monte Carlo Asset Return Simulator
    
    Features:
    - Fat-tailed distributions (Student-t, Skewed-t)
    - GARCH(1,1) volatility clustering
    - Mean reversion dynamics
    - Macro factor sensitivities
    - Parallel processing support
    
    Examples:
        >>> model = ProfessionalMCModel()
        >>> inputs = {
        ...     'baseMu': 10.0,
        ...     'baseSigma': 15.0,
        ...     'horizon': 1.0,
        ...     'iters': 10000,
        ...     'betas': {...}
        ... }
        >>> results = model.run(inputs)
        >>> print(f"Mean return: {results['stats']['mean']:.2f}%")
    
    References:
        - Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
        - McNeil, A. J., Frey, R., & Embrechts, P. (2015). "Quantitative Risk Management"
    """
    
    def __init__(self):
        self.use_multiprocessing = PERFORMANCE_CONFIG['enable_multiprocessing']
        self.n_processes = PERFORMANCE_CONFIG['n_processes'] or max(1, cpu_count() - 1)
        logger.info(f"Model initialized (multiprocessing: {self.use_multiprocessing}, processes: {self.n_processes})")

    def run(self, inputs: Dict) -> Dict:
        """
        Run Monte Carlo simulation
        
        Args:
            inputs: Dictionary with all simulation parameters
            
        Returns:
            Dictionary with stats, riskMetrics, percentiles, and results
        """
        # Validate inputs
        is_valid, errors = validate_inputs(inputs)
        if not is_valid:
            error_msgs = [e.message for e in errors if e.severity == 'error']
            raise ValueError("Validation failed:\n" + "\n".join(error_msgs))
        
        # Set random seed for reproducibility
        seed = inputs.get('seed')
        if seed is not None:
            np.random.seed(int(seed))
            logger.info(f"Using seed: {seed}")
        
        # Handle null values with defaults
        macro_factors = {
            'realRate': inputs.get('realRate', 0) or 0,
            'expRealRate': inputs.get('expRealRate', 0) or 0,
            'inflExp': inputs.get('inflExp', 0) or 0,
            'vix': inputs.get('vix', 15.0) or 15.0,
            'dxy': inputs.get('dxy', 100.0) or 100.0,
            'creditSpread': inputs.get('creditSpread', 100.0) or 100.0,
            'termSpread': inputs.get('termSpread', 0) or 0
        }

        # Adjust mean based on macro factors
        mu = inputs['baseMu']
        betas = inputs.get('betas', {})
        
        # Map macro factors to beta keys
        beta_mapping = {
            'realRate': 'real',
            'expRealRate': 'expReal',
            'inflExp': 'infl',
            'vix': 'vix',
            'dxy': 'dxy',
            'creditSpread': 'credit',
            'termSpread': 'term'
        }
        
        for macro_key, beta_key in beta_mapping.items():
            if beta_key in betas and macro_key in macro_factors:
                mu += betas[beta_key] * macro_factors[macro_key]

        # Adjust volatility based on VIX
        sigma = inputs['baseSigma']
        if 'vix' in betas and macro_factors['vix'] > 0:
            vix_adjustment = 1 + betas['vix'] * (macro_factors['vix'] / 15.0 - 1)
            sigma *= max(0.1, vix_adjustment)

        # Prepare simulation parameters
        horizon = inputs['horizon']
        iterations = int(inputs['iters'])
        mean_reversion = inputs.get('meanReversion', 0)
        dist_type = inputs.get('distType', 'normal')
        tdf = inputs.get('tdf', 5)
        enable_garch = inputs.get('enableGarch', False)
        
        logger.info(
            f"Starting simulation: {iterations} iterations, "
            f"horizon={horizon:.2f}y, μ={mu:.2f}%, σ={sigma:.2f}%"
        )

        # Run simulations (parallel or sequential)
        if self.use_multiprocessing and iterations >= 1000:
            returns = self._run_parallel(
                mu / 100, sigma / 100, horizon, mean_reversion,
                dist_type, tdf, enable_garch, inputs, iterations, seed
            )
        else:
            returns = self._run_sequential(
                mu / 100, sigma / 100, horizon, mean_reversion,
                dist_type, tdf, enable_garch, inputs, iterations
            )

        logger.info(f"Simulation complete: mean={np.mean(returns):.2f}%, std={np.std(returns):.2f}%")

        # Calculate and return statistics
        return calculate_stats(returns)

    def _run_sequential(
        self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
        enable_garch, inputs, iterations
    ) -> List[float]:
        """Run simulations sequentially (for small iteration counts)"""
        returns = []
        
        garch_params = (
            inputs.get('garchOmega', 0.0001),
            inputs.get('garchAlpha', 0.08),
            inputs.get('garchBeta', 0.90)
        ) if enable_garch else (0, 0, 0)
        
        for i in range(iterations):
            args = (
                mu, sigma, horizon, mean_reversion, dist_type, tdf,
                enable_garch, *garch_params, i
            )
            returns.append(_simulate_path(args))
        
        return returns

    def _run_parallel(
        self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
        enable_garch, inputs, iterations, base_seed
    ) -> List[float]:
        """Run simulations in parallel using multiprocessing"""
        logger.info(f"Running parallel simulation with {self.n_processes} processes")
        
        garch_params = (
            inputs.get('garchOmega', 0.0001),
            inputs.get('garchAlpha', 0.08),
            inputs.get('garchBeta', 0.90)
        ) if enable_garch else (0, 0, 0)
        
        # Create argument list for each simulation
        args_list = [
            (
                mu, sigma, horizon, mean_reversion, dist_type, tdf,
                enable_garch, *garch_params,
                (base_seed + i) if base_seed is not None else None
            )
            for i in range(iterations)
        ]
        
        # Run in parallel
        with Pool(processes=self.n_processes) as pool:
            returns = pool.map(_simulate_path, args_list)
        
        return returns

    def backtest(self, inputs: Dict) -> Dict:
        """
        Walk-forward backtest on historical data
        
        Uses each historical period to predict the next period
        """
        historical_data = inputs.get('historical_data', [])
        
        if not historical_data or len(historical_data) < 2:
            raise ValueError("Backtesting requires at least 2 historical returns")
        
        logger.info(f"Starting backtest with {len(historical_data)} historical periods")
        
        predictions = []
        
        # Walk-forward: use data up to period i to predict period i+1
        for i in range(len(historical_data) - 1):
            # Create inputs for this iteration
            temp_inputs = inputs.copy()
            temp_inputs['iters'] = 1000  # Reduce iterations for speed
            
            # Use historical volatility if we have enough data
            if i > 0:
                hist_vol = np.std(historical_data[:i+1])
                temp_inputs['baseSigma'] = max(1.0, hist_vol)
            
            # Run simulation
            try:
                result = self.run(temp_inputs)
                predictions.append(result['stats']['mean'])
                logger.debug(f"Backtest period {i+1}: predicted {result['stats']['mean']:.2f}%")
            except Exception as e:
                logger.warning(f"Backtest period {i+1} failed: {e}")
                # Use previous prediction or baseline
                if predictions:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(inputs.get('baseMu', 0))
        
        # Align predictions with actuals
        actuals = historical_data[1:]
        errors = np.array(predictions) - np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # R² calculation
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Hit rate: correct direction prediction
        hit_rate = np.mean(np.sign(predictions) == np.sign(actuals))
        
        logger.info(f"Backtest complete: R²={r2:.3f}, MAE={mae:.2f}%, Hit Rate={hit_rate*100:.1f}%")
        
        return {
            'years': list(range(1, len(actuals) + 1)),
            'actuals': actuals,
            'predictions': predictions,
            'stats': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'hitRate': float(hit_rate)
            }
        }

    def run_sensitivity_analysis(self, inputs: Dict) -> Dict:
        """
        Analyze sensitivity of returns to macro factors
        """
        logger.info("Starting sensitivity analysis")
        
        factors = ['vix', 'inflExp', 'realRate', 'creditSpread']
        
        ranges = {
            'vix': np.linspace(5, 50, 10),
            'inflExp': np.linspace(0, 5, 10),
            'realRate': np.linspace(-2, 5, 10),
            'creditSpread': np.linspace(0, 300, 10)
        }
        
        results = {
            'factor': [],
            'value': [],
            'mean_return': []
        }
        
        for factor in factors:
            if factor not in ranges:
                continue
            
            logger.debug(f"Analyzing sensitivity to {factor}")
            base_inputs = inputs.copy()
            base_inputs['iters'] = 1000  # Reduce iterations for speed
            
            for value in ranges[factor]:
                test_inputs = base_inputs.copy()
                test_inputs[factor] = float(value)
                
                try:
                    sim_result = self.run(test_inputs)
                    mean_return = sim_result['stats']['mean']
                except Exception as e:
                    logger.warning(f"Sensitivity test failed for {factor}={value}: {e}")
                    mean_return = 0.0
                
                results['factor'].append(factor)
                results['value'].append(float(value))
                results['mean_return'].append(float(mean_return))
        
        logger.info("Sensitivity analysis complete")
        return results
