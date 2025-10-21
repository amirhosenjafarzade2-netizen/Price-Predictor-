"""
Professional Monte Carlo Model with parallel processing
Supports GARCH volatility, fat-tailed distributions, and mean reversion
"""

import numpy as np
from scipy.stats import t, norm
from typing import Dict, List, Optional
from utils import validate_inputs, calculate_stats
from config import PERFORMANCE_CONFIG, MC_DEFAULTS, DISTRIBUTION_CONFIG, VALIDATION
import logging
from multiprocessing import Pool, cpu_count, Manager
import streamlit as st
import time

logger = logging.getLogger(__name__)

def _simulate_path(args: tuple) -> float:
    """
    Helper function for simulation
    """
    (mu, sigma, horizon, mean_reversion, dist_type, tdf,
     enable_garch, garch_omega, garch_alpha, garch_beta, skew, seed_offset) = args
    
    # Set unique seed for this simulation
    if seed_offset is not None:
        np.random.seed(seed_offset)
    
    steps = max(1, int(horizon * MC_DEFAULTS['steps_per_year']))
    dt = horizon / steps
    wealth = 1.0  # Start with initial wealth of 1 for compound return calculation
    
    mu_dec = mu / 100.0  # Convert percentage to decimal
    sigma_dec = sigma / 100.0  # Convert percentage to decimal
    theta = mean_reversion  # Annual reversion speed
    
    current_r = mu  # Start with annualized return rate at mean (%)
    
    if enable_garch:
        vol_dec = sigma_dec
        for i in range(steps):
            vol_dt_dec = vol_dec * np.sqrt(dt)
            
            # Generate shock
            if dist_type == 't':
                standardized_shock = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2)) if tdf > 2 else np.random.normal(0, 1)
            elif dist_type == 'skewt':
                base = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2)) if tdf > 2 else np.random.normal(0, 1)
                standardized_shock = base * (1 + skew * np.random.randn())
            else:
                standardized_shock = np.random.normal(0, 1)
            
            shock_dec = standardized_shock * vol_dt_dec
            
            # Update annualized return rate with mean reversion
            drift_dec = mu_dec * dt
            reversion_dec = theta * dt * (mu_dec - current_r / 100.0)
            current_r_dec = current_r / 100.0 + drift_dec + reversion_dec + shock_dec
            current_r = current_r_dec * 100.0  # Back to %
            
            # Incremental return (properly scaled for time step)
            inc_return_dec = current_r_dec * dt
            
            # Compound wealth with clipping to prevent numerical instability
            wealth *= (1 + np.clip(inc_return_dec, -0.5, 0.5))
            
            # Update volatility (GARCH)
            epsilon_squared = (shock_dec / np.sqrt(dt)) ** 2
            vol_squared_dec = garch_omega + garch_alpha * epsilon_squared + garch_beta * (vol_dec ** 2)
            vol_dec = np.sqrt(np.clip(vol_squared_dec, MC_DEFAULTS['min_vol'], MC_DEFAULTS['max_vol']))
            
            # Debug logging for extreme values
            if i % 100 == 0 and (wealth > 1000 or wealth < 0.01):
                logger.debug(f"Step {i}: wealth={wealth:.4f}, current_r={current_r:.2f}%, vol_dec={vol_dec*100:.2f}%")
    
    else:
        for i in range(steps):
            vol_dt_dec = sigma_dec * np.sqrt(dt)
            
            # Generate shock
            if dist_type == 't':
                shock_dec = t.rvs(df=tdf) * vol_dt_dec / np.sqrt(tdf / (tdf - 2)) if tdf > 2 else np.random.normal(0, vol_dt_dec)
            elif dist_type == 'skewt':
                base_shock = t.rvs(df=tdf) * vol_dt_dec / np.sqrt(tdf / (tdf - 2)) if tdf > 2 else np.random.normal(0, vol_dt_dec)
                shock_dec = base_shock * (1 + skew * np.random.randn())
            else:
                shock_dec = np.random.normal(0, vol_dt_dec)
            
            # Update annualized return rate with mean reversion
            drift_dec = mu_dec * dt
            reversion_dec = theta * dt * (mu_dec - current_r / 100.0)
            current_r_dec = current_r / 100.0 + drift_dec + reversion_dec + shock_dec
            current_r = current_r_dec * 100.0  # Back to %
            
            # Incremental return (properly scaled for time step)
            inc_return_dec = current_r_dec * dt
            
            # Compound wealth with clipping to prevent numerical instability
            wealth *= (1 + np.clip(inc_return_dec, -0.5, 0.5))
            
            # Debug logging for extreme values
            if i % 100 == 0 and (wealth > 1000 or wealth < 0.01):
                logger.debug(f"Step {i}: wealth={wealth:.4f}, current_r={current_r:.2f}%, vol_dec={sigma_dec*100:.2f}%")
    
    # Return cumulative compound return as percentage
    final_return = (wealth - 1.0) * 100.0
    if abs(final_return) > 10000:
        logger.warning(f"Extreme return detected: {final_return:.2f}%")
    return np.clip(final_return, -1000, 10000)  # Cap extreme returns

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
        self.use_multiprocessing = False  # Temporarily disable for debugging
        self.n_processes = PERFORMANCE_CONFIG['n_processes'] or max(1, cpu_count() - 1)
        logger.info(f"Model initialized (multiprocessing: {self.use_multiprocessing}, processes: {self.n_processes})")

    def run(self, inputs: Dict) -> Dict:
        """
        Run Monte Carlo simulation with progress bar
        
        Args:
            inputs: Dictionary with all simulation parameters
            
        Returns:
            Dictionary with stats, riskMetrics, percentiles, and results
        """
        try:
            logger.info("Starting Monte Carlo simulation")
            
            # Validate inputs
            is_valid, errors = validate_inputs(inputs)
            if not is_valid:
                error_msgs = [e.message for e in errors if e.severity == 'error']
                logger.error(f"Validation failed: {error_msgs}")
                raise ValueError("Validation failed:\n" + "\n".join(error_msgs))
            
            logger.debug(f"Inputs: {inputs}")
            
            # Additional validation for distribution parameters
            dist_type = inputs.get('distType', 'normal')
            tdf = inputs.get('tdf', DISTRIBUTION_CONFIG[dist_type]['default_tdf'] if dist_type in ['t', 'skewt'] else 5.0)
            if dist_type in ['t', 'skewt'] and (tdf < 2.1 or tdf > 30):
                logger.warning(f"Invalid tdf={tdf}, resetting to 5.0")
                tdf = 5.0
            
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

            # Adjust mean based on macro factors with horizon scaling
            mu = inputs['baseMu']
            betas = inputs.get('betas', {})
            
            beta_mapping = {
                'realRate': 'real',
                'expRealRate': 'expReal',
                'inflExp': 'infl',
                'vix': 'vix',
                'dxy': 'dxy',
                'creditSpread': 'credit',
                'termSpread': 'term'
            }
            
            horizon = inputs['horizon']
            macro_weight = np.exp(-horizon / 10.0)  # Exponential decay for long horizons
            mu += sum(
                betas.get(beta_mapping.get(factor, factor), 0) * value * macro_weight
                for factor, value in macro_factors.items()
                if beta_mapping.get(factor, factor) in betas and value is not None
            )

            # Adjust volatility with horizon scaling
            sigma = inputs['baseSigma']
            if 'vix' in betas:
                vix_factor = (macro_factors['vix'] / 15.0 - 1)
                sigma *= (1 + betas['vix'] * vix_factor * macro_weight)
            sigma = np.clip(sigma, VALIDATION['min_sigma'], VALIDATION['max_sigma'])

            # Simulation parameters
            horizon = inputs['horizon']
            iterations = int(inputs['iters'])
            mean_reversion = inputs.get('meanReversion', 0)
            dist_type = inputs.get('distType', 'normal')
            tdf = inputs.get('tdf', DISTRIBUTION_CONFIG[dist_type]['default_tdf'] if dist_type in ['t', 'skewt'] else 5.0)
            skew = inputs.get('skew', DISTRIBUTION_CONFIG[dist_type].get('default_skew', 0.2) if dist_type == 'skewt' else 0.0)
            enable_garch = inputs.get('enableGarch', False)
            
            # Validate GARCH parameters
            garch_omega = inputs.get('garchOmega', 0.0001) if enable_garch else 0
            garch_alpha = inputs.get('garchAlpha', 0.08) if enable_garch else 0
            garch_beta = inputs.get('garchBeta', 0.90) if enable_garch else 0
            if enable_garch and (garch_alpha + garch_beta >= 1.0):
                logger.warning(f"Unstable GARCH parameters (alpha+beta={garch_alpha+garch_beta}); disabling GARCH")
                enable_garch = False
                garch_omega = garch_alpha = garch_beta = 0
            
            logger.info(f"Simulation parameters: mu={mu:.2f}%, sigma={sigma:.2f}%, horizon={horizon}, iterations={iterations}, dist_type={dist_type}, enable_garch={enable_garch}")
            
            # Initialize progress bar
            progress_bar = st.progress(0)
            st.text("Running Monte Carlo simulation...")
            time.sleep(0.1)  # Brief pause to ensure UI updates

            # Run simulations (force sequential for debugging)
            logger.info("Starting simulations")
            returns = self._run_sequential(mu, sigma, horizon, mean_reversion, dist_type, tdf,
                                         enable_garch, inputs, iterations, seed, progress_bar)
            
            logger.info("Simulations completed")
            progress_bar.progress(1.0)
            
            # Calculate statistics
            logger.info("Calculating statistics")
            result = calculate_stats(returns)
            logger.info(f"Simulation results: mean={result['stats']['mean']:.2f}%, std={result['stats']['stdDev']:.2f}%")
            return result
        
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise

    def _run_sequential(self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
                       enable_garch, inputs, iterations, base_seed, progress_bar) -> List[float]:
        """Run simulations sequentially with progress bar"""
        logger.info(f"Running sequential simulation with {iterations} iterations")
        
        returns = []
        garch_omega = inputs.get('garchOmega', 0.0001) if enable_garch else 0
        garch_alpha = inputs.get('garchAlpha', 0.08) if enable_garch else 0
        garch_beta = inputs.get('garchBeta', 0.90) if enable_garch else 0
        skew = inputs.get('skew', DISTRIBUTION_CONFIG[dist_type].get('default_skew', 0.2) if dist_type == 'skewt' else 0.0)
        
        try:
            for i in range(iterations):
                args = (
                    mu, sigma, horizon, mean_reversion, dist_type, tdf,
                    enable_garch, garch_omega, garch_alpha, garch_beta, skew,
                    (base_seed + i) if base_seed is not None else None
                )
                returns.append(_simulate_path(args))
                progress_bar.progress((i + 1) / iterations)
                if i % 100 == 0:
                    logger.debug(f"Iteration {i}/{iterations}")
        except Exception as e:
            logger.error(f"Sequential simulation failed at iteration {i}: {str(e)}")
            raise
        
        return returns

    def _run_parallel(self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
                     enable_garch, inputs, iterations, base_seed, progress_bar) -> List[float]:
        """Run simulations in parallel using multiprocessing with progress bar"""
        logger.info(f"Running parallel simulation with {self.n_processes} processes")
        
        garch_omega = inputs.get('garchOmega', 0.0001) if enable_garch else 0
        garch_alpha = inputs.get('garchAlpha', 0.08) if enable_garch else 0
        garch_beta = inputs.get('garchBeta', 0.90) if enable_garch else 0
        skew = inputs.get('skew', DISTRIBUTION_CONFIG[dist_type].get('default_skew', 0.2) if dist_type == 'skewt' else 0.0)
        
        # Initialize shared counter for progress tracking
        manager = Manager()
        counter = manager.Value('i', 0)
        
        args_list = [
            (
                mu, sigma, horizon, mean_reversion, dist_type, tdf,
                enable_garch, garch_omega, garch_alpha, garch_beta, skew,
                (base_seed + i) if base_seed is not None else None,
                counter, iterations
            )
            for i in range(iterations)
        ]
        
        try:
            with Pool(processes=self.n_processes) as pool:
                returns = pool.map(_simulate_path, args_list)
        except Exception as e:
            logger.error(f"Parallel simulation failed: {str(e)}")
            raise
        
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
        horizon = inputs['horizon']
        iterations = inputs.get('iters', 1000)
        
        try:
            for i in range(len(historical_data) - 1):
                temp_inputs = inputs.copy()
                temp_inputs['iters'] = iterations
                
                if i > 0:
                    hist_vol = np.std(historical_data[:i+1])
                    temp_inputs['baseSigma'] = max(VALIDATION['min_sigma'], hist_vol)
                
                result = self.run(temp_inputs)
                predictions.append(result['stats']['mean'])
                logger.debug(f"Backtest period {i+1}: predicted {result['stats']['mean']:.2f}%")
        except Exception as e:
            logger.error(f"Backtest period {i+1} failed: {str(e)}")
            raise
        
        actuals = historical_data[1:]
        errors = np.array(predictions) - np.array(actuals)
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
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
        
        try:
            for factor in factors:
                if factor not in ranges:
                    continue
                
                logger.debug(f"Analyzing sensitivity to {factor}")
                base_inputs = inputs.copy()
                base_inputs['iters'] = 1000
                
                for value in ranges[factor]:
                    test_inputs = base_inputs.copy()
                    test_inputs[factor] = float(value)
                    
                    result = self.run(test_inputs)
                    mean_return = result['stats']['mean']
                    
                    results['factor'].append(factor)
                    results['value'].append(float(value))
                    results['mean_return'].append(float(mean_return))
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {str(e)}")
            raise
        
        logger.info("Sensitivity analysis complete")
        return results

    def fetch_live_macros(self):
        """
        Placeholder for fetching live macro data
        Currently returns empty dict as per user requirement
        """
        logger.info("Live macro data fetching disabled")
        return {}
