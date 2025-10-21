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
from multiprocessing import Pool, cpu_count
import os

logger = logging.getLogger(__name__)

def _simulate_path(args: tuple) -> float:
    """
    Helper function for parallel simulation
    Must be at module level for multiprocessing
    """
    (mu, sigma, horizon, mean_reversion, dist_type, tdf,
     enable_garch, garch_omega, garch_alpha, garch_beta, skew, seed_offset) = args
    
    # Set unique seed for this simulation
    if seed_offset is not None:
        np.random.seed(seed_offset)
    
    steps = max(1, int(horizon * MC_DEFAULTS['steps_per_year']))
    dt = horizon / steps
    wealth = 1.0  # Start with initial wealth of 1 for compound return calculation
    
    mu_dec = mu / 100.0
    sigma_dec = sigma / 100.0
    theta = mean_reversion  # Annual reversion speed
    
    current_r = mu  # Start with annualized return rate at mean (%)
    
    if enable_garch:
        vol_dec = sigma_dec
        for _ in range(steps):
            vol_dt_dec = vol_dec * np.sqrt(dt)
            
            # Generate shock
            if dist_type == 't':
                standardized_shock = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                base = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2))
                standardized_shock = base * (1 + skew * np.random.randn())
            else:
                standardized_shock = np.random.normal(0, 1)
            
            shock_dec = standardized_shock * vol_dt_dec
            
            # Update annualized return rate with mean reversion (always include drift)
            drift_dec = mu_dec * dt
            reversion_dec = theta * dt * (mu_dec - current_r / 100.0)  # Revert to mu_dec
            current_r_dec = current_r / 100.0 + drift_dec + reversion_dec + shock_dec
            current_r = current_r_dec * 100.0  # Back to %
            
            # Incremental return
            inc_return_dec = current_r_dec * dt  # Daily contribution
            
            # Compound wealth
            wealth *= (1 + inc_return_dec)
            
            # Update volatility (GARCH)
            epsilon_squared = (shock_dec / np.sqrt(dt)) ** 2
            vol_squared_dec = garch_omega + garch_alpha * epsilon_squared + garch_beta * (vol_dec ** 2)
            vol_dec = np.sqrt(np.clip(vol_squared_dec, MC_DEFAULTS['min_vol'], MC_DEFAULTS['max_vol']))
    else:
        for _ in range(steps):
            vol_dt_dec = sigma_dec * np.sqrt(dt)
            
            # Generate shock
            if dist_type == 't':
                shock_dec = t.rvs(df=tdf) * vol_dt_dec / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                base_shock = t.rvs(df=tdf) * vol_dt_dec / np.sqrt(tdf / (tdf - 2))
                shock_dec = base_shock * (1 + skew * np.random.randn())
            else:
                shock_dec = np.random.normal(0, vol_dt_dec)
            
            # Update annualized return rate with mean reversion (always include drift)
            drift_dec = mu_dec * dt
            reversion_dec = theta * dt * (mu_dec - current_r / 100.0)  # Revert to mu_dec
            current_r_dec = current_r / 100.0 + drift_dec + reversion_dec + shock_dec
            current_r = current_r_dec * 100.0  # Back to %
            
            # Incremental return
            inc_return_dec = current_r_dec * dt  # Daily contribution
            
            # Compound wealth
            wealth *= (1 + inc_return_dec)
    
    # Return cumulative compound return as percentage
    return (wealth - 1.0) * 100.0

class ProfessionalMCModel:
    # ... (rest of class unchanged)

    def run(self, inputs: Dict) -> Dict:
        # ... (validation and param setup unchanged)

        # Run simulations
        if self.use_multiprocessing and iterations >= PERFORMANCE_CONFIG['batch_size']:
            returns = self._run_parallel(mu, sigma, horizon, mean_reversion, dist_type, tdf,
                                        enable_garch, inputs, iterations, seed)
        else:
            returns = self._run_sequential(mu, sigma, horizon, mean_reversion, dist_type, tdf,
                                         enable_garch, inputs, iterations, seed)
        
        # Calculate statistics
        return calculate_stats(returns)

def _run_sequential(self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
                    enable_garch, inputs, iterations, base_seed) -> List[float]:
    """Run simulations sequentially"""
    logger.info(f"Running sequential simulation with {iterations} iterations")
    
    returns = []
    garch_omega = inputs.get('garchOmega', 0.0001) if enable_garch else 0
    garch_alpha = inputs.get('garchAlpha', 0.08) if enable_garch else 0
    garch_beta = inputs.get('garchBeta', 0.90) if enable_garch else 0
    skew = inputs.get('skew', DISTRIBUTION_CONFIG[dist_type].get('default_skew', 0.2) if dist_type == 'skewt' else 0.0)
    
    progress_bar = st.progress(0)
    for i in range(iterations):
        args = (
            mu, sigma, horizon, mean_reversion, dist_type, tdf,
            enable_garch, garch_omega, garch_alpha, garch_beta, skew,
            (base_seed + i) if base_seed is not None else None
        )
        returns.append(_simulate_path(args))
        progress_bar.progress((i + 1) / iterations)
    
    return returns

def _run_parallel(self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
                  enable_garch, inputs, iterations, base_seed) -> List[float]:
    """Run simulations in parallel using multiprocessing"""
    logger.info(f"Running parallel simulation with {self.n_processes} processes")
    
    garch_omega = inputs.get('garchOmega', 0.0001) if enable_garch else 0
    garch_alpha = inputs.get('garchAlpha', 0.08) if enable_garch else 0
    garch_beta = inputs.get('garchBeta', 0.90) if enable_garch else 0
    skew = inputs.get('skew', DISTRIBUTION_CONFIG[dist_type].get('default_skew', 0.2) if dist_type == 'skewt' else 0.0)
    
    args_list = [
        (
            mu, sigma, horizon, mean_reversion, dist_type, tdf,
            enable_garch, garch_omega, garch_alpha, garch_beta, skew,
            (base_seed + i) if base_seed is not None else None
        )
        for i in range(iterations)
    ]
    
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
                temp_inputs['baseSigma'] = max(VALIDATION['min_sigma'], hist_vol)
            
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

    def fetch_live_macros(self):
        """
        Placeholder for fetching live macro data
        Currently returns empty dict as per user requirement
        """
        logger.info("Live macro data fetching disabled")
        return {}
