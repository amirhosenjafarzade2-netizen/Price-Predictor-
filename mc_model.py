import numpy as np
from scipy.stats import t, norm
from typing import Dict, List
from utils import validate_inputs, calculate_stats


class ProfessionalMCModel:
    def __init__(self):
        self.asset_presets = {
            'equity': {
                'name': 'S&P 500',
                'mean': 9.2,
                'sigma': 16.5,
                'betas': {
                    'real': -0.35,
                    'expReal': -0.22,
                    'infl': 0.15,
                    'vix': 0.25,
                    'dxy': -0.02,
                    'credit': -0.08,
                    'term': 0.05
                },
                'meanReversion': 0.15
            },
            'bonds': {
                'name': '10Y Treasury',
                'mean': 4.5,
                'sigma': 7.8,
                'betas': {
                    'real': -0.85,
                    'expReal': -0.65,
                    'infl': -0.25,
                    'vix': 0.12,
                    'dxy': 0.03,
                    'credit': -0.15,
                    'term': 0.20
                },
                'meanReversion': 0.25
            },
            'reits': {
                'name': 'REITs',
                'mean': 8.5,
                'sigma': 20.2,
                'betas': {
                    'real': -0.45,
                    'expReal': -0.30,
                    'infl': 0.20,
                    'vix': 0.30,
                    'dxy': -0.03,
                    'credit': -0.12,
                    'term': 0.08
                },
                'meanReversion': 0.20
            },
            'commodities': {
                'name': 'Commodities',
                'mean': 6.8,
                'sigma': 22.5,
                'betas': {
                    'real': -0.20,
                    'expReal': -0.15,
                    'infl': 0.35,
                    'vix': 0.28,
                    'dxy': -0.05,
                    'credit': -0.05,
                    'term': 0.03
                },
                'meanReversion': 0.10
            },
            'custom': {
                'name': 'Custom Asset',
                'mean': 8.0,
                'sigma': 15.0,
                'betas': {
                    'real': -0.30,
                    'expReal': -0.20,
                    'infl': 0.10,
                    'vix': 0.20,
                    'dxy': -0.01,
                    'credit': -0.06,
                    'term': 0.04
                },
                'meanReversion': 0.15
            }
        }

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
            raise ValueError("\n".join(errors))
        
        # Set random seed for reproducibility
        seed = inputs.get('seed')
        if seed is not None:
            np.random.seed(int(seed))
        
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
            sigma *= max(0.1, vix_adjustment)  # Prevent negative/zero volatility

        # Generate simulations
        returns = []
        horizon = inputs['horizon']
        iterations = int(inputs['iters'])
        
        for i in range(iterations):
            if inputs.get('enableGarch', False):
                sim_returns = self._simulate_garch(
                    mu / 100,
                    sigma / 100,
                    horizon,
                    inputs.get('garchOmega', 0.0001),
                    inputs.get('garchAlpha', 0.08),
                    inputs.get('garchBeta', 0.90),
                    inputs.get('meanReversion', 0),
                    inputs.get('distType', 'normal'),
                    inputs.get('tdf', 5)
                )
            else:
                sim_returns = self._simulate_simple(
                    mu / 100,
                    sigma / 100,
                    horizon,
                    inputs.get('meanReversion', 0),
                    inputs.get('distType', 'normal'),
                    inputs.get('tdf', 5)
                )
            
            # Convert final return back to percentage
            returns.append(sim_returns[-1] * 100)

        # Calculate and return statistics
        return calculate_stats(returns)

    def _simulate_simple(
        self,
        mu: float,
        sigma: float,
        horizon: float,
        mean_reversion: float,
        dist_type: str,
        tdf: float
    ) -> List[float]:
        """
        Simple Monte Carlo path without GARCH
        
        Returns cumulative return path
        """
        steps = max(1, int(horizon * 252))  # Daily steps
        returns = [0.0]  # Start at zero return
        
        # Adjust parameters for time scaling
        dt = horizon / steps
        mu_dt = mu * dt
        sigma_dt = sigma * np.sqrt(dt)
        
        for _ in range(steps):
            # Generate random shock based on distribution type
            if dist_type == 't':
                # Standardized t-distribution
                shock = t.rvs(df=tdf) * sigma_dt / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                # Skewed t-distribution (simple approximation)
                base_shock = t.rvs(df=tdf) * sigma_dt / np.sqrt(tdf / (tdf - 2))
                skew_factor = 1 + 0.2 * np.random.randn()
                shock = base_shock * skew_factor
            else:
                # Normal distribution
                shock = np.random.normal(0, sigma_dt)
            
            # Mean reversion: pull towards long-term mean
            current_return = returns[-1]
            new_return = (1 - mean_reversion) * current_return + mean_reversion * mu_dt + shock
            returns.append(new_return)
        
        # Return cumulative returns
        cumulative = [sum(returns[:i+1]) for i in range(len(returns))]
        return cumulative

    def _simulate_garch(
        self,
        mu: float,
        sigma: float,
        horizon: float,
        omega: float,
        alpha: float,
        beta: float,
        mean_reversion: float,
        dist_type: str,
        tdf: float
    ) -> List[float]:
        """
        GARCH(1,1) volatility clustering simulation
        
        GARCH model: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
        """
        steps = max(1, int(horizon * 252))
        returns = [0.0]
        
        # Initialize volatility
        vol = sigma
        dt = horizon / steps
        mu_dt = mu * dt
        
        for _ in range(steps):
            # Time-scaled volatility
            vol_dt = vol * np.sqrt(dt)
            
            # Generate standardized shock
            if dist_type == 't':
                standardized_shock = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                base = t.rvs(df=tdf) / np.sqrt(tdf / (tdf - 2))
                standardized_shock = base * (1 + 0.2 * np.random.randn())
            else:
                standardized_shock = np.random.normal(0, 1)
            
            # Scale by current volatility
            shock = standardized_shock * vol_dt
            
            # Mean reversion
            current_return = returns[-1]
            new_return = (1 - mean_reversion) * current_return + mean_reversion * mu_dt + shock
            returns.append(new_return)
            
            # Update volatility using GARCH(1,1)
            # Vol is in annualized terms, shock is already time-scaled
            epsilon_squared = (shock / np.sqrt(dt)) ** 2
            vol_squared = omega + alpha * epsilon_squared + beta * (vol ** 2)
            vol = np.sqrt(max(0.0001, vol_squared))  # Prevent negative variance
        
        # Return cumulative returns
        cumulative = [sum(returns[:i+1]) for i in range(len(returns))]
        return cumulative

    def backtest(self, inputs: Dict) -> Dict:
        """
        Walk-forward backtest on historical data
        
        Uses each historical period to predict the next period
        """
        historical_data = inputs.get('historical_data', [])
        
        if not historical_data or len(historical_data) < 2:
            raise ValueError("Backtesting requires at least 2 historical returns")
        
        predictions = []
        
        # Walk-forward: use data up to period i to predict period i+1
        for i in range(len(historical_data) - 1):
            # Create inputs for this iteration
            temp_inputs = inputs.copy()
            temp_inputs['iters'] = 1000  # Reduce iterations for speed
            
            # Use historical volatility if available
            if i > 0:
                hist_vol = np.std(historical_data[:i+1])
                temp_inputs['baseSigma'] = max(1.0, hist_vol)
            
            # Run simulation
            try:
                result = self.run(temp_inputs)
                predictions.append(result['stats']['mean'])
            except Exception as e:
                # If simulation fails, use previous prediction or baseline
                if predictions:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(inputs.get('baseMu', 0))
        
        # Align predictions with actuals (predictions are for next period)
        actuals = historical_data[1:]  # Skip first period (no prediction for it)
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
            
            base_inputs = inputs.copy()
            base_inputs['iters'] = 1000  # Reduce iterations for speed
            
            for value in ranges[factor]:
                test_inputs = base_inputs.copy()
                test_inputs[factor] = float(value)
                
                try:
                    sim_result = self.run(test_inputs)
                    mean_return = sim_result['stats']['mean']
                except Exception:
                    mean_return = 0.0
                
                results['factor'].append(factor)
                results['value'].append(float(value))
                results['mean_return'].append(float(mean_return))
        
        return results
