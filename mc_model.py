import numpy as np
from scipy.stats import t, norm
from typing import Dict, List
import streamlit as st
from utils import validate_inputs, calculate_stats

class ProfessionalMCModel:
    def __init__(self):
        self.asset_presets = {
            'equity': {'name': 'S&P 500', 'mean': 9.2, 'sigma': 16.5, 'betas': {'real':-0.35, 'expReal':-0.22, 'infl':0.15, 'vix':0.25, 'dxy':-0.02, 'credit':-0.08, 'term':0.05}, 'meanReversion': 0.15},
            'bonds': {'name': '10Y Treasury', 'mean': 4.5, 'sigma': 7.8, 'betas': {'real':-0.85, 'expReal':-0.65, 'infl':-0.25, 'vix':0.12, 'dxy':0.03, 'credit':-0.15, 'term':0.20}, 'meanReversion': 0.25},
            'reits': {'name': 'REITs', 'mean': 8.5, 'sigma': 20.2, 'betas': {'real':-0.45, 'expReal':-0.30, 'infl':0.20, 'vix':0.30, 'dxy':-0.03, 'credit':-0.12, 'term':0.08}, 'meanReversion': 0.20},
            'commodities': {'name': 'Commodities', 'mean': 6.8, 'sigma': 22.5, 'betas': {'real':-0.20, 'expReal':-0.15, 'infl':0.35, 'vix':0.28, 'dxy':-0.05, 'credit':-0.05, 'term':0.03}, 'meanReversion': 0.10},
            'custom': {'name': 'Custom Asset', 'mean': 8.0, 'sigma': 15.0, 'betas': {'real':-0.30, 'expReal':-0.20, 'infl':0.10, 'vix':0.20, 'dxy':-0.01, 'credit':-0.06, 'term':0.04}, 'meanReversion': 0.15}
        }

    @st.cache_data
    def run(self, inputs: Dict) -> Dict:
        # Validate inputs
        is_valid, errors = validate_inputs(inputs)
        if not is_valid:
            raise ValueError("\n".join(errors))
        
        np.random.seed(inputs['seed'] if inputs['seed'] is not None else None)
        
        # Handle null values
        macro_factors = {
            'realRate': inputs['realRate'] if inputs['realRate'] is not None else 0,
            'expRealRate': inputs['expRealRate'] if inputs['expRealRate'] is not None else 0,
            'inflExp': inputs['inflExp'] if inputs['inflExp'] is not None else 0,
            'vix': inputs['vix'] if inputs['vix'] is not None else 15.0,
            'dxy': inputs['dxy'] if inputs['dxy'] is not None else 100.0,
            'creditSpread': inputs['creditSpread'] if inputs['creditSpread'] is not None else 100.0,
            'termSpread': inputs['termSpread'] if inputs['termSpread'] is not None else 0
        }

        # Validate extreme values
        for key, value in macro_factors.items():
            if value is not None and abs(value) > 1000:
                st.warning(f"Extreme value detected for {key}: {value}. Consider adjusting.")

        # Adjust mean based on macro factors
        mu = inputs['baseMu']
        mu += sum(
            inputs['betas'][k] * v for k, v in macro_factors.items()
            if k in inputs['betas'] and v is not None
        )

        # Adjust volatility
        sigma = inputs['baseSigma']
        sigma *= (1 + inputs['betas']['vix'] * (macro_factors['vix'] / 15.0 - 1))

        # Generate simulations
        returns = []
        last_return = mu / 100
        
        for _ in range(int(inputs['iters'])):
            if inputs['enableGarch']:
                sim_returns = self._simulate_garch(
                    mu / 100, sigma / 100, inputs['horizon'], inputs['garchOmega'],
                    inputs['garchAlpha'], inputs['garchBeta'], inputs['meanReversion'],
                    last_return, inputs['distType'], inputs['tdf']
                )
            else:
                sim_returns = self._simulate_simple(
                    mu / 100, sigma / 100, inputs['horizon'], inputs['meanReversion'],
                    last_return, inputs['distType'], inputs['tdf']
                )
            returns.append(sim_returns[-1] * 100)
            last_return = sim_returns[-1]

        # Calculate statistics
        return calculate_stats(returns)

    def _simulate_simple(self, mu, sigma, horizon, mean_reversion, last_return, dist_type, tdf):
        returns = [last_return]
        steps = int(horizon * 252)  # Assuming daily returns
        if steps < 1:
            raise ValueError("Simulation horizon too short for selected period unit")
        for _ in range(steps):
            if dist_type == 't':
                noise = t.rvs(df=tdf, size=1) * sigma / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                noise = t.rvs(df=tdf, size=1) * sigma / np.sqrt(tdf / (tdf - 2)) * (1 + 0.2 * np.random.randn())
            else:
                noise = norm.rvs(size=1) * sigma
            new_return = (1 - mean_reversion) * returns[-1] + mean_reversion * mu + noise
            returns.append(new_return)
        return returns

    def _simulate_garch(self, mu, sigma, horizon, omega, alpha, beta, mean_reversion, last_return, dist_type, tdf):
        returns = [last_return]
        vol = sigma
        steps = int(horizon * 252)
        if steps < 1:
            raise ValueError("Simulation horizon too short for selected period unit")
        for _ in range(steps):
            if dist_type == 't':
                noise = t.rvs(df=tdf, size=1) * vol / np.sqrt(tdf / (tdf - 2))
            elif dist_type == 'skewt':
                noise = t.rvs(df=tdf, size=1) * vol / np.sqrt(tdf / (tdf - 2)) * (1 + 0.2 * np.random.randn())
            else:
                noise = norm.rvs(size=1) * vol
            new_return = (1 - mean_reversion) * returns[-1] + mean_reversion * mu + noise
            returns.append(new_return)
            vol = np.sqrt(omega + alpha * noise**2 + beta * vol**2)
        return returns

    @st.cache_data
    def backtest(self, inputs: Dict) -> Dict:
        historical_data = inputs.get('historical_data', [])
        if not historical_data:
            historical_data = [10.3, -5.2, 18.7, 6.1, 12.2, 4.0, -8.1, 22.4, 9.9, 3.6, 15.2, -2.8, 11.5, 7.3, 13.8]
        if len(historical_data) < 2:
            raise ValueError("Backtesting requires at least 2 historical returns")
        
        predictions = []
        for i in range(len(historical_data)):
            temp_inputs = inputs.copy()
            temp_inputs['horizon'] = inputs['horizon']
            temp_inputs['iters'] = 1000
            result = self.run(temp_inputs)
            predictions.append(result['stats']['mean'])
        
        actuals = historical_data
        errors = np.array(predictions) - np.array(actuals)
        
        return {
            'years': list(range(len(historical_data))),
            'actuals': actuals,
            'predictions': predictions,
            'stats': {
                'mae': np.mean(np.abs(errors)),
                'rmse': np.sqrt(np.mean(errors**2)),
                'r2': 1 - np.sum(errors**2) / np.sum((actuals - np.mean(actuals))**2) if np.var(actuals) > 0 else 0,
                'hitRate': np.mean(np.sign(predictions) == np.sign(actuals))
            }
        }

    @st.cache_data
    def run_sensitivity_analysis(self, inputs: Dict) -> Dict:
        factors = ['vix', 'inflExp']
        ranges = {
            'vix': np.linspace(5, 50, 10),
            'inflExp': np.linspace(0, 5, 10)
        }
        results = {'factor': [], 'value': [], 'mean_return': []}
        
        for factor in factors:
            base_inputs = inputs.copy()
            for value in ranges[factor]:
                base_inputs[factor] = value
                sim_result = self.run(base_inputs)
                results['factor'].append(factor)
                results['value'].append(value)
                results['mean_return'].append(sim_result['stats']['mean'])
        
        return results

    def fetch_live_macros(self):
        # Placeholder for live data fetching
        return {
            'realRate': 2.1,
            'expRealRate': 1.8,
            'inflExp': 2.3,
            'vix': 15.5,
            'dxy': 103.2,
            'creditSpread': 85,
            'termSpread': 45
        }
