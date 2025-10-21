import numpy as np
import pandas as pd
import plotly.graph_objects as go

def parse_returns(text: str) -> list:
    return [float(x) for x in text.replace('\n', ',').split(',') if x.strip() and not np.isnan(float(x))]

def calculate_stats(data: dict) -> dict:
    returns = list(data.values()) if isinstance(data, dict) else data
    stats = {
        'mean': np.mean(returns),
        'stdDev': np.std(returns),
        'skew': pd.Series(returns).skew(),
        'kurtosis': pd.Series(returns).kurtosis()
    }
    
    sorted_returns = np.sort(returns)
    risk_metrics = {
        'var95': -np.percentile(returns, 5),
        'cvar95': -np.mean(sorted_returns[:int(0.05 * len(returns))]),
        'var99': -np.percentile(returns, 1),
        'cvar99': -np.mean(sorted_returns[:int(0.01 * len(returns))]),
        'sharpe': stats['mean'] / stats['stdDev'] if stats['stdDev'] > 0 else 0,
        'sortino': stats['mean'] / np.std([r for r in returns if r < 0]) if len([r for r in returns if r < 0]) > 0 else 0,
        'maxDD': np.max((np.maximum.accumulate(np.cumprod(1 + np.array(returns) / 100)) - np.cumprod(1 + np.array(returns) / 100)) / np.maximum.accumulate(np.cumprod(1 + np.array(returns) / 100))) * 100,
        'tailRatio': np.abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 1
    }
    
    percentiles = {f'p{p}': np.percentile(returns, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
    
    return {'stats': stats, 'riskMetrics': risk_metrics, 'percentiles': percentiles}

def create_convergence_plot(fitness_progress: list) -> go.Figure:
    fig = go.Figure()
    generations = [x['gen'] for x in fitness_progress]
    train_fitness = [x['train'] for x in fitness_progress]
    valid_fitness = [x['valid'] for x in fitness_progress]
    
    fig.add_trace(go.Scatter(x=generations, y=train_fitness, name="Training Fitness", line=dict(color="#0b74de")))
    fig.add_trace(go.Scatter(x=generations, y=valid_fitness, name="Validation Fitness", line=dict(color="#28a745")))
    fig.update_layout(
        title="Genetic Algorithm Convergence",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        template="plotly_white"
    )
    return fig