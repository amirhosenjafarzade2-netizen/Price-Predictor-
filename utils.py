import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Union
from config import UI_CONFIG, VALIDATION, PARAMETER_BOUNDS


def parse_returns(text: str) -> List[float]:
    """
    Parse comma or newline separated returns from text input
    
    Args:
        text: String containing returns separated by commas or newlines
        
    Returns:
        List of float returns
        
    Raises:
        ValueError: If no valid returns found
    """
    if not text or not text.strip():
        raise ValueError("No returns data provided")
    
    try:
        # Replace newlines with commas, split, clean, and convert
        returns = [
            float(x.strip()) 
            for x in text.replace('\n', ',').split(',') 
            if x.strip()
        ]
        # Filter out NaN and infinite values
        returns = [r for r in returns if not (np.isnan(r) or np.isinf(r))]
        
        if not returns:
            raise ValueError("No valid returns found")
        
        if len(returns) < VALIDATION['min_returns']:
            raise ValueError(
                f"At least {VALIDATION['min_returns']} returns required, got {len(returns)}"
            )
            
        return returns
        
    except (ValueError, TypeError) as e:
        if "could not convert" in str(e):
            raise ValueError(
                "Invalid returns format. Please enter numbers separated by commas or newlines"
            )
        raise


def calculate_stats(returns: Union[List[float], np.ndarray, Dict]) -> Dict:
    """
    Calculate comprehensive statistics for return distribution
    
    Args:
        returns: Array of returns (in %) or dict of returns
        
    Returns:
        Dictionary containing stats, riskMetrics, percentiles, and results
    """
    # Handle dict input (from GA optimizer)
    if isinstance(returns, dict):
        returns = list(returns.values())
    
    if not returns or len(returns) == 0:
        raise ValueError("No returns data provided")
    
    returns = np.array(returns, dtype=float)
    
    # Remove any NaN or inf values
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns after filtering")
    
    sorted_returns = np.sort(returns)
    
    # Basic statistics
    stats = {
        'mean': float(np.mean(returns)),
        'median': float(np.median(returns)),
        'stdDev': float(np.std(returns, ddof=1)),
        'skew': float(pd.Series(returns).skew()),
        'kurtosis': float(pd.Series(returns).kurtosis()),
        'pPos': float(np.mean(returns > 0) * 100),
        'min': float(np.min(returns)),
        'max': float(np.max(returns))
    }
    
    # Risk metrics
    var_95_idx = max(1, int(0.05 * len(sorted_returns)))
    var_99_idx = max(1, int(0.01 * len(sorted_returns)))
    
    risk_metrics = {
        'var95': float(-np.percentile(returns, 5)),
        'cvar95': float(-np.mean(sorted_returns[:var_95_idx])),
        'var99': float(-np.percentile(returns, 1)),
        'cvar99': float(-np.mean(sorted_returns[:var_99_idx])),
        'sharpe': float(stats['mean'] / stats['stdDev'] if stats['stdDev'] > 0 else 0),
        'sortino': float(calculate_sortino(returns)),
        'maxDD': float(calculate_max_drawdown(returns)),
        'tailRatio': float(calculate_tail_ratio(returns))
    }
    
    # Percentiles
    percentiles = {
        f'p{str(p).zfill(2)}': float(np.percentile(returns, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }
    
    return {
        'stats': stats,
        'riskMetrics': risk_metrics,
        'percentiles': percentiles,
        'results': returns.tolist()
    }


def calculate_sortino(returns: np.ndarray, target: float = 0) -> float:
    """Calculate Sortino ratio (return vs downside deviation)"""
    excess_returns = returns - target
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_dev == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_dev


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from return series"""
    if len(returns) == 0:
        return 0.0
    
    wealth = np.cumprod(1 + np.array(returns) / 100)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (peak - wealth) / peak
    
    return np.max(drawdowns) * 100


def calculate_tail_ratio(returns: np.ndarray) -> float:
    """Calculate tail ratio (95th percentile / 5th percentile)"""
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    
    if abs(p5) < 0.001:
        return 1.0
    
    return np.abs(p95 / p5)


def create_histogram_plot(
    returns: Union[List[float], np.ndarray],
    percentiles: Dict[str, float],
    title: str = "Return Distribution"
) -> go.Figure:
    """Create interactive histogram with percentile markers"""
    returns = np.array(returns)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Distribution',
        marker=dict(
            color='rgba(11, 116, 222, 0.8)',
            line=dict(color='rgba(11, 116, 222, 1)', width=1)
        )
    ))
    
    # Add percentile lines
    percentile_lines = [
        (percentiles['p50'], 'Median', 'green', 'dash'),
        (percentiles['p25'], '25th', 'blue', 'dot'),
        (percentiles['p75'], '75th', 'blue', 'dot'),
        (percentiles['p05'], '5th', 'red', 'dot'),
        (percentiles['p95'], '95th', 'orange', 'dot')
    ]
    
    for p_val, label, color, dash_style in percentile_lines:
        fig.add_vline(
            x=p_val,
            line_dash=dash_style,
            line_color=color,
            annotation_text=f"{label}: {p_val:.2f}%",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=UI_CONFIG['chart_height'],
        showlegend=False
    )
    
    return fig


def create_convergence_plot(fitness_progress: List[Dict]) -> go.Figure:
    """Create GA convergence plot showing training and validation fitness"""
    fig = go.Figure()
    
    generations = [x['generation'] for x in fitness_progress]
    train_fitness = [x['train'] for x in fitness_progress]
    valid_fitness = [x['valid'] for x in fitness_progress]
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=train_fitness,
        name="Training Fitness",
        line=dict(color="#0b74de", width=2),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=valid_fitness,
        name="Validation Fitness",
        line=dict(color="#28a745", width=2),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Genetic Algorithm Convergence",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        template="plotly_white",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig


def create_diversity_plot(convergence_history: List[Dict]) -> go.Figure:
    """Create population diversity plot"""
    fig = go.Figure()
    
    generations = [h['generation'] for h in convergence_history]
    diversity = [h['diversity'] for h in convergence_history]
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=diversity,
        name="Diversity",
        line=dict(color="#ff9900", width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 153, 0, 0.2)'
    ))
    
    fig.update_layout(
        title="Population Diversity Over Generations",
        xaxis_title="Generation",
        yaxis_title="Diversity Score",
        template="plotly_white",
        height=UI_CONFIG['chart_height']
    )
    
    return fig


def create_backtest_plot(backtest_results: Dict) -> go.Figure:
    """Create backtest plot with actual vs predicted returns and confidence intervals"""
    fig = go.Figure()
    
    years = backtest_results['years']
    actuals = backtest_results['actuals']
    predictions = backtest_results['predictions']
    rmse = backtest_results['stats']['rmse']
    
    # Confidence intervals (95%)
    ci_upper = [p + 1.96 * rmse for p in predictions]
    ci_lower = [p - 1.96 * rmse for p in predictions]
    
    # Add CI band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True
    ))
    
    # Actual returns
    fig.add_trace(go.Scatter(
        x=years,
        y=actuals,
        name="Actual",
        line=dict(color="#28a745", width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Predicted returns
    fig.add_trace(go.Scatter(
        x=years,
        y=predictions,
        name="Predicted",
        line=dict(color="#0b74de", width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title="Backtest: Actual vs Predicted Returns",
        xaxis_title="Period",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig


def create_sensitivity_plot(sensitivity_results: Dict) -> go.Figure:
    """Create sensitivity analysis plot"""
    df = pd.DataFrame(sensitivity_results)
    
    fig = px.line(
        df,
        x='value',
        y='mean_return',
        color='factor',
        title="Sensitivity Analysis: Mean Return vs Macro Factors",
        labels={'value': 'Factor Value', 'mean_return': 'Mean Return (%)'}
    )
    
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        template="plotly_white",
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig


def format_number(value: float, decimals: int = 2, suffix: str = '') -> str:
    """Format number with specified decimals and optional suffix"""
    if value is None or np.isnan(value):
        return 'N/A'
    return f"{value:.{decimals}f}{suffix}"


def validate_inputs(inputs: Dict) -> Tuple[bool, List[str]]:
    """
    Validate all simulation inputs
    
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check required fields
    if inputs.get('baseMu') is None:
        errors.append("Baseline mean (μ) is required")
    
    base_sigma = inputs.get('baseSigma')
    if base_sigma is None:
        errors.append("Baseline volatility (σ) is required")
    elif base_sigma <= 0:
        errors.append("Baseline volatility must be > 0")
    elif base_sigma > VALIDATION['max_sigma']:
        errors.append(f"Baseline volatility must be ≤ {VALIDATION['max_sigma']}%")
    
    # Check horizon
    horizon = inputs.get('horizon', 1)
    if horizon < VALIDATION['min_horizon']:
        errors.append(
            f"Horizon must be ≥ {VALIDATION['min_horizon']} years"
        )
    elif horizon > VALIDATION['max_horizon']:
        errors.append(
            f"Horizon must be ≤ {VALIDATION['max_horizon']} years"
        )
    
    # Check iterations
    iterations = inputs.get('iters', 10000)
    if iterations < UI_CONFIG['min_iterations']:
        errors.append(f"Iterations must be ≥ {UI_CONFIG['min_iterations']}")
    elif iterations > UI_CONFIG['max_iterations']:
        errors.append(f"Iterations must be ≤ {UI_CONFIG['max_iterations']}")
    
    # Check GARCH stability
    if inputs.get('enableGarch'):
        alpha = inputs.get('garchAlpha', 0)
        beta_g = inputs.get('garchBeta', 0)
        omega = inputs.get('garchOmega', 0)
        
        if omega < 0:
            errors.append("GARCH ω must be ≥ 0")
        if alpha < 0:
            errors.append("GARCH α must be ≥ 0")
        if beta_g < 0:
            errors.append("GARCH β must be ≥ 0")
        if alpha + beta_g >= 1:
            errors.append("GARCH parameters (α + β) must be < 1 for stability")
    
    # Check t-distribution DoF
    if inputs.get('distType') in ['t', 'skewt']:
        tdf = inputs.get('tdf', 5)
        if tdf < VALIDATION['min_tdf']:
            errors.append(f"Degrees of freedom must be ≥ {VALIDATION['min_tdf']}")
        elif tdf > VALIDATION['max_tdf']:
            errors.append(f"Degrees of freedom must be ≤ {VALIDATION['max_tdf']}")
    
    # Validate betas are within bounds
    betas = inputs.get('betas', {})
    for beta_name, beta_value in betas.items():
        param_key = f'beta_{beta_name}'
        if param_key in PARAMETER_BOUNDS:
            min_val, max_val = PARAMETER_BOUNDS[param_key]
            if beta_value < min_val or beta_value > max_val:
                errors.append(
                    f"Beta {beta_name} must be between {min_val} and {max_val}"
                )
    
    # Validate mean reversion
    mean_rev = inputs.get('meanReversion', 0)
    if mean_rev < 0 or mean_rev > PARAMETER_BOUNDS['mean_reversion'][1]:
        errors.append(
            f"Mean reversion must be between 0 and {PARAMETER_BOUNDS['mean_reversion'][1]}"
        )
    
    return len(errors) == 0, errors


def generate_summary_table(stats: Dict, risk_metrics: Dict) -> pd.DataFrame:
    """Generate summary statistics table"""
    return pd.DataFrame({
        'Metric': [
            'Mean Return',
            'Median Return',
            'Volatility (Std Dev)',
            'Skewness',
            'Kurtosis',
            'Probability Positive',
            'VaR 95%',
            'CVaR 95%',
            'VaR 99%',
            'CVaR 99%',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown',
            'Tail Ratio'
        ],
        'Value': [
            format_number(stats['mean'], 2, '%'),
            format_number(stats['median'], 2, '%'),
            format_number(stats['stdDev'], 2, '%'),
            format_number(stats['skew'], 3),
            format_number(stats['kurtosis'], 3),
            format_number(stats['pPos'], 1, '%'),
            format_number(risk_metrics['var95'], 2, '%'),
            format_number(risk_metrics['cvar95'], 2, '%'),
            format_number(risk_metrics['var99'], 2, '%'),
            format_number(risk_metrics['cvar99'], 2, '%'),
            format_number(risk_metrics['sharpe'], 2),
            format_number(risk_metrics['sortino'], 2),
            format_number(risk_metrics['maxDD'], 1, '%'),
            format_number(risk_metrics['tailRatio'], 2)
        ]
    })
