import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Union
from config import UI_CONFIG, VALIDATION

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
    try:
        # Replace newlines with commas, split, clean, and convert
        returns = [
            float(x.strip()) 
            for x in text.replace('\n', ',').split(',') 
            if x.strip()
        ]
        # Filter out NaN values
        returns = [r for r in returns if not np.isnan(r)]
        
        if not returns:
            raise ValueError("No valid returns found")
            
        return returns
        
    except (ValueError, TypeError) as e:
        raise ValueError(
            "Invalid returns format. Please enter numbers separated by commas or newlines"
        )

def calculate_stats(returns: Union[List[float], np.ndarray]) -> Dict:
    """
    Calculate comprehensive statistics for return distribution
    
    Args:
        returns: Array of returns (in %)
        
    Returns:
        Dictionary containing stats, risk metrics, and percentiles
    """
    if not returns or len(returns) == 0:
        raise ValueError("No returns data provided")
    
    returns = np.array(returns)
    sorted_returns = np.sort(returns)
    
    # Basic statistics
    stats = {
        'mean': np.mean(returns),
        'median': np.median(returns),
        'stdDev': np.std(returns),
        'skew': pd.Series(returns).skew(),
        'kurtosis': pd.Series(returns).kurtosis(),
        'pPos': np.mean(returns > 0) * 100,  # Probability of positive return
        'min': np.min(returns),
        'max': np.max(returns)
    }
    
    # Risk metrics
    var_95_idx = int(0.05 * len(sorted_returns))
    var_99_idx = int(0.01 * len(sorted_returns))
    
    risk_metrics = {
        'var95': -np.percentile(returns, 5),
        'cvar95': -np.mean(sorted_returns[:max(1, var_95_idx)]),
        'var99': -np.percentile(returns, 1),
        'cvar99': -np.mean(sorted_returns[:max(1, var_99_idx)]),
        'sharpe': stats['mean'] / stats['stdDev'] if stats['stdDev'] > 0 else 0,
        'sortino': calculate_sortino(returns),
        'maxDD': calculate_max_drawdown(returns),
        'tailRatio': calculate_tail_ratio(returns)
    }
    
    # Percentiles
    percentiles = {
        f'p{str(p).zfill(2)}': np.percentile(returns, p)
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }
    
    return {
        'stats': stats,
        'riskMetrics': risk_metrics,
        'percentiles': percentiles
    }

def calculate_sortino(returns: np.ndarray, target: float = 0) -> float:
    """Calculate Sortino ratio (return vs downside deviation)"""
    downside_returns = returns[returns < target]
    if len(downside_returns) == 0:
        return 0
    downside_dev = np.std(downside_returns)
    if downside_dev == 0:
        return 0
    return np.mean(returns - target) / downside_dev

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from return series"""
    wealth = np.cumprod(1 + np.array(returns) / 100)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (peak - wealth) / peak
    return np.max(drawdowns) * 100

def calculate_tail_ratio(returns: np.ndarray) -> float:
    """Calculate tail ratio (95th percentile / 5th percentile)"""
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    if p5 == 0:
        return 1.0
    return np.abs(p95 / p5)

def cholesky_decomposition(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Perform Cholesky decomposition for correlation matrix
    Used to generate correlated random variables
    
    Args:
        corr_matrix: Correlation matrix (must be positive semi-definite)
        
    Returns:
        Lower triangular matrix L where L @ L.T = corr_matrix
    """
    n = len(corr_matrix)
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_sq = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = np.sqrt(max(0.001, corr_matrix[i][i] - sum_sq))
            else:
                sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (corr_matrix[i][j] - sum_prod) / max(0.001, L[j][j])
    
    return L

def create_histogram_plot(
    returns: List[float],
    percentiles: Dict[str, float],
    title: str = "Return Distribution"
) -> go.Figure:
    """Create interactive histogram with percentile markers"""
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
    for p, label, color in [
        (percentiles['p50'], 'Median', 'green'),
        (percentiles['p25'], '25th', 'blue'),
        (percentiles['p75'], '75th', 'blue'),
        (percentiles['p05'], '5th', 'red'),
        (percentiles['p95'], '95th', 'orange')
    ]:
        fig.add_vline(
            x=p,
            line_dash="dash" if label == 'Median' else "dot",
            line_color=color,
            annotation_text=label,
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
        errors.append("Baseline mean is required")
    
    if inputs.get('baseSigma') is None or inputs.get('baseSigma', 0) <= 0:
        errors.append("Baseline volatility must be > 0")
    
    # Check horizon
    horizon = inputs.get('horizon', 1)
    if horizon < VALIDATION['min_horizon'] or horizon > VALIDATION['max_horizon']:
        errors.append(
            f"Horizon must be between {VALIDATION['min_horizon']} and {VALIDATION['max_horizon']} years"
        )
    
    # Check iterations
    iterations = inputs.get('iters', 10000)
    if iterations < VALIDATION['min_returns'] * 500 or iterations > 100000:
        errors.append("Iterations must be between 1,000 and 100,000")
    
    # Check GARCH stability
    if inputs.get('enableGarch'):
        alpha = inputs.get('garchAlpha', 0)
        beta = inputs.get('garchBeta', 0)
        if alpha + beta >= 1:
            errors.append("GARCH parameters (α + β) must be < 1 for stability")
        if alpha < 0 or beta < 0:
            errors.append("GARCH parameters must be non-negative")
    
    # Check t-distribution DoF
    if inputs.get('distType') in ['t', 'skewt']:
        tdf = inputs.get('tdf', 5)
        if tdf < VALIDATION['min_tdf']:
            errors.append(f"Degrees of freedom must be ≥ {VALIDATION['min_tdf']}")
    
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
