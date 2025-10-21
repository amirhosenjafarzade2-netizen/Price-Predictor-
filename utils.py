"""
Utility functions for Monte Carlo Asset Predictor
Includes validation, statistics, and visualization
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from config import (
    UI_CONFIG, VALIDATION, PARAMETER_BOUNDS, VIZ_CONFIG,
    RISK_FREE_RATE
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    field: str
    message: str
    suggested_value: Optional[Union[float, int, str]] = None
    severity: str = 'error'

def parse_returns(text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("No returns data provided")
    
    try:
        returns = [
            float(x.strip()) 
            for x in text.replace('\n', ',').split(',') 
            if x.strip()
        ]
        returns = [r for r in returns if not (np.isnan(r) or np.isinf(r))]
        
        if not returns:
            raise ValueError("No valid returns found after filtering")
        
        if len(returns) < VALIDATION['min_returns']:
            raise ValueError(
                f"At least {VALIDATION['min_returns']} returns required, got {len(returns)}"
            )
        
        logger.info(f"Parsed {len(returns)} historical returns")
        return returns
    except (ValueError, TypeError) as e:
        if "could not convert" in str(e):
            raise ValueError(
                "Invalid returns format. Please enter numbers separated by commas or newlines"
            )
        raise

def calculate_stats(returns: Union[List[float], np.ndarray, Dict]) -> Dict:
    if isinstance(returns, dict):
        returns = list(returns.values())
    
    if not returns or len(returns) == 0:
        raise ValueError("No returns data provided")
    
    returns = np.array(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns after filtering")
    
    if np.max(np.abs(returns)) > 1000:
        logger.warning(f"Extreme returns detected: min={np.min(returns):.2f}%, max={np.max(returns):.2f}%")
    
    sorted_returns = np.sort(returns)
    
    stats = {
        'mean': float(np.mean(returns)),
        'median': float(np.median(returns)),
        'stdDev': float(np.std(returns, ddof=1)),
        'skew': float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0,
        'kurtosis': float(pd.Series(returns).kurtosis() + 3.0) if len(returns) > 3 else 3.0,  # Raw kurtosis
        'pPos': float(np.mean(returns > 0) * 100),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'range': float(np.max(returns) - np.min(returns)),
        'iqr': float(np.percentile(returns, 75) - np.percentile(returns, 25))
    }
    
    var_95_idx = max(1, int(0.05 * len(sorted_returns)))
    var_99_idx = max(1, int(0.01 * len(sorted_returns)))
    
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(np.mean(sorted_returns[:var_95_idx]))
    var_99 = float(np.percentile(returns, 1))
    cvar_99 = float(np.mean(sorted_returns[:var_99_idx]))
    
    logger.debug(f"VaR 95%={var_95:.2f}%, CVaR 95%={cvar_95:.2f}%, VaR 99%={var_99:.2f}%, CVaR 99%={cvar_99:.2f}%")
    
    sharpe = (stats['mean'] - RISK_FREE_RATE) / stats['stdDev'] if stats['stdDev'] > 1e-6 else 0.0
    
    risk_metrics = {
        'var95': var_95,
        'cvar95': cvar_95,
        'var99': var_99,
        'cvar99': cvar_99,
        'sharpe': float(sharpe),
        'sortino': float(calculate_sortino(returns)),
        'calmar': float(calculate_calmar(returns)),
        'maxDD': float(calculate_max_drawdown(returns)),
        'tailRatio': float(calculate_tail_ratio(returns)),
        'gainLossRatio': float(calculate_gain_loss_ratio(returns))
    }
    
    percentiles = {
        f'p{str(p).zfill(2)}': float(np.percentile(returns, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }
    
    logger.debug(f"Calculated stats: mean={stats['mean']:.2f}%, std={stats['stdDev']:.2f}%")
    
    return {
        'stats': stats,
        'riskMetrics': risk_metrics,
        'percentiles': percentiles,
        'results': returns.tolist()
    }

def calculate_sortino(returns: np.ndarray, target: float = 0) -> float:
    excess_returns = returns - target
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.mean(downside_returns ** 2) < 1e-6:
        return 0.0
    
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    annualized_mean = np.mean(returns)
    return annualized_mean / downside_dev if downside_dev > 0 else 0.0

def calculate_calmar(returns: np.ndarray) -> float:
    max_dd = calculate_max_drawdown(returns)
    if abs(max_dd) < 1e-6:
        return 0.0
    return np.mean(returns) / max_dd if max_dd > 0 else 0.0

def calculate_max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    
    wealth = 1 + np.array(returns) / 100.0
    peak = np.maximum.accumulate(wealth)
    drawdowns = (peak - wealth) / peak
    max_dd = np.max(drawdowns) * 100.0
    if max_dd > 100:
        logger.warning(f"Unrealistic max drawdown: {max_dd:.2f}%")
        max_dd = min(max_dd, 100.0)
    return max_dd

def calculate_tail_ratio(returns: np.ndarray) -> float:
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    
    if abs(p5) < 0.001:
        return 1.0
    
    return np.abs(p95 / p5)

def calculate_gain_loss_ratio(returns: np.ndarray) -> float:
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(gains) == 0 or len(losses) == 0:
        return 1.0
    
    avg_gain = np.mean(gains)
    avg_loss = np.abs(np.mean(losses))
    
    if avg_loss == 0:
        return float('inf')
    
    return avg_gain / avg_loss

def create_histogram_plot(
    returns: Union[List[float], np.ndarray],
    percentiles: Dict[str, float],
    title: str = "Return Distribution"
) -> go.Figure:
    returns = np.array(returns)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Distribution',
        marker=dict(
            color=VIZ_CONFIG['color_palette']['primary'],
            line=dict(color='white', width=1)
        ),
        opacity=0.7
    ))
    
    if VIZ_CONFIG.get('show_kde', True):
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(returns)
            x_range = np.linspace(min(returns), max(returns), 200)
            kde_values = kde(x_range)
            
            hist, bins = np.histogram(returns, bins=50)
            kde_scaled = kde_values * (max(hist) / max(kde_values) * 0.8)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_scaled,
                mode='lines',
                name='Density',
                line=dict(color=VIZ_CONFIG['color_palette']['danger'], width=3),
                yaxis='y2'
            ))
        except ImportError:
            logger.warning("scipy not available, skipping KDE overlay")
    
    fig.add_vrect(
        x0=min(returns), x1=percentiles['p05'],
        fillcolor=VIZ_CONFIG['color_palette']['danger'],
        opacity=0.15,
        annotation_text="Extreme Loss<br>Zone",
        annotation_position="top left",
        annotation=dict(font_size=10)
    )
    
    fig.add_vrect(
        x0=percentiles['p05'], x1=percentiles['p25'],
        fillcolor=VIZ_CONFIG['color_palette']['warning'],
        opacity=0.1,
        annotation_text="High<br>Risk",
        annotation_position="top left",
        annotation=dict(font_size=10)
    )
    
    fig.add_vrect(
        x0=percentiles['p75'], x1=percentiles['p95'],
        fillcolor=VIZ_CONFIG['color_palette']['success'],
        opacity=0.1,
        annotation_text="High<br>Return",
        annotation_position="top right",
        annotation=dict(font_size=10)
    )
    
    percentile_lines = [
        (percentiles['p50'], 'Median', VIZ_CONFIG['color_palette']['success'], 'dash'),
        (percentiles['p05'], 'VaR 95%', VIZ_CONFIG['color_palette']['danger'], 'dot'),
        (percentiles['p95'], '95th %ile', VIZ_CONFIG['color_palette']['warning'], 'dot')
    ]
    
    for p_val, label, color, dash_style in percentile_lines:
        fig.add_vline(
            x=p_val,
            line_dash=dash_style,
            line_color=color,
            line_width=2,
            annotation_text=f"{label}: {p_val:.2f}%",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        yaxis2=dict(overlaying='y', side='right', showgrid=False),
        template=VIZ_CONFIG['theme'],
        height=UI_CONFIG['chart_height'],
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_convergence_plot(fitness_progress: List[Dict]) -> go.Figure:
    fig = go.Figure()
    
    generations = [x['generation'] for x in fitness_progress]
    train_fitness = [x['train'] for x in fitness_progress]
    valid_fitness = [x['valid'] for x in fitness_progress]
    avg_fitness = [x.get('avg_fitness', x['train']) for x in fitness_progress]
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=train_fitness,
        name="Best Training",
        line=dict(color=VIZ_CONFIG['color_palette']['primary'], width=3),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=valid_fitness,
        name="Best Validation",
        line=dict(color=VIZ_CONFIG['color_palette']['success'], width=3),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=avg_fitness,
        name="Avg Population",
        line=dict(color=VIZ_CONFIG['color_palette']['warning'], width=2, dash='dot'),
        mode='lines'
    ))
    
    fig.update_layout(
        title="Genetic Algorithm Convergence",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        template=VIZ_CONFIG['theme'],
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig

def create_diversity_plot(convergence_history: List[Dict]) -> go.Figure:
    fig = go.Figure()
    
    generations = [h['generation'] for h in convergence_history]
    diversity = [h['diversity'] for h in convergence_history]
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=diversity,
        name="Diversity",
        line=dict(color=VIZ_CONFIG['color_palette']['warning'], width=2),
        fill='tozeroy',
        fillcolor=f"rgba(255, 153, 0, 0.2)"
    ))
    
    fig.update_layout(
        title="Population Diversity Over Generations",
        xaxis_title="Generation",
        yaxis_title="Diversity Score",
        template=VIZ_CONFIG['theme'],
        height=UI_CONFIG['chart_height']
    )
    
    return fig

def create_backtest_plot(backtest_results: Dict) -> go.Figure:
    fig = go.Figure()
    
    years = backtest_results['years']
    actuals = backtest_results['actuals']
    predictions = backtest_results['predictions']
    rmse = backtest_results['stats']['rmse']
    
    if VIZ_CONFIG.get('show_confidence_intervals', True):
        ci_upper = [p + 1.96 * rmse for p in predictions]
        ci_lower = [p - 1.96 * rmse for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True
        ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=actuals,
        name="Actual",
        line=dict(color=VIZ_CONFIG['color_palette']['success'], width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=predictions,
        name="Predicted",
        line=dict(color=VIZ_CONFIG['color_palette']['primary'], width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="Zero Return"
    )
    
    fig.update_layout(
        title="Backtest: Actual vs Predicted Returns",
        xaxis_title="Period",
        yaxis_title="Return (%)",
        template=VIZ_CONFIG['theme'],
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig

def create_sensitivity_plot(sensitivity_results: Dict) -> go.Figure:
    df = pd.DataFrame(sensitivity_results)
    
    fig = px.line(
        df,
        x='value',
        y='mean_return',
        color='factor',
        title="Sensitivity Analysis: Mean Return vs Macro Factors",
        labels={'value': 'Factor Value', 'mean_return': 'Mean Return (%)'},
        color_discrete_sequence=[
            VIZ_CONFIG['color_palette']['primary'],
            VIZ_CONFIG['color_palette']['success'],
            VIZ_CONFIG['color_palette']['warning'],
            VIZ_CONFIG['color_palette']['danger']
        ]
    )
    
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        template=VIZ_CONFIG['theme'],
        height=UI_CONFIG['chart_height'],
        hovermode='x unified'
    )
    
    return fig

def format_number(value: float, decimals: int = 2, suffix: str = '') -> str:
    if value is None or np.isnan(value):
        return 'N/A'
    if np.isinf(value):
        return '∞' if value > 0 else '-∞'
    return f"{value:.{decimals}f}{suffix}"

def validate_inputs(inputs: Dict) -> Tuple[bool, List[ValidationError]]:
    errors = []
    
    if inputs.get('baseMu') is None:
        errors.append(ValidationError(
            'baseMu',
            'Baseline mean (μ) is required',
            suggested_value=8.0
        ))
    
    base_sigma = inputs.get('baseSigma')
    if base_sigma is None:
        errors.append(ValidationError(
            'baseSigma',
            'Baseline volatility (σ) is required',
            suggested_value=15.0
        ))
    elif base_sigma <= 0:
        errors.append(ValidationError(
            'baseSigma',
            'Baseline volatility must be positive',
            suggested_value=abs(base_sigma) if base_sigma else 15.0
        ))
    elif base_sigma > VALIDATION['max_sigma']:
        errors.append(ValidationError(
            'baseSigma',
            f'Volatility exceeds maximum ({VALIDATION["max_sigma"]}%)',
            suggested_value=VALIDATION['max_sigma']
        ))
    elif base_sigma > VALIDATION['warning_sigma']:
        errors.append(ValidationError(
            'baseSigma',
            f'Very high volatility ({base_sigma}%). This may produce extreme results.',
            severity='warning'
        ))
    
    horizon = inputs.get('horizon', 1)
    if horizon < VALIDATION['min_horizon']:
        errors.append(ValidationError(
            'horizon',
            f'Horizon too short (minimum: {VALIDATION["min_horizon"]} years)',
            suggested_value=VALIDATION['min_horizon']
        ))
    elif horizon > VALIDATION['max_horizon']:
        errors.append(ValidationError(
            'horizon',
            f'Horizon too long (maximum: {VALIDATION["max_horizon"]} years)',
            suggested_value=VALIDATION['max_horizon']
        ))
    
    iterations = inputs.get('iters', 10000)
    if iterations < UI_CONFIG['min_iterations']:
        errors.append(ValidationError(
            'iters',
            f'Too few iterations (minimum: {UI_CONFIG["min_iterations"]})',
            suggested_value=UI_CONFIG['min_iterations']
        ))
    elif iterations > UI_CONFIG['max_iterations']:
        errors.append(ValidationError(
            'iters',
            f'Too many iterations (maximum: {UI_CONFIG["max_iterations"]})',
            suggested_value=UI_CONFIG['max_iterations']
        ))
    
    if inputs.get('enableGarch'):
        alpha = inputs.get('garchAlpha', 0)
        beta_g = inputs.get('garchBeta', 0)
        omega = inputs.get('garchOmega', 0)
        
        if omega < 0:
            errors.append(ValidationError('garchOmega', 'GARCH ω must be non-negative', suggested_value=0.0001))
        if alpha < 0:
            errors.append(ValidationError('garchAlpha', 'GARCH α must be non-negative', suggested_value=0.08))
        if beta_g < 0:
            errors.append(ValidationError('garchBeta', 'GARCH β must be non-negative', suggested_value=0.90))
        if alpha + beta_g >= 1:
            errors.append(ValidationError(
                'garchParameters',
                f'GARCH unstable: α + β = {alpha + beta_g:.3f} ≥ 1',
                suggested_value='α=0.08, β=0.90'
            ))
    
    if inputs.get('distType') in ['t', 'skewt']:
        tdf = inputs.get('tdf', 5)
        if tdf < VALIDATION['min_tdf']:
            errors.append(ValidationError(
                'tdf',
                f'Degrees of freedom too low (minimum: {VALIDATION["min_tdf"]})',
                suggested_value=VALIDATION['min_tdf']
            ))
        elif tdf > VALIDATION['max_tdf']:
            errors.append(ValidationError(
                'tdf',
                f'Degrees of freedom too high (maximum: {VALIDATION["max_tdf"]})',
                suggested_value=VALIDATION['max_tdf']
            ))
    
    betas = inputs.get('betas', {})
    for beta_name, beta_value in betas.items():
        param_key = f'beta_{beta_name}'
        if param_key in PARAMETER_BOUNDS:
            min_val, max_val = PARAMETER_BOUNDS[param_key]
            if beta_value < min_val or beta_value > max_val:
                errors.append(ValidationError(
                    param_key,
                    f'Beta {beta_name} out of range [{min_val}, {max_val}]',
                    suggested_value=(min_val + max_val) / 2
                ))
    
    mean_rev = inputs.get('meanReversion', 0)
    min_mr, max_mr = PARAMETER_BOUNDS['mean_reversion']
    if mean_rev < min_mr or mean_rev > max_mr:
        errors.append(ValidationError(
            'meanReversion',
            f'Mean reversion must be in [{min_mr}, {max_mr}]',
            suggested_value=(min_mr + max_mr) / 2
        ))
    
    return len([e for e in errors if e.severity == 'error']) == 0, errors

def generate_summary_table(stats: Dict, risk_metrics: Dict) -> pd.DataFrame:
    return pd.DataFrame({
        'Metric': [
            'Mean Return',
            'Median Return',
            'Volatility (Std Dev)',
            'Skewness',
            'Kurtosis',
            'Probability Positive',
            'Min Return',
            'Max Return',
            'VaR 95%',
            'CVaR 95%',
            'VaR 99%',
            'CVaR 99%',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Max Drawdown',
            'Tail Ratio',
            'Gain/Loss Ratio'
        ],
        'Value': [
            format_number(stats['mean'], 2, '%'),
            format_number(stats['median'], 2, '%'),
            format_number(stats['stdDev'], 2, '%'),
            format_number(stats['skew'], 3),
            format_number(stats['kurtosis'], 3),
            format_number(stats['pPos'], 1, '%'),
            format_number(stats['min'], 2, '%'),
            format_number(stats['max'], 2, '%'),
            format_number(risk_metrics['var95'], 2, '%'),
            format_number(risk_metrics['cvar95'], 2, '%'),
            format_number(risk_metrics['var99'], 2, '%'),
            format_number(risk_metrics['cvar99'], 2, '%'),
            format_number(risk_metrics['sharpe'], 2),
            format_number(risk_metrics['sortino'], 2),
            format_number(risk_metrics['calmar'], 2),
            format_number(risk_metrics['maxDD'], 1, '%'),
            format_number(risk_metrics['tailRatio'], 2),
            format_number(risk_metrics['gainLossRatio'], 2)
        ]
    })

def setup_logging(log_level: str = 'INFO'):
    from config import LOGGING_CONFIG
    import os
    
    if LOGGING_CONFIG['log_to_file']:
        os.makedirs(LOGGING_CONFIG['log_dir'], exist_ok=True)
        
        from logging.handlers import RotatingFileHandler
        from datetime import datetime
        
        log_file = os.path.join(
            LOGGING_CONFIG['log_dir'],
            f"simulation_{datetime.now():%Y%m%d}.log"
        )
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOGGING_CONFIG['max_bytes'],
            backupCount=LOGGING_CONFIG['backup_count']
        )
        file_handler.setFormatter(
            logging.Formatter(LOGGING_CONFIG['format'])
        )
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=LOGGING_CONFIG['format'],
            handlers=[
                file_handler,
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=LOGGING_CONFIG['format']
        )
    
    logger.info("Logging initialized")
