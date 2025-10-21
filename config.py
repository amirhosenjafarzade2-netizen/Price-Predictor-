"""
Configuration module for Monte Carlo Asset Predictor
Centralizes all configuration parameters
"""

ASSET_PRESETS = {
    'equity': {
        'name': 'S&P 500',
        'mean': 9.2,
        'sigma': 16.5,
        'betas': {},  # Set to empty to avoid macro factor bias
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

PARAMETER_BOUNDS = {
    'beta_real': (-1.5, 0.5),
    'beta_expReal': (-1.5, 0.5),
    'beta_infl': (-0.5, 0.8),
    'beta_vix': (0.0, 0.5),
    'beta_dxy': (-0.15, 0.15),
    'beta_credit': (-0.3, 0.1),
    'beta_term': (-0.1, 0.3),
    'mean_reversion': (0.0, 0.95)
}

UI_CONFIG = {
    'chart_height': 400,
    'max_iterations': 100000,
    'min_iterations': 1000,
    'default_iterations': 10000,
    'results_per_page': 20
}

VALIDATION = {
    'min_horizon': 0.25,
    'max_horizon': 30.0,
    'min_returns': 2,
    'min_returns_ga': 5,
    'min_tdf': 2.5,
    'max_tdf': 30.0,
    'min_sigma': 0.1,  # Increased from 0.01
    'max_sigma': 200.0,
    'warning_sigma': 100.0
}

GA_CONFIG = {
    'weight_mean_error': 0.4,
    'weight_std_error': 0.3,
    'weight_sharpe': 0.2,
    'weight_regularization': 0.1,
    'convergence_threshold': 0.001,
    'convergence_window': 10,
    'default_generations': 50,
    'default_population': 100,
    'elite_ratio': 0.2,
    'mutation_rate': 0.15,
    'mutation_strength': 0.1,
    'validation_split': 0.3,
    'tournament_size': 5,
    'crossover_probability': 0.8,
    'fitness_iterations': 500
}

PERFORMANCE_CONFIG = {
    'enable_multiprocessing': True,
    'n_processes': None,
    'enable_caching': True,
    'cache_max_size': 1000,
    'batch_size': 100
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_dir': 'logs',
    'max_bytes': 10485760,
    'backup_count': 5
}

DATABASE_CONFIG = {
    'enabled': False,
    'path': 'data/simulation_results.db',
    'auto_save': True,
    'save_full_results': False
}

VIZ_CONFIG = {
    'theme': 'plotly_white',
    'color_palette': {
        'primary': '#0b74de',
        'secondary': '#28a745',
        'warning': '#ff9900',
        'danger': '#dc3545',
        'success': '#28a745'
    },
    'show_kde': True,
    'show_confidence_intervals': True,
    'animation_duration': 500
}

PERIOD_MULTIPLIERS = {
    'Day': 1/252,
    'Week': 1/52,
    'Month': 1/12,
    '3 Months': 1/4,
    '4 Months': 1/3,
    'Quarter': 1/4,
    'Year': 1
}

DISTRIBUTION_CONFIG = {
    'normal': {
        'name': 'Normal (Gaussian)',
        'parameters': []
    },
    't': {
        'name': 'Student-t',
        'parameters': ['tdf'],
        'default_tdf': 5.0
    },
    'skewt': {
        'name': 'Skewed Student-t',
        'parameters': ['tdf', 'skew'],
        'default_tdf': 5.0,
        'default_skew': 0.0  # Changed from 0.2
    }
}

RISK_FREE_RATE = 2.0  # 2% annual, scaled by horizon in calculations

MC_DEFAULTS = {
    'steps_per_year': 252,
    'min_vol': 0.0001,
    'max_vol': 0.3  # Reduced from 3.0
}
