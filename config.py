ASSET_PRESETS = {
    'equity': {'name': 'S&P 500', 'mean': 9.2, 'sigma': 16.5, 'betas': {'real':-0.35, 'expReal':-0.22, 'infl':0.15, 'vix':0.25, 'dxy':-0.02, 'credit':-0.08, 'term':0.05}, 'meanReversion': 0.15},
    'bonds': {'name': '10Y Treasury', 'mean': 4.5, 'sigma': 7.8, 'betas': {'real':-0.85, 'expReal':-0.65, 'infl':-0.25, 'vix':0.12, 'dxy':0.03, 'credit':-0.15, 'term':0.20}, 'meanReversion': 0.25},
    'reits': {'name': 'REITs', 'mean': 8.5, 'sigma': 20.2, 'betas': {'real':-0.45, 'expReal':-0.30, 'infl':0.20, 'vix':0.30, 'dxy':-0.03, 'credit':-0.12, 'term':0.08}, 'meanReversion': 0.20},
    'commodities': {'name': 'Commodities', 'mean': 6.8, 'sigma': 22.5, 'betas': {'real':-0.20, 'expReal':-0.15, 'infl':0.35, 'vix':0.28, 'dxy':-0.05, 'credit':-0.05, 'term':0.03}, 'meanReversion': 0.10},
    'custom': {'name': 'Custom Asset', 'mean': 8.0, 'sigma': 15.0, 'betas': {'real':-0.30, 'expReal':-0.20, 'infl':0.10, 'vix':0.20, 'dxy':-0.01, 'credit':-0.06, 'term':0.04}, 'meanReversion': 0.15}
}

PARAMETER_BOUNDS = {
    'beta_real': (-1.5, 0.5),
    'beta_expReal': (-1.5, 0.5),
    'beta_infl': (-0.5, 0.8),
    'beta_vix': (0, 0.5),
    'beta_dxy': (-0.15, 0.15),
    'beta_credit': (-0.3, 0.1),
    'beta_term': (-0.1, 0.3),
    'mean_reversion': (0, 0.5)
}
