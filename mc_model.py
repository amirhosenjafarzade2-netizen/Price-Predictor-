"""
Professional Monte Carlo Model (fixed)
Supports GARCH volatility, fat-tailed distributions, and mean reversion
"""

import numpy as np
from scipy.stats import t
from typing import Dict, List
from utils import validate_inputs, calculate_stats
from config import PERFORMANCE_CONFIG, MC_DEFAULTS, DISTRIBUTION_CONFIG, VALIDATION
import logging
from multiprocessing import Pool, cpu_count, Manager
import streamlit as st
import time

logger = logging.getLogger(__name__)

# ============================================================
# FIXED _simulate_path FUNCTION
# ============================================================

def _simulate_path(args: tuple) -> float:
    """
    Robust path simulator:
      - mu, sigma inputs in percent; converted to decimal internally
      - Mean reversion via Ornstein-Uhlenbeck
      - Optional GARCH(1,1)
      - Compounds geometrically using exp(r_t * dt)
    """
    (mu, sigma, horizon, mean_reversion, dist_type, tdf,
     enable_garch, garch_omega, garch_alpha, garch_beta, skew, seed_offset) = args

    # Per-path RNG seed
    if seed_offset is not None:
        np.random.seed(int(seed_offset))

    # Simulation setup
    steps = max(1, int(horizon * MC_DEFAULTS.get("steps_per_year", 252)))
    dt = float(horizon) / steps

    mu_dec = float(mu) / 100.0
    sigma_dec = float(sigma) / 100.0
    theta = float(mean_reversion)
    r_t = mu_dec  # current annualized return rate (decimal)
    wealth = 1.0

    # GARCH variance (annualized)
    var_t = max(sigma_dec**2, MC_DEFAULTS.get("min_vol", 1e-6)**2)

    def draw_standardized_t(df):
        if df is None or df <= 2:
            return np.random.normal(0.0, 1.0)
        x = t.rvs(df=df)
        return x / np.sqrt(df / (df - 2.0))

    for i in range(steps):
        # Draw shock
        if dist_type == "t":
            eps = draw_standardized_t(tdf)
        elif dist_type == "skewt":
            eps = draw_standardized_t(tdf) * (1 + np.clip(skew * np.random.normal(0.0, 0.3), -0.5, 0.5))
        else:
            eps = np.random.normal(0.0, 1.0)

        vol_annual = np.sqrt(var_t)
        vol_step = vol_annual * np.sqrt(dt)

        # OU mean reversion
        dr = theta * (mu_dec - r_t) * dt + vol_step * eps
        r_t = r_t + dr
        r_t = np.clip(r_t, -0.5, 0.5)  # Tighter clipping

        # Compound wealth geometrically
        inc_log = r_t * dt
        inc_log = np.clip(inc_log, -0.3, 0.3)  # Tighter clipping for stability
        wealth *= np.exp(inc_log)

        # GARCH update
        if enable_garch:
            eps2 = eps**2  # Use standardized shock
            var_next = garch_omega + garch_alpha * eps2 * var_t + garch_beta * var_t
            var_t = np.clip(
                var_next,
                MC_DEFAULTS.get("min_vol", 1e-6)**2,
                MC_DEFAULTS.get("max_vol", 0.5)**2  # Tighter max variance
            )

        # Debug extreme values
        if i % 100 == 0 and (wealth > 100 or wealth < 0.01):
            logger.debug(f"Step {i}: wealth={wealth:.4f}, r_t={r_t*100:.2f}%, vol_annual={vol_annual*100:.2f}%")

    final_return = (wealth - 1.0) * 100.0
    if not np.isfinite(final_return):
        logger.warning(f"Non-finite return detected; setting to -1000%")
        final_return = -1000.0
    return float(np.clip(final_return, -1000.0, 1000.0))  # Tighter return cap

# ============================================================
# MAIN CLASS
# ============================================================

class ProfessionalMCModel:
    """
    Professional Monte Carlo Asset Return Simulator (fixed)
    """

    def __init__(self):
        self.use_multiprocessing = False  # Disabled for stability
        self.n_processes = PERFORMANCE_CONFIG.get("n_processes") or max(1, cpu_count() - 1)
        logger.info(f"Model initialized (multiprocessing={self.use_multiprocessing})")

    def run(self, inputs: Dict) -> Dict:
        """
        Run Monte Carlo simulation with progress bar
        """
        try:
            logger.info("Starting Monte Carlo simulation")

            is_valid, errors = validate_inputs(inputs)
            if not is_valid:
                msgs = [e.message for e in errors if e.severity == "error"]
                logger.error(f"Validation failed: {msgs}")
                raise ValueError("Validation failed:\n" + "\n".join(msgs))

            dist_type = inputs.get("distType", "normal")
            tdf = inputs.get("tdf", DISTRIBUTION_CONFIG.get(dist_type, {}).get("default_tdf", 5.0))
            if dist_type in ["t", "skewt"] and (tdf < 2.1 or tdf > 30):
                logger.warning(f"Invalid tdf={tdf}, resetting to 5.0")
                tdf = 5.0

            seed = inputs.get("seed")
            if seed is not None:
                np.random.seed(int(seed))
                logger.info(f"Using seed {seed}")

            # Macro factor adjustments
            mu = float(inputs["baseMu"])
            sigma = float(inputs["baseSigma"])
            horizon = float(inputs["horizon"])
            iterations = int(inputs["iters"])
            mean_reversion = float(inputs.get("meanReversion", 0.0))

            betas = inputs.get("betas", {})
            macro_factors = {
                "realRate": inputs.get("realRate", 0) or 0,
                "expRealRate": inputs.get("expRealRate", 0) or 0,
                "inflExp": inputs.get("inflExp", 0) or 0,
                "vix": inputs.get("vix", 15.0) or 15.0,
                "dxy": inputs.get("dxy", 100.0) or 100.0,
                "creditSpread": inputs.get("creditSpread", 100.0) or 100.0,
                "termSpread": inputs.get("termSpread", 0) or 0,
            }

            beta_map = {
                "realRate": "real",
                "expRealRate": "expReal",
                "inflExp": "infl",
                "vix": "vix",
                "dxy": "dxy",
                "creditSpread": "credit",
                "termSpread": "term",
            }

            macro_weight = np.exp(-horizon / 10.0)
            mu += sum(
                betas.get(beta_map.get(f, f), 0) * val * macro_weight
                for f, val in macro_factors.items()
                if beta_map.get(f, f) in betas
            )

            if "vix" in betas:
                vix_factor = (macro_factors["vix"] / 15.0 - 1)
                sigma *= (1 + betas["vix"] * vix_factor * macro_weight)
            sigma = np.clip(sigma, VALIDATION["min_sigma"], VALIDATION["max_sigma"])

            # GARCH parameters
            enable_garch = bool(inputs.get("enableGarch", False))
            garch_omega = inputs.get("garchOmega", 0.0001) if enable_garch else 0
            garch_alpha = inputs.get("garchAlpha", 0.08) if enable_garch else 0
            garch_beta = inputs.get("garchBeta", 0.90) if enable_garch else 0
            if enable_garch and garch_alpha + garch_beta >= 1.0:
                logger.warning("Unstable GARCH params; disabling GARCH")
                enable_garch = False
                garch_omega = garch_alpha = garch_beta = 0

            skew = inputs.get(
                "skew",
                DISTRIBUTION_CONFIG.get(dist_type, {}).get("default_skew", 0.2)
                if dist_type == "skewt"
                else 0.0,
            )

            # Run simulation
            progress_bar = st.progress(0)
            st.text("Running Monte Carlo simulation...")
            time.sleep(0.05)

            logger.info(
                f"Params: mu={mu:.2f}%, sigma={sigma:.2f}%, horizon={horizon}, iters={iterations}, dist={dist_type}, GARCH={enable_garch}, skew={skew}"
            )

            returns = self._run_sequential(
                mu, sigma, horizon, mean_reversion, dist_type, tdf,
                enable_garch, garch_omega, garch_alpha, garch_beta, skew,
                iterations, seed, progress_bar
            )

            progress_bar.progress(1.0)
            result = calculate_stats(returns)
            logger.info(f"Results: mean={result['stats']['mean']:.2f}%, std={result['stats']['stdDev']:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise

    def _run_sequential(
        self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
        enable_garch, garch_omega, garch_alpha, garch_beta, skew,
        iterations, base_seed, progress_bar
    ) -> List[float]:
        """
        Run simulations sequentially with frequent progress updates
        """
        logger.info(f"Running sequential simulation with {iterations} iterations")
        returns = []
        try:
            for i in range(iterations):
                args = (
                    mu, sigma, horizon, mean_reversion, dist_type, tdf,
                    enable_garch, garch_omega, garch_alpha, garch_beta, skew,
                    (base_seed + i) if base_seed is not None else None
                )
                ret = _simulate_path(args)
                returns.append(ret)
                if (i + 1) % max(10, iterations // 100) == 0:
                    progress_bar.progress(min(1.0, (i + 1) / iterations))
                    time.sleep(0.05)
                    logger.debug(f"Iteration {i+1}/{iterations}, return={ret:.2f}%")
        except Exception as e:
            logger.error(f"Sequential simulation failed at iteration {i+1}: {str(e)}")
            raise
        return returns

    def _run_parallel(
        self, mu, sigma, horizon, mean_reversion, dist_type, tdf,
        enable_garch, garch_omega, garch_alpha, garch_beta, skew,
        iterations, base_seed
    ) -> List[float]:
        """
        Run simulations in parallel with basic progress tracking
        """
        logger.info(f"Running parallel simulation with {self.n_processes} processes")
        manager = Manager()
        counter = manager.Value('i', 0)

        args_list = [
            (
                mu, sigma, horizon, mean_reversion, dist_type, tdf,
                enable_garch, garch_omega, garch_alpha, garch_beta, skew,
                (base_seed + i) if base_seed is not None else None
            )
            for i in range(iterations)
        ]
        try:
            with Pool(processes=self.n_processes) as pool:
                returns = pool.map(_simulate_path, args_list)
            logger.info("Parallel simulation completed")
            return returns
        except Exception as e:
            logger.error(f"Parallel simulation failed: {str(e)}")
            raise

    def backtest(self, inputs: Dict) -> Dict:
        """
        Walk-forward backtest on historical data
        """
        try:
            historical = inputs.get("historical_data", [])
            if not historical or len(historical) < 2:
                raise ValueError("Backtest requires ≥2 historical returns")

            logger.info(f"Starting backtest with {len(historical)} periods")
            predictions = []
            horizon = inputs["horizon"]
            iters = inputs.get("iters", 1000)

            for i in range(len(historical) - 1):
                temp_inputs = inputs.copy()
                temp_inputs["iters"] = iters
                if i > 0:
                    hist_vol = np.std(historical[:i + 1])
                    temp_inputs["baseSigma"] = max(VALIDATION["min_sigma"], hist_vol)
                result = self.run(temp_inputs)
                predictions.append(result["stats"]["mean"])
                logger.debug(f"Backtest period {i+1}: predicted {result['stats']['mean']:.2f}%")

            actuals = historical[1:]
            errors = np.array(predictions) - np.array(actuals)
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            hit = np.mean(np.sign(predictions) == np.sign(actuals))

            logger.info(f"Backtest complete: R²={r2:.3f}, MAE={mae:.2f}%, Hit Rate={hit*100:.1f}%")
            return {
                "years": list(range(1, len(actuals) + 1)),
                "actuals": actuals,
                "predictions": predictions,
                "stats": {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "hitRate": float(hit),
                },
            }
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise

    def run_sensitivity_analysis(self, inputs: Dict) -> Dict:
        """
        Analyze sensitivity of returns to macro factors
        """
        try:
            logger.info("Running sensitivity analysis")
            factors = ["vix", "inflExp", "realRate", "creditSpread"]
            ranges = {
                "vix": np.linspace(5, 50, 10),
                "inflExp": np.linspace(0, 5, 10),
                "realRate": np.linspace(-2, 5, 10),
                "creditSpread": np.linspace(0, 300, 10),
            }

            results = {"factor": [], "value": [], "mean_return": []}
            for factor in factors:
                if factor not in ranges:
                    continue
                logger.debug(f"Analyzing sensitivity to {factor}")
                base_inputs = inputs.copy()
                base_inputs["iters"] = 500
                for value in ranges[factor]:
                    test_inputs = base_inputs.copy()
                    test_inputs[factor] = float(value)
                    out = self.run(test_inputs)
                    results["factor"].append(factor)
                    results["value"].append(float(value))
                    results["mean_return"].append(float(out["stats"]["mean"]))
            logger.info("Sensitivity analysis complete")
            return results
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {str(e)}")
            raise

    def fetch_live_macros(self):
        """
        Placeholder for fetching live macro data
        """
        logger.info("Live macro fetch disabled")
        return {}
