import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from mc_model import ProfessionalMCModel
from ga_optimizer import GeneticOptimizer
from utils import (
    parse_returns, calculate_stats, create_histogram_plot,
    create_convergence_plot, create_diversity_plot, create_backtest_plot,
    create_sensitivity_plot, validate_inputs, generate_summary_table
)
from config import ASSET_PRESETS, PARAMETER_BOUNDS, UI_CONFIG, VALIDATION
import random

st.set_page_config(page_title="Monte Carlo Asset Predictor", layout="wide")


def main():
    st.title("🎯 Professional Monte Carlo Asset Predictor")
    st.caption("Enterprise-grade: Fat-tailed distributions • GARCH volatility • Mean reversion • Full backtesting")

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None

    # Sidebar for mode selection and period settings
    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox("Analysis Mode", ["Monte Carlo", "Genetic Algorithm"])
        period_unit = st.selectbox("Period Unit", ["Day", "Month", "3 Months", "4 Months", "Year"])
        period_count = st.number_input("Number of Periods", min_value=1, max_value=100, value=10, step=1)

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1) Asset Configuration")
        asset_type = st.selectbox(
            "Asset Type",
            list(ASSET_PRESETS.keys()),
            format_func=lambda x: ASSET_PRESETS[x]['name']
        )
        
        hist_returns = st.text_area(
            "Historical Returns (%)",
            placeholder="e.g. 8.2, -3.1, 12.5, 7.8, -1.2, ...",
            value="10.3, -5.2, 18.7, 6.1, 12.2, 4.0, -8.1, 22.4, 9.9, 3.6, 15.2, -2.8, 11.5, 7.3, 13.8",
            help=f"Enter returns separated by commas (minimum {VALIDATION['min_returns_ga']} returns for Genetic Algorithm)"
        )

        col_mean, col_sigma, col_reversion = st.columns(3)
        with col_mean:
            baseline_mean = st.number_input(
                "Baseline μ (%)",
                step=0.01,
                value=ASSET_PRESETS[asset_type]['mean'],
                key="baseline_mean"
            )
        with col_sigma:
            baseline_sigma = st.number_input(
                "Baseline σ (%)",
                step=0.01,
                min_value=VALIDATION['min_sigma'],
                value=ASSET_PRESETS[asset_type]['sigma'],
                key="baseline_sigma"
            )
        with col_reversion:
            mean_reversion = st.number_input(
                "Mean Reversion φ",
                min_value=0.0,
                max_value=PARAMETER_BOUNDS['mean_reversion'][1],
                step=0.01,
                value=ASSET_PRESETS[asset_type]['meanReversion'],
                key="mean_reversion"
            )

        st.subheader("2) Macro Environment")
        
        col_real, col_exp_real, col_infl = st.columns(3)
        with col_real:
            real_rate = st.number_input("Real Rate (%)", step=0.01, value=2.1, key="real_rate")
        with col_exp_real:
            exp_real_rate = st.number_input("Exp. Real Rate (%)", step=0.01, value=1.8, key="exp_real_rate")
        with col_infl:
            infl_exp = st.number_input("Inflation Exp. (%)", step=0.01, value=2.3, key="infl_exp")

        col_vix, col_dxy, col_credit = st.columns(3)
        with col_vix:
            vix = st.number_input("VIX", step=0.1, min_value=5.0, max_value=100.0, value=15.5, key="vix")
        with col_dxy:
            dxy = st.number_input("DXY", step=0.1, min_value=70.0, max_value=150.0, value=103.2, key="dxy")
        with col_credit:
            credit_spread = st.number_input(
                "Credit Spread (bps)",
                step=1.0,
                min_value=0.0,
                max_value=500.0,
                value=85.0,
                key="credit_spread"
            )

        col_term, col_horizon = st.columns(2)
        with col_term:
            term_spread = st.number_input("Term Spread (bps)", step=1.0, value=45.0, key="term_spread")
        with col_horizon:
            horizon = st.number_input(
                "Forecast Horizon (years)",
                min_value=VALIDATION['min_horizon'],
                max_value=VALIDATION['max_horizon'],
                step=0.25,
                value=float(period_count),
                key="horizon"
            )

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            iterations = st.slider(
                "Iterations",
                UI_CONFIG['min_iterations'],
                UI_CONFIG['max_iterations'],
                UI_CONFIG['default_iterations'],
                step=1000
            )
            seed = st.number_input("Random Seed", min_value=0, step=1, value=None, key="seed")
            dist_type = st.selectbox("Distribution Type", ["normal", "t", "skewt"], index=1)
            tdf = st.number_input(
                "t DoF",
                min_value=VALIDATION['min_tdf'],
                max_value=VALIDATION['max_tdf'],
                step=0.5,
                value=5.0
            )
            
            st.subheader("Sensitivity Coefficients (Betas)")
            col_beta1, col_beta2 = st.columns(2)
            betas = {}
            for idx, (param, value) in enumerate(ASSET_PRESETS[asset_type]['betas'].items()):
                with col_beta1 if idx % 2 == 0 else col_beta2:
                    min_val, max_val = PARAMETER_BOUNDS[f"beta_{param}"]
                    betas[param] = st.number_input(
                        f"{param.capitalize()} Beta",
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01,
                        value=value,
                        key=f"beta_{param}"
                    )

            st.subheader("Dynamic Correlations")
            col_corr1, col_corr2 = st.columns(2)
            with col_corr1:
                corr_vix_infl = st.number_input(
                    "VIX-Inflation Corr",
                    min_value=-1.0,
                    max_value=1.0,
                    step=0.01,
                    value=0.25
                )
                corr_real_vix = st.number_input(
                    "Real Rate-VIX Corr",
                    min_value=-1.0,
                    max_value=1.0,
                    step=0.01,
                    value=-0.15
                )
            with col_corr2:
                corr_credit_vix = st.number_input(
                    "Credit-VIX Corr",
                    min_value=-1.0,
                    max_value=1.0,
                    step=0.01,
                    value=0.40
                )

            st.subheader("GARCH Volatility Clustering")
            enable_garch = st.selectbox("Enable GARCH(1,1)", ["Disabled", "Enabled"], index=0) == "Enabled"
            col_garch1, col_garch2, col_garch3 = st.columns(3)
            with col_garch1:
                garch_omega = st.number_input("ω (omega)", step=0.0001, value=0.0001, min_value=0.0)
            with col_garch2:
                garch_alpha = st.number_input("α (alpha)", step=0.01, value=0.08, min_value=0.0, max_value=0.99)
            with col_garch3:
                garch_beta = st.number_input("β (beta)", step=0.01, value=0.90, min_value=0.0, max_value=0.99)

        # Control buttons
        col_run, col_backtest, col_sensitivity, col_export = st.columns(4)
        with col_run:
            run_button = st.button("⚡ Run Simulation")
        with col_backtest:
            backtest_button = st.button("🧪 Run Backtest")
        with col_sensitivity:
            sensitivity_button = st.button("🔍 Run Sensitivity Analysis")
        with col_export:
            export_button = st.button("📥 Export Results")

        # Randomize button
        if st.button("🎲 Randomize Inputs"):
            st.session_state.baseline_mean = random.uniform(0, 15)
            st.session_state.baseline_sigma = random.uniform(5, 25)
            st.session_state.mean_reversion = random.uniform(0, PARAMETER_BOUNDS['mean_reversion'][1])
            st.session_state.real_rate = random.uniform(-2, 5)
            st.session_state.exp_real_rate = random.uniform(-2, 5)
            st.session_state.infl_exp = random.uniform(0, 6)
            st.session_state.vix = random.uniform(5, 100)
            st.session_state.dxy = random.uniform(70, 150)
            st.session_state.credit_spread = random.uniform(0, 500)
            st.session_state.term_spread = random.uniform(-100, 100)
            st.session_state.horizon = random.uniform(VALIDATION['min_horizon'], VALIDATION['max_horizon'])
            
            for param in ASSET_PRESETS[asset_type]['betas']:
                min_val, max_val = PARAMETER_BOUNDS[f"beta_{param}"]
                st.session_state[f"beta_{param}"] = random.uniform(min_val, max_val)
            
            st.session_state.seed = random.randint(0, 1000000)
            st.rerun()

    with col2:
        st.subheader("📊 Simulation Results")
        results_placeholder = st.empty()
        backtest_placeholder = st.empty()
        sensitivity_placeholder = st.empty()

    # Initialize model
    model = ProfessionalMCModel()

    # Validate and parse historical returns
    try:
        historical_data = parse_returns(hist_returns)
        
        # Check minimum requirements for GA
        if mode == "Genetic Algorithm" and len(historical_data) < VALIDATION['min_returns_ga']:
            st.error(
                f"Genetic Algorithm requires at least {VALIDATION['min_returns_ga']} "
                f"historical returns, got {len(historical_data)}"
            )
            return
            
    except ValueError as e:
        st.error(str(e))
        return

    # Convert period unit to years
    period_multipliers = {
        'Day': 1/252,
        'Month': 1/12,
        '3 Months': 1/4,
        '4 Months': 1/3,
        'Year': 1
    }

    # Prepare inputs
    inputs = {
        'baseMu': baseline_mean,
        'baseSigma': baseline_sigma,
        'realRate': real_rate,
        'expRealRate': exp_real_rate,
        'inflExp': infl_exp,
        'vix': vix,
        'dxy': dxy,
        'creditSpread': credit_spread,
        'termSpread': term_spread,
        'horizon': horizon * period_multipliers[period_unit],
        'iters': iterations,
        'seed': int(seed) if seed is not None else None,
        'distType': dist_type,
        'tdf': tdf,
        'meanReversion': mean_reversion,
        'enableGarch': enable_garch,
        'garchOmega': garch_omega,
        'garchAlpha': garch_alpha,
        'garchBeta': garch_beta,
        'betas': betas,
        'corrVixInfl': corr_vix_infl,
        'corrRealVix': corr_real_vix,
        'corrCreditVix': corr_credit_vix,
        'historical_data': historical_data
    }

    # Validate inputs
    is_valid, errors = validate_inputs(inputs)
    if not is_valid:
        for error in errors:
            st.error(error)
        return

    # Run simulation
    if run_button:
        with st.spinner("Running simulation..."):
            try:
                if mode == "Monte Carlo":
                    results = model.run(inputs)
                    st.session_state.results = results
                    st.success(f"✅ {iterations} simulations complete!")
                else:
                    # Genetic Algorithm mode
                    optimizer = GeneticOptimizer(model, historical_data)
                    ga_results = optimizer.optimize(inputs, historical_data)
                    st.session_state.results = optimizer.export_results(ga_results)
                    st.success(
                        f"✅ Genetic Algorithm optimization complete! "
                        f"Validation Score: {st.session_state.results['validationScore']:.3f}"
                    )
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    # Run backtest
    if backtest_button:
        with st.spinner("Running backtest..."):
            try:
                backtest_inputs = inputs.copy()
                backtest_inputs['horizon'] = period_multipliers[period_unit]
                backtest_inputs['iters'] = 1000
                backtest_inputs['seed'] = 12345
                
                backtest_results = model.backtest(backtest_inputs)
                st.session_state.backtest_results = backtest_results
                st.success(f"✅ Backtest complete! R²: {backtest_results['stats']['r2']:.3f}")
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    # Run sensitivity analysis
    if sensitivity_button:
        with st.spinner("Running sensitivity analysis..."):
            try:
                sensitivity_inputs = inputs.copy()
                sensitivity_inputs['iters'] = 1000  # Reduce for speed
                
                sensitivity_results = model.run_sensitivity_analysis(sensitivity_inputs)
                st.session_state.sensitivity_results = sensitivity_results
                st.success("✅ Sensitivity analysis complete!")
            except Exception as e:
                st.error(f"Sensitivity analysis failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        with results_placeholder.container():
            # Get the returns data
            returns_data = results.get('results', [])
            
            if returns_data:
                st.plotly_chart(
                    create_histogram_plot(returns_data, results['percentiles']),
                    use_container_width=True
                )

            st.subheader("Key Statistics")
            summary_table = generate_summary_table(results['stats'], results['riskMetrics'])
            st.table(summary_table)

            if mode == "Genetic Algorithm":
                st.subheader("Optimized Parameters")
                opt_betas = results.get('optimizedBetas', {})
                improvement = results.get('improvement', {})
                
                param_df = pd.DataFrame({
                    'Parameter': list(opt_betas.keys()) + ['mean_reversion'],
                    'Optimized Value': list(opt_betas.values()) + [results.get('optimizedMeanReversion', 0)],
                    'Change': list(improvement.values())
                })
                st.dataframe(param_df, use_container_width=True)
                
                st.subheader("Convergence History")
                if 'diagnostics' in results and 'fitnessProgress' in results['diagnostics']:
                    st.plotly_chart(
                        create_convergence_plot(results['diagnostics']['fitnessProgress']),
                        use_container_width=True
                    )
                    st.plotly_chart(
                        create_diversity_plot(results['diagnostics']['fitnessProgress']),
                        use_container_width=True
                    )

    # Display backtest results
    if st.session_state.backtest_results:
        with backtest_placeholder.container():
            st.subheader("Backtest Results")
            st.plotly_chart(
                create_backtest_plot(st.session_state.backtest_results),
                use_container_width=True
            )

            col_backtest1, col_backtest2 = st.columns(2)
            with col_backtest1:
                st.metric("MAE", f"{st.session_state.backtest_results['stats']['mae']:.2f}%")
                st.metric("RMSE", f"{st.session_state.backtest_results['stats']['rmse']:.2f}%")
            with col_backtest2:
                st.metric("R²", f"{st.session_state.backtest_results['stats']['r2']:.3f}")
                st.metric(
                    "Hit Rate",
                    f"{st.session_state.backtest_results['stats']['hitRate']*100:.1f}%"
                )

    # Display sensitivity analysis
    if st.session_state.sensitivity_results:
        with sensitivity_placeholder.container():
            st.subheader("Sensitivity Analysis")
            st.plotly_chart(
                create_sensitivity_plot(st.session_state.sensitivity_results),
                use_container_width=True
            )

    # Export results
    if export_button and st.session_state.results:
        results = st.session_state.results
        summary_table = generate_summary_table(results['stats'], results['riskMetrics'])
        csv = summary_table.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"mc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
