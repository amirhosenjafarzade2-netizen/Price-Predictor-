"""
Enhanced Monte Carlo Asset Predictor Application
Professional-grade Streamlit interface with logging and error handling
"""

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
    create_sensitivity_plot, validate_inputs, generate_summary_table,
    ValidationError, setup_logging
)
from config import (
    ASSET_PRESETS, PARAMETER_BOUNDS, UI_CONFIG, VALIDATION,
    PERIOD_MULTIPLIERS, GA_CONFIG, PERFORMANCE_CONFIG
)
import random
import logging
import traceback

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Monte Carlo Asset Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


def display_validation_errors(errors: list):
    """Display validation errors with appropriate styling"""
    for error in errors:
        if error.severity == 'error':
            st.error(f"❌ **{error.field}**: {error.message}")
        elif error.severity == 'warning':
            st.warning(f"⚠️ **{error.field}**: {error.message}")
        else:
            st.info(f"ℹ️ **{error.field}**: {error.message}")
        
        if error.suggested_value is not None:
            st.info(f"💡 Suggested value: {error.suggested_value}")


def annualize_returns(returns: list, period_type: str) -> list:
    """Convert returns from their period type to annual returns"""
    multiplier = PERIOD_MULTIPLIERS[period_type]
    # Annualization factor (how many periods in a year)
    periods_per_year = 1 / multiplier
    
    # Convert to annualized returns
    annualized = [(((1 + r/100) ** periods_per_year) - 1) * 100 for r in returns]
    return annualized


def main():
    st.title("🎯 Professional Monte Carlo Asset Predictor")
    st.caption(
        "Enterprise-grade: Fat-tailed distributions • GARCH volatility • "
        "Mean reversion • Full backtesting • Parallel processing"
    )

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None
    if 'last_run_time' not in st.session_state:
        st.session_state.last_run_time = None
    if 'historical_macro_data' not in st.session_state:
        st.session_state.historical_macro_data = []

    # Sidebar for mode selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.selectbox(
            "Analysis Mode",
            ["Monte Carlo", "Genetic Algorithm"],
            help="Monte Carlo: Direct simulation | GA: Parameter optimization"
        )
        
        st.markdown("---")
        st.subheader("📅 Forecast Horizon")
        
        period_unit = st.selectbox(
            "Period Type",
            list(PERIOD_MULTIPLIERS.keys()),
            help="Time unit for forecast horizon"
        )
        
        period_count = st.number_input(
            "Number of Periods",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Number of periods to forecast"
        )
        
        st.markdown("---")
        st.subheader("🚀 Performance")
        
        use_parallel = st.checkbox(
            "Enable Parallel Processing",
            value=PERFORMANCE_CONFIG['enable_multiprocessing'],
            help="Use multiple CPU cores for faster simulation"
        )
        
        if use_parallel:
            n_processes = st.slider(
                "Number of Processes",
                min_value=1,
                max_value=16,
                value=PERFORMANCE_CONFIG['n_processes'] or 4,
                help="Number of parallel processes"
            )
            PERFORMANCE_CONFIG['n_processes'] = n_processes
        
        PERFORMANCE_CONFIG['enable_multiprocessing'] = use_parallel
        
        st.markdown("---")
        if st.session_state.last_run_time:
            st.metric("Last Run Time", f"{st.session_state.last_run_time:.2f}s")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1️⃣ Asset Configuration")
        
        asset_type = st.selectbox(
            "Asset Type",
            list(ASSET_PRESETS.keys()),
            format_func=lambda x: ASSET_PRESETS[x]['name'],
            help="Select a preset or use custom parameters"
        )
        
        st.markdown("**📊 Historical Returns Data**")
        
        # Add period selection for historical data
        col_hist_period, col_hist_count = st.columns([2, 1])
        with col_hist_period:
            hist_period_type = st.selectbox(
                "Historical Data Period Type",
                list(PERIOD_MULTIPLIERS.keys()),
                index=list(PERIOD_MULTIPLIERS.keys()).index("Year"),
                help="What timeframe do your historical returns represent?"
            )
        with col_hist_count:
            st.metric(
                "Periods to Year", 
                f"{1/PERIOD_MULTIPLIERS[hist_period_type]:.0f}x",
                help=f"There are {1/PERIOD_MULTIPLIERS[hist_period_type]:.0f} {hist_period_type.lower()}s per year"
            )
        
        hist_returns = st.text_area(
            f"Historical Returns (%) - {hist_period_type}ly",
            placeholder=f"e.g. 8.2, -3.1, 12.5, 7.8, -1.2, ... (one return per {hist_period_type.lower()})",
            value="10.3, -5.2, 18.7, 6.1, 12.2, 4.0, -8.1, 22.4, 9.9, 3.6, 15.2, -2.8, 11.5, 7.3, 13.8",
            help=f"Enter {hist_period_type.lower()}ly returns separated by commas (min {VALIDATION['min_returns_ga']} for GA)",
            height=100
        )
        
        auto_annualize = st.checkbox(
            f"Auto-annualize {hist_period_type.lower()}ly returns",
            value=True,
            help=f"Convert {hist_period_type.lower()}ly returns to annual equivalent for baseline calculations"
        )

        col_mean, col_sigma, col_reversion = st.columns(3)
        with col_mean:
            baseline_mean = st.number_input(
                "Baseline μ (%)",
                step=0.01,
                value=ASSET_PRESETS[asset_type]['mean'],
                key="baseline_mean",
                help="Expected annual return"
            )
        with col_sigma:
            baseline_sigma = st.number_input(
                "Baseline σ (%)",
                step=0.01,
                min_value=VALIDATION['min_sigma'],
                value=ASSET_PRESETS[asset_type]['sigma'],
                key="baseline_sigma",
                help="Annual volatility (standard deviation)"
            )
        with col_reversion:
            mean_reversion = st.number_input(
                "Mean Reversion φ",
                min_value=0.0,
                max_value=PARAMETER_BOUNDS['mean_reversion'][1],
                step=0.01,
                value=ASSET_PRESETS[asset_type]['meanReversion'],
                key="mean_reversion",
                help="Speed of reversion to mean (0=none, 1=instant)"
            )

        # Parse historical returns early to know how many periods we have
        try:
            historical_data = parse_returns(hist_returns)
            num_historical_periods = len(historical_data)
        except:
            historical_data = []
            num_historical_periods = 0

        st.subheader("2️⃣ Macro Environment")
        
        # Toggle between single current macro or historical macro data
        macro_input_mode = st.radio(
            "Macro Data Input Mode",
            ["Current Only (for forecasting, monte carlo)", "Historical (for each period, genetic algorithm)"],
            help="Current: Single set of macro conditions for forecast | Historical: Macro conditions for each historical period"
        )
        
        if macro_input_mode == "Current Only (for forecasting)":
            # Single set of macro inputs (original behavior)
            st.caption("📍 Current macro conditions for forecasting")
            
            col_real, col_exp_real, col_infl = st.columns(3)
            with col_real:
                real_rate = st.number_input(
                    "Real Rate (%)",
                    step=0.01,
                    value=2.1,
                    key="real_rate",
                    help="Current real interest rate"
                )
            with col_exp_real:
                exp_real_rate = st.number_input(
                    "Exp. Real Rate (%)",
                    step=0.01,
                    value=1.8,
                    key="exp_real_rate",
                    help="Expected future real rate"
                )
            with col_infl:
                infl_exp = st.number_input(
                    "Inflation Exp. (%)",
                    step=0.01,
                    value=2.3,
                    key="infl_exp",
                    help="Expected inflation rate"
                )

            col_vix, col_dxy, col_credit = st.columns(3)
            with col_vix:
                vix = st.number_input(
                    "VIX",
                    step=0.1,
                    min_value=5.0,
                    max_value=100.0,
                    value=15.5,
                    key="vix",
                    help="Volatility index (market fear gauge)"
                )
            with col_dxy:
                dxy = st.number_input(
                    "DXY",
                    step=0.1,
                    min_value=70.0,
                    max_value=150.0,
                    value=103.2,
                    key="dxy",
                    help="Dollar index"
                )
            with col_credit:
                credit_spread = st.number_input(
                    "Credit Spread (bps)",
                    step=1.0,
                    min_value=0.0,
                    max_value=500.0,
                    value=85.0,
                    key="credit_spread",
                    help="Credit spread in basis points"
                )

            col_term, _ = st.columns(2)
            with col_term:
                term_spread = st.number_input(
                    "Term Spread (bps)",
                    step=1.0,
                    value=45.0,
                    key="term_spread",
                    help="Yield curve slope (10Y - 2Y)"
                )
            
            # Store as single-item list for consistency
            historical_macro_list = None
            
        else:
            # Historical macro data for each period
            if num_historical_periods == 0:
                st.warning("⚠️ Please enter historical returns first to determine number of periods")
                historical_macro_list = []
            else:
                st.caption(f"📊 Enter macro conditions for each of {num_historical_periods} {hist_period_type.lower()}(s)")
                
                # Option to upload CSV or enter manually
                input_method = st.radio(
                    "Input Method",
                    ["Manual Entry", "Upload CSV"],
                    horizontal=True
                )
                
                if input_method == "Upload CSV":
                    st.info(
                        "📋 CSV should have columns: period, real_rate, exp_real_rate, infl_exp, vix, dxy, credit_spread, term_spread"
                    )
                    uploaded_file = st.file_uploader("Upload Macro Data CSV", type=['csv'])
                    
                    if uploaded_file is not None:
                        try:
                            macro_df = pd.read_csv(uploaded_file)
                            required_cols = ['real_rate', 'exp_real_rate', 'infl_exp', 'vix', 'dxy', 'credit_spread', 'term_spread']
                            
                            if all(col in macro_df.columns for col in required_cols):
                                historical_macro_list = []
                                for idx, row in macro_df.iterrows():
                                    if idx >= num_historical_periods:
                                        break
                                    historical_macro_list.append({
                                        'realRate': float(row['real_rate']),
                                        'expRealRate': float(row['exp_real_rate']),
                                        'inflExp': float(row['infl_exp']),
                                        'vix': float(row['vix']),
                                        'dxy': float(row['dxy']),
                                        'creditSpread': float(row['credit_spread']),
                                        'termSpread': float(row['term_spread'])
                                    })
                                st.success(f"✅ Loaded macro data for {len(historical_macro_list)} periods")
                                st.session_state.historical_macro_data = historical_macro_list
                            else:
                                st.error(f"❌ CSV missing required columns: {required_cols}")
                                historical_macro_list = []
                        except Exception as e:
                            st.error(f"❌ Error reading CSV: {str(e)}")
                            historical_macro_list = []
                    else:
                        historical_macro_list = st.session_state.historical_macro_data if st.session_state.historical_macro_data else []
                
                else:
                    # Manual entry with expander for each period
                    st.info("💡 Tip: Use 'Fill All Periods' button to quickly set the same values for all periods")
                    
                    # Quick fill option
                    with st.expander("⚡ Quick Fill All Periods", expanded=False):
                        col_qf1, col_qf2, col_qf3 = st.columns(3)
                        with col_qf1:
                            qf_real = st.number_input("Real Rate (%)", value=2.0, key="qf_real")
                            qf_exp_real = st.number_input("Exp. Real Rate (%)", value=1.8, key="qf_exp_real")
                        with col_qf2:
                            qf_infl = st.number_input("Inflation Exp. (%)", value=2.3, key="qf_infl")
                            qf_vix = st.number_input("VIX", value=15.5, key="qf_vix")
                        with col_qf3:
                            qf_dxy = st.number_input("DXY", value=103.0, key="qf_dxy")
                            qf_credit = st.number_input("Credit Spread (bps)", value=85.0, key="qf_credit")
                        
                        qf_term = st.number_input("Term Spread (bps)", value=45.0, key="qf_term")
                        
                        if st.button("📝 Fill All Periods with These Values"):
                            st.session_state.historical_macro_data = [
                                {
                                    'realRate': qf_real,
                                    'expRealRate': qf_exp_real,
                                    'inflExp': qf_infl,
                                    'vix': qf_vix,
                                    'dxy': qf_dxy,
                                    'creditSpread': qf_credit,
                                    'termSpread': qf_term
                                }
                                for _ in range(num_historical_periods)
                            ]
                            st.success(f"✅ Filled {num_historical_periods} periods!")
                            st.rerun()
                    
                    # Initialize if empty
                    if len(st.session_state.historical_macro_data) != num_historical_periods:
                        st.session_state.historical_macro_data = [
                            {
                                'realRate': 2.0,
                                'expRealRate': 1.8,
                                'inflExp': 2.3,
                                'vix': 15.5,
                                'dxy': 103.0,
                                'creditSpread': 85.0,
                                'termSpread': 45.0
                            }
                            for _ in range(num_historical_periods)
                        ]
                    
                    # Display in a scrollable container with tabs for better UX
                    st.markdown("---")
                    
                    # Show as table for easier viewing
                    if st.checkbox("📊 Show as Editable Table", value=True):
                        macro_df = pd.DataFrame(st.session_state.historical_macro_data)
                        macro_df.insert(0, 'Period', range(1, len(macro_df) + 1))
                        
                        edited_df = st.data_editor(
                            macro_df,
                            use_container_width=True,
                            num_rows="fixed",
                            height=400,
                            column_config={
                                "Period": st.column_config.NumberColumn("Period", disabled=True),
                                "realRate": st.column_config.NumberColumn("Real Rate (%)", format="%.2f"),
                                "expRealRate": st.column_config.NumberColumn("Exp. Real Rate (%)", format="%.2f"),
                                "inflExp": st.column_config.NumberColumn("Inflation Exp. (%)", format="%.2f"),
                                "vix": st.column_config.NumberColumn("VIX", format="%.1f"),
                                "dxy": st.column_config.NumberColumn("DXY", format="%.1f"),
                                "creditSpread": st.column_config.NumberColumn("Credit Spread (bps)", format="%.0f"),
                                "termSpread": st.column_config.NumberColumn("Term Spread (bps)", format="%.0f")
                            }
                        )
                        
                        # Update session state from edited table
                        st.session_state.historical_macro_data = edited_df.drop('Period', axis=1).to_dict('records')
                    
                    else:
                        # Individual period inputs (old method)
                        for period_idx in range(num_historical_periods):
                            with st.expander(f"Period {period_idx + 1} - Return: {historical_data[period_idx]:.2f}%", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.session_state.historical_macro_data[period_idx]['realRate'] = st.number_input(
                                        "Real Rate (%)",
                                        value=st.session_state.historical_macro_data[period_idx]['realRate'],
                                        key=f"hist_real_{period_idx}",
                                        step=0.1
                                    )
                                    st.session_state.historical_macro_data[period_idx]['expRealRate'] = st.number_input(
                                        "Exp. Real Rate (%)",
                                        value=st.session_state.historical_macro_data[period_idx]['expRealRate'],
                                        key=f"hist_exp_real_{period_idx}",
                                        step=0.1
                                    )
                                
                                with col2:
                                    st.session_state.historical_macro_data[period_idx]['inflExp'] = st.number_input(
                                        "Inflation Exp. (%)",
                                        value=st.session_state.historical_macro_data[period_idx]['inflExp'],
                                        key=f"hist_infl_{period_idx}",
                                        step=0.1
                                    )
                                    st.session_state.historical_macro_data[period_idx]['vix'] = st.number_input(
                                        "VIX",
                                        value=st.session_state.historical_macro_data[period_idx]['vix'],
                                        key=f"hist_vix_{period_idx}",
                                        min_value=5.0,
                                        max_value=100.0,
                                        step=0.5
                                    )
                                
                                with col3:
                                    st.session_state.historical_macro_data[period_idx]['dxy'] = st.number_input(
                                        "DXY",
                                        value=st.session_state.historical_macro_data[period_idx]['dxy'],
                                        key=f"hist_dxy_{period_idx}",
                                        min_value=70.0,
                                        max_value=150.0,
                                        step=0.5
                                    )
                                    st.session_state.historical_macro_data[period_idx]['creditSpread'] = st.number_input(
                                        "Credit Spread (bps)",
                                        value=st.session_state.historical_macro_data[period_idx]['creditSpread'],
                                        key=f"hist_credit_{period_idx}",
                                        min_value=0.0,
                                        max_value=500.0,
                                        step=5.0
                                    )
                                
                                st.session_state.historical_macro_data[period_idx]['termSpread'] = st.number_input(
                                    "Term Spread (bps)",
                                    value=st.session_state.historical_macro_data[period_idx]['termSpread'],
                                    key=f"hist_term_{period_idx}",
                                    step=5.0
                                )
                    
                    historical_macro_list = st.session_state.historical_macro_data
            
            # Use first period's values as "current" for forecast
            if historical_macro_list and len(historical_macro_list) > 0:
                real_rate = historical_macro_list[-1]['realRate']  # Use most recent
                exp_real_rate = historical_macro_list[-1]['expRealRate']
                infl_exp = historical_macro_list[-1]['inflExp']
                vix = historical_macro_list[-1]['vix']
                dxy = historical_macro_list[-1]['dxy']
                credit_spread = historical_macro_list[-1]['creditSpread']
                term_spread = historical_macro_list[-1]['termSpread']
            else:
                # Defaults
                real_rate = 2.1
                exp_real_rate = 1.8
                infl_exp = 2.3
                vix = 15.5
                dxy = 103.2
                credit_spread = 85.0
                term_spread = 45.0

        # Forecast horizon display
        st.markdown("---")
        col_horizon_display = st.columns(1)[0]
        with col_horizon_display:
            horizon_years = period_count * PERIOD_MULTIPLIERS[period_unit]
            st.metric(
                "📅 Forecast Horizon",
                f"{horizon_years:.2f} years",
                delta=f"{period_count} {period_unit.lower()}(s)",
                help="Calculated from period settings in sidebar"
            )

        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.subheader("Simulation Parameters")
            
            col_iter, col_seed = st.columns(2)
            with col_iter:
                iterations = st.slider(
                    "Iterations",
                    UI_CONFIG['min_iterations'],
                    UI_CONFIG['max_iterations'],
                    UI_CONFIG['default_iterations'],
                    step=1000,
                    help="Number of Monte Carlo simulations"
                )
            with col_seed:
                use_seed = st.checkbox("Use Random Seed", value=False)
                if use_seed:
                    seed = st.number_input(
                        "Seed Value",
                        min_value=0,
                        step=1,
                        value=42,
                        key="seed"
                    )
                else:
                    seed = None
            
            col_dist, col_tdf = st.columns(2)
            with col_dist:
                dist_type = st.selectbox(
                    "Distribution Type",
                    ["normal", "t", "skewt"],
                    index=1,
                    help="Normal: Gaussian | t: Fat tails | skewt: Asymmetric fat tails"
                )
            with col_tdf:
                if dist_type in ['t', 'skewt']:
                    tdf = st.number_input(
                        "t Degrees of Freedom",
                        min_value=VALIDATION['min_tdf'],
                        max_value=VALIDATION['max_tdf'],
                        step=0.5,
                        value=5.0,
                        help="Lower = fatter tails (more extreme events)"
                    )
                else:
                    tdf = 5.0
            
            st.subheader("Sensitivity Coefficients (Betas)")
            st.caption("How macro factors affect returns")
            
            col_beta1, col_beta2 = st.columns(2)
            betas = {}
            for idx, (param, value) in enumerate(ASSET_PRESETS[asset_type]['betas'].items()):
                with col_beta1 if idx % 2 == 0 else col_beta2:
                    min_val, max_val = PARAMETER_BOUNDS[f"beta_{param}"]
                    betas[param] = st.number_input(
                        f"{param.capitalize()} β",
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01,
                        value=value,
                        key=f"beta_{param}",
                        help=f"Sensitivity to {param}"
                    )

            st.subheader("GARCH Volatility Clustering")
            st.caption("Model volatility clustering (volatility breeds volatility)")
            
            enable_garch = st.checkbox(
                "Enable GARCH(1,1)",
                value=False,
                help="Model time-varying volatility"
            )
            
            if enable_garch:
                col_garch1, col_garch2, col_garch3 = st.columns(3)
                with col_garch1:
                    garch_omega = st.number_input(
                        "ω (omega)",
                        step=0.0001,
                        value=0.0001,
                        min_value=0.0,
                        format="%.4f",
                        help="Long-run variance"
                    )
                with col_garch2:
                    garch_alpha = st.number_input(
                        "α (alpha)",
                        step=0.01,
                        value=0.08,
                        min_value=0.0,
                        max_value=0.99,
                        help="ARCH coefficient (shock impact)"
                    )
                with col_garch3:
                    garch_beta = st.number_input(
                        "β (beta)",
                        step=0.01,
                        value=0.90,
                        min_value=0.0,
                        max_value=0.99,
                        help="GARCH coefficient (persistence)"
                    )
                
                # Check stability
                if garch_alpha + garch_beta >= 1:
                    st.error(f"⚠️ GARCH unstable: α + β = {garch_alpha + garch_beta:.3f} ≥ 1")
            else:
                garch_omega = 0.0001
                garch_alpha = 0.08
                garch_beta = 0.90

        # Control buttons
        st.markdown("---")
        col_run, col_backtest, col_sensitivity, col_export = st.columns(4)
        
        with col_run:
            run_button = st.button(
                "⚡ Run Simulation",
                type="primary",
                use_container_width=True
            )
        with col_backtest:
            backtest_button = st.button(
                "🧪 Run Backtest",
                use_container_width=True
            )
        with col_sensitivity:
            sensitivity_button = st.button(
                "🔍 Sensitivity",
                use_container_width=True
            )
        with col_export:
            export_button = st.button(
                "📥 Export",
                use_container_width=True
            )

        # Randomize button
        if st.button("🎲 Randomize Inputs", use_container_width=True):
            st.session_state.baseline_mean = random.uniform(0, 15)
            st.session_state.baseline_sigma = random.uniform(5, 25)
            st.session_state.mean_reversion = random.uniform(0, PARAMETER_BOUNDS['mean_reversion'][1])
            
            if macro_input_mode == "Current Only (for forecasting)":
                st.session_state.real_rate = random.uniform(-2, 5)
                st.session_state.exp_real_rate = random.uniform(-2, 5)
                st.session_state.infl_exp = random.uniform(0, 6)
                st.session_state.vix = random.uniform(5, 100)
                st.session_state.dxy = random.uniform(70, 150)
                st.session_state.credit_spread = random.uniform(0, 500)
                st.session_state.term_spread = random.uniform(-100, 100)
            
            for param in ASSET_PRESETS[asset_type]['betas']:
                min_val, max_val = PARAMETER_BOUNDS[f"beta_{param}"]
                st.session_state[f"beta_{param}"] = random.uniform(min_val, max_val)
            
            if use_seed:
                st.session_state.seed = random.randint(0, 1000000)
            
            logger.info("Inputs randomized")
            st.rerun()

    with col2:
        st.subheader("📊 Simulation Results")
        results_placeholder = st.empty()
        
        st.subheader("📈 Additional Analysis")
        backtest_placeholder = st.empty()
        sensitivity_placeholder = st.empty()

    # Initialize model
    try:
        model = ProfessionalMCModel()
        logger.info("Model initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        logger.error(f"Model initialization failed: {traceback.format_exc()}")
        return

    # Validate and parse historical returns
    try:
        historical_data = parse_returns(hist_returns)
        
        # Display info about historical data
        st.info(
            f"ℹ️ Loaded {len(historical_data)} {hist_period_type.lower()}ly returns | "
            f"Mean: {np.mean(historical_data):.2f}% | "
            f"Std Dev: {np.std(historical_data):.2f}%"
        )
        
        # Annualize if requested
        if auto_annualize and hist_period_type != "Year":
            annualized_data = annualize_returns(historical_data, hist_period_type)
            st.success(
                f"✅ Auto-annualized: {hist_period_type}ly → Annual | "
                f"Annual Mean: {np.mean(annualized_data):.2f}% | "
                f"Annual Std Dev: {np.std(annualized_data):.2f}%"
            )
            # Use annualized data for model
            historical_data_for_model = annualized_data
        else:
            historical_data_for_model = historical_data
        
        # Check minimum requirements for GA
        if mode == "Genetic Algorithm" and len(historical_data) < VALIDATION['min_returns_ga']:
            st.error(
                f"❌ Genetic Algorithm requires at least {VALIDATION['min_returns_ga']} "
                f"historical returns, got {len(historical_data)}"
            )
            return
            
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        return

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
        'horizon': horizon_years,
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
        'historical_data': historical_data_for_model,
        'historical_macro_data': historical_macro_list  # New: list of macro conditions per period
    }

    # Validate inputs
    is_valid, errors = validate_inputs(inputs)
    if not is_valid:
        st.error("⚠️ Input Validation Failed")
        display_validation_errors(errors)
        return
    
    # Display warnings if any
    warnings = [e for e in errors if e.severity == 'warning']
    if warnings:
        display_validation_errors(warnings)

    # Run simulation
    if run_button:
        start_time = datetime.now()
        
        with st.spinner(f"Running {mode} simulation..."):
            try:
                logger.info(f"Starting {mode} simulation")
                
                if mode == "Monte Carlo":
                    results = model.run(inputs)
                    st.session_state.results = results
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.session_state.last_run_time = elapsed
                    
                    st.success(
                        f"✅ {iterations:,} simulations complete in {elapsed:.2f}s! "
                        f"Mean return: {results['stats']['mean']:.2f}%"
                    )
                    logger.info(f"MC simulation completed in {elapsed:.2f}s")
                    
                else:
                    # Genetic Algorithm mode
                    optimizer = GeneticOptimizer(model, historical_data_for_model)
                    ga_results = optimizer.optimize(inputs, historical_data_for_model)
                    st.session_state.results = optimizer.export_results(ga_results)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.session_state.last_run_time = elapsed
                    
                    st.success(
                        f"✅ GA optimization complete in {elapsed:.2f}s! "
                        f"Validation Score: {st.session_state.results['validationScore']:.3f} | "
                        f"Cache Hit Rate: {st.session_state.results['cacheHitRate']:.1%}"
                    )
                    logger.info(f"GA optimization completed in {elapsed:.2f}s")
                    
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                logger.error(f"Simulation failed: {traceback.format_exc()}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                return

    # Run backtest
    if backtest_button:
        with st.spinner("Running backtest..."):
            try:
                logger.info("Starting backtest")
                backtest_inputs = inputs.copy()
                backtest_inputs['horizon'] = PERIOD_MULTIPLIERS[hist_period_type]
                backtest_inputs['iters'] = 1000
                backtest_inputs['seed'] = 12345
                
                backtest_results = model.backtest(backtest_inputs)
                st.session_state.backtest_results = backtest_results
                
                st.success(
                    f"✅ Backtest complete! R²: {backtest_results['stats']['r2']:.3f}, "
                    f"Hit Rate: {backtest_results['stats']['hitRate']*100:.1f}%"
                )
                logger.info("Backtest completed successfully")
                
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                logger.error(f"Backtest failed: {traceback.format_exc()}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                return

    # Run sensitivity analysis
    if sensitivity_button:
        with st.spinner("Running sensitivity analysis..."):
            try:
                logger.info("Starting sensitivity analysis")
                sensitivity_inputs = inputs.copy()
                sensitivity_inputs['iters'] = 1000
                
                sensitivity_results = model.run_sensitivity_analysis(sensitivity_inputs)
                st.session_state.sensitivity_results = sensitivity_results
                
                st.success("✅ Sensitivity analysis complete!")
                logger.info("Sensitivity analysis completed successfully")
                
            except Exception as e:
                st.error(f"❌ Sensitivity analysis failed: {str(e)}")
                logger.error(f"Sensitivity analysis failed: {traceback.format_exc()}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
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

            st.subheader("📋 Key Statistics")
            summary_table = generate_summary_table(results['stats'], results['riskMetrics'])
            st.dataframe(summary_table, use_container_width=True, hide_index=True)

            if mode == "Genetic Algorithm":
                st.subheader("🧬 Optimized Parameters")
                opt_betas = results.get('optimizedBetas', {})
                improvement = results.get('improvement', {})
                
                param_df = pd.DataFrame({
                    'Parameter': list(opt_betas.keys()) + ['mean_reversion'],
                    'Optimized Value': [f"{v:.4f}" for v in opt_betas.values()] + [f"{results.get('optimizedMeanReversion', 0):.4f}"],
                    'Change': [f"{v:+.4f}" for v in improvement.values()]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                
                col_ga1, col_ga2 = st.columns(2)
                with col_ga1:
                    st.metric("Training Score", f"{results.get('trainingScore', 0):.4f}")
                with col_ga2:
                    st.metric("Converged At", f"Gen {results.get('convergedAt', 0)}")
                
                st.subheader("📈 Convergence History")
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
            st.plotly_chart(
                create_backtest_plot(st.session_state.backtest_results),
                use_container_width=True
            )

            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
            with col_bt1:
                st.metric("MAE", f"{st.session_state.backtest_results['stats']['mae']:.2f}%")
            with col_bt2:
                st.metric("RMSE", f"{st.session_state.backtest_results['stats']['rmse']:.2f}%")
            with col_bt3:
                st.metric("R²", f"{st.session_state.backtest_results['stats']['r2']:.3f}")
            with col_bt4:
                st.metric(
                    "Hit Rate",
                    f"{st.session_state.backtest_results['stats']['hitRate']*100:.1f}%"
                )

    # Display sensitivity analysis
    if st.session_state.sensitivity_results:
        with sensitivity_placeholder.container():
            st.plotly_chart(
                create_sensitivity_plot(st.session_state.sensitivity_results),
                use_container_width=True
            )

    # Export results
    if export_button and st.session_state.results:
        try:
            results = st.session_state.results
            summary_table = generate_summary_table(results['stats'], results['riskMetrics'])
            
            # Add metadata
            metadata = pd.DataFrame({
                'Parameter': ['Mode', 'Date', 'Iterations', 'Asset Type', 'Horizon', 'Historical Period Type', 'Historical Data Points', 'Macro Input Mode'],
                'Value': [
                    mode,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    iterations,
                    ASSET_PRESETS[asset_type]['name'],
                    f"{horizon_years:.2f} years ({period_count} {period_unit})",
                    hist_period_type,
                    len(historical_data),
                    macro_input_mode
                ]
            })
            
            # Combine tables
            export_df = pd.concat([
                metadata,
                pd.DataFrame({'Parameter': [''], 'Value': ['']}),  # Spacer
                summary_table.rename(columns={'Metric': 'Parameter'})
            ], ignore_index=True)
            
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"mc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            logger.info("Results exported successfully")
            st.success("✅ Results ready for download!")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            logger.error(f"Export failed: {traceback.format_exc()}")


if __name__ == "__main__":
    logger.info("Application started")
    try:
        main()
    except Exception as e:
        st.error("💥 Fatal error occurred")
        logger.critical(f"Fatal error: {traceback.format_exc()}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
