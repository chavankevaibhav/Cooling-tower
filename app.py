# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from main import CoolingTowerDesign

st.set_page_config(layout="wide", page_title="Cooling Tower Design Tool")

st.title("❄️ Cooling Tower Design & Analysis Tool")
st.markdown("""
This application provides a preliminary design and analysis for industrial cooling towers.
Adjust the parameters in the sidebar to see how they affect the design and performance.
""")

# Initialize CoolingTowerDesign object from main.py
if 'ct' not in st.session_state:
    st.session_state.ct = CoolingTowerDesign()
ct = st.session_state.ct

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Input Parameters")

# Example Presets
st.sidebar.subheader("🔖 Example Presets")
preset = st.sidebar.selectbox("Load Example Scenario", ["Custom", "Small Industrial", "Large Power Plant"])
if preset == "Small Industrial":
    ct.heat_load = 1500
    ct.T_hot = 45.0
    ct.T_cold = 32.0
    ct.T_wb = 27.0
elif preset == "Large Power Plant":
    ct.heat_load = 50000
    ct.T_hot = 60.0
    ct.T_cold = 35.0
    ct.T_wb = 28.0

st.sidebar.subheader("Base Thermal Parameters")
ct.heat_load = st.sidebar.number_input("Heat Load (kW)", min_value=100, max_value=100000, value=ct.heat_load, step=100)
ct.T_hot = st.sidebar.slider("Hot Water Temperature (°C)", min_value=30.0, max_value=90.0, value=ct.T_hot, step=0.5)

# Use attributes directly, no conversion needed
t_cold_value = st.sidebar.slider(
    "Target Cold Water Temperature (°C)", 
    min_value=10.0, 
    max_value=ct.T_hot - 1.0 if ct.T_hot > 10 else 40.0,
    value=min(ct.T_cold, ct.T_hot - 1.0 if ct.T_hot > 10 else 40.0),
    step=0.5
)
ct.T_cold = t_cold_value

t_wb_value = st.sidebar.slider(
    "Wet-Bulb Temperature (°C)", 
    min_value=5.0, 
    max_value=ct.T_cold - 1.0 if ct.T_cold > 5 else 25.0,
    value=min(ct.T_wb, ct.T_cold - 1.0 if ct.T_cold > 5 else 25.0),
    step=0.5
)
ct.T_wb = t_wb_value

st.sidebar.subheader("Advanced Design Parameters")
ct.fill_type = st.sidebar.selectbox("Fill Type", ["Film", "Splash", "Structured"], index=["Film", "Splash", "Structured"].index(ct.fill_type))
ct.pressure_drop = st.sidebar.number_input("Target Air Pressure Drop (Pa)", min_value=50, max_value=500, value=ct.pressure_drop, step=10)
ct.efficiency = st.sidebar.slider("Fan System Efficiency (decimal)", min_value=0.5, max_value=0.95, value=ct.efficiency, step=0.01)
ct.cycles_of_concentration = st.sidebar.slider(
    "Cycles of Concentration (COC)", 
    min_value=1.5, 
    max_value=10.0, 
    value=ct.cycles_of_concentration, 
    step=0.1
)
ct.drift_loss_rate = st.sidebar.number_input("Drift Loss Rate (fraction of water flow)", min_value=0.0001, max_value=0.02, value=ct.drift_loss_rate, step=0.0001, format="%.4f")

# Real-Time Calculated Results
st.sidebar.subheader("📈 Key Calculated Results")
try:
    ct.calculate_water_flow()
    ct.air_flow_requirement()
    ct.calculate_fill_height()
    st.sidebar.metric("Water Flow (m³/hr)", f"{ct.water_flow:.2f}")
    st.sidebar.metric("Air Flow (m³/s)", f"{ct.air_flow:.2f}")
    st.sidebar.metric("Fill Height (m)", f"{ct.fill_height:.2f}")
except Exception as e:
    st.sidebar.error(f"Calculation error: {e}")

# Input Validation Feedback
if ct.T_hot <= ct.T_cold:
    st.sidebar.warning("Hot water temperature must be greater than cold water temperature.")
if ct.T_cold <= ct.T_wb:
    st.sidebar.warning("Cold water temperature must be greater than wet-bulb temperature.")

# --- Main Application Area ---
try:
    ct.select_fill()
    ct.calculate_water_flow()
    ct.air_flow_requirement()
    ct.calculate_fill_height()
    ct.estimate_costs()

    tab_summary, tab_costs, tab_water_balance, tab_performance_plots, tab_components, tab_optimization = st.tabs([
        "📊 Design Summary", "💲 Cost Analysis", "💧 Water Balance",
        "📈 Performance Curves", "🛠️ Component Analysis", "💡 Optimization"
    ])

    with tab_summary:
        st.header("Cooling Tower Design Summary")
        summary_text = ct.design_summary()
        st.markdown(f"```\n{summary_text}\n```")
        st.download_button("Download Design Summary", summary_text, file_name="cooling_tower_summary.txt")

    with tab_costs:
        st.header("Economic Analysis Details")
        _ = ct.estimate_costs()
        if ct.water_flow and ct.fill_data and ct.fill_height and ct.air_flow and ct.total_cost is not None:
            fill_area = ct.water_flow / 6
            fill_cost_val = fill_area * ct.fill_data["cost"] * ct.fill_height
            fan_cost_val = 5000 * (ct.air_flow ** 0.8)
            basin_cost_val = 300 * fill_area

            st.markdown(f"- **Fill Material Cost:** ${fill_cost_val:,.0f}")
            st.markdown(f"- **Fan System Cost:** ${fan_cost_val:,.0f}")
            st.markdown(f"- **Basin Cost:** ${basin_cost_val:,.0f}")
            st.markdown(f"---")
            st.markdown(f"- **Total Estimated Capital Cost:** ${ct.total_cost:,.0f}")
        else:
            st.warning("Cost components cannot be displayed. Please check input parameters.")

    with tab_water_balance:
        st.header("Water Balance Analysis")
        evap_loss = ct.calculate_evaporation_loss()
        drift_loss = ct.calculate_drift_loss()
        st.write(f"**Evaporation Loss:** {evap_loss:.2f} m³/hr")
        st.write(f"**Drift Loss:** {drift_loss:.2f} m³/hr ({ct.drift_loss_rate*100:.3f}%)")
        blowdown_loss = ct.calculate_blowdown_loss()
        if np.isnan(blowdown_loss):
            st.error("Blowdown loss cannot be calculated. Cycles of Concentration must be > 1.")
            makeup_water = np.nan
        else:
            st.write(f"**Blowdown Loss:** {blowdown_loss:.2f} m³/hr (COC = {ct.cycles_of_concentration})")
            makeup_water = ct.estimate_makeup_water()
        if not np.isnan(makeup_water):
            st.write(f"**Total Estimated Makeup Water:** {makeup_water:.2f} m³/hr")
        else:
            st.write(f"**Total Estimated Makeup Water:** Calculation pending valid COC.")

    with tab_performance_plots:
        st.header("Performance & Sensitivity Curves")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cost vs. Cooling Tower Approach")
            if st.button("Generate Cost vs. Approach Plot", key="cost_approach_plot_btn"):
                if ct.T_wb >= ct.T_hot:
                    st.error("Wet bulb temp must be less than hot water temp for this plot.")
                else:
                    with st.spinner("Generating plot..."):
                        fig = ct.plot_performance_curve()
                        st.pyplot(fig)
                        plt.close(fig)
            st.subheader("Thermal Performance Curve")
            if st.button("Generate Thermal Performance Plot", key="thermal_perf_plot_btn"):
                with st.spinner("Generating plot..."):
                    fig = ct.thermal_performance_curve()
                    st.pyplot(fig)
                    plt.close(fig)
        with col2:
            st.subheader("Hypothetical Efficiency vs. Fill Height")
            if st.button("Generate Efficiency vs. Fill Height Plot", key="eff_fill_plot_btn"):
                with st.spinner("Generating plot..."):
                    fig = ct.plot_efficiency_vs_fill_height()
                    st.pyplot(fig)
                    plt.close(fig)
            st.warning("This plot uses a *simplified hypothetical model* for efficiency change with fill height.")
            st.subheader("Parameter Sensitivity Analysis")
            param = st.selectbox("Select Parameter", ["Heat Load", "Hot Water Temp", "Cold Water Temp"])
            if st.button("Show Sensitivity Plot", key="sens_plot_btn"):
                with st.spinner("Generating sensitivity plot..."):
                    fig = ct.plot_sensitivity(param)
                    st.pyplot(fig)
                    plt.close(fig)

    with tab_components:
        st.header("Component Specific Analysis")
        st.subheader("Fill Type Details")
        ct.select_fill()
        st.write(f"Selected Fill Type: **{ct.fill_type}**")
        st.write(f"- Characteristic Ka value: {ct.fill_data['Ka']}")
        st.write(f"- Cost parameter: ${ct.fill_data['cost']}/m²")
        st.write(f"Calculated Fill Height: {ct.fill_height:.2f} m")
        st.subheader("Fan Type Comparison (Illustrative)")
        fan_comparison_data = ct.compare_fan_types()
        st.table(fan_comparison_data)

    with tab_optimization:
        st.header("Design Optimization")
        st.subheader("Optimize Approach Temperature for Cost")
        opt_approach_min = st.slider("Min Approach for Opt. (°C)", 1.0, 7.0, 3.0, 0.1, key="opt_min")
        opt_approach_max = st.slider("Max Approach for Opt. (°C)", opt_approach_min + 0.1, 15.0, 10.0, 0.1, key="opt_max")
        if st.button("Find Optimal Approach Temperature", key="optimize_btn"):
            with st.spinner("Optimizing..."):
                ct_for_opt = CoolingTowerDesign()
                ct_for_opt.heat_load = ct.heat_load
                ct_for_opt.T_hot = ct.T_hot
                ct_for_opt.T_wb = ct.T_wb
                ct_for_opt.Cp_water = ct.Cp_water
                ct_for_opt.air_density = ct.air_density
                ct_for_opt.water_density = ct.water_density
                ct_for_opt.fill_type = ct.fill_type
                ct_for_opt.pressure_drop = ct.pressure_drop
                ct_for_opt.efficiency = ct.efficiency
                optimal_approach = ct_for_opt.optimize_approach(target_range_min=opt_approach_min, target_range_max=opt_approach_max)
                if optimal_approach is not None:
                    st.success(f"Optimization Complete!")
                    st.metric("Optimal Approach Temperature", f"{optimal_approach:.1f} °C")
                    st.metric("Resulting Cold Water Temperature", f"{ct_for_opt.T_cold:.1f} °C")
                    st.metric("Estimated Cost at Optimal Approach", f"${ct_for_opt.total_cost:,.0f}")
                    st.info(f"To apply this, manually set Target Cold Water Temp. to {ct_for_opt.T_cold:.1f}°C.")
                else:
                    st.error("Optimization failed. Try different bounds or check parameters.")

except ValueError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.warning("Please adjust input parameters in the sidebar. Common issues include:\n"
               "- Hot Water Temperature not being greater than Cold Water Temperature.\n"
               "- Cold Water Temperature not being greater than Wet-Bulb Temperature.")
except Exception as e:
    st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")

st.sidebar.markdown("---")
st.sidebar.info("Cooling Tower Design Tool v1.0")
