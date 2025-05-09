# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from main import CoolingTowerDesign # Make sure main.py with CoolingTowerDesign class is in the same directory

# Initialize or retrieve CoolingTowerDesign object from session state
if 'ct' not in st.session_state:
    st.session_state.ct = CoolingTowerDesign()
ct = st.session_state.ct

st.title("❄️ Cooling Tower Design & Analysis Tool")
st.markdown("""
This application provides a preliminary design and analysis for industrial cooling towers.
Adjust the parameters in the sidebar to see how they affect the design and performance.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Input Parameters")
st.sidebar.subheader("Base Thermal Parameters")

# --- Heat Load ---
# Ensure ct.heat_load is scalar for number_input
current_heat_load = getattr(ct, 'heat_load', 1000) # Default if not set
if not isinstance(current_heat_load, (int, float)):
    st.warning(f"Heat load had an unexpected type ({type(current_heat_load)}), resetting to default.")
    current_heat_load = 1000
ct.heat_load = st.sidebar.number_input(
    "Heat Load (kW)",
    min_value=100,
    max_value=100000,
    value=int(current_heat_load),
    step=100,
    key="heat_load_input"
)

# --- T_hot slider (Error Source was around here) ---
t_hot_val_from_state = getattr(ct, 'T_hot', 40.0) # Default if not set
t_hot_min_val, t_hot_max_val = 30.0, 90.0

if isinstance(t_hot_val_from_state, list):
    st.warning(f"Session state for T_hot was a list ({t_hot_val_from_state}). Attempting to recover.")
    try:
        t_hot_initial = float(t_hot_val_from_state[0]) if t_hot_val_from_state else t_hot_min_val
    except (IndexError, TypeError, ValueError):
        t_hot_initial = 40.0  # Fallback default
elif not isinstance(t_hot_val_from_state, (int, float)):
    st.warning(f"Session state for T_hot had an unexpected type ({type(t_hot_val_from_state)}). Resetting.")
    t_hot_initial = 40.0 # Fallback default
else:
    t_hot_initial = float(t_hot_val_from_state)

# Ensure the initial value is within bounds
t_hot_initial = max(t_hot_min_val, min(t_hot_initial, t_hot_max_val))

ct.T_hot = st.sidebar.slider(
    "Hot Water Temperature (°C)",
    min_value=t_hot_min_val,
    max_value=t_hot_max_val,
    value=t_hot_initial,  # This is now guaranteed to be a float
    step=0.5,
    key="t_hot_slider"
)

# --- T_cold slider ---
# ct.T_hot is now a scalar float from the slider above
t_cold_min_val = 10.0
# Ensure max_value for T_cold is strictly less than T_hot
t_cold_max_val = ct.T_hot - 0.5 # Use a small delta like 0.5 or 1.0
if t_cold_max_val < t_cold_min_val:
    t_cold_max_val = t_cold_min_val # Prevent max < min

t_cold_val_from_state = getattr(ct, 'T_cold', 30.0)
if isinstance(t_cold_val_from_state, list):
    st.warning(f"Session state for T_cold was a list ({t_cold_val_from_state}). Attempting to recover.")
    try:
        t_cold_initial = float(t_cold_val_from_state[0]) if t_cold_val_from_state else t_cold_min_val
    except (IndexError, TypeError, ValueError):
        t_cold_initial = 30.0
elif not isinstance(t_cold_val_from_state, (int, float)):
    st.warning(f"Session state for T_cold had an unexpected type ({type(t_cold_val_from_state)}). Resetting.")
    t_cold_initial = 30.0
else:
    t_cold_initial = float(t_cold_val_from_state)

t_cold_initial = max(t_cold_min_val, min(t_cold_initial, t_cold_max_val))

ct.T_cold = st.sidebar.slider(
    "Target Cold Water Temperature (°C)",
    min_value=t_cold_min_val,
    max_value=t_cold_max_val,
    value=t_cold_initial,
    step=0.5,
    key="t_cold_slider",
    disabled=(t_cold_min_val >= t_cold_max_val + 0.1) # Disable if no valid range
)

# --- T_wb slider ---
# ct.T_cold is now a scalar float
t_wb_min_val = 5.0
# Ensure max_value for T_wb is strictly less than T_cold
t_wb_max_val = ct.T_cold - 0.5
if t_wb_max_val < t_wb_min_val:
    t_wb_max_val = t_wb_min_val

t_wb_val_from_state = getattr(ct, 'T_wb', 25.0)
if isinstance(t_wb_val_from_state, list):
    st.warning(f"Session state for T_wb was a list ({t_wb_val_from_state}). Attempting to recover.")
    try:
        t_wb_initial = float(t_wb_val_from_state[0]) if t_wb_val_from_state else t_wb_min_val
    except (IndexError, TypeError, ValueError):
        t_wb_initial = 25.0
elif not isinstance(t_wb_val_from_state, (int, float)):
    st.warning(f"Session state for T_wb had an unexpected type ({type(t_wb_val_from_state)}). Resetting.")
    t_wb_initial = 25.0
else:
    t_wb_initial = float(t_wb_val_from_state)

t_wb_initial = max(t_wb_min_val, min(t_wb_initial, t_wb_max_val))

ct.T_wb = st.sidebar.slider(
    "Wet-Bulb Temperature (°C)",
    min_value=t_wb_min_val,
    max_value=t_wb_max_val,
    value=t_wb_initial,
    step=0.5,
    key="t_wb_slider",
    disabled=(t_wb_min_val >= t_wb_max_val + 0.1) # Disable if no valid range
)

st.sidebar.subheader("Advanced Design Parameters")
# Assuming these are initialized correctly in the class or handled similarly if needed
ct.fill_type = st.sidebar.selectbox("Fill Type", ["Film", "Splash", "Structured"], index=["Film", "Splash", "Structured"].index(getattr(ct, 'fill_type', 'Film')), key="fill_type_select")
ct.pressure_drop = st.sidebar.number_input("Target Air Pressure Drop (Pa)", min_value=50, max_value=500, value=int(getattr(ct, 'pressure_drop', 150)), step=10, key="pressure_drop_input")
ct.efficiency = st.sidebar.slider("Fan System Efficiency (decimal)", min_value=0.5, max_value=0.95, value=float(getattr(ct, 'efficiency', 0.7)), step=0.01, key="efficiency_slider")
ct.cycles_of_concentration = st.sidebar.slider("Cycles of Concentration (COC)", min_value=1.5, max_value=10.0, value=float(getattr(ct, 'cycles_of_concentration', 3.0)), step=0.1, key="coc_slider")
ct.drift_loss_rate = st.sidebar.number_input("Drift Loss Rate (fraction)", min_value=0.0001, max_value=0.02, value=float(getattr(ct, 'drift_loss_rate', 0.002)), step=0.0001, format="%.4f", key="drift_loss_input")


# --- Main Application Area ---
try:
    # Trigger calculations to update the state of the `ct` object
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
        summary_text = ct.design_summary() # This method should handle potential internal errors
        if "Error" in summary_text:
            st.error(summary_text)
        else:
            st.markdown(f"```\n{summary_text}\n```")
        st.info("""
        **Note on Design Summary:**
        - **Approach:** $T_{cold} - T_{wb}$
        - **Range:** $T_{hot} - T_{cold}$
        """)

    with tab_costs:
        st.header("Economic Analysis Details")
        _ = ct.estimate_costs() # Ensure costs are up-to-date
        if all(v is not None for v in [ct.water_flow, ct.fill_data, ct.fill_height, ct.air_flow, ct.total_cost]):
            fill_area = float(np.asarray(ct.water_flow).item()) / 6
            fill_cost_val = fill_area * ct.fill_data["cost"] * float(np.asarray(ct.fill_height).item())
            fan_cost_val = 5000 * (float(np.asarray(ct.air_flow).item()) ** 0.8)
            basin_cost_val = 300 * fill_area

            st.markdown(f"- **Fill Material Cost:** ${fill_cost_val:,.0f}")
            st.markdown(f"- **Fan System Cost:** ${fan_cost_val:,.0f}")
            st.markdown(f"- **Basin Cost:** ${basin_cost_val:,.0f}")
            st.markdown(f"---")
            st.markdown(f"- **Total Estimated Capital Cost:** ${ct.total_cost:,.0f}")
        else:
            st.warning("Cost components cannot be fully displayed. Please check input parameters or calculation results.")
        st.info("""
        **Cost Assumptions:**
        - Fill loading: 6 m³/hr of water per m² of fill plan area.
        - Fan & Basin costs are empirical.
        """)

    with tab_water_balance:
        st.header("Water Balance Analysis")
        # Ensure calculations are safe
        try:
            evap_loss = ct.calculate_evaporation_loss()
            drift_loss = ct.calculate_drift_loss()
            st.write(f"**Evaporation Loss:** {evap_loss:.2f} m³/hr")
            st.write(f"**Drift Loss:** {drift_loss:.2f} m³/hr ({ct.drift_loss_rate*100:.3f}%)")

            blowdown_loss = ct.calculate_blowdown_loss() # Returns np.nan if COC <= 1
            if np.isnan(blowdown_loss):
                st.error("Blowdown loss cannot be calculated. Cycles of Concentration must be > 1.")
                makeup_water = np.nan
            else:
                st.write(f"**Blowdown Loss:** {blowdown_loss:.2f} m³/hr (COC = {ct.cycles_of_concentration})")
                makeup_water = ct.estimate_makeup_water() # Also handles nan blowdown

            if not np.isnan(makeup_water):
                st.write(f"**Total Estimated Makeup Water:** {makeup_water:.2f} m³/hr")
            else:
                st.write(f"**Total Estimated Makeup Water:** Calculation pending valid COC.")
            st.latex(r"M = E + B + D") # Example formula
        except Exception as e:
            st.error(f"Error in water balance calculation: {e}")


    with tab_performance_plots:
        st.header("Performance & Sensitivity Curves")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cost vs. Cooling Tower Approach")
            if st.button("Generate Cost vs. Approach Plot", key="cost_approach_plot_btn"):
                if ct.T_wb >= ct.T_hot -1: # Ensure some gap
                     st.error("Wet bulb temp too high relative to hot water temp for this plot.")
                else:
                    with st.spinner("Generating plot..."):
                        fig = ct.plot_performance_curve()
                        st.pyplot(fig)
                        plt.close(fig) # Good practice to close fig
            
            st.subheader("Thermal Performance Curve")
            if st.button("Generate Thermal Performance Plot", key="thermal_perf_plot_btn"):
                if ct.T_wb >= ct.T_hot -1:
                    st.error("Wet bulb temp too high relative to hot water temp for this plot.")
                else:
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
            st.warning("This plot uses a *simplified hypothetical model* for efficiency change with fill height for illustrative purposes.")

    with tab_components:
        st.header("Component Specific Analysis")
        st.subheader("Fill Type Details")
        ct.select_fill() # Ensure ct.fill_data is current
        st.write(f"Selected Fill Type: **{ct.fill_type}**")
        st.write(f"- Characteristic Ka value: {ct.fill_data['Ka']}")
        st.write(f"- Cost parameter: ${ct.fill_data['cost']}/m²")
        if ct.fill_height is not None:
             st.write(f"Calculated Fill Height: {ct.fill_height:.2f} m")
        
        st.subheader("Fan Type Comparison (Illustrative)")
        try:
            fan_comparison_data = ct.compare_fan_types()
            st.table(fan_comparison_data)
        except Exception as e:
            st.error(f"Could not generate fan comparison: {e}")


    with tab_optimization:
        st.header("Design Optimization")
        st.subheader("Optimize Approach Temperature for Cost")
        opt_approach_min = st.slider("Min Approach for Opt. (°C)", 1.0, 7.0, 3.0, 0.1, key="opt_min")
        # Ensure max is greater than min
        opt_approach_max_min_val = opt_approach_min + 0.1 
        opt_approach_max = st.slider("Max Approach for Opt. (°C)", opt_approach_max_min_val, 15.0, max(opt_approach_max_min_val, 10.0), 0.1, key="opt_max")


        if st.button("Find Optimal Approach Temperature", key="optimize_btn"):
            if ct.T_wb >= ct.T_hot - (opt_approach_min + 1.0) : # Check if optimization is feasible
                st.error("Current T_wb and T_hot settings may not allow for the specified optimization range. Adjust temperatures or optimization bounds.")
            else:
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
                    # cycles_of_concentration, drift_loss_rate are not directly in cost_function path for optimization
                    
                    optimal_approach = ct_for_opt.optimize_approach(target_range_min=opt_approach_min, target_range_max=opt_approach_max)
                    if optimal_approach is not None and ct_for_opt.total_cost is not None:
                        st.success(f"Optimization Complete!")
                        st.metric("Optimal Approach Temperature", f"{optimal_approach:.1f} °C")
                        st.metric("Resulting Cold Water Temperature", f"{ct_for_opt.T_cold:.1f} °C")
                        st.metric("Estimated Cost at Optimal Approach", f"${ct_for_opt.total_cost:,.0f}")
                        st.info(f"To apply this, manually set Target Cold Water Temp. to {ct_for_opt.T_cold:.1f}°C.")
                    else:
                        st.error("Optimization failed. Could not find an optimal approach. Try different optimization bounds or check input parameters like T_hot and T_wb.")

except ValueError as e: # Catch specific ValueErrors from your class's validations
    st.error(f"⚠️ Input Error: {e}")
    st.warning("Please adjust input parameters in the sidebar. Common issues include:\n"
               "- Hot Water Temperature not being greater than Cold Water Temperature.\n"
               "- Cold Water Temperature not being greater than Wet-Bulb Temperature.")
except Exception as e: # Catch any other unexpected errors
    st.error(f"An unexpected application error occurred: {type(e).__name__} - {e}")
    st.exception(e) # This will print the full traceback in the Streamlit app for debugging

st.sidebar.markdown("---")
st.sidebar.info("Cooling Tower Design Tool")
