# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from main import CoolingTowerDesign # Import your class

st.set_page_config(layout="wide", page_title="Cooling Tower Design Tool")

st.title("❄️ Cooling Tower Design & Analysis Tool")
st.markdown("""
This application provides a preliminary design and analysis for industrial cooling towers.
Adjust the parameters in the sidebar to see how they affect the design and performance.
""")

# Initialize CoolingTowerDesign object from main.py
# It's good practice to create one instance and update its attributes from Streamlit widgets
if 'ct' not in st.session_state:
    st.session_state.ct = CoolingTowerDesign()
ct = st.session_state.ct


# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Input Parameters")

st.sidebar.subheader("Base Thermal Parameters")
ct.heat_load = st.sidebar.number_input("Heat Load (kW)", min_value=100, max_value=100000, value=ct.heat_load, step=100)
ct.T_hot = st.sidebar.slider("Hot Water Temperature (°C)", min_value=30.0, max_value=90.0, value=ct.T_hot, step=0.5)

# Ensure T_cold < T_hot and T_wb < T_cold dynamically
t_cold_value = st.sidebar.slider("Target Cold Water Temperature (°C)", 
                                 min_value=10.0, 
                                 max_value=float(ct.T_hot - 1.0) if ct.T_hot > 10 else 40.0, # Ensure max is valid
                                 value=min(ct.T_cold, float(ct.T_hot - 1.0) if ct.T_hot > 10 else 40.0), # Ensure value is valid
                                 step=0.5)
ct.T_cold = t_cold_value

t_wb_value = st.sidebar.slider("Wet-Bulb Temperature (°C)", 
                               min_value=5.0, 
                               max_value=float(ct.T_cold - 1.0) if ct.T_cold > 5 else 25.0, # Ensure max is valid
                               value=min(ct.T_wb, float(ct.T_cold - 1.0) if ct.T_cold > 5 else 25.0), # Ensure value is valid
                               step=0.5)
ct.T_wb = t_wb_value


st.sidebar.subheader("Advanced Design Parameters")
ct.fill_type = st.sidebar.selectbox("Fill Type", ["Film", "Splash", "Structured"], index=["Film", "Splash", "Structured"].index(ct.fill_type))
ct.pressure_drop = st.sidebar.number_input("Target Air Pressure Drop (Pa)", min_value=50, max_value=500, value=ct.pressure_drop, step=10)
ct.efficiency = st.sidebar.slider("Fan System Efficiency (decimal)", min_value=0.5, max_value=0.95, value=ct.efficiency, step=0.01)
ct.cycles_of_concentration = st.sidebar.slider("Cycles of Concentration (COC)", min_value=1.5, max_value=10.0, value=ct.cycles_of_concentration, step=0.1)
ct.drift_loss_rate = st.sidebar.number_input("Drift Loss Rate (fraction of water flow)", min_value=0.0001, max_value=0.02, value=ct.drift_loss_rate, step=0.0001, format="%.4f")

# --- Main Application Area ---
try:
    # Ensure dependent calculations are triggered if inputs change
    # These methods update the state of the `ct` object
    ct.select_fill() # Needs to be called to set ct.fill_data based on ct.fill_type
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
        # ... (add your info/notes for summary as before)

    with tab_costs:
        st.header("Economic Analysis Details")
        # Recalculate costs here or ensure ct object is up to date.
        # The call to ct.estimate_costs() above should suffice if no params changed since.
        _ = ct.estimate_costs() # Ensure costs are up-to-date
        if ct.water_flow and ct.fill_data and ct.fill_height and ct.air_flow and ct.total_cost is not None:
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
            st.warning("Cost components cannot be displayed. Please check input parameters.")
        # ... (add your info/notes for costs as before)


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
        # ... (add your latex formulas and notes as before)

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
        # ... (add captions as before)

    with tab_components:
        st.header("Component Specific Analysis")
        st.subheader("Fill Type Details")
        ct.select_fill() # Ensure ct.fill_data is current
        st.write(f"Selected Fill Type: **{ct.fill_type}**")
        st.write(f"- Characteristic Ka value: {ct.fill_data['Ka']}")
        st.write(f"- Cost parameter: ${ct.fill_data['cost']}/m²")
        st.write(f"Calculated Fill Height: {ct.fill_height:.2f} m")
        
        st.subheader("Fan Type Comparison (Illustrative)")
        fan_comparison_data = ct.compare_fan_types()
        st.table(fan_comparison_data)
        # ... (add captions as before)

    with tab_optimization:
        st.header("Design Optimization")
        st.subheader("Optimize Approach Temperature for Cost")
        opt_approach_min = st.slider("Min Approach for Opt. (°C)", 1.0, 7.0, 3.0, 0.1, key="opt_min")
        opt_approach_max = st.slider("Max Approach for Opt. (°C)", opt_approach_min + 0.1, 15.0, 10.0, 0.1, key="opt_max")

        if st.button("Find Optimal Approach Temperature", key="optimize_btn"):
            with st.spinner("Optimizing..."):
                # Create a temporary copy for optimization to avoid altering main `ct` state unexpectedly
                # or ensure optimize_approach method handles state appropriately.
                # The current optimize_approach modifies the instance it's called on.
                ct_for_opt = CoolingTowerDesign()
                # Copy relevant parameters from the main 'ct' object to 'ct_for_opt'
                ct_for_opt.heat_load = ct.heat_load
                ct_for_opt.T_hot = ct.T_hot
                # ct_for_opt.T_cold will be set by the optimizer
                ct_for_opt.T_wb = ct.T_wb
                ct_for_opt.Cp_water = ct.Cp_water # etc. for all params used in cost_function
                ct_for_opt.air_density = ct.air_density
                ct_for_opt.water_density = ct.water_density
                ct_for_opt.fill_type = ct.fill_type
                ct_for_opt.pressure_drop = ct.pressure_drop
                ct_for_opt.efficiency = ct.efficiency
                # cycles_of_concentration, drift_loss_rate (if they affect cost indirectly)

                optimal_approach = ct_for_opt.optimize_approach(target_range_min=opt_approach_min, target_range_max=opt_approach_max)
                if optimal_approach is not None:
                    st.success(f"Optimization Complete!")
                    st.metric("Optimal Approach Temperature", f"{optimal_approach:.1f} °C")
                    st.metric("Resulting Cold Water Temperature", f"{ct_for_opt.T_cold:.1f} °C")
                    st.metric("Estimated Cost at Optimal Approach", f"${ct_for_opt.total_cost:,.0f}")
                    st.info(f"To apply this, manually set Target Cold Water Temp. to {ct_for_opt.T_cold:.1f}°C.")
                else:
                    st.error("Optimization failed. Try different bounds or check parameters.")
        # ... (add notes as before)

except ValueError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.warning("Please adjust input parameters in the sidebar. Common issues include:\n"
               "- Hot Water Temperature not being greater than Cold Water Temperature.\n"
               "- Cold Water Temperature not being greater than Wet-Bulb Temperature.")
except Exception as e:
    st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
    # st.exception(e) # Uncomment for full traceback during development

st.sidebar.markdown("---")
st.sidebar.info("Cooling Tower Design Tool v1.0")
# ... (add suggestions for more functionalities if desired)