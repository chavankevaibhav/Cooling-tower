import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Advanced Cooling Tower Design Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CoolingTowerDesign:
    def __init__(self):
        # Base parameters (SI units)
        self.heat_load = 1000  # kW
        self._T_hot = 45.0
        self._T_cold = 32.0
        self._T_wb = 27.0
        self._cycles_of_concentration = 3
        self.Cp_water = 4.18  # kJ/kg·°C
        self.air_density = 1.2  # kg/m³
        self.water_density = 997  # kg/m³

        # Advanced parameters
        self.fill_type = "Film"  # Film/Splash/Structured
        self.pressure_drop = 150  # Pa
        self.efficiency = 0.7  # Fan efficiency
        self._cycles_of_concentration = 3  # Default value
        self.drift_loss_rate = 0.002  # Typical drift loss rate (0.2% of water flow)
        
        self.water_flow = None
        self.air_flow = None
        self.fill_height = None
        self.fill_data = None
        self.total_cost = None

    @property
    def T_cold(self):
        return float(self._T_cold)
    @T_cold.setter
    def T_cold(self, value):
        self._T_cold = float(value)

    @property
    def T_hot(self):
        return float(self._T_hot)
    @T_hot.setter
    def T_hot(self, value):
        self._T_hot = float(value)

    @property
    def T_wb(self):
        return float(self._T_wb)
    @T_wb.setter
    def T_wb(self, value):
        self._T_wb = float(value)

    @property
    def cycles_of_concentration(self):
        return float(self._cycles_of_concentration)
    @cycles_of_concentration.setter
    def cycles_of_concentration(self, value):
        self._cycles_of_concentration = float(value)

    def calculate_water_flow(self):
        delta_T = self.T_hot - self.T_cold
        if delta_T <= 0:
            raise ValueError("Delta T must be positive. Check T_hot and T_cold values.")
        self.water_flow = (self.heat_load / (self.Cp_water * delta_T)) * 3600
        return round(self.water_flow, 1)

    def air_flow_requirement(self, L_G_ratio=1.0):
        if self.water_flow is None:
            self.calculate_water_flow()
        water_flow_kg_s = (self.water_flow / 3600) * self.water_density
        self.air_flow = water_flow_kg_s / (L_G_ratio * self.air_density)
        return round(self.air_flow, 1)

    def select_fill(self):
        fill_properties = {
            "Splash": {"Ka": 0.6, "cost": 80},
            "Film": {"Ka": 0.8, "cost": 120},
            "Structured": {"Ka": 1.1, "cost": 200}
        }
        self.fill_data = fill_properties.get(self.fill_type, fill_properties["Film"])
        return self.fill_data

    def calculate_fill_height(self):
        if self.fill_data is None:
            self.select_fill()
        target_KaVL = 1.5
        self.fill_height = target_KaVL / self.fill_data["Ka"]
        return round(self.fill_height, 2)

    def estimate_costs(self):
        if self.water_flow is None: self.calculate_water_flow()
        if self.fill_data is None: self.select_fill()
        if self.fill_height is None: self.calculate_fill_height()
        fill_area = self.water_flow / 6
        fill_cost = fill_area * self.fill_data["cost"] * self.fill_height
        if self.air_flow is None: self.air_flow_requirement()
        fan_cost = 5000 * (self.air_flow ** 0.8)
        basin_cost = 300 * fill_area
        self.total_cost = fill_cost + fan_cost + basin_cost
        return round(self.total_cost, 0)

    def optimize_approach(self, target_range_min=3, target_range_max=10):
        original_T_cold = self.T_cold
        def cost_function(approach_array):
            approach = approach_array[0]
            self.T_cold = self.T_wb + approach
            if self.T_cold >= self.T_hot: return 1e12
            try:
                self.calculate_water_flow()
                self.air_flow_requirement()
                self.calculate_fill_height()
                return self.estimate_costs()
            except ValueError: return 1e12

        res = minimize(cost_function, x0=np.array([(target_range_min + target_range_max) / 2]), bounds=[(target_range_min, target_range_max)])
        
        if res.success:
            optimal_approach = round(float(np.asarray(res.x).item()), 1)
            self.T_cold = self.T_wb + optimal_approach
            self.calculate_water_flow()
            self.air_flow_requirement()
            self.calculate_fill_height()
            self.estimate_costs()
            return optimal_approach
        else:
            self.T_cold = original_T_cold
            return None

    def plot_performance_curve(self):
        approaches = np.linspace(3, 10, 8)
        costs = []
        original_T_cold = self.T_cold
        for app_val in approaches:
            self.T_cold = self.T_wb + app_val
            if self.T_cold >= self.T_hot:
                costs.append(np.nan)
            else:
                try:
                    self.calculate_water_flow()
                    self.air_flow_requirement()
                    self.calculate_fill_height()
                    costs.append(self.estimate_costs())
                except ValueError:
                    costs.append(np.nan)
        self.T_cold = original_T_cold
        self.calculate_water_flow()
        self.air_flow_requirement()
        self.calculate_fill_height()
        self.estimate_costs()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(approaches, costs, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel("Approach Temperature (°C)", fontsize=12)
        ax.set_ylabel("Total Cost ($)", fontsize=12)
        ax.set_title("Cost vs. Cooling Tower Approach", fontsize=14)
        ax.grid(True, alpha=0.3)
        return fig

    def fan_power(self):
        if self.air_flow is None: self.air_flow_requirement()
        return round((float(np.asarray(self.air_flow).item()) * self.pressure_drop) / (self.efficiency * 1000), 1)

    def calculate_evaporation_loss(self):
        latent_heat = 2450 
        evaporation_loss_kg_hr = (self.heat_load * 3600) / latent_heat
        evaporation_loss_m3_hr = evaporation_loss_kg_hr / self.water_density
        return round(float(np.asarray(evaporation_loss_m3_hr).item()), 2)

    def calculate_blowdown_loss(self):
        if self.cycles_of_concentration <= 1: return np.nan
        evaporation_loss = self.calculate_evaporation_loss()
        blowdown_loss = evaporation_loss / (self.cycles_of_concentration - 1)
        return round(float(np.asarray(blowdown_loss).item()), 2)

    def calculate_drift_loss(self):
        if self.water_flow is None: self.calculate_water_flow()
        drift_loss = float(np.asarray(self.water_flow).item()) * self.drift_loss_rate
        return round(drift_loss, 2)

    def estimate_makeup_water(self):
        evaporation_loss = self.calculate_evaporation_loss()
        blowdown_loss = self.calculate_blowdown_loss()
        drift_loss = self.calculate_drift_loss()
        if np.isnan(blowdown_loss): return np.nan
        makeup_water = evaporation_loss + blowdown_loss + drift_loss
        return round(float(np.asarray(makeup_water).item()), 2)

    def plot_water_loss_breakdown(self):
        cycles = np.arange(2, 8)
        evaporation = []
        blowdown = []
        drift = []
        total_makeup = []

        original_cycles = self.cycles_of_concentration
        for c in cycles:
            self.cycles_of_concentration = c
            evaporation.append(self.calculate_evaporation_loss())
            blowdown.append(self.calculate_blowdown_loss())
            drift.append(self.calculate_drift_loss())
            total_makeup.append(self.estimate_makeup_water())
        self.cycles_of_concentration = original_cycles

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cycles, evaporation, 'b^-', label='Evaporation Loss', linewidth=2, markersize=8)
        ax.plot(cycles, blowdown, 'ro-', label='Blowdown Loss', linewidth=2, markersize=8)
        ax.plot(cycles, drift, 'gs-', label='Drift Loss', linewidth=2, markersize=8)
        ax.plot(cycles, total_makeup, 'k*-', label='Total Makeup Water', linewidth=2, markersize=8)
        ax.set_xlabel("Cycles of Concentration", fontsize=12)
        ax.set_ylabel("Water Loss (m³/hr)", fontsize=12)
        ax.set_title("Water Loss Breakdown vs Cycles of Concentration", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def plot_fan_power_vs_air_flow(self):
        air_flows = np.linspace(5, 50, 10)
        fan_powers = []
        original_air_flow = self.air_flow
        for af in air_flows:
            self.air_flow = af
            fan_powers.append(self.fan_power())
        self.air_flow = original_air_flow

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(air_flows, fan_powers, 'ms-', linewidth=2, markersize=8)
        ax.set_xlabel("Air Flow (m³/s)", fontsize=12)
        ax.set_ylabel("Fan Power (kW)", fontsize=12)
        ax.set_title("Fan Power vs Air Flow", fontsize=14)
        ax.grid(True, alpha=0.3)
        return fig

    def design_summary(self):
        try:
            self.calculate_water_flow()
            self.air_flow_requirement()
            self.select_fill()
            self.calculate_fill_height()
            total_cost_val = self.estimate_costs()
            approach_temp = self.T_cold - self.T_wb
            fan_pwr = self.fan_power()
            range_temp = self.T_hot - self.T_cold

            return {
                "heat_load": self.heat_load,
                "water_flow": self.water_flow,
                "air_flow": self.air_flow,
                "T_hot": self.T_hot,
                "T_cold": self.T_cold,
                "T_wb": self.T_wb,
                "approach": approach_temp,
                "range": range_temp,
                "fill_type": self.fill_type,
                "ka_value": self.fill_data['Ka'],
                "fill_height": self.fill_height,
                "fan_power": fan_pwr,
                "pressure_drop": self.pressure_drop,
                "efficiency": self.efficiency,
                "fill_cost_param": self.fill_data['cost'],
                "total_cost": total_cost_val,
                "evaporation_loss": self.calculate_evaporation_loss(),
                "blowdown_loss": self.calculate_blowdown_loss(),
                "drift_loss": self.calculate_drift_loss(),
                "makeup_water": self.estimate_makeup_water(),
                "cycles": self.cycles_of_concentration
            }
        except ValueError as e:
            return {"error": f"Error in generating summary: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

# Initialize session state
if 'tower' not in st.session_state:
    st.session_state.tower = CoolingTowerDesign()

def main():
    st.title("🌊 Advanced Cooling Tower Design Tool")
    st.markdown("Design and optimize industrial cooling towers with comprehensive analysis")

    # Sidebar for input parameters
    st.sidebar.header("🔧 Design Parameters")
    
    # Basic thermal parameters
    st.sidebar.subheader("Thermal Parameters")
    heat_load = st.sidebar.number_input("Heat Load (kW)", min_value=100, max_value=10000, value=1000, step=100)
    T_hot = st.sidebar.number_input("Hot Water Temperature (°C)", min_value=30.0, max_value=80.0, value=45.0, step=0.5)
    T_wb = st.sidebar.number_input("Wet Bulb Temperature (°C)", min_value=15.0, max_value=35.0, value=27.0, step=0.5)
    T_cold = st.sidebar.number_input("Cold Water Temperature (°C)", min_value=20.0, max_value=50.0, value=32.0, step=0.5)
    
    # Advanced parameters
    st.sidebar.subheader("Design Specifications")
    fill_type = st.sidebar.selectbox("Fill Type", ["Film", "Splash", "Structured"], index=0)
    cycles = st.sidebar.slider("Cycles of Concentration", min_value=2.0, max_value=8.0, value=3.0, step=0.5)
    pressure_drop = st.sidebar.number_input("Pressure Drop (Pa)", min_value=50, max_value=500, value=150, step=25)
    fan_efficiency = st.sidebar.slider("Fan Efficiency", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

    # Update tower object
    tower = st.session_state.tower
    tower.heat_load = heat_load
    tower.T_hot = T_hot
    tower.T_cold = T_cold
    tower.T_wb = T_wb
    tower.fill_type = fill_type
    tower.cycles_of_concentration = cycles
    tower.pressure_drop = pressure_drop
    tower.efficiency = fan_efficiency

    # Validation
    if T_cold <= T_wb:
        st.error("⚠️ Cold water temperature must be higher than wet bulb temperature!")
        return
    if T_hot <= T_cold:
        st.error("⚠️ Hot water temperature must be higher than cold water temperature!")
        return

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Design Summary", "📈 Performance Analysis", "💧 Water Balance", "⚡ Power Analysis", "🎯 Optimization"])

    with tab1:
        st.header("Design Summary")
        
        try:
            summary = tower.design_summary()
            if "error" in summary:
                st.error(summary["error"])
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Thermal Performance")
                    thermal_data = {
                        "Parameter": ["Heat Load", "Water Flow", "Air Flow", "Hot Water Temp", "Cold Water Temp", "Wet Bulb Temp", "Approach", "Range"],
                        "Value": [f"{summary['heat_load']} kW", f"{summary['water_flow']:.1f} m³/hr", f"{summary['air_flow']:.1f} m³/s", 
                                f"{summary['T_hot']}°C", f"{summary['T_cold']}°C", f"{summary['T_wb']}°C", 
                                f"{summary['approach']:.1f}°C", f"{summary['range']:.1f}°C"]
                    }
                    st.dataframe(pd.DataFrame(thermal_data), use_container_width=True)
                    
                    st.subheader("Water Balance")
                    water_data = {
                        "Loss Type": ["Evaporation", "Blowdown", "Drift", "Total Makeup"],
                        "Flow Rate (m³/hr)": [f"{summary['evaporation_loss']:.2f}", f"{summary['blowdown_loss']:.2f}", 
                                            f"{summary['drift_loss']:.2f}", f"{summary['makeup_water']:.2f}"]
                    }
                    st.dataframe(pd.DataFrame(water_data), use_container_width=True)

                with col2:
                    st.subheader("Mechanical Design")
                    mechanical_data = {
                        "Parameter": ["Fill Type", "Ka Value", "Fill Height", "Fan Power", "Pressure Drop", "Fan Efficiency"],
                        "Value": [summary['fill_type'], f"{summary['ka_value']}", f"{summary['fill_height']:.2f} m", 
                                f"{summary['fan_power']:.1f} kW", f"{summary['pressure_drop']} Pa", f"{summary['efficiency']*100:.0f}%"]
                    }
                    st.dataframe(pd.DataFrame(mechanical_data), use_container_width=True)
                    
                    st.subheader("Economic Analysis")
                    economic_data = {
                        "Parameter": ["Fill Cost Parameter", "Total Estimated Cost"],
                        "Value": [f"${summary['fill_cost_param']}/m²", f"${summary['total_cost']:,.0f}"]
                    }
                    st.dataframe(pd.DataFrame(economic_data), use_container_width=True)

                # Key metrics in columns
                st.subheader("Key Performance Indicators")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Approach Temperature", f"{summary['approach']:.1f}°C")
                with metric_col2:
                    st.metric("Range Temperature", f"{summary['range']:.1f}°C")
                with metric_col3:
                    st.metric("Total Cost", f"${summary['total_cost']:,.0f}")
                with metric_col4:
                    st.metric("Fan Power", f"{summary['fan_power']:.1f} kW")

        except Exception as e:
            st.error(f"Error generating design summary: {e}")

    with tab2:
        st.header("Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost vs Approach Temperature")
            try:
                fig1 = tower.plot_performance_curve()
                st.pyplot(fig1)
                plt.close()
            except Exception as e:
                st.error(f"Error plotting performance curve: {e}")
        
        with col2:
            st.subheader("Current Operating Point")
            try:
                summary = tower.design_summary()
                if "error" not in summary:
                    efficiency = (summary['range'] / (summary['range'] + summary['approach'])) * 100
                    st.metric("Tower Efficiency", f"{efficiency:.1f}%")
                    st.metric("Approach/Range Ratio", f"{summary['approach']/summary['range']:.2f}")
                    
                    # Performance indicators
                    st.write("**Performance Assessment:**")
                    if summary['approach'] < 4:
                        st.success("✅ Excellent approach temperature")
                    elif summary['approach'] < 6:
                        st.info("ℹ️ Good approach temperature")
                    else:
                        st.warning("⚠️ High approach temperature - consider optimization")
            except Exception as e:
                st.error(f"Error calculating performance metrics: {e}")

    with tab3:
        st.header("Water Balance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Water Losses vs Concentration Cycles")
            try:
                fig2 = tower.plot_water_loss_breakdown()
                st.pyplot(fig2)
                plt.close()
            except Exception as e:
                st.error(f"Error plotting water loss breakdown: {e}")
        
        with col2:
            st.subheader("Water Management Insights")
            try:
                summary = tower.design_summary()
                if "error" not in summary:
                    st.write("**Water Loss Distribution:**")
                    total_loss = summary['evaporation_loss'] + summary['blowdown_loss'] + summary['drift_loss']
                    evap_pct = (summary['evaporation_loss'] / total_loss) * 100
                    blow_pct = (summary['blowdown_loss'] / total_loss) * 100
                    drift_pct = (summary['drift_loss'] / total_loss) * 100
                    
                    st.write(f"• Evaporation: {evap_pct:.1f}%")
                    st.write(f"• Blowdown: {blow_pct:.1f}%")
                    st.write(f"• Drift: {drift_pct:.1f}%")
                    
                    st.write("**Recommendations:**")
                    if summary['cycles'] < 4:
                        st.info("💡 Consider increasing cycles of concentration to reduce blowdown")
                    if summary['drift_loss'] > 0.5:
                        st.warning("⚠️ High drift losses - check drift eliminators")
            except Exception as e:
                st.error(f"Error calculating water insights: {e}")

    with tab4:
        st.header("Power Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fan Power vs Air Flow")
            try:
                fig3 = tower.plot_fan_power_vs_air_flow()
                st.pyplot(fig3)
                plt.close()
            except Exception as e:
                st.error(f"Error plotting fan power analysis: {e}")
        
        with col2:
            st.subheader("Power Consumption Analysis")
            try:
                summary = tower.design_summary()
                if "error" not in summary:
                    annual_power = summary['fan_power'] * 8760  # kWh/year
                    power_cost_rate = st.number_input("Electricity Rate ($/kWh)", min_value=0.05, max_value=0.50, value=0.12, step=0.01)
                    annual_cost = annual_power * power_cost_rate
                    
                    st.metric("Annual Power Consumption", f"{annual_power:,.0f} kWh")
                    st.metric("Annual Power Cost", f"${annual_cost:,.0f}")
                    
                    # Power efficiency metrics
                    power_per_ton = summary['fan_power'] / (summary['heat_load'] / 3.517)  # kW per ton
                    st.metric("Power per Cooling Ton", f"{power_per_ton:.2f} kW/ton")
                    
                    if power_per_ton < 0.02:
                        st.success("✅ Excellent power efficiency")
                    elif power_per_ton < 0.035:
                        st.info("ℹ️ Good power efficiency")
                    else:
                        st.warning("⚠️ Consider optimization for better efficiency")
            except Exception as e:
                st.error(f"Error calculating power analysis: {e}")

    with tab5:
        st.header("Design Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Approach Temperature Optimization")
            if st.button("🎯 Optimize Approach Temperature"):
                with st.spinner("Optimizing..."):
                    try:
                        min_approach = st.number_input("Minimum Approach (°C)", min_value=2.0, max_value=8.0, value=3.0, step=0.5)
                        max_approach = st.number_input("Maximum Approach (°C)", min_value=5.0, max_value=15.0, value=10.0, step=0.5)
                        
                        optimal_approach = tower.optimize_approach(min_approach, max_approach)
                        if optimal_approach:
                            st.success(f"✅ Optimal approach temperature: {optimal_approach}°C")
                            st.success(f"💰 Optimized total cost: ${tower.total_cost:,.0f}")
                            st.info(f"🌡️ Optimized cold water temperature: {tower.T_cold:.1f}°C")
                        else:
                            st.error("❌ Optimization failed - check input parameters")
                    except Exception as e:
                        st.error(f"Optimization error: {e}")
        
        with col2:
            st.subheader("Design Recommendations")
            try:
                summary = tower.design_summary()
                if "error" not in summary:
                    st.write("**Current Design Assessment:**")
                    
                    # Approach temperature assessment
                    if summary['approach'] < 4:
                        st.write("🟢 **Approach**: Excellent (< 4°C)")
                    elif summary['approach'] < 6:
                        st.write("🟡 **Approach**: Good (4-6°C)")
                    else:
                        st.write("🔴 **Approach**: Consider optimization (> 6°C)")
                    
                    # Range assessment
                    if summary['range'] > 8:
                        st.write("🟢 **Range**: Good thermal load distribution")
                    else:
                        st.write("🟡 **Range**: Consider higher range for efficiency")
                    
                    # Fill selection
                    if tower.fill_type == "Structured":
                        st.write("🟢 **Fill**: High-performance structured fill")
                    elif tower.fill_type == "Film":
                        st.write("🟡 **Fill**: Standard film fill - good balance")
                    else:
                        st.write("🟡 **Fill**: Basic splash fill - consider upgrade")
                    
                    st.write("**Optimization Suggestions:**")
                    if summary['total_cost'] > 100000:
                        st.write("💡 High cost - consider lower approach or different fill")
                    if summary['fan_power'] > 50:
                        st.write("💡 High fan power - optimize air flow requirements")
                    if summary['cycles'] < 4:
                        st.write("💡 Increase concentration cycles to reduce water costs")
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

    # Footer
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("""
    This advanced cooling tower design tool provides comprehensive analysis including:
    - **Thermal Design**: Heat load calculations and temperature optimization
    - **Mechanical Design**: Fill selection, fan sizing, and pressure drop analysis  
    - **Water Balance**: Evaporation, blowdown, and drift loss calculations
    - **Economic Analysis**: Cost estimation and optimization
    - **Performance Optimization**: Automated approach temperature optimization
    
    Built with Streamlit for interactive engineering design and analysis.
    """)

if __name__ == "__main__":
    main()
