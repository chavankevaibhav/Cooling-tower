import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import the CoolingTowerDesign class (assuming it's in main.py)
# If main.py is in the same directory, use: from main import CoolingTowerDesign
# For this example, I'll include a simplified version of the class

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
        ax.set_xlabel("Approach Temperature (°C)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title("Cost vs. Cooling Tower Approach")
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

    def calculate_tower_efficiency(self, T_hot, T_cold, T_wb):
        if T_hot <= T_cold or T_cold <= T_wb: return 0
        range_temp = T_hot - T_cold
        approach_temp = T_cold - T_wb
        if (range_temp + approach_temp) == 0: return 0
        efficiency = (range_temp / (range_temp + approach_temp)) * 100
        return round(float(np.asarray(efficiency).item()), 2)

    def thermal_performance_curve(self):
        original_T_cold = self.T_cold
        min_approach_for_plot = 1.0
        min_range_for_plot = 1.0
        
        possible_T_cold_values = np.linspace(max(self.T_wb + min_approach_for_plot, self.T_wb + 0.1), 
                                             min(self.T_hot - min_range_for_plot, self.T_hot - 0.1), 
                                             15)
        
        ranges = []
        approaches = []

        for tc_val in possible_T_cold_values:
            if tc_val <= self.T_wb or tc_val >= self.T_hot: continue

            current_range = self.T_hot - tc_val
            current_approach = tc_val - self.T_wb
            
            if current_range > 0 and current_approach > 0:
                ranges.append(current_range)
                approaches.append(current_approach)
            
        self.T_cold = original_T_cold

        fig, ax = plt.subplots(figsize=(10, 6))
        if ranges and approaches:
            sorted_pairs = sorted(zip(ranges, approaches))
            if sorted_pairs:
                sorted_ranges, sorted_approaches = zip(*sorted_pairs)
                ax.plot(sorted_ranges, sorted_approaches, 'ro-', linewidth=2, markersize=8)
            else:
                ax.text(0.5, 0.5, "No valid points for plotting.", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Not enough valid points to plot.", ha='center', va='center')

        ax.set_xlabel("Range (T_hot - T_cold) (°C)")
        ax.set_ylabel("Approach (T_cold - T_wb) (°C)")
        ax.set_title(f"Thermal Performance (T_hot={self.T_hot}°C, T_wb={self.T_wb}°C)")
        ax.grid(True, alpha=0.3)
        if ranges: ax.invert_xaxis()
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

            return f"""
            ADVANCED COOLING TOWER DESIGN
            =============================
            Thermal Parameters:
            - Heat load:       {self.heat_load} kW
            - Water flow:      {self.water_flow:.1f} m³/hr
            - Air flow:        {self.air_flow:.1f} m³/s
            - Hot Water Temp:  {self.T_hot}°C
            - Cold Water Temp: {self.T_cold}°C
            - Wet Bulb Temp:   {self.T_wb}°C
            - Approach:        {approach_temp:.1f} °C
            - Range:           {range_temp:.1f} °C
            
            Mechanical Design:
            - Fill type:       {self.fill_type} (Ka = {self.fill_data['Ka']})
            - Fill height:     {self.fill_height:.2f} m
            - Fan power:       {fan_pwr:.1f} kW (Estimated)
            - Pressure drop:   {self.pressure_drop} Pa
            - Fan Efficiency:  {self.efficiency*100:.0f}%
            
            Economic Analysis:
            - Fill cost param: ${self.fill_data['cost']}/m²
            - Total cost:      ${total_cost_val:,.0f} (Estimated)
            
            Water Balance Estimates:
            - Evaporation Loss: {self.calculate_evaporation_loss():.2f} m³/hr
            - Blowdown Loss:    {self.calculate_blowdown_loss():.2f} m³/hr (at {self.cycles_of_concentration} cycles)
            - Drift Loss:       {self.calculate_drift_loss():.2f} m³/hr
            - Total Makeup Water: {self.estimate_makeup_water():.2f} m³/hr
            """
        except ValueError as e:
            return f"Error in generating summary: {e}."
        except Exception as e:
            return f"An unexpected error occurred: {e}"


# Streamlit App
def main():
    st.set_page_config(
        page_title="Cooling Tower Design Tool",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏭 Advanced Cooling Tower Design Tool")
    st.markdown("Design and optimize cooling towers with advanced thermal and economic analysis")
    
    # Initialize session state
    if 'tower' not in st.session_state:
        st.session_state.tower = CoolingTowerDesign()
    
    tower = st.session_state.tower
    
    # Sidebar for inputs
    st.sidebar.header("Design Parameters")
    
    # Basic thermal parameters
    st.sidebar.subheader("Thermal Parameters")
    heat_load = st.sidebar.number_input("Heat Load (kW)", min_value=100, max_value=10000, value=1000, step=100)
    T_hot = st.sidebar.number_input("Hot Water Temperature (°C)", min_value=30.0, max_value=80.0, value=45.0, step=0.5)
    T_cold = st.sidebar.number_input("Cold Water Temperature (°C)", min_value=15.0, max_value=50.0, value=32.0, step=0.5)
    T_wb = st.sidebar.number_input("Wet Bulb Temperature (°C)", min_value=10.0, max_value=35.0, value=27.0, step=0.5)
    
    # Advanced parameters
    st.sidebar.subheader("Advanced Parameters")
    fill_type = st.sidebar.selectbox("Fill Type", ["Film", "Splash", "Structured"])
    pressure_drop = st.sidebar.number_input("Pressure Drop (Pa)", min_value=50, max_value=500, value=150, step=10)
    fan_efficiency = st.sidebar.slider("Fan Efficiency", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    cycles_of_concentration = st.sidebar.number_input("Cycles of Concentration", min_value=2.0, max_value=8.0, value=3.0, step=0.5)
    
    # Update tower parameters
    tower.heat_load = heat_load
    tower.T_hot = T_hot
    tower.T_cold = T_cold
    tower.T_wb = T_wb
    tower.fill_type = fill_type
    tower.pressure_drop = pressure_drop
    tower.efficiency = fan_efficiency
    tower.cycles_of_concentration = cycles_of_concentration
    
    # Validation
    if T_cold <= T_wb:
        st.error("❌ Cold water temperature must be higher than wet bulb temperature!")
        return
    if T_hot <= T_cold:
        st.error("❌ Hot water temperature must be higher than cold water temperature!")
        return
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Design Summary", "📈 Performance Analysis", "🔧 Optimization", "💰 Economic Analysis", "🌊 Water Balance"])
    
    with tab1:
        st.header("Design Summary")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                summary = tower.design_summary()
                st.text(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
        
        with col2:
            st.subheader("Key Metrics")
            try:
                tower.calculate_water_flow()
                tower.air_flow_requirement()
                tower.calculate_fill_height()
                tower.estimate_costs()
                
                approach = tower.T_cold - tower.T_wb
                range_temp = tower.T_hot - tower.T_cold
                efficiency = tower.calculate_tower_efficiency(tower.T_hot, tower.T_cold, tower.T_wb)
                
                st.metric("Approach Temperature", f"{approach:.1f} °C")
                st.metric("Range Temperature", f"{range_temp:.1f} °C")
                st.metric("Tower Efficiency", f"{efficiency:.1f} %")
                st.metric("Total Cost", f"${tower.total_cost:,.0f}")
                
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
    
    with tab2:
        st.header("Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost vs Approach Temperature")
            try:
                fig1 = tower.plot_performance_curve()
                st.pyplot(fig1)
                plt.close(fig1)
            except Exception as e:
                st.error(f"Error generating performance curve: {e}")
        
        with col2:
            st.subheader("Thermal Performance")
            try:
                fig2 = tower.thermal_performance_curve()
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.error(f"Error generating thermal curve: {e}")
    
    with tab3:
        st.header("Optimization")
        
        st.subheader("Approach Temperature Optimization")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            min_approach = st.number_input("Minimum Approach (°C)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
            max_approach = st.number_input("Maximum Approach (°C)", min_value=4.0, max_value=15.0, value=10.0, step=0.5)
            
            if st.button("🎯 Optimize Approach", type="primary"):
                with st.spinner("Optimizing..."):
                    try:
                        optimal_approach = tower.optimize_approach(min_approach, max_approach)
                        if optimal_approach:
                            st.success(f"✅ Optimal approach temperature: {optimal_approach:.1f} °C")
                            st.success(f"✅ Optimized cold water temperature: {tower.T_wb + optimal_approach:.1f} °C")
                            st.success(f"✅ Estimated total cost: ${tower.total_cost:,.0f}")
                        else:
                            st.error("❌ Optimization failed. Try different bounds.")
                    except Exception as e:
                        st.error(f"Error during optimization: {e}")
        
        with col2:
            st.subheader("Optimization Results")
            try:
                # Show current vs optimal comparison
                current_cost = tower.estimate_costs()
                current_approach = tower.T_cold - tower.T_wb
                
                optimization_data = {
                    "Parameter": ["Current Approach (°C)", "Current Cost ($)", "Water Flow (m³/hr)", "Air Flow (m³/s)"],
                    "Value": [f"{current_approach:.1f}", f"{current_cost:,.0f}", f"{tower.water_flow:.1f}", f"{tower.air_flow:.1f}"]
                }
                
                df_opt = pd.DataFrame(optimization_data)
                st.dataframe(df_opt, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error displaying optimization results: {e}")
    
    with tab4:
        st.header("Economic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost Breakdown")
            try:
                tower.calculate_water_flow()
                tower.air_flow_requirement()
                tower.select_fill()
                tower.calculate_fill_height()
                
                fill_area = tower.water_flow / 6
                fill_cost = fill_area * tower.fill_data["cost"] * tower.fill_height
                fan_cost = 5000 * (tower.air_flow ** 0.8)
                basin_cost = 300 * fill_area
                
                cost_data = {
                    "Component": ["Fill", "Fan", "Basin", "Total"],
                    "Cost ($)": [f"{fill_cost:,.0f}", f"{fan_cost:,.0f}", f"{basin_cost:,.0f}", f"{fill_cost + fan_cost + basin_cost:,.0f}"]
                }
                
                df_cost = pd.DataFrame(cost_data)
                st.dataframe(df_cost, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating cost breakdown: {e}")
        
        with col2:
            st.subheader("Fill Type Comparison")
            fill_comparison = {
                "Fill Type": ["Splash", "Film", "Structured"],
                "Ka Value": [0.6, 0.8, 1.1],
                "Cost ($/m²)": [80, 120, 200],
                "Efficiency": ["Low", "Medium", "High"]
            }
            
            df_fill = pd.DataFrame(fill_comparison)
            st.dataframe(df_fill, use_container_width=True)
    
    with tab5:
        st.header("Water Balance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Water Loss Components")
            try:
                evap_loss = tower.calculate_evaporation_loss()
                blowdown_loss = tower.calculate_blowdown_loss()
                drift_loss = tower.calculate_drift_loss()
                makeup_water = tower.estimate_makeup_water()
                
                water_data = {
                    "Loss Type": ["Evaporation", "Blowdown", "Drift", "Total Makeup"],
                    "Flow Rate (m³/hr)": [f"{evap_loss:.2f}", f"{blowdown_loss:.2f}", f"{drift_loss:.2f}", f"{makeup_water:.2f}"]
                }
                
                df_water = pd.DataFrame(water_data)
                st.dataframe(df_water, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating water balance: {e}")
        
        with col2:
            st.subheader("Water Balance Chart")
            try:
                evap_loss = tower.calculate_evaporation_loss()
                blowdown_loss = tower.calculate_blowdown_loss()
                drift_loss = tower.calculate_drift_loss()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                losses = [evap_loss, blowdown_loss, drift_loss]
                labels = ['Evaporation', 'Blowdown', 'Drift']
                colors = ['skyblue', 'lightcoral', 'lightgreen']
                
                ax.pie(losses, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Water Loss Distribution')
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error generating water balance chart: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Cooling Tower Design Tool** - Advanced thermal and economic analysis for industrial cooling systems")


if __name__ == "__main__":
    main()
