# main.py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CoolingTowerDesign:
    def __init__(self):
        # Base parameters (SI units)
        self.heat_load = 1000  # kW
        self.T_hot = 40  # °C
        self.T_cold = 30  # °C
        self.T_wb = 25  # Wet-bulb temp (°C)
        self.Cp_water = 4.18  # kJ/kg·°C
        self.air_density = 1.2  # kg/m³
        self.water_density = 997  # kg/m³

        # Advanced parameters
        self.fill_type = "Film"  # Film/Splash/Structured
        self.pressure_drop = 150  # Pa
        self.efficiency = 0.7  # Fan efficiency
        self.cycles_of_concentration = 3  # Default value
        self.drift_loss_rate = 0.002  # Typical drift loss rate (0.2% of water flow)
        
        self.water_flow = None
        self.air_flow = None
        self.fill_height = None
        self.fill_data = None
        self.total_cost = None

    def calculate_water_flow(self):
        delta_T = self.T_hot - self.T_cold
        if delta_T <= 0:
            raise ValueError("Delta T must be positive. Check T_hot and T_cold values.")
        self.water_flow = (self.heat_load / (self.Cp_water * delta_T)) * 3600
        return round(float(np.asarray(self.water_flow).item()), 1)

    def air_flow_requirement(self, L_G_ratio=1.0):
        if self.water_flow is None:
            self.calculate_water_flow()
        water_flow_kg_s = (float(np.asarray(self.water_flow).item()) / 3600) * self.water_density
        self.air_flow = water_flow_kg_s / (L_G_ratio * self.air_density)
        return round(float(np.asarray(self.air_flow).item()), 1)

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
        return round(float(np.asarray(self.fill_height).item()), 2)

    def estimate_costs(self):
        if self.water_flow is None: self.calculate_water_flow()
        if self.fill_data is None: self.select_fill() # Ensure fill_data is set
        if self.fill_height is None: self.calculate_fill_height()
        
        fill_area = float(np.asarray(self.water_flow).item()) / 6
        # Ensure fill_data['cost'] and self.fill_height are scalars for multiplication
        fill_cost = fill_area * self.fill_data["cost"] * float(np.asarray(self.fill_height).item())

        if self.air_flow is None: self.air_flow_requirement()
        fan_cost = 5000 * (float(np.asarray(self.air_flow).item()) ** 0.8)
        basin_cost = 300 * fill_area
        self.total_cost = fill_cost + fan_cost + basin_cost
        return round(float(np.asarray(self.total_cost).item()), 0)

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
            self.calculate_water_flow() # Recalculate with optimal T_cold
            self.air_flow_requirement()
            self.calculate_fill_height()
            self.estimate_costs()
            return optimal_approach
        else:
            self.T_cold = original_T_cold # Restore on failure
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
        self.calculate_water_flow() # Reset state
        self.air_flow_requirement()
        self.calculate_fill_height()
        self.estimate_costs()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(approaches, costs, 'bo-')
        ax.set_xlabel("Approach Temperature (°C)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title("Cost vs. Cooling Tower Approach")
        ax.grid(True)
        return fig # Return figure for Streamlit

    def design_summary(self):
        try:
            self.calculate_water_flow()
            self.air_flow_requirement()
            self.select_fill()
            self.calculate_fill_height()
            total_cost_val = self.estimate_costs() # Ensure cost is calculated and current
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

    def plot_efficiency_vs_fill_height(self):
        fill_heights = np.linspace(1, 5, 10)
        efficiencies = []
        # original_fill_height = self.fill_height # Not strictly needed to restore if T_cold isn't changed based on height
        
        for height_val in fill_heights:
            # Hypothetical: Higher fill allows achieving better (lower) approach
            # This is a simplified model for plotting purposes ONLY.
            hypothetical_approach_factor = 0.2 
            current_ka = self.fill_data['Ka'] if self.fill_data else self.select_fill()['Ka']
            # Simulating that a taller fill reduces approach temperature
            # This requires a more complex model to be accurate.
            # For this plot, assume T_cold changes based on a hypothetical relation to height
            base_approach = self.T_cold - self.T_wb # Current approach
            # Reduce approach proportionally to increase in height beyond a baseline (e.g. 1m)
            # This is very simplified.
            sim_approach = max(1.0, base_approach / (1 + hypothetical_approach_factor * current_ka * (height_val - 1.0)))
            temp_T_cold = self.T_wb + sim_approach
            temp_T_cold = max(self.T_wb + 1, min(temp_T_cold, self.T_hot - 1)) # Ensure valid temps

            current_efficiency = self.calculate_tower_efficiency(self.T_hot, temp_T_cold, self.T_wb)
            efficiencies.append(current_efficiency)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fill_heights, efficiencies, 'go-')
        ax.set_xlabel("Fill Height (m)")
        ax.set_ylabel("Cooling Tower Efficiency (%)")
        ax.set_title("Hypothetical Efficiency vs Fill Height")
        ax.grid(True)
        return fig

    def calculate_tower_efficiency(self, T_hot, T_cold, T_wb):
        if T_hot <= T_cold or T_cold <= T_wb: return 0
        range_temp = T_hot - T_cold
        approach_temp = T_cold - T_wb
        if (range_temp + approach_temp) == 0: return 0
        efficiency = (range_temp / (range_temp + approach_temp)) * 100
        return round(float(np.asarray(efficiency).item()), 2)

    def compare_fan_types(self):
        if self.air_flow is None: self.air_flow_requirement()
        current_fan_power = self.fan_power()
        axial_power = current_fan_power
        axial_cost = 5000 * (float(np.asarray(self.air_flow).item()) ** 0.8)
        
        centrifugal_efficiency_factor = 0.9
        centrifugal_power = axial_power / centrifugal_efficiency_factor
        centrifugal_cost_factor = 1.5
        centrifugal_cost = axial_cost * centrifugal_cost_factor
        
        return {
            "Axial Fan (Baseline)": {"Power (kW)": axial_power, "Estimated Cost ($)": round(axial_cost,0)},
            "Centrifugal Fan (Example)": {"Power (kW)": round(centrifugal_power,1), "Estimated Cost ($)": round(centrifugal_cost,0)}
        }

    def thermal_performance_curve(self):
        original_T_cold = self.T_cold
        min_approach_for_plot = 1.0
        min_range_for_plot = 1.0
        
        # Generate T_cold values ensuring they are within valid bounds
        # T_wb < T_cold < T_hot
        possible_T_cold_values = np.linspace(max(self.T_wb + min_approach_for_plot, self.T_wb + 0.1), 
                                             min(self.T_hot - min_range_for_plot, self.T_hot - 0.1), 
                                             15)
        
        ranges = []
        approaches = []

        for tc_val in possible_T_cold_values:
            if tc_val <= self.T_wb or tc_val >= self.T_hot: continue # Should be caught by linspace bounds

            current_range = self.T_hot - tc_val
            current_approach = tc_val - self.T_wb
            
            # Ensure range and approach are positive
            if current_range > 0 and current_approach > 0:
                ranges.append(current_range)
                approaches.append(current_approach)
            
        self.T_cold = original_T_cold # Restore

        fig, ax = plt.subplots(figsize=(8, 4))
        if ranges and approaches:
            sorted_pairs = sorted(zip(ranges, approaches))
            if sorted_pairs: # check if sorting resulted in non-empty list
                sorted_ranges, sorted_approaches = zip(*sorted_pairs)
                ax.plot(sorted_ranges, sorted_approaches, 'ro-')
            else:
                ax.text(0.5, 0.5, "No valid points for plotting.", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Not enough valid points to plot.", ha='center', va='center')

        ax.set_xlabel("Range (T_hot - T_cold) (°C)")
        ax.set_ylabel("Approach (T_cold - T_wb) (°C)")
        ax.set_title(f"Thermal Performance (T_hot={self.T_hot}°C, T_wb={self.T_wb}°C)")
        ax.grid(True)
        if ranges: ax.invert_xaxis() # Common representation
        return fig
