# Core logic for Cooling Tower Design

class CoolingTowerDesign:
    def __init__(self, T_hot, T_cold, T_wb, flow_rate):
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.T_wb = T_wb
        self.flow_rate = flow_rate

    def design_summary(self):
        delta_T = self.T_hot - self.T_cold
        effectiveness = delta_T / (self.T_hot - self.T_wb)
        return {
            "Hot Water Temp (°C)": self.T_hot,
            "Cold Water Temp (°C)": self.T_cold,
            "Wet Bulb Temp (°C)": self.T_wb,
            "Flow Rate (m³/h)": self.flow_rate,
            "Temperature Drop (°C)": delta_T,
            "Cooling Effectiveness": round(effectiveness, 3)
        }
