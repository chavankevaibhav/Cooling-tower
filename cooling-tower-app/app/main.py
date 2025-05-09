import streamlit as st
from app.design import CoolingTowerDesign

st.title("Cooling Tower Design App")

T_hot = st.number_input("Hot Water Temperature (°C)", value=40.0)
T_cold = st.number_input("Cold Water Temperature (°C)", value=30.0)
T_wb = st.number_input("Wet Bulb Temperature (°C)", value=25.0)
flow_rate = st.number_input("Water Flow Rate (m³/h)", value=100.0)

if st.button("Calculate Design Summary"):
    design = CoolingTowerDesign(T_hot, T_cold, T_wb, flow_rate)
    summary = design.design_summary()
    st.subheader("Design Summary")
    for k, v in summary.items():
        st.write(f"**{k}:** {v}")
