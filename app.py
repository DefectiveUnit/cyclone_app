import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate cutsize
def calculate_cutsize(D_vortexFinder, D_cyclone, D_spigot, solids_volume, ReynoldsNumber, D_inlet, L_cylinder, ConeAngle, InclinationAngle, rho_solid, rho_fluid, K_d=1):
    term1 = (D_vortexFinder / D_cyclone) ** 1.093
    term2 = (D_spigot / D_cyclone) ** -1.00
    term3 = ((1 - solids_volume) ** 2 / 10 ** (1.82 * solids_volume)) ** -0.703
    term4 = ReynoldsNumber ** -0.436
    term5 = (D_inlet / D_cyclone) ** -0.936
    term6 = (L_cylinder / D_cyclone) ** 0.187
    term7 = (1 / np.tan(np.radians(ConeAngle / 2))) ** -1.988
    term8 = (np.cos(np.radians(InclinationAngle / 2))) ** -1.034
    term9 = ((rho_solid - rho_fluid) / rho_fluid) ** -0.217
    return K_d * term1 * term2 * term3 * term4 * term5 * term6 * term7 * term8 * term9

# Function to calculate alpha
def calculate_alpha(D_vortexFinder, D_cyclone, TerminalVelocity, Rmax, InclinationAngle, solids_volume, D_spigot, rho_solid, rho_pulp, viscosity_slurry, viscosity_fluid, ConeAngle, L_cylinder, K_alpha=1):
    term1 = (D_vortexFinder / D_cyclone) ** 0.27
    term2 = (TerminalVelocity / (9.81 * Rmax)) ** 0.016
    term3 = (np.cos(np.radians(InclinationAngle / 2))) ** 0.868
    term4 = ((1 - solids_volume) ** 2 / 10 ** (1.82 * solids_volume)) ** 0.72
    term5 = (D_spigot / D_cyclone) ** 0.567
    term6 = ((rho_solid - rho_pulp) / rho_solid) ** 1.837
    term7 = (viscosity_slurry / viscosity_fluid) ** 0.127
    term8 = (1 / np.tan(np.radians(ConeAngle / 2))) ** 0.182
    term9 = (L_cylinder / D_cyclone) ** 0.2
    return K_alpha * term1 * term2 * term3 * term4 / (term5 * term6 * term7 * term8 * term9)

# Function to calculate Underflow Probability
def calculate_underflow_probability(alpha, x):
    return (np.exp(alpha * x) - 1) / (np.exp(alpha * x) + np.exp(alpha) - 2) * 100

# Helper function to map slider values to a log scale with unique key
def power_of_10_slider(label, default_value, key):
    slider_val = st.slider(label, -3, 3, 0, key=key)  # Unique key
    return default_value * 10 ** slider_val

# Reset functionality
def reset_defaults():
    default_values = {
        'D_vortexFinder': 0.5, 'D_cyclone': 1.0, 'D_spigot': 0.5, 'D_inlet': 0.5,
        'L_cylinder': 2.0, 'ConeAngle': 45, 'InclinationAngle': 45, 'solids_volume': 0.5,  # Changed to 0.5
        'ReynoldsNumber': 5000, 'rho_solid': 2500, 'rho_fluid': 1000, 'rho_pulp': 1000,
        'viscosity_slurry': 0.005, 'viscosity_fluid': 0.005, 'TerminalVelocity': 2.0, 'Rmax': 2.0
    }
    for key, default_value in default_values.items():
        st.session_state[key] = default_value

# Streamlit App
st.title("Cyclone Separation Model")

# --- Move the Reset Button to the Top ---
#if st.button("Reset to Defaults"):
#    reset_defaults()
#    st.rerun()

# --- Initialize the default values only once ---
if 'cutsize_0' not in st.session_state:
    # Set default solid density for the initial/default curve
    default_rho_solid = 2500
    
    # Calculate the default cutsize and alpha
    cutsize_0 = calculate_cutsize(0.5, 1.0, 0.5, 0.5, 5000, 0.5, 2.0, 45, 45, default_rho_solid, 1000)
    alpha_0 = calculate_alpha(0.5, 1.0, 2.0, 2.0, 45, 0.5, 0.5, default_rho_solid, 1000, 0.005, 0.005, 45, 2.0)
    
    st.session_state['cutsize_0'] = cutsize_0
    st.session_state['alpha_0'] = alpha_0

# Retrieve cutsize_0 and alpha_0 from session state
cutsize_0 = st.session_state['cutsize_0']
alpha_0 = st.session_state['alpha_0']

# --- Initialize other default values ---
if 'D_vortexFinder' not in st.session_state:
    # Define default values for all parameters
    reset_defaults()

# Assign default values to variables and store them back in session state as sliders update
D_vortexFinder = st.session_state['D_vortexFinder']
D_cyclone = st.session_state['D_cyclone']
D_spigot = st.session_state['D_spigot']
D_inlet = st.session_state['D_inlet']
L_cylinder = st.session_state['L_cylinder']
ConeAngle = st.session_state['ConeAngle']
InclinationAngle = st.session_state['InclinationAngle']
solids_volume = st.session_state['solids_volume']
ReynoldsNumber = st.session_state['ReynoldsNumber']
rho_solid = st.session_state['rho_solid']
rho_fluid = st.session_state['rho_fluid']
rho_pulp = st.session_state['rho_pulp']
viscosity_slurry = st.session_state['viscosity_slurry']
viscosity_fluid = st.session_state['viscosity_fluid']
TerminalVelocity = st.session_state['TerminalVelocity']
Rmax = st.session_state['Rmax']

# --- Dropdown to select parameter to vary ---
parameter_to_vary = st.selectbox(
    "Select a parameter to vary:",
    [
        "Vortex Finder Diameter", "Cyclone Diameter", "Spigot Diameter", 
        "Inlet Diameter", "Cylinder Length", "Cone Angle", "Inclination Angle",
        "Solids Volume", "Reynolds Number", "Solid Density", "Fluid Density",
        "Pulp Density", "Slurry Viscosity", "Fluid Viscosity", "Terminal Velocity", "Rmax"
    ]
)

# --- Display single slider based on the selected parameter ---
if parameter_to_vary == "Vortex Finder Diameter":
    st.session_state['D_vortexFinder'] = power_of_10_slider('Vortex Finder Diameter (D_vortexFinder)', D_vortexFinder, key="vortexFinder")
elif parameter_to_vary == "Cyclone Diameter":
    st.session_state['D_cyclone'] = power_of_10_slider('Cyclone Diameter (D_cyclone)', D_cyclone, key="cycloneDiameter")
elif parameter_to_vary == "Spigot Diameter":
    st.session_state['D_spigot'] = power_of_10_slider('Spigot Diameter (D_spigot)', D_spigot, key="spigotDiameter")
elif parameter_to_vary == "Inlet Diameter":
    st.session_state['D_inlet'] = power_of_10_slider('Inlet Diameter (D_inlet)', D_inlet, key="inletDiameter")
elif parameter_to_vary == "Cylinder Length":
    st.session_state['L_cylinder'] = power_of_10_slider('Cylinder Length (L_cylinder)', L_cylinder, key="cylinderLength")
elif parameter_to_vary == "Cone Angle":
    st.session_state['ConeAngle'] = st.slider('Cone Angle (degrees)', 1, 90, ConeAngle, step=1, key="coneAngle")
elif parameter_to_vary == "Inclination Angle":
    st.session_state['InclinationAngle'] = st.slider('Inclination Angle (degrees)', 0, 90, InclinationAngle, step=1, key="inclinationAngle")
elif parameter_to_vary == "Solids Volume":
    st.session_state['solids_volume'] = st.slider('Solids Volume (%)', 0.0, 1.0, solids_volume, step=0.01, key="solidsVolume")
elif parameter_to_vary == "Reynolds Number":
    st.session_state['ReynoldsNumber'] = power_of_10_slider('Reynolds Number', ReynoldsNumber, key="reynoldsNumber")
elif parameter_to_vary == "Solid Density":
    # Solid Density reverted to logarithmic scale (-3 to 3)
    st.session_state['rho_solid'] = power_of_10_slider('Solid Density (ρ_solid)', rho_solid, key="solidDensity")
elif parameter_to_vary == "Fluid Density":
    st.session_state['rho_fluid'] = power_of_10_slider('Fluid Density (ρ_fluid)', rho_fluid, key="fluidDensity")
elif parameter_to_vary == "Pulp Density":
    st.session_state['rho_pulp'] = power_of_10_slider('Pulp Density (ρ_pulp)', rho_pulp, key="pulpDensity")
elif parameter_to_vary == "Slurry Viscosity":
    st.session_state['viscosity_slurry'] = power_of_10_slider('Slurry Viscosity (Pa.s)', viscosity_slurry, key="slurryViscosity")
elif parameter_to_vary == "Fluid Viscosity":
    st.session_state['viscosity_fluid'] = power_of_10_slider('Fluid Viscosity (Pa.s)', viscosity_fluid, key="fluidViscosity")
elif parameter_to_vary == "Terminal Velocity":
    st.session_state['TerminalVelocity'] = power_of_10_slider('Terminal Velocity (m/s)', TerminalVelocity, key="terminalVelocity")
elif parameter_to_vary == "Rmax":
    st.session_state['Rmax'] = power_of_10_slider('Rmax (m)', Rmax, key="rmax")

# --- First Chart: Separation Efficiency ---
log_x_values = np.logspace(-2, 2, 100)  # log scale from 10^-2 to 10^2
x_values_0 = log_x_values  # x = d / d50_0 (based on initial cutsize)
y_values_0 = calculate_underflow_probability(alpha_0, x_values_0)

# Recalculate current cutsize and alpha based on the sliders
cutsize_current = calculate_cutsize(
    st.session_state['D_vortexFinder'], st.session_state['D_cyclone'], 
    st.session_state['D_spigot'], st.session_state['solids_volume'], 
    st.session_state['ReynoldsNumber'], st.session_state['D_inlet'], 
    st.session_state['L_cylinder'], st.session_state['ConeAngle'], 
    st.session_state['InclinationAngle'], st.session_state['rho_solid'], 
    st.session_state['rho_fluid']
)

alpha_current = calculate_alpha(
    st.session_state['D_vortexFinder'], st.session_state['D_cyclone'], 
    st.session_state['TerminalVelocity'], st.session_state['Rmax'], 
    st.session_state['InclinationAngle'], st.session_state['solids_volume'], 
    st.session_state['D_spigot'], st.session_state['rho_solid'], 
    st.session_state['rho_pulp'], st.session_state['viscosity_slurry'], 
    st.session_state['viscosity_fluid'], st.session_state['ConeAngle'], 
    st.session_state['L_cylinder']
)

# Calculate Y values for the current curve
x_values_current = log_x_values * (cutsize_current / cutsize_0)  # Shift the x-axis based on new cutsize
y_values_current = calculate_underflow_probability(alpha_current, x_values_current)

# Plot the first curve
fig1, ax1 = plt.subplots()
ax1.plot(x_values_0, y_values_0, color='grey', linestyle='--', label='Default (initial)', alpha=0.5)
ax1.plot(x_values_0, y_values_current, label='Current')
ax1.set_xscale('log')
ax1.set_xlabel('d / d50_0 (log scale)')
ax1.set_ylabel('Underflow Probability (%)')
ax1.set_title('Separation Efficiency vs d/d50_0')
ax1.legend()
st.pyplot(fig1)

# --- Second Chart: Cutsize and Alpha ---
fig2, ax2 = plt.subplots()
ax2.scatter([cutsize_0, cutsize_current], [alpha_0, alpha_current], color=['grey', 'blue'])
ax2.set_xlabel('Cut Size (d50)')
ax2.set_ylabel('Alpha')
ax2.set_title('Initial vs Current Cut Size and Alpha')
st.pyplot(fig2)

# --- Additional Parameters ---
with st.expander("Additional Sliders for Other Parameters"):
    st.session_state['D_vortexFinder'] = power_of_10_slider('Vortex Finder Diameter (D_vortexFinder)', D_vortexFinder, key="additional_vortexFinder")
    st.session_state['D_cyclone'] = power_of_10_slider('Cyclone Diameter (D_cyclone)', D_cyclone, key="additional_cycloneDiameter")
    st.session_state['D_spigot'] = power_of_10_slider('Spigot Diameter (D_spigot)', D_spigot, key="additional_spigotDiameter")
    st.session_state['D_inlet'] = power_of_10_slider('Inlet Diameter (D_inlet)', D_inlet, key="additional_inletDiameter")
    st.session_state['L_cylinder'] = power_of_10_slider('Cylinder Length (L_cylinder)', L_cylinder, key="additional_cylinderLength")
    st.session_state['ConeAngle'] = st.slider('Cone Angle (degrees)', 1, 90, ConeAngle, step=1, key="additional_coneAngle")
    st.session_state['InclinationAngle'] = st.slider('Inclination Angle (degrees)', 0, 90, InclinationAngle, step=1, key="additional_inclinationAngle")
    st.session_state['solids_volume'] = st.slider('Solids Volume (%)', 0.0, 1.0, solids_volume, step=0.01, key="additional_solidsVolume")
    st.session_state['ReynoldsNumber'] = power_of_10_slider('Reynolds Number', ReynoldsNumber, key="additional_reynoldsNumber")
    st.session_state['rho_solid'] = power_of_10_slider('Solid Density (ρ_solid)', rho_solid, key="additional_solidDensity")
    st.session_state['rho_fluid'] = power_of_10_slider('Fluid Density (ρ_fluid)', rho_fluid, key="additional_fluidDensity")
    st.session_state['rho_pulp'] = power_of_10_slider('Pulp Density (ρ_pulp)', rho_pulp, key="additional_pulpDensity")
    st.session_state['viscosity_slurry'] = power_of_10_slider('Slurry Viscosity (Pa.s)', viscosity_slurry, key="additional_slurryViscosity")
    st.session_state['viscosity_fluid'] = power_of_10_slider('Fluid Viscosity (Pa.s)', viscosity_fluid, key="additional_fluidViscosity")
    st.session_state['TerminalVelocity'] = power_of_10_slider('Terminal Velocity (m/s)', TerminalVelocity, key="additional_terminalVelocity")
    st.session_state['Rmax'] = power_of_10_slider('Rmax (m)', Rmax, key="additional_rmax")
