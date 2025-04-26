import matplotlib.pyplot as plt
import numpy as np
# Volcano parameters
dome_x = 0.0       # Volcano summit at x=0
dome_z = 0.0       # Define summit elevation as 0 for simplicity
slope = -0.2        # 20% slope = 0.2 km vertical drop per 1 km horizontal
source_depth_km_true = 3.0  # true source depth

# Station positions (same as in previous simulation)
station_horizontal = np.array([+3.0, +5.5, +8.0, -3.0, -5.5, -8.0])
station_vertical = slope * np.abs(station_horizontal)  # Always positive slope down from summit

# Create figure
fig, ax = plt.subplots(figsize=(10,6))

# Plot volcano slope (linear on both sides)
x_vals = np.linspace(-10, 10, 500)
z_vals = slope * np.abs(x_vals)
ax.plot(x_vals, z_vals, 'k-', label="Volcano Surface (20% Slope)")

# Plot stations
ax.scatter(station_horizontal, station_vertical, color='blue', s=100, edgecolor='black', label="Stations")

# Plot true source location
ax.scatter(0, -source_depth_km_true, color='red', s=150, marker='*', label="True Source Location")

# Label the stations
for i, (x, z) in enumerate(zip(station_horizontal, station_vertical)):
    ax.text(x, z + 0.1, f"STA{i+1}", ha='center', fontsize=9)

# Diagram styling
ax.set_xlim(-10, 10)
ax.set_ylim(-4, 4)
ax.axhline(0, color='black', linestyle='--')
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Horizontal Distance from Dome (km)')
ax.set_ylabel('Elevation relative to Dome Summit (km)')
ax.set_title('Simulation Setup: Stations on Volcano Flanks')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('simulation_setup_volcano_stations.png')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def compute_distance(horizontal_km, vertical_km, depth_km):
    """Compute source-to-station distance in km."""
    return np.sqrt(horizontal_km**2 + (vertical_km + depth_km)**2)

def compute_energy_magnitude_error(R_true, R_assumed):
    """Compute energy magnitude error from true and assumed distances."""
    return np.log10(R_assumed / R_true)

# --- Define the 6 stations on opposite sides of the volcano with 20% slope ---
stations = [
    {"name": "STA1", "horiz_km": +3.0, "vert_km": 0.2 * 3.0},
    {"name": "STA2", "horiz_km": +5.5, "vert_km": 0.2 * 5.5},
    {"name": "STA3", "horiz_km": +8.0, "vert_km": 0.2 * 8.0},
    {"name": "STA4", "horiz_km": -3.0, "vert_km": 0.2 * 3.0},
    {"name": "STA5", "horiz_km": -5.5, "vert_km": 0.2 * 5.5},
    {"name": "STA6", "horiz_km": -8.0, "vert_km": 0.2 * 8.0},
]

source_depth_km_true = 3.0  # True source depth beneath dome

# --- Set up the mislocation grid ---
depth_errors = np.linspace(-3.0, 3.0, 100)         # Depth errors from -3 km to +3 km
epicenter_offsets = np.linspace(-3.0, 3.0, 100)    # Epicenter offsets from -3 km to +3 km

X, Y = np.meshgrid(epicenter_offsets, depth_errors)

# Initialize grid for network-averaged ΔMₑ
delta_Me_network_grid = np.zeros_like(X)

# --- Main loop over mislocation grid ---
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        delta_Me_list = []

        for sta in stations:
            # True distance
            R_true = compute_distance(
                sta['horiz_km'],
                sta['vert_km'],
                source_depth_km_true
            )

            # Assumed distance (mislocated source)
            R_assumed = compute_distance(
                sta['horiz_km'] + X[i, j],       # apply epicenter shift
                sta['vert_km'],                  # station elevation stays fixed
                source_depth_km_true + Y[i, j]    # apply depth shift
            )

            delta_Me = compute_energy_magnitude_error(R_true, R_assumed)
            delta_Me_list.append(delta_Me)

        # Average ΔMₑ across all stations
        delta_Me_network_grid[i, j] = np.mean(delta_Me_list)

# --- Plot the network-averaged ΔMₑ heatmap ---
fig, ax = plt.subplots(figsize=(10,8))

cp = ax.contourf(X, Y, delta_Me_network_grid, levels=20, cmap='coolwarm', extend='both')
cbar = fig.colorbar(cp, ax=ax, label='Network-Averaged Energy Magnitude Error ΔMₑ')

ax.set_xlabel('Epicenter Mislocation (km)')
ax.set_ylabel('Depth Mislocation (km)')
ax.set_title('Network-Averaged Energy Magnitude Error ΔMₑ\n(6 Stations, ±8 km, 20% Slope)')
ax.grid(True, linestyle='--', alpha=0.6)

# Mark perfect location (0,0)
ax.plot(0, 0, 'k+', markersize=12, label='True Source')

# Draw ±2 km mislocation guideline lines
ax.axhline(2, color='black', linestyle='--', alpha=0.5)
ax.axhline(-2, color='black', linestyle='--', alpha=0.5)
ax.axvline(2, color='black', linestyle='--', alpha=0.5)
ax.axvline(-2, color='black', linestyle='--', alpha=0.5)

ax.legend()

plt.tight_layout()
plt.savefig('network_energy_magnitude_error_6stations_slope.png')
plt.show()
