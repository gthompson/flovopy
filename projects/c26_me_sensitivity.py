import numpy as np
import matplotlib.pyplot as plt

def compute_distance(horizontal_km, vertical_km, depth_km):
    return np.sqrt(horizontal_km**2 + (vertical_km + depth_km)**2)

def compute_energy_magnitude_error(R_true, R_assumed):
    return np.log10(R_assumed / R_true)

# Set up a typical station
station_horizontal_km = 5.0   # 5 km horizontally from dome
station_vertical_km = 1.0     # 1 km below dome summit
source_depth_km_true = 3.0    # True depth is 3 km below dome

# True distance
R_true = compute_distance(station_horizontal_km, station_vertical_km, source_depth_km_true)

# Set up a grid of depth errors and epicenter offsets
depth_errors = np.linspace(-3.0, 3.0, 100)         # Depth errors from -3 km to +3 km
epicenter_offsets = np.linspace(-3.0, 3.0, 100)    # Horizontal epicenter offsets from -3 km to +3 km

# Create a meshgrid for plotting
X, Y = np.meshgrid(epicenter_offsets, depth_errors)

# Compute ΔMₑ for each point in the grid
delta_Me_grid = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        R_assumed = compute_distance(
            station_horizontal_km + X[i, j],
            station_vertical_km,
            source_depth_km_true + Y[i, j]
        )
        delta_Me_grid[i, j] = compute_energy_magnitude_error(R_true, R_assumed)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10,8))

cp = ax.contourf(X, Y, delta_Me_grid, levels=20, cmap='coolwarm', extend='both')
cbar = fig.colorbar(cp, ax=ax, label='Energy Magnitude Error ΔMₑ')

ax.set_xlabel('Epicenter Mislocation (km)')
ax.set_ylabel('Depth Mislocation (km)')
ax.set_title('Energy Magnitude Error ΔMₑ vs Source Mislocation (Typical Station)')
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('energy_magnitude_error_contour_typical_station.png')
plt.show()
