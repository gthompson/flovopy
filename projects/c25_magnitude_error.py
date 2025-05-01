import numpy as np

def compute_energy_magnitude_error(
    station_horizontal_km,
    station_vertical_km,
    source_depth_km_true,
    source_depth_km_assumed,
    epicenter_offset_km=0.0
):
    """
    Estimate error in energy magnitude caused by source mislocation.

    Parameters:
    - station_horizontal_km: horizontal distance from station to dome (km)
    - station_vertical_km: elevation difference (positive if station is below dome) (km)
    - source_depth_km_true: true source depth beneath dome (km)
    - source_depth_km_assumed: assumed source depth (km)
    - epicenter_offset_km: horizontal mislocation error (km)

    Returns:
    - delta_Me: estimated error in energy magnitude
    """

    # True source-to-station distance
    R_true = np.sqrt(
        (station_horizontal_km)**2 +
        (station_vertical_km + source_depth_km_true)**2
    )

    # Assumed (wrong) source-to-station distance
    R_assumed = np.sqrt(
        (station_horizontal_km + epicenter_offset_km)**2 +
        (station_vertical_km + source_depth_km_assumed)**2
    )

    # Compute magnitude error: log10(R_assumed / R_true)
    delta_Me = np.log10(R_assumed / R_true)

    print(f"True distance: {R_true:.2f} km")
    print(f"Assumed distance: {R_assumed:.2f} km")
    print(f"Estimated magnitude error (ΔMₑ): {delta_Me:.3f}")

    return delta_Me

# Example usage:
if __name__ == "__main__":
    # Example 1: Close station
    compute_energy_magnitude_error(
        station_horizontal_km=3.0,
        station_vertical_km=1.0,
        source_depth_km_true=2.0,
        source_depth_km_assumed=4.0,
        epicenter_offset_km=0.0
    )

    # Example 2: Distant station
    compute_energy_magnitude_error(
        station_horizontal_km=10.0,
        station_vertical_km=1.0,
        source_depth_km_true=2.0,
        source_depth_km_assumed=4.0,
        epicenter_offset_km=0.0
    )

    # Example 3: Horizontal offset only
    compute_energy_magnitude_error(
        station_horizontal_km=3.0,
        station_vertical_km=1.0,
        source_depth_km_true=2.0,
        source_depth_km_assumed=2.0,
        epicenter_offset_km=2.0
    )
