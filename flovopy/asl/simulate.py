# flovopy/asl/simulate.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Inventory
from obspy.geodetics import locations2degrees, degrees2kilometers
from flovopy.processing.sam import VSAM, DSAM
from flovopy.stationmetadata.utils import inventory2traceid
from flovopy.asl.map import topo_map

def synthetic_source_from_grid(grid, sampling_interval: float = 60.0, DR_cm2: float = 100.0, t0: UTCDateTime | None = None, order: str = "C"):
    if t0 is None:
        t0 = UTCDateTime(0)
    lat = grid.gridlat.ravel(order=order).astype(float)
    lon = grid.gridlon.ravel(order=order).astype(float)
    npts = lat.size
    return {"lat": lat, "lon": lon, "DR": np.full(npts, float(DR_cm2)), "t": [t0 + i * sampling_interval for i in range(npts)]}

def simulate_SAM(inv: Inventory, source, units="m/s", assume_surface_waves=False, wave_speed_kms=1.5, peakf=8.0, Q=None, noise_level_percent=0.0, verbose=False):
    sam_class = VSAM if units == "m/s" else DSAM
    if not isinstance(inv, Inventory):
        return None
    seed_ids = inventory2traceid(inv)

    npts = len(source["DR"])
    dataframes = {}
    for sid in seed_ids:
        net, sta, loc, chan = sid.split(".")
        coords = inv.get_coordinates(sid)
        dist_km = degrees2kilometers(locations2degrees(coords["latitude"], coords["longitude"], source["lat"], source["lon"]))
        gsc = sam_class.compute_geometrical_spreading_correction(dist_km, chan, surfaceWaves=assume_surface_waves, wavespeed_kms=wave_speed_kms, peakf=peakf)
        isc = sam_class.compute_inelastic_attenuation_correction(dist_km, peakf, wave_speed_kms, Q)
        times = [UTCDateTime().timestamp + i for i in range(npts)]
        amplitude = source["DR"] / (gsc * isc) * 1e-7
        if noise_level_percent > 0.0:
            amplitude += amplitude * (noise_level_percent / 100.0) * np.random.uniform(0, 1, size=npts)
        dataframes[sid] = pd.DataFrame({"time": times, "mean": amplitude})

    return sam_class(dataframes=dataframes, sampling_interval=1.0, verbose=verbose)

def plot_SAM(samobj, gridobj, K=3, metric="mean", colors=None, seed=None, topo_kw=None, show_map=True):
    rng = np.random.default_rng(seed)
    total_nodes = gridobj.gridlat.size
    K = max(1, min(K, total_nodes))
    chosen_nodes = rng.choice(total_nodes, size=K, replace=False)

    stations = [sid.split(".")[1] for sid in samobj.dataframes]
    st = samobj.to_stream(metric=metric)
    y_matrix = np.vstack([[tr.data[node] for tr in st] for node in chosen_nodes])

    x = np.arange(len(stations))
    if colors is None:
        base = ["yellow", "red", "magenta", "dodgerblue", "limegreen", "orange", "purple"]
        colors = (base * ((K + len(base) - 1) // len(base)))[:K]
    barw = 0.9 / K
    offsets = (np.arange(K) - (K - 1) / 2.0) * barw

    plt.figure()
    for i in range(K):
        plt.bar(x + offsets[i], y_matrix[i, :], width=barw, color=colors[i], edgecolor="black", linewidth=0.5, label=f"Node {chosen_nodes[i]}")
    plt.xticks(x, stations, rotation=45, fontsize=8)
    plt.xlabel("Station"); plt.ylabel(metric); plt.title(f"{metric} by station for {K} source node(s)")
    plt.legend(ncols=min(K, 4), fontsize=8, frameon=False)

    fig_map = None
    if show_map:
        fig_map = topo_map(**topo_kw)
        lonf, latf = gridobj.gridlon.ravel(), gridobj.gridlat.ravel()
        for i, node in enumerate(chosen_nodes):
            fig_map.plot(x=[lonf[node]], y=[latf[node]], style="c0.6c", pen=f"1p,{colors[i]}", fill=colors[i])
        fig_map.show()
    return fig_map, chosen_nodes