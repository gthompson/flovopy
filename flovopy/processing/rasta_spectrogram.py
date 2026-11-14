"""Render a UTC interval from a RASTA site's archived minute PSD and RSAM."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from .spectrograms import SpectrogramResult


def generate(site, volcano, start, end, output, channels=None):
    from matplotlib import dates, pyplot as plt
    from flovopy.plotting.spectrograms import plot_waveform_spectrograms

    site = Path(site)
    if not volcano or Path(volcano).name != volcano or volcano in {".", ".."}:
        raise ValueError("Select a volcano from the archive")
    a, b = (UTCDateTime(str(t).replace("/", "-")).timestamp for t in (start, end))
    if b <= a or a % 60 or b % 60:
        raise ValueError("Use increasing UTC times aligned to whole minutes")
    a,b = int(a),int(b)
    root = site/"history"/"panels"/"python"/volcano/"10m"
    minutes = (b-a)//60
    stride = max(1, math.ceil(minutes/1200))
    bins = math.ceil(minutes/stride)
    sums, counts, grids, profiles, units = {}, {}, {}, {}, {}
    manifests = []
    for stamp in sorted(root.glob("*")):
        t = int(UTCDateTime.strptime(stamp.name,"%Y%m%dT%H%M%SZ").timestamp)
        if t < b and t+600 > a:
            manifest = max(stamp.glob("*/panel.json"), key=lambda p:(p.stat().st_mtime_ns,str(p)), default=None)
            if manifest:
                manifests.append((t, manifest))
    for t, manifest in manifests:
        roster = site/"system"/volcano/f"{t//3600*3600}.json"
        ids = channels or json.loads((roster if roster.exists() else manifest).read_text())["display" if roster.exists() else "channels"]
        for channel in ids:
            if Path(channel).name != channel:
                raise ValueError("Invalid channel ID")
            file = manifest.parent/(channel+".npz")
            if not file.exists():
                continue
            with np.load(file) as z:
                if channel not in grids:
                    grids[channel] = z["hz"]
                    profiles[channel] = str(z["profile"])
                    units[channel] = str(z["units"])
                    sums[channel] = np.zeros((len(grids[channel]),bins))
                    counts[channel] = np.zeros(bins)
                if (not np.array_equal(grids[channel],z["hz"]) or profiles[channel]!=str(z["profile"])
                        or units[channel]!=str(z["units"])):
                    raise ValueError(f"{channel}: incompatible spectral profiles across interval")
                for time, row, weight in zip(z["minute"],z["mean_psd"],z["valid_columns"]):
                    if a<=time<b and weight>0 and np.all(np.isfinite(row)):
                        i = (int(time)-a)//(60*stride)
                        sums[channel][:,i] += row*weight
                        counts[channel][i] += weight
    if not any(v.any() for v in counts.values()):
        raise ValueError("No archived minute spectra for this volcano and interval")
    rsam = {c:{} for c in grids}
    windows = site/"replay"
    # Each chart run is scoped to the volcano and the source policy revision.
    for t in sorted({stamp//3600*3600 for stamp,_ in manifests}):
        roster = site/"system"/volcano/f"{t}.json"
        if not roster.exists():
            continue
        policy = json.loads(roster.read_text())["policy"]
        sources = (windows/policy/volcano).glob(f"charts-*/windows/{t}/complete.json")
        manifest = max(sources,key=lambda p:(p.stat().st_mtime_ns,str(p)),default=None)
        if not manifest:
            continue
        for item in json.loads(manifest.read_text())["metadata"]["outputs"]:
            channel = item["trace_id"]
            if channel not in grids or item["product"]!="RSAM":
                continue
            with (manifest.parent/item["filename"]).open() as f:
                for row in csv.DictReader(f):
                    time = int(float(row["time"]))
                    if a<=time<b and row["mean"] and float(row["coverage"] or 0)>0:
                        rsam[channel][time] = float(row["mean"])
    stream, spectra = Stream(), {}
    for c,freq in grids.items():
        net,sta,loc,cha = c.split(".")
        data = np.full(minutes,np.nan)
        for time,value in rsam[c].items():
            data[(time-a)//60] = value
        stream += Trace(data,header=dict(network=net,station=sta,location=loc,channel=cha,
                                         starttime=UTCDateTime(a+30),delta=60))
        n = counts[c]
        matrix = np.divide(sums[c],n,out=np.full_like(sums[c],np.nan),where=n>0)
        spectra[c] = SpectrogramResult(c,a+(np.arange(bins)+0.5)*stride*60,
            freq,matrix,(n>0).astype(float),units=units[c],metadata={"step_seconds":stride*60})
    fig = plot_waveform_spectrograms(stream,spectra,cmap="inferno",color_scale="robust",
        frequency_limits=(0,min(12,max(f[-1] for f in grids.values()))),
        title=f"{volcano} · archived spectra · {stride} min per column")
    try:
        for (wave,spec),c in zip(fig._flovopy_axes,grids):
            wave.set_ylabel(f"{c}\nRSAM (counts)",rotation=0,ha="right")
            if not rsam[c]:
                wave.text(.01,.6,"No archived RSAM",transform=wave.transAxes)
            spec.set_xlim(UTCDateTime(a).datetime,UTCDateTime(b).datetime)
            spec.xaxis.set_major_formatter(dates.DateFormatter("%Y/%m/%d\n%H:%M"))
            if spec.collections:
                spec.collections[0].colorbar.set_label(f"PSD · dB re 1 ({units[c]})²/Hz")
        fig.savefig(output,format="png",dpi=120)
    finally:
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("site","volcano","start","end","output"):
        p.add_argument(name)
    p.add_argument("--channels",nargs="+",help="Override the hourly display roster with these SEED IDs")
    args=p.parse_args()
    try:
        generate(**vars(args))
    except ValueError as e:
        p.error(str(e))


if __name__=="__main__":
    main()
