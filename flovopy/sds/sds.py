"""
flovopy.sds.sds

Modern SDS archive manager built on EnhancedSDSClient.

Design goals
------------
- Preserve existing SDSobj API
- Centralize SDS mechanics in EnhancedSDSClient
- Keep plotting, grouping, metadata logic here
- Enable gradual deprecation without breaking workflows
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

from obspy import Stream, Trace, UTCDateTime

# --- flovopy ---
from flovopy.enhanced.sdsclient import EnhancedSDSClient
from flovopy.core.miniseed_io import smart_merge, downsample_stream_to_common_rate
from flovopy.core.trace_utils import remove_empty_traces
from flovopy.stationmetadata.utils import build_dataframe_from_table
from flovopy.enhanced.stream import EnhancedStream


# =============================================================================
# SDSobj (compatibility-preserving façade)
# =============================================================================

class SDSobj:
    """
    High-level SDS archive manager.

    This class preserves historical flovopy SDSobj semantics while delegating
    low-level SDS mechanics to EnhancedSDSClient.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        basedir: str | Path,
        sds_type: str = "D",
        format: str = "MSEED",
        streamobj: Optional[Stream] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        self.basedir = Path(basedir).expanduser().resolve()
        self.client = EnhancedSDSClient(self.basedir, sds_type=sds_type, format=format)
        self.stream: Stream = streamobj or Stream() or EnhancedStream()
        self.metadata = metadata

    # ------------------------------------------------------------------
    # Reading (preserves legacy behavior)
    # ------------------------------------------------------------------

    def read(
        self,
        startt: UTCDateTime,
        endt: UTCDateTime,
        trace_ids: Optional[Sequence[str]] = None,
        skip_low_rate_channels: bool = True,
        speed: int = 2,
        verbose: bool = False,
        progress: bool = False,
        min_sampling_rate: float = 50.0,
        max_sampling_rate: float = 250.0,
        merge_strategy: str = "obspy",
        final_merge: str = "always",
    ) -> int:
        """
        Read SDS data into self.stream.

        Preserves legacy semantics:
        - optional trace_id discovery
        - downsampling
        - smart_merge
        - trimming
        """

        if trace_ids is None:
            trace_ids = self.client.iter_trace_ids(startt, endt, skip_low_rate=skip_low_rate_channels)

        st = Stream()
        iterator = tqdm(trace_ids, desc="Reading SDS") if progress else trace_ids

        for tid in iterator:
            net, sta, loc, chan = tid.split(".")

            if skip_low_rate_channels and chan.startswith("L"):
                continue

            try:
                traces = self.client.get_waveforms(
                    net, sta, loc, chan,
                    startt, endt,
                    merge=0
                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to read {tid}: {e}")
                continue

            # Cull low-rate traces
            if min_sampling_rate is not None:
                for tr in list(traces):
                    if tr.stats.sampling_rate < min_sampling_rate:
                        traces.remove(tr)

            # Downsample + merge per ID
            ds = downsample_stream_to_common_rate(
                traces,
                max_sampling_rate=max_sampling_rate
            )
            smart_merge(ds, strategy=merge_strategy)
            st += ds

        remove_empty_traces(st, inplace=True)

        if len(st):
            st.trim(startt, endt)

            if final_merge == "always":
                smart_merge(st, strategy=merge_strategy)

            for tr in st:
                tr.stats.processing = (tr.stats.processing or []) + [
                    "flovopy:smart_merge_v1"
                ]

        self.stream = st
        gc.collect()
        return 0 if len(st) else 1

    # ------------------------------------------------------------------
    # Writing (delegated)
    # ------------------------------------------------------------------

    def write(self, overwrite: bool = False) -> List[Path]:
        """
        Write current stream to SDS archive.
        """
        return self.client.write_stream(self.stream, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Availability (modern backend, legacy interface)
    # ------------------------------------------------------------------

    def get_availability(
        self,
        startday: UTCDateTime,
        endday: UTCDateTime,
        trace_ids: Optional[Sequence[str]] = None,
        skip_low_rate_channels: bool = True,
        progress: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Return (availability_df, trace_ids).

        availability_df:
            columns = ['date'] + trace_ids
            values in [0, 1]
        """

        if trace_ids is None:
            trace_ids = self.client.iter_trace_ids(
                startday, endday, skip_low_rate=skip_low_rate_channels
            )

        days = []
        t = startday
        while t < endday:
            days.append(t)
            t += 86400

        records = []
        iterator = tqdm(days, desc="Availability") if progress else days

        for day in iterator:
            for tid in trace_ids:
                net, sta, loc, chan = tid.split(".")
                frac = self.client.availability_fraction_for_day(
                    net, sta, loc, chan, day
                )
                records.append((day.date, tid, frac))

        df = pd.DataFrame(records, columns=["date", "trace_id", "value"])
        wide = df.pivot_table(
            index="date",
            columns="trace_id",
            values="value",
            aggfunc="mean"
        ).reset_index()

        return wide, list(trace_ids)

    # ------------------------------------------------------------------
    # Plotting (unchanged API)
    # ------------------------------------------------------------------

    def plot_availability(
        self,
        availabilityDF: Optional[pd.DataFrame] = None,
        outfile: Optional[str] = None,
        figsize=(12, 9),
        fontsize=10,
        cmap="gray_r",
        show_counts=True,
        count_min_fraction=0.0,
    ):
        """
        Plot availability heatmap.
        """

        if availabilityDF is None or availabilityDF.empty:
            print("No availability data to plot.")
            return

        A_df = availabilityDF.set_index("date").T
        dates = pd.to_datetime(A_df.columns)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            A_df.to_numpy(),
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(np.arange(len(dates)))
        ax.set_xticklabels(dates.strftime("%Y-%m-%d"), rotation=90, fontsize=fontsize)
        ax.set_yticks(np.arange(len(A_df.index)))
        ax.set_yticklabels(A_df.index, fontsize=fontsize)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        if outfile:
            fig.savefig(outfile, dpi=300)
            print(f"Saved availability plot to {outfile}")

        return fig

    # ------------------------------------------------------------------
    # Metadata helpers (unchanged)
    # ------------------------------------------------------------------

    def load_metadata_from_excel(self, excel_path, sheet_name=0):
        self.metadata = build_dataframe_from_table(excel_path, sheet_name)

    def load_metadata_from_stationxml(self, xml_path):
        from obspy.core.inventory import Inventory
        inv = Inventory.read(xml_path, format="stationxml")
        rows = []

        for net in inv:
            for sta in net:
                for chan in sta:
                    rows.append({
                        "network": net.code,
                        "station": sta.code,
                        "location": chan.location_code,
                        "channel": chan.code,
                        "latitude": sta.latitude,
                        "longitude": sta.longitude,
                    })

        self.metadata = pd.DataFrame(rows)

    def build_file_list(
        self,
        parameters=None,
        starttime=None,
        endtime=None,
        return_failed_list_too=False,
    ):
        """
        Build a list of SDS MiniSEED files matching criteria.

        This method is retained for backward compatibility but now
        delegates file discovery to `discover_files()`.

        Parameters
        ----------
        parameters : dict, optional
            Filtering parameters:
            {
                'network': [...],
                'station': [...],
                'location': [...],
                'channel': [...]
            }
        starttime, endtime : UTCDateTime, optional
            Time window filter.
        return_failed_list_too : bool
            If True, return (file_list, failed_list)

        Returns
        -------
        list or (list, list)
            file_list [, failed_list]
        """

        from flovopy.sds.sds_utils import discover_files
        from flovopy.sds.sds_utils import is_valid_sds_filename

        file_list = []
        failed_list = []

        discovered = discover_files(
            sds_root=self.sds_root,
            use_sds=True,
            filterdict=parameters,
            starttime=starttime,
            endtime=endtime,
        )

        for f in discovered:
            fname = os.path.basename(f)
            if is_valid_sds_filename(fname):
                file_list.append(f)
            else:
                failed_list.append(f)

        if return_failed_list_too:
            return file_list, failed_list
        else:
            return file_list
