"""Legacy Montserrat pipeline orchestration (SDSobj-free at this layer).

This module retains the original ``small_sausage`` and ``big_sausage`` entry
points while isolating legacy processing backends. It does *not* claim to
replace them with archive_metrics: those APIs need to be checked separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from obspy import UTCDateTime

LOG = logging.getLogger(__name__)


def _intervals(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(float(v) <= 0 for v in values):
        raise ValueError("sampling_interval must contain positive values")
    return list(values)


def _requested_metrics(paths, net, startt, endt, intervals, ext, invfile, do_metric):
    if do_metric is not None:
        return dict(do_metric)  # An explicit empty dict means 'do nothing'.
    from flovopy.pipeline.check_data_requirements import check_what_to_do
    # The legacy checker may accept a list, so preserve that calling convention.
    interval_arg = intervals[0] if len(intervals) == 1 else intervals
    return dict(check_what_to_do(
        paths, net, startt, endt,
        sampling_interval=interval_arg, ext=ext, invfile=invfile,
    ))


def _years(startt, endt):
    # [start, end): do not include the year at an exclusive Jan 1 end.
    last = endt - 0.000001
    return range(startt.year, last.year + 1)


def small_sausage(
    paths, startt, endt, sampling_interval=60, source=None, invfile=None,
    Q=None, ext="pickle", net=None, do_metric=None,
):
    """Process existing SDS waveforms into legacy SAM/reduced metrics.

    Processing implementations are imported only when required. They remain
    legacy dependencies and should be migrated independently.
    """
    startt, endt = UTCDateTime(startt), UTCDateTime(endt)
    if endt <= startt:
        raise ValueError("endt must be later than startt")
    intervals = _intervals(sampling_interval)
    wanted = _requested_metrics(
        paths, net, startt, endt, intervals, ext, invfile, do_metric,
    )
    if not any(wanted.values()):
        LOG.info("No metrics requested")
        return

    inventory_ok = invfile is not None and Path(invfile).is_file()
    calibrated = ("SDS_DISP", "VSAM", "VSEM", "DSAM", "ER", "DR", "DRS")
    if not inventory_ok and any(wanted.get(k) for k in calibrated):
        raise FileNotFoundError(
            "Requested calibrated/reduced metrics require a valid StationXML "
            f"inventory; received {invfile!r}"
        )

    from flovopy.pipeline.compute_metrics import (
        compute_raw_metrics, compute_SDS_DISP,
        compute_velocity_metrics, compute_displacement_metrics,
    )

    if wanted.get("SDS_DISP"):
        compute_SDS_DISP(paths, startt, endt, invfile)

    for delta in intervals:
        if wanted.get("RSAM"):
            compute_raw_metrics(
                paths, startt, endt, sampling_interval=delta,
                do_RSAM=True, net=net,
            )
        if wanted.get("VSAM") or wanted.get("VSEM"):
            compute_velocity_metrics(
                paths, startt, endt, sampling_interval=delta,
                do_VSAM=bool(wanted.get("VSAM")),
                do_VSEM=bool(wanted.get("VSEM")), net=net, ext=ext,
            )
        if wanted.get("DSAM"):
            compute_displacement_metrics(
                paths, startt, endt, sampling_interval=delta,
                do_DSAM=True, net=net, ext=ext,
            )

    reduced = ("ER", "DR", "DRS")
    if any(wanted.get(k) for k in reduced):
        if source is None:
            raise ValueError("Reduced metrics requested but 'source' is missing")
        from flovopy.pipeline.reduce_metrics import reduce_to_1km
        for year in _years(startt, endt):
            for delta in intervals:
                reduce_to_1km(
                    paths, year,
                    do_ER=bool(wanted.get("ER")),
                    do_DR=bool(wanted.get("DR")),
                    do_DRS=bool(wanted.get("DRS")),
                    sampling_interval=delta, invfile=invfile,
                    source=source, Q=Q, ext=ext,
                )


def big_sausage(
    seisandbdir, paths, startt, endt, sampling_interval=60,
    source=None, invfile=None, Q=None, ext="pickle", dbout=None,
    round_sampling_rate=True, net=None, do_metric=None, MBWHZ_only=False,
):
    """Convert SEISAN to SDS if requested, then run the metrics pipeline."""
    startt, endt = UTCDateTime(startt), UTCDateTime(endt)
    if endt <= startt:
        raise ValueError("endt must be later than startt")
    intervals = _intervals(sampling_interval)
    wanted = _requested_metrics(
        paths, net, startt, endt, intervals, ext, invfile, do_metric,
    )
    if wanted.get("SDS_RAW"):
        from flovopy.pipeline.seisan_to_sds import seisandb2SDS
        seisandb2SDS(
            seisandbdir, paths["SDS_DIR"], startt, endt, net,
            dbout=dbout, round_sampling_rate=round_sampling_rate,
            MBWHZ_only=MBWHZ_only,
        )
    small_sausage(
        paths, startt, endt, sampling_interval=intervals,
        source=source, invfile=invfile, Q=Q, ext=ext,
        net=net, do_metric=wanted,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import small_sausage/big_sausage from a configured driver. "
        "The old hard-coded Montserrat example has been removed."
    )
