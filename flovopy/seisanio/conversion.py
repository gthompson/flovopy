"""SEISAN WAV-to-SDS conversion using the authoritative EnhancedSDSClient.

The converter does not modify source files. MVO-specific identifier corrections
are opt-in via the existing archive ``fixid`` behavior or a supplied reader.
"""
from __future__ import annotations
from pathlib import Path
import json
from collections import defaultdict
from obspy import Stream, UTCDateTime, read
from flovopy.enhanced.sdsclient import EnhancedSDSClient
from flovopy.seisanio.validation import validate_stream_inventory

def archive_to_sds(
        archive,
        starttime,
        endtime,
        sds_root,
        db=None,
        reader=read,
        fixid: bool = False,
        preprocess: bool = False,
        waveform_transform=None,
        transform_audit=None,
        inventory=None,
        manifest_path=None,
        verify_writes=False,
        preprocess_fn=None,
        preprocess_kwargs=None,
        write_mode: str = "merge",
        write_preprocess: bool = True,
        return_counts: bool = False,
        verbose: bool = False,
        **write_kwargs,
    ):
        """
        Convert a Seisan waveform archive to SDS format.

        This method groups Seisan waveform files by filename UTC day before writing,
        so that each SDS day file is merged/re-written at most once per call
        per day group.

        Parameters
        ----------
        starttime, endtime : UTCDateTime or compatible
            Time range to process.
        sds_root : str or Path
            Output SDS root directory.
        db : str or None
            Seisan database name (defaults to self.db_cont).
        reader : callable
            Function used to read waveform files (default: obspy.read).
        fixid : bool
            If True, apply Seisan/MVO trace-id fixing during read.
        waveform_transform : callable or None
            Optional Stream -> Stream (or in-place returning None) transformation,
            applied after reading and before time clipping. Never enabled by default.
        transform_audit : callable or None
            Called with (source_path, changes) after successful transformation.
            Each change has original/corrected NSLC, start time and sample rate.
        inventory : obspy.Inventory or None
            Optional advisory channel/epoch/response validation.
        manifest_path : str or Path or None
            Append one JSONL record per source file, including write status.
        verify_writes : bool
            Read back SDS outputs and verify expected trace sample values. This
            is slower and is most suitable for small migration batches.
        preprocess : bool
            If True, apply `preprocess_fn` during read.
        preprocess_fn : callable or None
            Optional read-side preprocessing function.
            Signature:
                preprocess_fn(stream, **kwargs) -> stream
        preprocess_kwargs : dict or None
            Optional kwargs for `preprocess_fn`.
        write_mode : str
            SDS write mode passed to EnhancedSDSClient.write_stream():
                - "fail"
                - "overwrite"
                - "merge"   (recommended; default)
        write_preprocess : bool
            If True, apply FLOVOpy pre-write processing inside
            EnhancedSDSClient.write_stream() before writing to SDS.
        return_counts : bool
            If True, return summary statistics.
        verbose : bool
            Print progress messages.
        **write_kwargs
            Additional kwargs passed to EnhancedSDSClient.write_stream().

            Examples:
                merge=True
                merge_strategy="both"
                harmonize_rates=True
                max_sampling_rate=100.0
                fill_value=0.0
                encoding="STEIM2"
                reclen=4096

        Returns
        -------
        dict or None
            If return_counts=True, returns a summary dictionary with keys:
                - files_read
                - files_failed
                - traces_read
                - days_written
                - sds_files_written
                - failed_files
        """


        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        sds_root = Path(sds_root)
        sds_root.mkdir(parents=True, exist_ok=True)

        client = EnhancedSDSClient(str(sds_root))

        if endtime <= starttime:
            raise ValueError("endtime must be after starttime")
        if write_mode not in {"merge", "fail", "overwrite"}:
            raise ValueError(f"Unsupported write_mode: {write_mode}")

        # Process bounded batches rather than collecting a full day of waveforms
        # in RAM. The SDS writer itself splits traces at UTC day boundaries.
        # 'merge' is the safe default for files overlapping output day files.
        files_read = traces_read = days_written = sds_files_written = 0
        failed_files = []
        written_paths = set()
        batch = Stream()
        batch_paths = []
        batch_records = []
        batch_size = int(write_kwargs.pop("batch_files", 32))
        lookback_days = int(write_kwargs.pop("lookback_days", 1))
        if batch_size < 1:
            raise ValueError("batch_files must be >= 1")
        if write_mode == "overwrite":
            raise ValueError(
                "overwrite is unsafe for incremental SEISAN conversion; "
                "use mode='merge' or explicitly rebuild an empty SDS archive"
            )

        def record_manifest(record):
            if manifest_path is None:
                return
            target = Path(manifest_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, sort_keys=True, default=str) + "\n")

        def verify_batch(expected, paths):
            """Verify actual written samples; don't infer success from filenames."""
            from obspy import read as read_mseed
            actual = Stream()
            for output in paths:
                actual += read_mseed(str(output))
            for expected_trace in expected:
                matches = actual.select(network=expected_trace.stats.network,
                                        station=expected_trace.stats.station,
                                        location=expected_trace.stats.location,
                                        channel=expected_trace.stats.channel)
                if not matches:
                    raise ValueError(f"SDS verification: missing {expected_trace.id}")
                # Check every expected sample at its exact UTC timestamp; this
                # detects missing samples and conflicting overlaps.
                for i, value in enumerate(expected_trace.data):
                    when = expected_trace.stats.starttime + i / expected_trace.stats.sampling_rate
                    found = False
                    for candidate in matches:
                        offset = (when - candidate.stats.starttime) * candidate.stats.sampling_rate
                        j = round(offset)
                        if abs(offset - j) < 1e-5 and 0 <= j < candidate.stats.npts:
                            if candidate.data[j] == value:
                                found = True
                                break
                    if not found:
                        raise ValueError(f"SDS verification: sample mismatch {expected_trace.id} at {when}")

        def flush():
            nonlocal batch, batch_paths, batch_records, sds_files_written, days_written
            if not batch:
                return
            try:
                written = client.write_stream(
                    batch, mode=write_mode, preprocess=write_preprocess,
                    verbose=verbose, **write_kwargs,
                ) or []
                if verify_writes:
                    if not written:
                        raise ValueError("SDS writer returned no output paths")
                    verify_batch(batch, written)
                for record in batch_records:
                    record_manifest({**record, "status": "verified" if verify_writes else "written",
                                     "sds_outputs": [str(x) for x in written]})
                for output in written:
                    written_paths.add(str(output))
                sds_files_written += len(written)
                days_written = len({Path(path).name.rsplit(".", 1)[-1] + ":" + Path(path).parts[-3] for path in written_paths})
            except Exception as exc:
                # A failed write must never be reported as a successful conversion.
                failed_files.extend(str(path) for path in batch_paths)
                for record in batch_records:
                    record_manifest({**record, "status": "write_failed", "error": str(exc)})
                if verbose:
                    print(f"[WARN] SDS write failed for {batch_paths}: {exc}")
            finally:
                batch = Stream()
                batch_paths = []
                batch_records = []

        for path in archive.iter_waveform_files(
            starttime, endtime, db=db, lookback_days=lookback_days
        ):
            try:
                st = archive.read_waveform_file(
                    path, reader=reader, fixid=fixid, preprocess=preprocess,
                    preprocess_fn=preprocess_fn,
                    preprocess_kwargs=preprocess_kwargs, verbose=verbose,
                )
                if not st:
                    raise ValueError("No readable traces")
                changes = []
                if waveform_transform is not None:
                    # Transform a copy so callers never mutate their source stream.
                    st = st.copy()
                    before = [(tr.id, str(tr.stats.starttime), float(tr.stats.sampling_rate)) for tr in st]
                    transformed = waveform_transform(st)
                    if transformed is not None:
                        st = transformed
                    if not isinstance(st, Stream):
                        raise TypeError("waveform_transform must return an ObsPy Stream or None")
                    if len(st) != len(before):
                        raise ValueError("waveform_transform changed trace count; provenance cannot be paired")
                    changes = []
                    for old, tr in zip(before, st):
                        new = (tr.id, str(tr.stats.starttime), float(tr.stats.sampling_rate))
                        if old != new:
                            changes.append(dict(original_id=old[0], corrected_id=new[0],
                                                original_starttime=old[1], corrected_starttime=new[1],
                                                original_sampling_rate=old[2], corrected_sampling_rate=new[2]))
                    if transform_audit is not None:
                        transform_audit(str(path), changes)
                # Crop to the requested half-open interval, using actual trace
                # timestamps rather than SEISAN filename-derived start times.
                cropped = Stream()
                for trace in st:
                    if trace.stats.endtime < starttime or trace.stats.starttime >= endtime:
                        continue
                    tr = trace.copy()
                    tr.trim(starttime=starttime, endtime=endtime, nearest_sample=False)
                    # ObsPy trim includes endtime; exclude sample at endtime.
                    if tr.stats.npts and tr.stats.endtime >= endtime:
                        tr.data = tr.data[:-1]
                    if tr.stats.npts:
                        cropped += tr
                if not cropped:
                    continue
                validation = validate_stream_inventory(cropped, inventory) if inventory is not None else []
                batch_records.append({"source_file": str(path), "corrections": changes,
                                      "stationxml_validation": validation,
                                      "traces": [{"trace_id": t.id, "starttime": str(t.stats.starttime),
                                                  "endtime": str(t.stats.endtime), "npts": int(t.stats.npts),
                                                  "sampling_rate": float(t.stats.sampling_rate)}
                                                 for t in cropped]})
                batch += cropped
                batch_paths.append(path)
                files_read += 1
                traces_read += len(cropped)
                if len(batch_paths) >= batch_size:
                    flush()
            except Exception as exc:
                failed_files.append(str(path))
                record_manifest({"source_file": str(path), "status": "read_or_transform_failed",
                                 "error": str(exc)})
                if verbose:
                    print(f"[WARN] Failed to read {path}: {exc}")
        flush()
        if verbose:
            print(f"[DONE] Read {files_read} files, {traces_read} traces; "
                  f"wrote {sds_files_written} SDS outputs; "
                  f"{len(failed_files)} failures")
        if return_counts:
            return dict(files_read=files_read, files_failed=len(failed_files),
                        traces_read=traces_read, days_written=days_written,
                        sds_files_written=sds_files_written,
                        failed_files=failed_files)



def seisan_to_sds(seisan_root, sds_root, starttime, endtime, *, db_cont=None, **kwargs):
    """Convenience entry point for a generic continuous SEISAN WAV database.

    ``starttime``/``endtime`` are a half-open UTC interval. For MVO-specific
    corrections, construct ``MVOSeisanArchive`` and call ``archive_to_sds``.
    """
    from flovopy.seisanio.core.seisanarchive import SeisanArchive
    archive = SeisanArchive(seisan_root, db_cont=db_cont)
    return archive_to_sds(archive, starttime, endtime, sds_root, **kwargs)
