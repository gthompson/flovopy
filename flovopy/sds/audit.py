import os
import pandas as pd
from obspy import read
from flovopy.core.mvo import fix_trace_mvo_wrapper
from flovopy.sds.sds import is_valid_sds_filename
import atexit

def audit_trace_ids_in_sds(sds_root, output_csv="trace_id_audit.csv", verbose=True):
    """
    Audit legacy vs fixed trace IDs in an SDS archive using head-only MiniSEED reads.
    Periodically writes partial results to CSV to protect against crashes.
    """
    audit_records = []
    partial_csv = output_csv + ".partial"

    def autosave():
        if audit_records:
            df = pd.DataFrame(audit_records)
            df.to_csv(partial_csv, index=False)
            print(f"🛟 Autosaved audit results to {partial_csv} (may be partial)")

    atexit.register(autosave)

    print(f'📁 Auditing {sds_root}')
    file_counter = 0

    for root, dirs, files in os.walk(sds_root):
        dirs.sort()
        files.sort()

        for fname in files:
            full_path = os.path.join(root, fname)
            if not is_valid_sds_filename(os.path.basename(full_path)):
                if verbose:
                    print(f"⛔ {full_path} not a valid SDS filename")
                continue

            try:
                if verbose:
                    print(f"📄 Reading {full_path}")
                st = read(full_path, headonly=True, format='MSEED')
            except Exception as e:
                if verbose:
                    print(f"⚠️ Failed to read {full_path}: {e}")
                continue

            for tr in st:
                try:
                    original_id = tr.id
                    samplerate = tr.stats.sampling_rate
                    fix_trace_mvo_wrapper(tr)
                    fixed_id = tr.id

                    audit_records.append({
                        "filepath": full_path,
                        "original_id": original_id,
                        "sampling_rate": samplerate,
                        "fixed_id": fixed_id
                    })

                    file_counter += 1
                    if file_counter % 100 == 0:
                        pd.DataFrame(audit_records).to_csv(partial_csv, index=False)
                        if verbose:
                            print(f"💾 Saved {file_counter} records to {partial_csv}")

                except Exception as e:
                    if verbose:
                        print(f"❌ Failed to fix trace ID in {full_path}: {e}")
                    continue

    # Final save
    df = pd.DataFrame(audit_records)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Audit complete. Saved to {output_csv}. Total traces processed: {len(df)}")

    # Remove partial CSV if final is written
    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    return df


def export_grouped_summary(df, output_csv="trace_id_mapping_summary.csv"):
    """
    Export a grouped summary showing how each original_id maps to one or more fixed_id values.

    Parameters
    ----------
    df : pandas.DataFrame
        The audit DataFrame with columns: original_id, fixed_id, sampling_rate
    output_csv : str
        Path to the CSV file to write.
    """
    summary = (
        df.groupby("original_id")
        .agg(
            fixed_ids=("fixed_id", lambda x: sorted(set(x))),
            num_fixed_ids=("fixed_id", lambda x: len(set(x))),
            sampling_rates=("sampling_rate", lambda x: sorted(set(x)))
        )
        .reset_index()
    )

    # Convert list columns to strings
    summary["fixed_ids"] = summary["fixed_ids"].apply(lambda x: ", ".join(x))
    summary["sampling_rates"] = summary["sampling_rates"].apply(lambda x: ", ".join(str(s) for s in x))

    summary.to_csv(output_csv, index=False)
    print(f"📄 Grouped summary saved to {output_csv}. Total unique original_ids: {len(summary)}")



def export_per_file_metadata(df, output_csv="per_file_trace_metadata.csv"):
    """
    Export detailed per-file trace metadata from the audit DataFrame.
    Safely saves partial progress on crash or interruption.
    """
    records = []
    partial_csv = output_csv + ".partial"

    def autosave():
        if records:
            pd.DataFrame(records).to_csv(partial_csv, index=False)
            print(f"🛟 Autosaved per-file metadata to {partial_csv} (may be partial)")
    atexit.register(autosave)

    for i, row in df.iterrows():
        try:
            st = read(row["filepath"], format="MSEED")
            for tr in st:
                orig_id = tr.id
                orig_sr = tr.stats.sampling_rate
                orig_start = tr.stats.starttime.datetime
                orig_end = tr.stats.endtime.datetime
                npts = tr.stats.npts
                fix_trace_mvo_wrapper(tr)
                fixed_id = tr.id
                fixed_sr = tr.stats.sampling_rate
                records.append({
                    "filepath": row["filepath"],
                    "original_id": orig_id,
                    "starttime": orig_start,
                    "endtime": orig_end,
                    "npts": npts,
                    "sampling_rate": orig_sr,
                    "fixed_id": fixed_id,
                    "fixed_sampling_rate": fixed_sr
                })

            # Periodic flush every 100 files
            if i % 100 == 0 and records:
                pd.DataFrame(records).to_csv(partial_csv, index=False)
                print(f"💾 Wrote partial per-file metadata to {partial_csv} after {i} files")

        except Exception as e:
            print(f"⚠️ Failed to process {row['filepath']}: {e}")
            continue

    metadata_df = pd.DataFrame(records)
    metadata_df.sort_values(by=["fixed_id", "starttime"], inplace=True)
    metadata_df.to_csv(output_csv, index=False)
    print(f"📄 Per-file metadata saved to {output_csv}. Total rows: {len(metadata_df)}")

    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    return metadata_df


def compute_contiguous_ranges(df, output_csv="trace_segment_ranges.csv", gap_threshold=1.0, rate_tolerance=1.0):
    """
    Compute contiguous time ranges for each trace ID, with autosave protection.
    """
    from datetime import datetime
    records = []
    partial_csv = output_csv + ".partial"

    def autosave():
        if records:
            pd.DataFrame(records).to_csv(partial_csv, index=False)
            print(f"🛟 Autosaved contiguous ranges to {partial_csv} (may be partial)")
    atexit.register(autosave)

    for i, (trace_id, group) in enumerate(df.groupby("fixed_id")):
        group = group.sort_values(by="starttime")
        segment_start = group.iloc[0]["starttime"]
        segment_end = group.iloc[0]["endtime"]
        segment_sr = group.iloc[0]["fixed_sampling_rate"]
        total_npts = group.iloc[0]["npts"]

        for j in range(1, len(group)):
            row = group.iloc[j]
            gap = (pd.to_datetime(row["starttime"]) - pd.to_datetime(segment_end)).total_seconds()
            sr_diff = abs(row["fixed_sampling_rate"] - segment_sr)

            if gap > gap_threshold or sr_diff > rate_tolerance:
                records.append({
                    "fixed_id": trace_id,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "total_npts": total_npts,
                    "sampling_rate": segment_sr
                })
                segment_start = row["starttime"]
                total_npts = 0
                segment_sr = row["fixed_sampling_rate"]

            segment_end = max(segment_end, row["endtime"])
            total_npts += row["npts"]

        records.append({
            "fixed_id": trace_id,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "total_npts": total_npts,
            "sampling_rate": segment_sr
        })

        if i % 100 == 0 and records:
            pd.DataFrame(records).to_csv(partial_csv, index=False)
            print(f"💾 Wrote partial ranges after {i} trace IDs")

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"📄 Contiguous ranges saved to {output_csv}. Total rows: {len(out_df)}")

    if os.path.exists(partial_csv):
        os.remove(partial_csv)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Audit legacy vs fixed trace IDs in an SDS archive.")
    parser.add_argument("sds_root", help="Root path of the SDS archive (e.g., /raid/data/SDS_Montserrat)")
    parser.add_argument("--audit_csv", default="trace_id_audit.csv", help="Path to write detailed audit CSV")
    parser.add_argument("--summary_csv", default="trace_id_mapping_summary.csv", help="Path to write grouped summary CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--gap_threshold", type=float, default=1.0, help="Max gap (s) before splitting segment")
    parser.add_argument("--rate_tolerance", type=float, default=1.0, help="Max change in Hz before splitting segment")
    parser.add_argument("--perfile_csv", default="per_file_trace_metadata.csv", help="Per-file trace metadata CSV")
    parser.add_argument("--ranges_csv", default="trace_segment_ranges.csv", help="Contiguous trace ranges CSV")
    args = parser.parse_args()

    df = audit_trace_ids_in_sds(
        sds_root=args.sds_root,
        output_csv=args.audit_csv,
        verbose=not args.quiet
    )

    export_grouped_summary(
        df,
        output_csv=args.summary_csv
    )

    perfile_df = export_per_file_metadata(df, output_csv=args.perfile_csv)

    compute_contiguous_ranges(
        perfile_df,
        output_csv=args.ranges_csv,
        gap_threshold=args.gap_threshold,
        rate_tolerance=args.rate_tolerance
    )   
