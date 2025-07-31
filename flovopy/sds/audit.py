import os
import pandas as pd
from obspy import read
from flovopy.core.mvo import fix_trace_mvo_wrapper
from flovopy.sds.sds import is_valid_sds_filename

def audit_trace_ids_in_sds(sds_root, output_csv="trace_id_audit.csv", verbose=True):
    """
    Audit legacy vs fixed trace IDs in an SDS archive using head-only MiniSEED reads.

    Parameters
    ----------
    sds_root : str
        Path to the root of the SDS archive.
    output_csv : str
        Path to output CSV file with audit results.
    verbose : bool
        If True, print progress and warnings to the console.

    Returns
    -------
    pandas.DataFrame
        DataFrame of audit results.
    """
    audit_records = []

    for root, dirs, files in os.walk(sds_root):
        for fname in files:
            if not fname.endswith(".D"):
                continue

            full_path = os.path.join(root, fname)
            if not is_valid_sds_filename(full_path):
                continue

            try:
                st = read(full_path, headonly=True)
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
                except Exception as e:
                    if verbose:
                        print(f"❌ Failed to fix trace ID in {full_path}: {e}")
                    continue

    df = pd.DataFrame(audit_records)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Audit complete. Saved to {output_csv}. Total traces processed: {len(df)}")
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Audit legacy vs fixed trace IDs in an SDS archive.")
    parser.add_argument("sds_root", help="Root path of the SDS archive (e.g., /raid/data/SDS_Montserrat)")
    parser.add_argument("--audit_csv", default="trace_id_audit.csv", help="Path to write detailed audit CSV")
    parser.add_argument("--summary_csv", default="trace_id_mapping_summary.csv", help="Path to write grouped summary CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

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
