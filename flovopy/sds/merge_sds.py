import os
import shutil
import sqlite3
import pandas as pd
from obspy import UTCDateTime
from flovopy.core.miniseed_io import read_mseed, write_mseed, smart_merge
from flovopy.core.trace_utils import streams_equal, summarize_stream
import json
import numpy as np

from sds_utils import (
    convert_numpy_types,
    setup_merge_tracking_db,
    restore_backup_file,
    is_valid_sds_dir, 
    is_valid_sds_filename
)


def merge_two_sds_archives(source1_sds_dir, source2_sds_dir, dest_sds_dir, 
                           db_path="merge_tracking.sqlite", interactive=True, merge_strategy='obspy'):
    """
    Merge SDS files from source1_sds_dir into dest_sds_dir.

    Behavior:
    ---------
    - If source2_sds_dir is None:
        - source1_sds_dir is merged into an existing SDS archive at dest_sds_dir.
        - dest_sds_dir must already exist and contain valid SDS structure.
    - If both source1_sds_dir and source2_sds_dir are given:
        - source1 and source2 are merged into a new or empty destination directory.
        - source1_sds_dir is copied first to dest_sds_dir.
        - Then, source2_sds_dir is merged into dest_sds_dir.

    Parameters:
    -----------
    source1_sds_dir : str
        Path to first SDS archive (required).

    source2_sds_dir : str or None
        Optional path to second SDS archive. If None, only source1 is merged into dest.

    dest_sds_dir : str
        Path to destination SDS archive. Must be empty or non-existent if source2 is provided.

    db_path : str, optional
        SQLite file path used for tracking merge operations. Defaults to "merge_tracking.sqlite".

    interactive : bool, optional
        If True, prompts user for confirmation before proceeding. Defaults to True.
    """
    # Ensure source1 is distinct and source2 is optional (can be the same as dest)
    abs1 = os.path.abspath(source1_sds_dir)
    abs2 = os.path.abspath(source2_sds_dir) if source2_sds_dir else None
    abs_dest = os.path.abspath(dest_sds_dir)

    if abs1 in [abs2, abs_dest] or (abs2 == abs_dest and abs2 is not None):
        raise ValueError("source1_sds_dir must be distinct, and no two input directories can be the same.")
    db_path=os.path.join(dest_sds_dir, db_path)

    if interactive:
        print(f"🔍 Preparing to merge\n- Source1: {abs1}\n- Source2: {abs2 or '(none)'}\n- Destination: {abs_dest}")
        proceed = input("❓ Proceed with merge? (Y/N): ").strip().lower()
        if proceed != 'y':
            print("❌ Merge cancelled by user.")
            return

    conn = setup_merge_tracking_db(db_path)
    c = conn.cursor()
    session_start = UTCDateTime().isoformat()

    c.execute("""
        INSERT INTO merge_sessions (started_at, source1, source2, destination, resolved, unresolved)
        VALUES (?, ?, ?, ?, 0, 0)
    """, (source1_sds_dir, source2_sds_dir or '', dest_sds_dir, session_start))
    session_id = c.lastrowid
    conn.commit()

    conflicts_resolved = 0
    conflicts_remaining = 0
    unresolved_conflicts = []

    if not source2_sds_dir:
        source2_sds_dir = source1_sds_dir

    print(f"⚡ Using rsync to quickly copy new files from {source2_sds_dir} to {dest_sds_dir}")
    os.system(f'rsync -av --ignore-existing --exclude "/*.*"  "{source2_sds_dir}/" "{dest_sds_dir}/"')

    for root, _, files in os.walk(source2_sds_dir):
        if not is_valid_sds_dir(root):
            continue
        numfiles = len(files)
        print(f'Directory: {root} Got {numfiles} files')
        for filenum, file in enumerate(files):
            if not is_valid_sds_filename(file):
                continue
            print(f'Processing file {filenum} of {numfiles}')

            source_file = os.path.join(root, file)
            rel_path = os.path.relpath(source_file, source2_sds_dir)
            dest_file = os.path.join(dest_sds_dir, rel_path)

            action = "copied"
            reason = "new file"
            st1_summary = ""
            st2_summary = ""
            time_shifts = {}
            fallback_max = []

            try:
                if os.path.exists(dest_file):
                    st1 = read_mseed(dest_file)
                    st2 = read_mseed(source_file)
                    merged = st1 + st2
                    report = smart_merge(merged, strategy=merge_strategy)

                    st1_summary = json.dumps(convert_numpy_types(summarize_stream(st1)))
                    st2_summary = json.dumps(convert_numpy_types(summarize_stream(st2)))

                    status_summary = report.get("summary", {})
                    status_by_id = report.get("status_by_id", {})
                    time_shifts = report.get("time_shifts", {})
                    fallback_max = report.get("fallback_to_max", [])

                    if not streams_equal(merged, st1):
                        # Only back up if we're about to overwrite
                        backup_file = dest_file + f".bak_session{session_id}"
                        shutil.copy2(dest_file, backup_file)
                        write_mseed(merged, dest_file, overwrite_ok=True)
                        action = "merged"
                        if any(status in ("timeshifted", "max") for status in status_by_id.values()):
                            reason = f"resolved via {','.join(set(status_by_id.values()))}"
                        else:
                            reason = "overlap resolved"
                        conflicts_resolved += 1
                    else:
                        action = "merged"
                        reason = "equal"

                else: # this should never get called now that i run rsync
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    shutil.copy2(source_file, dest_file)
                    action = "copied"
                    reason = "new file"
                    st1_summary = ""
                    st2_summary = ""
                    time_shifts = {}
                    fallback_max = []

                c.execute("""
                    INSERT INTO merge_files (session_id, rel_path, source_file, dest_file, action, reason, st1_metadata, st2_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, rel_path, source_file, dest_file, action, reason, st1_summary, st2_summary))

            except Exception as e:
                conflicts_remaining += 1
                reason = str(e)
                unresolved_conflicts.append({
                    "relative_path": rel_path,
                    "source_file": source_file,
                    "dest_file": dest_file,
                    "reason": reason
                })
                c.execute("""
                    INSERT INTO merge_files (session_id, rel_path, source_file, dest_file, action, reason, st1_metadata, st2_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, rel_path, source_file, dest_file, "error", reason, "", ""))

    session_end = UTCDateTime().isoformat()
    c.execute("""
        UPDATE merge_sessions
        SET ended_at = ?, resolved = ?, unresolved = ?
        WHERE id = ?
    """, (session_end, conflicts_resolved, conflicts_remaining, session_id))
    conn.commit()
    conn.close()

    if unresolved_conflicts:
        log_path = os.path.join(dest_sds_dir, f"conflicts_unresolved_session_{session_id}.csv")
        pd.DataFrame(unresolved_conflicts).to_csv(log_path, index=False)
        print(f"⚠️ Logged {len(unresolved_conflicts)} unresolved conflicts to {log_path}")

    print("✅ SDS merge complete.")
    print(f"✔️ Conflicts resolved and merged: {conflicts_resolved}")
    print(f"⚠️ Conflicts remaining:           {conflicts_remaining}")


def rollback_merge_session(session_id, db_path="merge_tracking.sqlite"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        SELECT dest_file FROM merge_files
        WHERE session_id = ? AND action = 'merged'
    """, (session_id,))
    merged_files = c.fetchall()

    rollback_count = 0
    for (dest_file,) in merged_files:
        backup_file = dest_file + f".bak_session{session_id}"
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, dest_file)
            print(f"🔄 Rolled back {dest_file} ← {backup_file}")
            rollback_count += 1
        else:
            print(f"⚠️ Backup missing: {backup_file}")

    conn.close()
    print(f"✅ Rollback complete. Restored {rollback_count} files.")


def merge_sds_archives(
    source1_sds_dir,
    dest_sds_dir,
    source2_sds_dir=None,
    db_path="merge_tracking.sqlite",
    interactive=True,
    merge_strategy='obspy'
):
    """
    Merge one or two SDS archives into a destination directory.

    Parameters
    ----------
    source1_sds_dir : str
        First SDS archive to merge. Required.
    dest_sds_dir : str
        Destination SDS archive path.
        - If `source2_sds_dir` is None: must already exist.
        - If `source2_sds_dir` is given: must be empty or non-existent (will be created).
    source2_sds_dir : str or None
        Optional second archive to merge alongside source1.
    """
    if not os.path.isdir(source1_sds_dir):
        raise ValueError("source1_sds_dir does not exist or is not a directory.")
    if source2_sds_dir and not os.path.isdir(source2_sds_dir):
        raise ValueError("source2_sds_dir does not exist or is not a directory.")

    abs1 = os.path.abspath(source1_sds_dir)
    abs2 = os.path.abspath(source2_sds_dir) if source2_sds_dir else None
    abs_dest = os.path.abspath(dest_sds_dir)

    if abs1 in [abs2, abs_dest] or (abs2 == abs_dest and abs2 is not None):
        raise ValueError("Input SDS directories must all be distinct.")

    if source2_sds_dir:
        # Two-source mode: destination must not exist or must be empty
        if os.path.exists(abs_dest) and os.listdir(abs_dest):
            raise ValueError("Destination directory must be empty or not exist when merging two source archives.")

        print(f"📂 Creating initial destination from {abs1}")
        #shutil.copytree(abs1, abs_dest, dirs_exist_ok=True,  copy_function=copytree_without_overwrite)
        os.system(f'rsync -av --ignore-existing --exclude "/*.*" "{abs1}/" "{abs_dest}/"')

        # Merge second archive into the destination
        print('Calling merge_two_sds_archives()')
        return merge_two_sds_archives(
            source1_sds_dir=abs2,
            source2_sds_dir=None,
            dest_sds_dir=abs_dest,
            db_path=db_path,
            interactive=interactive,
            merge_strategy=merge_strategy
        )

    else:
        # One-source mode: merging into an existing SDS archive
        if not os.path.exists(abs_dest) or not os.listdir(abs_dest):
            raise ValueError("Destination directory must already exist and contain an SDS archive.")
        
        print('Calling merge_two_sds_archives()')
        return merge_two_sds_archives(
            source1_sds_dir=abs1,
            source2_sds_dir=None,
            dest_sds_dir=abs_dest,
            db_path=db_path,
            interactive=interactive,
            merge_strategy=merge_strategy
        )


def merge_multiple_sds_archives(source_sds_dirs, dest_sds_dir, db_path="merge_tracking.sqlite", merge_strategy='obspy'):
    """
    Merge multiple SDS archives into a single destination SDS directory.

    Parameters
    ----------
    source_sds_dirs : list of str
        List of source SDS archive paths to merge. The first will initialize the destination if it doesn't exist.
    dest_sds_dir : str
        Path to the destination SDS archive. Must be distinct from all sources.
    db_path : str
        Path to the SQLite tracking database (relative to dest_sds_dir). Default is 'merge_tracking.sqlite'.
    """
    if not source_sds_dirs:
        print("❌ No source directories provided.")
        return

    abs_dest = os.path.abspath(dest_sds_dir)
    abs_sources = [os.path.abspath(s) for s in source_sds_dirs]

    if len(set(abs_sources + [abs_dest])) != len(abs_sources) + 1:
        raise ValueError("All source directories and destination must be distinct.")

    print(f"\n🔍 Ready to merge {len(abs_sources)} archive(s) into: {dest_sds_dir}")
    proceed = input("❓ Proceed with merge? (Y/N): ").strip().lower()
    if proceed != 'y':
        print("❌ Merge cancelled.")
        return

    if not os.path.exists(dest_sds_dir):
        if len(abs_sources) == 1:
            # Just copy the one source
            print(f"📂 Initializing destination by calling merge_sds_archives() from {abs_sources[0]} to {dest_sds_dir}")
            merge_sds_archives(
                source1_sds_dir=abs_sources[0],
                dest_sds_dir=abs_dest,
                source2_sds_dir=None,
                db_path=db_path,
                interactive=False,
                merge_strategy=merge_strategy
            )
            abs_sources = []
        else:
            # Merge first two sources into empty destination
            print(f"📂 Merging {abs_sources[0]} and {abs_sources[1]} into {dest_sds_dir} with merge_sds_archives()")
            merge_sds_archives(
                source1_sds_dir=abs_sources[0],
                source2_sds_dir=abs_sources[1],
                dest_sds_dir=abs_dest,
                db_path=db_path,
                interactive=False,
                merge_strategy=merge_strategy
            )
            abs_sources = abs_sources[2:]
    else:
        # Destination already exists, do not reinitialize
        pass

    # Merge remaining sources one by one
    for i, src in enumerate(abs_sources, start=1):
        print(f"\n🔄 Merging archive {i}/{len(abs_sources)}: {src} into {dest_sds_dir} by calling merge_sds_archives()")
        merge_sds_archives(
            source1_sds_dir=src,
            dest_sds_dir=abs_dest,
            source2_sds_dir=None,
            db_path=db_path,
            interactive=False,
            merge_strategy=merge_strategy
        )

    print("\n✅ All SDS archives merged successfully.")

'''
def copytree_without_overwrite(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

'''


