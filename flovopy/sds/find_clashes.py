import os
import argparse
import random
from obspy import read, Stream
from flovopy.core.trace_utils import streams_equal
from multiprocessing import Pool, cpu_count
from flovopy.core.miniseed_io import smart_merge, read_mseed, write_mseed, compare_mseed_files
from pprint import pprint
from flovopy.sds.sds_utils import write_csv



def process_file(args):
    rel_path, source_root, dest_root = args
    src_path = os.path.join(source_root, rel_path)
    dest_path = os.path.join(dest_root, rel_path)

    if not os.path.exists(dest_path):
        return ("safe_to_copy", [rel_path, os.path.getsize(src_path)])

    src_size = os.path.getsize(src_path)
    dest_size = os.path.getsize(dest_path)
    result = {"rel_path": rel_path, "src_size": src_size, "dest_size": dest_size}

    if src_size == dest_size:
        result["size_status"] = "same"
    else:
        result["size_status"] = "clashed"

    same, err = compare_mseed_files(src_path, dest_path)
    if err:
        result["content_status"] = "error"
        result["error"] = err
    else:
        result["content_status"] = "same" if same else "clashed"

    return ("compared", result)

def find_clashes_parallel(source_root, dest_root, outdir=".", nproc=6, sample_size=None):
    print(f"🔍 Scanning source directory: {source_root}")
    all_files = []
    for root, _, files in os.walk(source_root):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, source_root)
            all_files.append(rel_path)

    if not all_files:
        print("❌ No files found in source directory.")
        return

    print(f"📦 Total files found: {len(all_files)}")
    if sample_size:
        random.shuffle(all_files)
        all_files = all_files[:sample_size]
        print(f"🎯 Sampling {sample_size} files.")

    # Prepare argument list for multiprocessing
    args = [(rel_path, source_root, dest_root) for rel_path in all_files]

    print(f"🚀 Starting parallel processing with {nproc} workers...")
    with Pool(processes=nproc) as pool:
        results = pool.map(process_file, args)

    # Organize results
    clashed = []
    same_size = []
    safe_to_copy = []
    same_contents = []
    clashed_contents = []
    read_errors = []

    for tag, item in results:
        if tag == "safe_to_copy":
            safe_to_copy.append(item)
        elif tag == "compared":
            rel_path = item["rel_path"]
            src_size = item["src_size"]
            dest_size = item["dest_size"]

            if item["size_status"] == "same":
                same_size.append([rel_path, src_size])
            else:
                clashed.append([rel_path, src_size, dest_size])

            if item["content_status"] == "same":
                same_contents.append([rel_path, src_size])
            elif item["content_status"] == "clashed":
                clashed_contents.append([rel_path, src_size, dest_size])
            elif item["content_status"] == "error":
                read_errors.append([rel_path, item["error"]])

    # Write output
    os.makedirs(outdir, exist_ok=True)

    write_csv(os.path.join(outdir, "find_clashes_clashed.csv"),
              ["Relative Path", "Source Size (bytes)", "Destination Size (bytes)"],
              clashed)

    write_csv(os.path.join(outdir, "find_clashes_samesize.csv"),
              ["Relative Path", "Size (bytes)"],
              same_size)

    write_csv(os.path.join(outdir, "find_clashes_same_contents.csv"),
              ["Relative Path", "Size (bytes)"],
              same_contents)

    write_csv(os.path.join(outdir, "find_clashes_clashed_contents.csv"),
              ["Relative Path", "Source Size (bytes)", "Destination Size (bytes)"],
              clashed_contents)

    write_csv(os.path.join(outdir, "find_clashes_safetocopy.csv"),
              ["Relative Path", "Source Size (bytes)"],
              safe_to_copy)

    if read_errors:
        write_csv(os.path.join(outdir, "find_clashes_read_errors.csv"),
                  ["Relative Path", "Error Message"],
                  read_errors)

    # Summary
    print("\n✅ Parallel Clash Check Complete.")
    print(f"Total compared:                    {len(all_files)}")
    print(f"  ├── Not in dest (safe):          {len(safe_to_copy)}")
    print(f"  ├── Same size:                   {len(same_size)}")
    print(f"  │     ├── Same content:          {len(same_contents)}")
    print(f"  │     └── Content mismatch:      {len(clashed_contents)}")
    print(f"  └── Size mismatch:               {len(clashed)}")
    print(f"⚠️  Read errors:                   {len(read_errors)}")

    # Convert both to dicts using rel_path as key
    clashed_dict = {entry[0]: entry for entry in clashed}
    clashed_contents_dict = {entry[0]: entry for entry in clashed_contents}

    # Merge the dictionaries
    combined_clashes = {**clashed_dict, **clashed_contents_dict}

    # Extract combined list
    all_clash_entries = list(combined_clashes.values())

    merge_conflicts = []
    merged_output_root = os.path.join(outdir, "merged_good")

    print("\n🔄 Attempting smart_merge() on content clashes...")

    for rel_path, _, _ in all_clash_entries:
        src_file = os.path.join(source_root, rel_path)
        dest_file = os.path.join(dest_root, rel_path)
        try:
            st = read_mseed(src_file) + read_mseed(dest_file)
            merged_stream = st.copy()
            report = smart_merge(merged_stream)
            if report['status']=='ok':
                out_file = os.path.join(merged_output_root, rel_path)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                write_mseed(merged_stream, out_file)
                print(f"✅ Merged: {rel_path}")
            else:
                pprint(report)
                print(f"❌ Merge failed: {rel_path} ")
                merge_conflicts.append([rel_path, report['status']])
        except Exception as e:
            print(f"❌ Merge failed: {rel_path} — {e}")
            merge_conflicts.append([rel_path, str(e)])

    if merge_conflicts:
        merge_log = os.path.join(outdir, "find_clashes_merge_conflicts.csv")
        write_csv(merge_log, ["Relative Path", "Error Message"], merge_conflicts)
        print(f"\n⚠️ Merge conflicts logged to: {merge_log}")
    else:
        print("\n✅ All content clashes successfully merged.")

    # Optional: Suggest rsync command to copy missing files
    safecopy_list_path = os.path.join(outdir, "find_clashes_safetocopy.csv")
    if os.path.exists(safecopy_list_path):
        print("\n💡 To copy only missing files from source to destination, use:")
        print("rsync -av --ignore-existing --files-from=<(cut -d, -f1 {} | tail -n +2) {} {}".format(
            safecopy_list_path, source_root, dest_root))
        print("⚠️ Run this in a Bash-compatible shell (Linux/macOS).")
    else:
        print("ℹ️ No safe-to-copy file list found; rsync suggestion skipped.")

    # Optional: Suggest rsync command to overwrite clashed files with smart_merge results
    if os.path.exists(merged_output_root):
        print("\n💡 To copy merged files (that previously clashed) into the destination archive, use:")
        print(f"rsync -av --backup --backup-dir=overwritten_files_backup_dir {merged_output_root}/ {dest_root}/")
        print("⚠️ This will overwrite files that clashed but were successfully merged.")

def main():
    parser = argparse.ArgumentParser(description="Parallel MiniSEED clash checker.")
    parser.add_argument("source", help="Source SDS directory")
    parser.add_argument("destination", help="Destination SDS directory")
    parser.add_argument("-o", "--outdir", default=".", help="Output directory for CSVs")
    parser.add_argument("-n", "--sample-size", type=int, default=None, help="Random sample size")
    parser.add_argument("-p", "--processes", type=int, default=6, help="Number of parallel processes")
    args = parser.parse_args()

    find_clashes_parallel(args.source, args.destination, args.outdir, args.processes, args.sample_size)

if __name__ == "__main__":
    main()