#!/usr/bin/env python3

import os
from obsolete.asl import extract_asl_diagnostics
from obspy import UTCDateTime

# === Set top-level directory ===
TOPDIR = "/data/b18_waveform_processing"

# === Output path ===
timestamp = int(UTCDateTime().timestamp)
OUTFILE = f"/home/thompsong/Dropbox/ASL_results_{timestamp}.csv"

# === Run extractor ===
print(f"[INFO] Extracting ASL results from: {TOPDIR}")
df = extract_asl_diagnostics(topdir=TOPDIR, output_csv=OUTFILE)

# === Optional: summary ===
print("\n[✓] Extraction complete.")
print(df.describe(include='all'))
