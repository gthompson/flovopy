from obspy import read_events, Catalog
from collections import Counter
import pandas as pd
import glob
import os

QML_DIR = "/data/SEISAN_DB/json/MVOE_"
CATALOG_ALL_QML = "/data/SEISAN_DB/catalogall.qml"
CATALOG_MAG_QML = "/data/SEISAN_DB/catalogmag.qml"

def load_all_qml_files():
    qml_files = sorted(glob.glob(os.path.join(QML_DIR, "*", "*", "*.qml")))
    if os.path.isfile(CATALOG_ALL_QML):
        catalog = read_events(CATALOG_ALL_QML)
    else:
        print(f"[INFO] Reading {len(qml_files)} QuakeML files...")
        catalog = Catalog()
        for qml in qml_files:
            try:
                catalog += read_events(qml)
            except Exception as e:
                print(f"[WARN] Failed to read {qml}: {e}")
        catalog.write(CATALOG_ALL_QML, format="QUAKEML")
    return catalog, qml_files

def summarize_magnitudes(catalog, qml_files):
    event_summaries = {}
    mag_events = []

    for i, event in enumerate(catalog):
        evid = event.resource_id.id.split('/')[-1]
        mag_list = event.magnitudes
        if not mag_list:
            continue

        mag = mag_list[0]
        mag_value = mag.mag
        mag_type = getattr(mag, 'magnitude_type', None)
        if mag_type is None and hasattr(mag.creation_info, 'agency_id'):
            mag_type = mag.creation_info.agency_id

        mag_author = getattr(mag.creation_info, 'author', None)
        try:
            mag_class = event.event_descriptions[0].text.strip()
        except (IndexError, AttributeError):
            mag_class = None

        mag_qml_path = qml_files[i] if i < len(qml_files) else None

        event_summaries[evid] = {
            'event_id': evid,
            'qml': mag_qml_path,
            'mainclass': mag_class,
            'type': mag_type,
            'magnitude': mag_value,
            'author': mag_author
        }

        mag_events.append(event)

    return event_summaries, Catalog(events=mag_events)

def export_summary_to_csv(event_summaries, output_csv="/data/SEISAN_DB/magnitude_summary.csv"):
    df = pd.DataFrame.from_dict(event_summaries, orient="index")
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Exported {len(df)} rows to {output_csv}")
    return df

# === Run everything ===
if os.path.isfile(CATALOG_MAG_QML):
    catalogmag = read_events(CATALOG_MAG_QML)
    catalogall, qml_files = None, None  # not used in this case
else:
    catalogall, qml_files = load_all_qml_files()
    event_summaries, catalogmag = summarize_magnitudes(catalogall, qml_files)
    catalogmag.write(CATALOG_MAG_QML, format="QUAKEML")

# If we just loaded catalogmag from file, regenerate qml_files for summary
if catalogall is None:
    qml_files = sorted(glob.glob(os.path.join(QML_DIR, "*", "*", "*.qml")))
    event_summaries, _ = summarize_magnitudes(catalogmag, qml_files)

# Magnitude types summary
mag_types = Counter()
for event in catalogmag:
    for mag in event.magnitudes:
        mtype = getattr(mag, 'magnitude_type', None)
        if not mtype:
            mtype = getattr(mag.creation_info, 'agency_id', 'UNKNOWN')
        mag_types[mtype] += 1

print("\n=== Magnitude Type Counts ===")
for k, v in mag_types.items():
    print(f"{k}: {v}")

# Export to CSV
df = export_summary_to_csv(event_summaries)

# Loop over the dataframe and load the corresponding json file which contains AEF information
# - to compute energy magnitude, we need to 
