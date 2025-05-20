# a13_merge_continuous.py

import sqlite3
from obspy import read, Stream, UTCDateTime
from flovopy.config_projects import get_config
from flovopy.sds.sds import SDSobj

def merge_continuous_data(dbfile, output_sds_dir, start_date, end_date):
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    current_date = UTCDateTime(start_date)
    end_date = UTCDateTime(end_date)

    while current_date < end_date:
        next_date = current_date + 86400
        print(f"\nProcessing {current_date.date}...")

        cursor.execute("""
            SELECT path, start_time, end_time FROM wav_files 
            WHERE file_type IN ('continuous', 'pseudo_continuous')
              AND (start_time < ? AND end_time > ?)
            ORDER BY start_time
        """, (str(next_date), str(current_date)))
        rows = cursor.fetchall()

        combined_stream = Stream()
        for path, start, end in rows:
            try:
                st = read(path)
                combined_stream += st
            except Exception as e:
                print(f"Failed to read {path}: {e}")

        if not combined_stream:
            print("No data for this day.")
            current_date = next_date
            continue

        combined_stream.merge(method=1, fill_value=0)
        combined_stream.trim(current_date, next_date)

        sds = SDSobj(output_sds_dir)
        sds.stream = combined_stream
        sds.write()

        current_date = next_date

    conn.close()

if __name__ == "__main__":
    config = get_config()
    dbfile = config['mvo_seisan_index_db']
    output_sds_dir = config['sds_top']
    start_date = "1996-08-01"
    end_date = "1996-08-04"

    merge_continuous_data(dbfile, output_sds_dir, start_date, end_date)