import os
import sqlite3
from obspy import read, Stream
from obspy.core import UTCDateTime
from pathlib import Path
from flovopy.config_projects import get_config
from tqdm import tqdm

GAP_THRESHOLD = 1  # second tolerance for gap detection

def nearest_sampling_rate(tr):
    return 75.0 if abs(tr.stats.sampling_rate - 75.0) < 5 else 100.0

def detect_gaps(wav_rows):
    """Returns list of (gap_start, gap_end) UTCDateTimes"""
    gaps = []
    for i in range(len(wav_rows) - 1):
        end = UTCDateTime(wav_rows[i][3])
        next_start = UTCDateTime(wav_rows[i + 1][2])
        if (next_start - end) > GAP_THRESHOLD:
            gaps.append((end, next_start))
    return gaps

def find_event_fillers(conn, gap_start, gap_end):
    cur = conn.cursor()
    cur.execute("""
        SELECT path FROM wav_files
        WHERE file_type='event'
        AND (
            (start_time BETWEEN ? AND ?)
            OR (end_time BETWEEN ? AND ?)
        )
    """, (gap_start.isoformat(), gap_end.isoformat(), gap_start.isoformat(), gap_end.isoformat()))
    return [bytes(row[0]).decode("utf-8") if isinstance(row[0], bytes) else row[0] for row in cur.fetchall()]

def fill_gap(gap_start, gap_end, event_paths, mseed_output_dir, conn):
    st = Stream()
    for path in event_paths:
        try:
            st += read(path)
        except Exception as e:
            print(f"[!] Failed to read {path}: {e}")

    st = st.trim(gap_start, gap_end)
    if len(st) == 0:
        print(f"No data could be extracted for gap {gap_start} – {gap_end}")
        return

    st.merge(method=1, fill_value=0)

    # Group by network.station.location
    nsl_groups = defaultdict(Stream)
    for tr in st:
        key = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}"
        tr.stats.sampling_rate = 75.0 if abs(tr.stats.sampling_rate - 75.0) < 5 else 100.0
        nsl_groups[key].append(tr)

    for nsl_key, group_stream in nsl_groups.items():
        chunk_start = gap_start
        while chunk_start < gap_end:
            chunk_end = min(chunk_start + 20 * 60, gap_end)
            chunk = group_stream.copy().trim(starttime=chunk_start, endtime=chunk_end, pad=True, fill_value=0)

            # Write full Stream chunk
            fname = f"{nsl_key.replace('.', '_')}_{chunk_start.isoformat().replace(':','')}.mseed"
            outpath = Path(mseed_output_dir) / fname
            os.makedirs(outpath.parent, exist_ok=True)
            chunk.write(str(outpath), format="MSEED")

            # Insert to DB
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO wav_files (path, filetime, start_time, end_time,
                    associated_sfile, used_in_event_id, network, parsed_successfully, 
                    parsed_time, error, file_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(outpath),
                chunk_start.isoformat(),
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                None, None,
                chunk[0].stats.network,
                1,
                UTCDateTime.utcnow().isoformat(),
                None,
                "pseudo_continuous"
            ))
            conn.commit()

            print(f"[✓] Wrote filled stream: {outpath}")
            chunk_start = chunk_end

def main():
    config = get_config()
    dbfile = config['mvo_seisan_index_db']
    mseed_output_dir = config['mvo_seisan_mseed_dir']

    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT path, filetime, start_time, end_time FROM wav_files
        WHERE file_type='continuous'
        ORDER BY start_time
    """)
    wav_rows = cursor.fetchall()

    print(f"Loaded {len(wav_rows)} continuous WAV file records")
    gaps = detect_gaps(wav_rows)
    print(f"Detected {len(gaps)} gaps")

    for gap_start, gap_end in tqdm(gaps, desc="Filling gaps"):
        event_paths = find_event_fillers(conn, gap_start, gap_end)
        if event_paths:
            print(f"Found {len(event_paths)} event file(s) for gap {gap_start} to {gap_end}")
            fill_gap(UTCDateTime(gap_start), UTCDateTime(gap_end), event_paths, mseed_output_dir, conn)
        else:
            print(f"No event files for gap {gap_start} to {gap_end}")

    conn.close()

if __name__ == "__main__":
    main()