import sqlite3
import pandas as pd
from obspy import read
from flovopy.core.enhanced import EnhancedStream, EnhancedCatalog  # adjust path if needed

def get_event_info(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sfiles", conn)
    conn.close()
    return df

if __name__ == '__main__':
    # Step 0: Load your index database
    db_path = "/home/thompsong/public_html/index_mvoe4.sqlite"
    sfiles_df = get_event_info(db_path)
    print(sfiles_df)

    # Let's take the first entry as an example
    for _, row in sfiles_df.iterrows():
        sfile_path = row['filepath']      # adjust if column name is different
        wav_path = row['dsnwavfile']      # this is the corresponding waveform file

        print(f"Processing: S-file = {sfile_path}, WAV = {wav_path}")

        # Step 1: Parse the S-file for metadata
        #sfile_event_metadata = parse_sfile(sfile_path)

        # Step 2: Load waveform file
        raw_stream = read(wav_path)
        print(raw_stream)

        # Step 3: Convert to EnhancedStream
        enh_st = EnhancedStream(raw_stream)
        print(enh_st)

        break  # remove this if processing all events
