import sqlite3
from flovopy.wrappers.MVOE_conversion_pipeline.db_backup import backup_db
DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
if not backup_db(DB_PATH, __file__):
    exit()
    
# === Configuration ===
TEST_MODE = False
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# === Update 'time' and 'endtime' fields in mseed_file_status ===
update_mseed_time = """
UPDATE mseed_file_status
SET time = time || 'Z'
WHERE time IS NOT NULL AND time NOT LIKE '%Z';
"""
update_mseed_endtime = """
UPDATE mseed_file_status
SET endtime = endtime || 'Z'
WHERE endtime IS NOT NULL AND endtime NOT LIKE '%Z';
"""

# === Update 'time' and 'endtime' fields in wfdisc ===
update_wfdisc_time = """
UPDATE wfdisc
SET time = time || 'Z'
WHERE time IS NOT NULL AND time NOT LIKE '%Z';
"""
update_wfdisc_endtime = """
UPDATE wfdisc
SET endtime = endtime || 'Z'
WHERE endtime IS NOT NULL AND endtime NOT LIKE '%Z';
"""

# Execute updates
cur.execute(update_mseed_time)
cur.execute(update_mseed_endtime)
cur.execute(update_wfdisc_time)
cur.execute(update_wfdisc_endtime)
if not TEST_MODE:
    conn.commit()

# === Verification ===
def count_non_z_suffix(table, column):
    query = f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL AND {column} NOT LIKE '%Z';"
    cur.execute(query)
    return cur.fetchone()[0]

tables_columns = [
    ('mseed_file_status', 'time'),
    ('mseed_file_status', 'endtime'),
    ('wfdisc', 'time'),
    ('wfdisc', 'endtime'),
]

for table, column in tables_columns:
    count = count_non_z_suffix(table, column)
    print(f"Remaining entries in {table}.{column} without 'Z' suffix: {count}")

conn.close()
