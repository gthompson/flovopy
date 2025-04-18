import sqlite3
import os
import shutil

# === Configuration ===
DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
TEST_MODE = True
TEST_DB_COPY = DB_PATH.replace('.sqlite', '_test.sqlite')

# === Setup Database Connection ===
if TEST_MODE:
    if os.path.exists(TEST_DB_COPY):
        os.remove(TEST_DB_COPY)
    shutil.copy(DB_PATH, TEST_DB_COPY)
    DB_PATH = TEST_DB_COPY
    print(f"[TEST MODE] Using temporary DB: {DB_PATH}")

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
