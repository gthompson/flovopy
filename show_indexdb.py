import sqlite3
import os
os.system("clear")
# Connect to your SQLite database file
#conn = sqlite3.connect("/Volumes/tachyon/from_hal/SEISAN_DB/index_mvoe4.sqlite")
conn = sqlite3.connect("/data/SEISAN_DB/index_mvoe4.sqlite")
cursor = conn.cursor()

# Query for all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print results
for (table_name,) in tables:
    print(f"\nTable: {table_name}")
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = cursor.fetchall()
    for col in columns:
        cid, name, col_type, notnull, dflt_value, pk = col
        print(f"  {name} ({col_type}){' PRIMARY KEY' if pk else ''}")

    # Show number of rows
    try:
        cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
        count = cursor.fetchone()[0]
        print(f"  → Rows: {count}")
    except Exception as e:
        print(f"  → Could not count rows: {e}")
conn.close()
