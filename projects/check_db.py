import sqlite3

conn = sqlite3.connect("/Volumes/tachyon/from_hal/SEISAN_DB/seisan_db.sqlite")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in database:")
for table in tables:
    print(table[0])

cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Table name | Row count")
print("-" * 30)
for (table_name,) in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
        row_count = cursor.fetchone()[0]
        print(f"{table_name:<20} {row_count}")
    except Exception as e:
        print(f"{table_name:<20} ERROR: {e}")

conn.close()

