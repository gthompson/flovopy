import sqlite3
import os

def add_file_type_column(db_path):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the column already exists
    cursor.execute("PRAGMA table_info(wav_files);")
    columns = [col[1] for col in cursor.fetchall()]
    if 'file_type' in columns:
        print("[✓] Column 'file_type' already exists.")
    else:
        try:
            cursor.execute("ALTER TABLE wav_files ADD COLUMN file_type TEXT DEFAULT 'event';")
            conn.commit()
            print("[✓] Added 'file_type' column with default value 'event'.")
        except sqlite3.OperationalError as e:
            print(f"[ERROR] Failed to add column: {e}")
            conn.close()
            return

    # Optional: Verify summary of rows by file_type
    print("\n[INFO] Row count by file_type:")
    try:
        for file_type in ['event', 'continuous']:
            cursor.execute("SELECT COUNT(*) FROM wav_files WHERE file_type = ?", (file_type,))
            count = cursor.fetchone()[0]
            print(f"  {file_type:<12}: {count}")
    except sqlite3.OperationalError:
        print("  [WARN] Could not count by file_type — maybe the column hasn't been populated yet.")

    conn.close()

if __name__ == "__main__":
    from flovopy.config_projects import get_config
    config = get_config()
    dbfile = config['mvo_seisan_index_db']
    add_file_type_column(dbfile)