#!/usr/bin/env python3
# remove_unique_constraint.py

import sqlite3

# Path to your SQLite database
DB_PATH = "/home/thompsong/public_html/seiscomp_like_test.sqlite"

def remove_unique_constraint():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        # Disable foreign key constraints
        cur.execute("PRAGMA foreign_keys=OFF;")

        # Begin transaction
        cur.execute("BEGIN TRANSACTION;")

        # Rename the existing table
        cur.execute("ALTER TABLE event_classifications RENAME TO old_event_classifications;")

        # Create the new table without the UNIQUE constraint on dfile
        cur.execute('''
            CREATE TABLE event_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                dfile TEXT,
                mainclass TEXT,
                subclass TEXT,
                author TEXT,
                time TEXT,
                source TEXT,
                FOREIGN KEY(event_id) REFERENCES events(public_id),
                FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile)
            );
        ''')

        # Copy data from the old table to the new table
        cur.execute('''
            INSERT INTO event_classifications (id, event_id, dfile, mainclass, subclass, author, time, source)
            SELECT id, event_id, dfile, mainclass, subclass, author, time, source
            FROM old_event_classifications;
        ''')

        # Drop the old table
        cur.execute("DROP TABLE old_event_classifications;")

        # Commit the transaction
        conn.commit()

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        # Re-enable foreign key constraints
        cur.execute("PRAGMA foreign_keys=ON;")
        conn.close()

if __name__ == "__main__":
    remove_unique_constraint()
