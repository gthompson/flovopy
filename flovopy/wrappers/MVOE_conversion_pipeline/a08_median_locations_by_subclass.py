import sqlite3
import pandas as pd

DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
conn = sqlite3.connect(DB_PATH)

query1 = """
SELECT ec.mainclass
FROM event_classifications ec
"""

query2 = """
SELECT ec.subclass
FROM event_classifications ec
WHERE ec.mainclass = 'LV'
"""

query3 = """
SELECT ec.subclass, o.latitude, o.longitude, o.depth
FROM event_classifications ec
JOIN origins o ON ec.event_id = o.event_id
WHERE ec.mainclass = 'LV'
  AND o.latitude IS NOT NULL
  AND o.longitude IS NOT NULL
  AND o.depth IS NOT NULL
"""

query4 = """
SELECT ec.subclass
FROM event_classifications ec
WHERE NOT ec.mainclass = 'LV'
  AND ec.subclass IS NOT NULL
"""

df1 = pd.read_sql_query(query1, conn)
df2 = pd.read_sql_query(query2, conn)
df3 = pd.read_sql_query(query3, conn)
df4 = pd.read_sql_query(query4, conn)

conn.close()

mainclass_df = df1.groupby("mainclass")
for name,group in mainclass_df:
    print(name, len(group))
print()

subclass_df = df2.groupby("subclass")
for name,group in subclass_df:
    print(name, len(group))
print()

# Compute medians
located = df3.groupby("subclass")
summary_df = located[["latitude", "longitude", "depth"]].median().reset_index()

# Add count of located events per subclass
summary_df["n_located"] = located.size().values

print(summary_df)
print()

subclass_df = df4.groupby("subclass")
for name,group in subclass_df:
    print(name, len(group))