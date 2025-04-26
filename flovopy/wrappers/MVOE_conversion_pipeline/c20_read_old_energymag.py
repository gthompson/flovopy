import os
import glob
import pandas as pd
TOP_DIR = '/data/Montserrat/mbwh_catalog'
all_year_month_dats = sorted(glob.glob(os.path.join(TOP_DIR, '*??????.dat' )))
df_list = []
for thisdat in all_year_month_dats:
    try:
        df_list.append(pd.read_csv(thisdat, sep=r"\s+", engine='python', header=None, names=['year', 'month', 'day', 'hour', 'minute', 'second', 'subclass', 'ME']))
    except Exception as e:
        print(thisdat, e)
alldf = pd.concat(df_list, ignore_index=True)
alldf.to_csv('/home/thompsong/Dropbox/old_energymag.csv', index=None)