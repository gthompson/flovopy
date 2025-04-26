import os 
import pandas as pd
from flovopy.wrappers.MVOE_conversion_pipeline.b17_compute_station_corrections import estimate_station_corrections
# Read your input file
os.chdir('flovopy/wrappers/MVOE_conversion_pipeline')
print(os.getcwd())
df = pd.read_csv('all_regionals_station_corrections.csv')

# Initialize empty DataFrame for all years
all_years_df = pd.DataFrame()

# Loop over years
for year in range(1996, 2010):
    corrections, stats = estimate_station_corrections(df, year=year, min_num_picks=6)
    if corrections.any():
        #stats = stats.reset_index()
        # Make sure the station IDs are the index
        print(stats.columns)
        #stats = stats.set_index('trace_id')
        # Use the '50%' (median) value
        all_years_df[year] = stats['50%']


# Drop stations that have NaNs for all years
all_years_df = all_years_df.dropna(how='all')
all_years_df = all_years_df.sort_index()

# Save the combined table
all_years_df.to_csv('station_corrections_by_year.csv', index=True)
print(all_years_df)

# Reference station
ref_id = 'MV.MBGB..BHZ'
for year in range(1996, 2005):
    all_years_df[year] = all_years_df[year] / all_years_df[year].loc[ref_id]
print(all_years_df)

ref_id2 = 'MV.MBGB..HHZ'
ratio = all_years_df[2005].loc[ref_id2] / all_years_df[2005].loc[ref_id]
print(ratio)
# Normalize all years by the reference station  

for year in range(2006, 2009):
    all_years_df[year] = all_years_df[year] / all_years_df[year].loc[ref_id2] * ratio
print(all_years_df)

