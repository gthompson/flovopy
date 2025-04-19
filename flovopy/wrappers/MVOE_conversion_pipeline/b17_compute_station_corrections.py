import pandas as pd
import numpy as np

def estimate_station_corrections(df, year, min_num_picks=3, verbose=True):
    """
    Estimate relative station corrections from a DataFrame of arrival times.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'num_picks' column, a 'time' datetime index, 
        and columns for each station's relative pick time in seconds.
    year : int
        Year to subset the DataFrame for.
    min_num_picks : int
        Minimum number of picks required per event to include it in the analysis.
    verbose : bool
        If True, print reference station and summary.

    Returns
    -------
    correction_factors : pd.Series
        Series of correction factors per station (relative to reference station).
    stats : pd.DataFrame
        Descriptive statistics of the correction ratios per station.
    """

    # Ensure 'time' is datetime and set as index
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

    df = df.set_index('time')


    # Subset by year and minimum number of picks
    df_year = df[(df.index.year == year) & (df['num_picks'] >= min_num_picks)]
    if len(df_year)==0:
        return pd.Series(), pd.DataFrame()

    # Drop metadata columns and work only on station columns
    station_cols = [col for col in df.columns if col not in ['num_picks', 'dfile']]
    df_sta = df_year[station_cols]


    # Find reference station: the one with the most non-NaN values
    ref_id = df_sta.count().idxmax()

    # Compute relative correction ratios
    ratios = df_sta.div(df_sta[ref_id], axis=0)
    # Remove extreme or invalid pick values (<0.1s or >10s)
    ratios = ratios.where((ratios >= 0.1) & (ratios <= 10.0))

    # Describe statistics per station
    stats = ratios.describe().transpose()
    stats = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'count']]

    # Output correction factors (mean ratio per station)
    correction_factors = stats['mean']

    if verbose:
        print(f"\n[INFO] Reference station: {ref_id}")
        print(f"[INFO] Used {len(df_sta)} events in {year} with ≥ {min_num_picks} picks.")
        print(stats[['mean', 'std', 'count']])

    return correction_factors, stats

df = pd.read_csv('all_regionals_station_corrections.csv')
for year in range(1996,2010,1):
    corrections, stats = estimate_station_corrections(df, year=year, min_num_picks=6)
    if corrections.any():
        stats.reset_index()
        stats.to_csv(f'station_corrections_{year}.csv', index=True)

import os
def get_correction(year, trace_id):
    csv = f'station_corrections_{year}.csv'
    if os.path.isfile(csv):
        df2 = pd.read_csv(csv, index_col=0)
        return df2.loc['MV.MBWH..SHZ', 'mean']
    else:
        return None

print('\n\n\n')
trace_id = 'MV.MBWH..SHZ'
for year in range(1996,2010,1):
    value = get_correction(year, trace_id)
    print(f'the correction for {trace_id} in year {year} is {value}')