import pandas as pd
import numpy as np
import os
import pprint

def estimate_station_corrections(df, ref_id,  min_num_picks=3, verbose=True):
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
    df = df.dropna(subset=[ref_id])

    # Drop metadata columns and work only on station columns
    station_cols = [col for col in df.columns if col not in ['num_picks', 'dfile']]
    df_sta = df[station_cols]
    print('df_sta', df_sta)

    # Compute relative correction ratios
    ratios = df_sta.div(df_sta[ref_id], axis=0)
    # Remove extreme or invalid pick values (<0.1s or >10s)
    ratios = ratios.where((ratios >= 0.05) & (ratios <= 20.0))

    # Describe statistics per station
    stats = ratios.describe().transpose()
    stats = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'count']]

    # Output correction factors (mean ratio per station)
    correction_factors = stats['mean']

    if verbose:
        print(f"\n[INFO] Reference station: {ref_id}")
        print(f"[INFO] Used {len(df_sta)} events with ≥ {min_num_picks} picks.")
        print(stats[['mean', 'std', 'count']])

    return correction_factors, stats




if __name__ == "__main__":
    # Initialize a dictionary to hold DataFrames
    ref_corrections = {}
    ref_id1 = 'MV.MBGB..BHZ'
    ref_id2 = 'MV.MBGB..HHZ'

    # Change working directory
    os.chdir('flovopy/wrappers/MVOE_conversion_pipeline')
    print("Working directory:", os.getcwd())

    # Read your input file
    df = pd.read_csv('all_regionals_station_corrections.csv')

    # Loop over both ref_ids
    for ref_id in [ref_id1, ref_id2]:
        # Estimate corrections
        corrections, stats = estimate_station_corrections(df, ref_id, min_num_picks=6)
        if corrections.any():

            #stats = stats[['trace_id', '50%']]  # keep only trace_id and 50%
            stats = stats[['50%']]
            stats = stats.rename(columns={'50%': ref_id})  # rename 50% to ref_id name
            #stats = stats.set_index('trace_id')  # set trace_id as index
            ref_corrections[ref_id] = stats

    # Now join the two correction DataFrames
    summarydf = ref_corrections[ref_id1].join(ref_corrections[ref_id2], how='outer')

    # (Optional) Add a difference column
    summarydf['ratio'] = summarydf[ref_id2] / summarydf[ref_id1]
    mean_ratio = summarydf['ratio'].mean()

    # iterate  
    summarydf[ref_id2] = summarydf[ref_id2] / mean_ratio  # Normalize by mean ratio
    summarydf['ratio'] = summarydf[ref_id2] / summarydf[ref_id1]
    mean_ratio = summarydf['ratio'].mean()
    print(f"Mean ratio: {mean_ratio}")

    # Sort by trace_id
    summarydf = summarydf.sort_index()
    summarydf.dropna(how='all', inplace=True)  # Drop rows with all NaN values

    # Print and/or save
    pprint.pprint(summarydf)
    summarydf.to_csv('summary_station_corrections.csv')
