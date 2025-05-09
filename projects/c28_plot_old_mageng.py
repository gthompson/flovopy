import pandas as pd
import matplotlib.pyplot as plt
from flovopy.core.enhanced import bin_events, plot_event_statistics

def generate_subclass_plots(df, starttime, endtime, binsize='4W', save_prefix=''):
    """
    Generate and save subclass plots (individual and combined) for a time range and bin size.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'time', 'subclass', 'magnitude', etc.
    starttime : str or datetime
        Start time for filtering (e.g., '2000-01-01').
    endtime : str or datetime
        End time for filtering (e.g., '2008-12-31').
    binsize : str
        Bin size (e.g., 'W', '4W', 'M').
    save_prefix : str
        Optional prefix for output filenames.
    """

    main_subclasses = ['r', 'e', 'l', 'h', 't']

    # Ensure datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Filter to requested time window
    mask = (df['time'] >= pd.to_datetime(starttime)) & (df['time'] <= pd.to_datetime(endtime))
    df_filtered = df.loc[mask].copy()

    # Assign 'o' to other subclasses
    df_filtered['subclass'] = df_filtered['subclass'].apply(lambda x: x if x in main_subclasses else 'o')

    # Bin events
    binned = bin_events(df_filtered, interval=binsize, groupby='subclass')

    # === Plot each subclass separately ===
    num_subclasses = len(main_subclasses)
    fig_stat, axs_stat = plt.subplots(num_subclasses, 1, figsize=(12, 18), sharex=True)
    fig_energy, axs_energy = plt.subplots(num_subclasses, 1, figsize=(12, 18), sharex=True)

    for i, subclass in enumerate(main_subclasses):
        subset = binned[binned['subclass'] == subclass].copy()
        subset['time'] = pd.to_datetime(subset['time'])
        subset = subset.sort_values('time')
        print(f"Processing subclass: {subclass}")

        # Plot event statistics
        plot_event_statistics(subset, ax=axs_stat[i], label=subclass, secondary_y='cumulative_magnitude')
        axs_stat[i].set_title(f'Subclass: {subclass}')

        # Plot energy statistics
        plot_event_statistics(subset, ax=axs_energy[i], label=subclass, secondary_y='thresholded_count')
        axs_energy[i].set_title(f'Subclass: {subclass}')
        axs_energy[i].legend()

    plt.tight_layout()
    stat_filename = f"{save_prefix}subclass_statistics_{starttime}_{endtime}_{binsize}.png".replace(':', '-')
    energy_filename = f"{save_prefix}subclass_energy_{starttime}_{endtime}_{binsize}.png".replace(':', '-')
    fig_stat.savefig(stat_filename)
    fig_energy.savefig(energy_filename)

    # === Create and plot the "ALL" bin ===
    df_all = df_filtered[df_filtered['subclass'].isin(main_subclasses)].copy()
    df_all['subclass'] = 'all'

    binned_all = bin_events(df_all, interval=binsize, groupby='subclass')

    fig_all, ax_all = plt.subplots(1, 1, figsize=(12, 4))

    subset_all = binned_all[binned_all['subclass'] == 'all'].copy()
    subset_all['time'] = pd.to_datetime(subset_all['time'])
    subset_all = subset_all.sort_values('time')

    plot_event_statistics(subset_all, ax=ax_all, label='all', secondary_y='cumulative_magnitude')
    ax_all.set_title('All Main Subclasses Combined')
    ax_all.legend()

    plt.tight_layout()
    all_filename = f"{save_prefix}all_events_statistics_{starttime}_{endtime}_{binsize}.png".replace(':', '-')
    fig_all.savefig(all_filename)

    print(f"Plots saved: {stat_filename}, {energy_filename}, {all_filename}")

if __name__ == "__main__":      
    import os
    import pandas as pd

    # Load your energy magnitude CSV
    csvfile = "/Users/GlennThompson/Dropbox/old_energymag_converted.csv"
    if os.path.isfile(csvfile):
        df = pd.read_csv(csvfile)
    else:
        df = pd.read_csv("/Users/GlennThompson/Dropbox/old_energymag.csv")
        # Fix the time columns if necessary
        from obspy import UTCDateTime
        df['time'] = df.apply(
            lambda row: UTCDateTime(
                int(row['year']),
                int(row['month']),
                int(row['day']),
                int(row['hour']),
                int(row['minute']),
                int(row['second'])
            ).datetime, axis=1
        )
        df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
        df.rename(columns={'ME': 'magnitude'}, inplace=True)
        df.to_csv(csvfile, index=False)

    # Call the function to generate the '4W' binned plots
    generate_subclass_plots(
        df,
        starttime="1996-01-01",
        endtime="2008-12-31",
        binsize='4W',
        save_prefix='regionals_'
    )