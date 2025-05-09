import pandas as pd
import matplotlib.pyplot as plt
from flovopy.core.enhanced import bin_events, plot_event_statistics

def analyze_event_data(csvfile, starttime=None, endtime=None, binsize='4W', save_prefix='events_'):
    """
    Analyze seismic event energy and magnitude data.

    Parameters:
    -----------
    csvfile : str
        Path to the CSV file containing event-level data.
    starttime : str or None
        Start time (e.g., '2000-01-01'). If None, use earliest.
    endtime : str or None
        End time (e.g., '2010-01-01'). If None, use latest.
    binsize : str
        Binning interval (e.g., 'W', '4W', 'M').
    save_prefix : str
        Filename prefix for saving plots.
    """

    # Load data
    df = pd.read_csv(csvfile)
    print(df.columns.tolist())
    print(df.head())
    df['time'] = pd.to_datetime(df['event_time'])
    df['magnitude'] = pd.to_numeric(df['ME_median'], errors='coerce')

    # Filter by time
    if starttime:
        df = df[df['time'] >= pd.to_datetime(starttime, utc=True)]
    if endtime:
        df = df[df['time'] <= pd.to_datetime(endtime, utc=True)]

    # Bin the events
    binned = bin_events(df, interval=binsize)

    # Plot event statistics
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_event_statistics(binned, ax=ax, secondary_y='cumulative_magnitude')
    ax.set_title(f"Event Statistics ({binsize} bins)")
    ax.legend()
    plt.tight_layout()

    # Save plot
    output_filename = f"{save_prefix}statistics_{starttime}_{endtime}_{binsize}.png".replace(':', '-')
    fig.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

def plot_magnitude_vs_time(csvfile, save_prefix='events_'):
    """
    Plot a stem plot of magnitude vs time for each individual event.

    Parameters:
    -----------
    csvfile : str
        Path to the CSV file.
    save_prefix : str
        Prefix for saved plot filename.
    """

    # Load data
    df = pd.read_csv(csvfile)
    
    # Fix datetime
    df['time'] = pd.to_datetime(df['event_time'])

    # Create magnitude column (choose ME or ML)
    if 'ME_median' in df.columns:
        df['magnitude'] = df['ME_median']
    elif 'ML_median' in df.columns:
        df['magnitude'] = df['ML_median']
    else:
        raise ValueError("No usable magnitude column found!")

    # Sort by time (just to be safe)
    df = df.sort_values('time')

    # Create stem plot
    fig, ax = plt.subplots(figsize=(14, 6))
    markerline, stemlines, baseline = ax.stem(df['time'], df['magnitude'], linefmt='grey', markerfmt='o', basefmt=" ")
    
    # Style
    plt.setp(markerline, markersize=4, color='black')
    plt.setp(stemlines, linewidth=0.5, color='grey')
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.set_title("Magnitude vs Time (No Binning)")
    ax.grid(True)
    plt.tight_layout()

    # Save
    output_filename = f"{save_prefix}magnitude_vs_time_stemplot.png"
    fig.savefig(output_filename)
    print(f"Stem plot saved as {output_filename}")

import numpy as np
from scipy.stats import linregress

def plot_ml_vs_me(csvfile, save_prefix='events_'):
    """
    Plot ML vs ME and fit a linear regression line.

    Parameters:
    -----------
    csvfile : str
        Path to the CSV file.
    save_prefix : str
        Prefix for saved plot filename.
    """

    # Load data
    df = pd.read_csv(csvfile)
    
    # Check if necessary columns exist
    if not {'ML_median', 'ME_median'}.issubset(df.columns):
        raise ValueError("CSV must contain 'ML_median' and 'ME_median' columns!")

    x = df['ML_median']
    y = df['ME_median']

    # Remove NaNs or bad values
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color='black', s=10, label='Events')
    ax.plot(x, slope*x + intercept, color='red', linewidth=2, label=f'Fit: ME = {slope:.2f} ML + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}')
    ax.set_xlabel('ML_median')
    ax.set_ylabel('ME_median')
    ax.set_title('ML vs ME with Linear Fit')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    # Save plot
    output_filename = f"{save_prefix}ML_vs_ME_fit.png"
    fig.savefig(output_filename)
    print(f"ML vs ME plot saved as {output_filename}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_ml_vs_me_by_subclass(csvfile, save_prefix='events_'):
    """
    Plot ML vs ME by subclass, with individual linear fits.

    Parameters:
    -----------
    csvfile : str
        Path to the CSV file.
    save_prefix : str
        Prefix for saved plot filename.
    """

    # Load data
    df = pd.read_csv(csvfile)
    
    # Check necessary columns
    if not {'ML_median', 'ME_median', 'subclass'}.issubset(df.columns):
        raise ValueError("CSV must contain 'ML_median', 'ME_median', and 'subclass' columns!")
    
    # Fix to get ME and ML aligned. 
    df['logEseismic'] = 1.5*df['ME_median']+3.2
    df['ME_corrected'] = 0.5 * df['logEseismic'] - 5.0

    subclasses = ['r', 'e', 'l', 'h', 't']
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    fig, ax = plt.subplots(figsize=(10, 8))

    for subclass, color in zip(subclasses, colors):
        df_sub = df[df['subclass'] == subclass]
        x = df_sub['ML_median']
        y = df_sub['ME_corrected']

        # Clean NaNs
        mask = (~x.isna()) & (~y.isna())
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            continue  # Need at least 2 points to fit

        # Fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Scatter plot
        ax.scatter(x, y, label=f"{subclass} (n={len(x)})", s=10, color=color)

        # Plot regression line
        xfit = np.linspace(x.min(), x.max(), 100)
        yfit = slope * xfit + intercept
        ax.plot(xfit, yfit, color=color, linestyle='--', linewidth=2,
                label=f"{subclass} fit: ME={slope:.2f} ML+{intercept:.2f} (R²={r_value**2:.2f})")

    # Style
    ax.set_xlabel('ML_median')
    ax.set_ylabel('ME_median')
    ax.set_title('ML vs ME by Subclass with Linear Fits')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best', ncol=1)
    plt.tight_layout()

    # Save
    output_filename = f"{save_prefix}ML_vs_ME_by_subclass_fit.png"
    fig.savefig(output_filename)
    print(f"Subclass ML vs ME plot saved as {output_filename}")

if __name__ == "__main__":
    csvfile = "Event_Level_ML_ME_summary_1745702673-autosave.csv"
    

if __name__ == "__main__":
    import os
    os.chdir(os.getenv('HOME') + '/Dropbox')
    csvfile = "Event_Level_ML_ME_summary_1745702673-autosave.csv"
    analyze_event_data(csvfile, starttime="1996-01-01", endtime="2008-12-31", binsize='W', save_prefix='ME_new_')
    plot_magnitude_vs_time(csvfile, save_prefix='ME_new_')
    plot_ml_vs_me(csvfile, save_prefix='linregress_')
    plot_ml_vs_me_by_subclass(csvfile, save_prefix='linregress_')