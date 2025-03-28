import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bin_counts(df, bin_edges):
    """
    DEPRECATED: Use EnhancedCatalog.plot_eventrate instead.

    Count the number of events in each bin.
    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'datetime' column.
    bin_edges : array-like
        Edges of time bins (as datetime or matplotlib date numbers).

    Returns
    -------
    counts : np.ndarray
        Count of events per bin.
    """
    warnings.warn("bin_counts is deprecated. Use EnhancedCatalog.plot_eventrate instead.", DeprecationWarning)
    return np.histogram(pd.to_datetime(df['datetime']), bins=bin_edges)[0]


def bin_irregular(df, ycol, bin_edges):
    """
    DEPRECATED: Use DataFrame.resample instead.

    Sum values of ycol in each time bin.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'datetime' and specified ycol.
    ycol : str
        Column name to bin (e.g. 'energy').
    bin_edges : array-like
        Time bins (datetime or mpl datenum).

    Returns
    -------
    binned : np.ndarray
        Summed ycol per bin.
    """
    warnings.warn("bin_irregular is deprecated. Use DataFrame.resample instead.", DeprecationWarning)
    df = df.set_index('datetime')
    df = df.sort_index()
    df["bin"] = pd.cut(df.index, bins=pd.to_datetime(bin_edges))
    return df.groupby("bin")[ycol].sum().values


def plot_counts(df, ax, bin_edges, snum=None, enum=None):
    """
    DEPRECATED: Use EnhancedCatalog.plot_eventrate instead.

    Plot histogram of event counts and cumulative count.

    Parameters
    ----------
    df : pd.DataFrame with 'datetime'
    ax : matplotlib axis
    bin_edges : array-like of datetime-like values
    snum : datetime (optional)
    enum : datetime (optional)
    """
    warnings.warn("plot_counts is deprecated. Use EnhancedCatalog.plot_eventrate instead.", DeprecationWarning)
    df = df.sort_values('datetime')
    time = pd.to_datetime(df['datetime'])
    cumcounts = np.arange(1, len(time)+1)
    binsize_str = f"{pd.to_timedelta(bin_edges[1] - bin_edges[0])}"

    counts, _, _ = ax.hist(time, bins=bin_edges, histtype='bar', color='black')
    ax.grid(True)
    ax.set_ylabel(f"# Earthquakes\n{binsize_str}", fontsize=8)
    if snum and enum:
        ax.set_xlim(snum, enum)

    ax2 = ax.twinx()
    ax2.plot(time, cumcounts, 'g', lw=2.5)
    ax2.set_ylabel("Cumulative\n# Earthquakes", fontsize=8)
    ax2.yaxis.get_label().set_color('g')
    for label in ax2.get_yticklabels():
        label.set_color('g')


def plot_energy(df, ax, bin_edges, snum=None, enum=None):
    """
    DEPRECATED: Use EnhancedCatalog.plot_eventrate instead.

    Plot energy release histogram and cumulative energy.

    Parameters
    ----------
    df : pd.DataFrame with 'datetime' and 'ml'
    ax : matplotlib axis
    bin_edges : array-like
    snum, enum : datetime, optional
    """
    warnings.warn("plot_energy is deprecated. Use EnhancedCatalog.plot_eventrate instead.", DeprecationWarning)
    df = df.copy()
    df = df.sort_values('datetime')
    df['energy'] = 10**(1.5 * df['ml'])
    time = pd.to_datetime(df['datetime'])
    cumenergy = df['energy'].cumsum().values
    binned_energy = bin_irregular(df, 'energy', bin_edges)
    barwidth = pd.to_datetime(bin_edges[1:]).to_numpy() - pd.to_datetime(bin_edges[:-1]).to_numpy()
    binsize_str = f"{pd.to_timedelta(barwidth[0])}"

    ax.bar(bin_edges[:-1], binned_energy, width=barwidth, color='black')
    ax.set_ylabel(f"Energy\n(unit: Ml)\n{binsize_str}", fontsize=8)
    if snum and enum:
        ax.set_xlim(snum, enum)

    ax2 = ax.twinx()
    ax2.plot(time, cumenergy, 'g', lw=2.5)
    ax2.set_ylabel("Cumulative Energy\n(unit: Ml)", fontsize=8)
    ax2.yaxis.get_label().set_color('g')
    for label in ax2.get_yticklabels():
        label.set_color('g')
