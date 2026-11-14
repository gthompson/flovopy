"""Optional plotting without global backend changes or implicit file writes."""
import numpy as np

def plot_series(dataset, *, series=None, columns=None, connect=False):
    """Independent axes per positional value; missing values remain gaps."""
    import matplotlib.pyplot as plt
    frame=dataset.data
    if series is not None:frame=frame[frame['series'].eq(series)]
    elif frame['series'].nunique()>1:raise ValueError('Select one series explicitly; unrelated sources are not merged for plotting.')
    columns=columns or [c for c in frame if c.startswith('value_') and not c.endswith(('_raw','_missing'))]
    if frame.empty or not columns:raise ValueError('No selected value records to plot.')
    frame=frame.sort_values('recorded_time')
    fig,axes=plt.subplots(len(columns),1,sharex=True,squeeze=False,figsize=(11,max(3,2*len(columns))))
    for ax,col in zip(axes[:,0],columns):
        ax.plot(frame['recorded_time'],frame[col],linewidth=.7,
                linestyle='-' if connect else 'None', marker=None if connect else '.', markersize=2)
        ax.set_ylabel(col+'\nsource units');ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Recorded calendar label; no clock correction')
    return fig

def plot_ssam(dataset, *, series, interval_seconds, start=None, end=None, ax=None):
    """Draw individual band/time cells, leaving absent rows as gaps.

    Caller supplies display interval; band positions are indices unless explicit
    frequency boundaries were passed to read_ssam. No log/unit conversion occurs.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import PolyCollection
    if interval_seconds<=0:raise ValueError('interval_seconds must be positive.')
    frame=dataset.data[dataset.data['series'].eq(series)].copy()
    if start is not None:frame=frame[frame.recorded_time>=pd.Timestamp(start)]
    if end is not None:frame=frame[frame.recorded_time<=pd.Timestamp(end)]
    frame=frame[frame.recorded_time.notna()].sort_values('recorded_time')
    if frame.empty:raise ValueError('No selected SSAM records.')
    if frame.recorded_time.duplicated().any():raise ValueError('Resolve duplicate timestamps explicitly before plotting SSAM.')
    bands=dataset.metadata['bands'];edges=dataset.metadata.get('band_edges_hz',list(range(bands+1)))
    vertices=[];colors=[]
    for row in frame.itertuples(index=False):
        x=mdates.date2num(row.recorded_time.to_pydatetime());dx=interval_seconds/86400
        for i in range(bands):
            value=getattr(row,f'value_{i+1:02d}')
            if not np.isfinite(value):continue
            vertices.append([(x,edges[i]),(x+dx,edges[i]),(x+dx,edges[i+1]),(x,edges[i+1])]);colors.append(value)
    if not vertices:raise ValueError('No finite selected band values.')
    if ax is None:_,ax=plt.subplots(figsize=(12,4))
    collection=PolyCollection(vertices,array=np.asarray(colors),edgecolors='none')
    ax.add_collection(collection);ax.autoscale_view();ax.xaxis_date()
    ax.set_ylabel('Frequency (Hz)' if 'band_edges_hz' in dataset.metadata else 'Positional band index')
    ax.set_xlabel('Recorded calendar label; cell width explicitly supplied')
    return ax,collection
