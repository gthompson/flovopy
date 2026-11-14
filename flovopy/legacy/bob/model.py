"""Containers preserve reported values separately from usable measurements."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import pandas as pd

@dataclass
class BOBDataset:
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)

    def coverage(self):
        """Rows with >=1 usable value; not uptime, calibration or valid signal."""
        columns = [c for c in self.data if c.startswith('value_') and not c.endswith(('_raw', '_missing'))]
        if self.data.empty or not columns:
            return pd.DataFrame(columns=['series', 'records', 'nonmissing_records', 'first', 'last'])
        frame = self.data.copy()
        frame['_usable'] = frame[columns].notna().any(axis=1)
        rows = []
        for series, group in frame.groupby('series', dropna=False):
            times = group.loc[group['_usable'], 'recorded_time'].dropna()
            rows.append(dict(series=series, records=len(group), nonmissing_records=int(group['_usable'].sum()),
                             first=times.min() if len(times) else pd.NaT,
                             last=times.max() if len(times) else pd.NaT,
                             unique_nonmissing_timestamps=times.nunique()))
        return pd.DataFrame(rows)

    def write(self, output_dir, *, prefix='bob', overwrite=False):
        """Explicit CSV/JSON export; no output paths or side effects on read."""
        output = Path(output_dir)
        names = {key: output / (prefix + suffix) for key, suffix in
                 [('records', '_records.csv'), ('metadata', '_metadata.json'), ('errors', '_errors.csv')]}
        if not overwrite and any(path.exists() for path in names.values()):
            raise FileExistsError('BOB output already exists; select another prefix or explicitly allow overwrite.')
        output.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(names['records'], index=False)
        self.errors.to_csv(names['errors'], index=False)
        names['metadata'].write_text(json.dumps(self.metadata, indent=2, default=str) + '\n')
        return names
