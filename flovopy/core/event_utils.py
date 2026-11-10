"""Small, dataset-independent utilities for event metadata and time matching."""
from __future__ import annotations

import math
import pandas as pd

__all__ = ['parse_seed_id', 'get_comment_value', 'associate_events_by_time']


def parse_seed_id(seed_id):
    """Split NET.STA.LOC.CHA, allowing empty fields; return four Nones otherwise.

    This checks the four-field structure only, not SEED code semantics.
    It does not supply a network, orientation, gain or other defaults.
    """
    if not isinstance(seed_id, str) or not seed_id.strip():
        return None, None, None, None
    parts = seed_id.split('.')
    return tuple(parts) if len(parts) == 4 else (None, None, None, None)


def get_comment_value(event, prefix):
    """Return the first stripped, case-sensitive comment value matching prefix.

    The caller supplies the complete prefix, usually including a delimiter
    such as 'event_class:'. Return None when no comment matches.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError('prefix must be a nonempty string')
    for comment in event.comments or []:
        text = (comment.text or '').strip()
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return None


def associate_events_by_time(
    left, right, *, left_id_col, left_time_col, right_id_col, right_time_col,
    tolerance_seconds=10.0, policy='one_to_one',
):
    """Associate two event tables within an inclusive time tolerance.

    Return columns left_id, left_time, right_id, right_time, abs_dt_s.
    Unmatched rows are omitted. IDs must be unique, non-null and hashable
    within each table; output preserves their values. Missing times are
    excluded; invalid non-null timestamps raise. Naive timestamps are
    interpreted as UTC. Inputs are copied, never modified.

    one_to_one: greedily accept pairs by increasing absolute time difference,
    then left time, right time and ID sort keys. Each ID is used at most once.
    This is deterministic, but is not a globally optimal assignment.

    nearest: choose the nearest right row for each left row, permitting reuse.
    Equidistant times prefer the earlier right time. Equal-time right rows
    retain input order (merge_asof's backward match takes the last such row).

    Candidate enumeration for one_to_one can be expensive with dense tables
    and large tolerances. Choose a tolerance appropriate to the dataset.
    """
    if policy not in ('one_to_one', 'nearest'):
        raise ValueError("policy must be 'one_to_one' or 'nearest'")
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError('tolerance_seconds must be finite and nonnegative')
    columns = ['left_id', 'left_time', 'right_id', 'right_time', 'abs_dt_s']

    def prepare(table, id_col, time_col, side):
        if id_col == time_col:
            raise ValueError('ID and time columns must differ')
        missing = {id_col, time_col} - set(table.columns)
        if missing:
            raise ValueError(f'Missing {side} columns: {sorted(missing)}')
        ids = table[id_col]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f'{side} IDs must be unique and non-null')
        for value in ids:
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(f'{side} IDs must be hashable') from exc
        times = pd.to_datetime(table[time_col], format='mixed', utc=True, errors='raise')
        times = times.astype('datetime64[ns, UTC]')
        result = pd.DataFrame({f'{side}_id': ids.to_numpy(copy=True),
                               f'{side}_time': times.to_numpy(copy=True)})
        return result.dropna(subset=[f'{side}_time']).sort_values(
            f'{side}_time', kind='stable').reset_index(drop=True)

    w = prepare(left, left_id_col, left_time_col, 'left')
    h = prepare(right, right_id_col, right_time_col, 'right')
    if w.empty or h.empty:
        return pd.DataFrame(columns=columns)
    tol = pd.Timedelta(seconds=tolerance_seconds)
    if policy == 'nearest':
        # Object dtype prevents missing matches coercing large integer IDs to float.
        h['right_id'] = h['right_id'].astype(object)
        result = pd.merge_asof(w, h, left_on='left_time', right_on='right_time',
                               tolerance=tol, direction='nearest')
        result = result.dropna(subset=['right_id']).copy()
        result['abs_dt_s'] = (result.left_time - result.right_time).abs().dt.total_seconds()
        return result[columns].reset_index(drop=True)

    candidates = []
    for row in w.itertuples(index=False):
        start = h.right_time.searchsorted(row.left_time - tol, side='left')
        stop = h.right_time.searchsorted(row.left_time + tol, side='right')
        for other in h.iloc[start:stop].itertuples(index=False):
            candidates.append(dict(left_id=row.left_id, left_time=row.left_time,
                                   right_id=other.right_id, right_time=other.right_time,
                                   abs_dt_s=abs((row.left_time-other.right_time).total_seconds())))
    if not candidates:
        return pd.DataFrame(columns=columns)
    cand = pd.DataFrame(candidates)
    # String IDs retain lexical ordering; mixed ID types have explicit sort keys.
    for side in ('left', 'right'):
        cand[f'_{side}_key'] = cand[f'{side}_id'].map(
            lambda value: (type(value).__name__, value) if isinstance(value, (str, int, float))
            else (type(value).__name__, repr(value)))
    cand = cand.sort_values(['abs_dt_s', 'left_time', 'right_time', '_left_key', '_right_key'], kind='stable')
    used_left, used_right, accepted = set(), set(), []
    for row in cand.to_dict('records'):
        if row['left_id'] in used_left or row['right_id'] in used_right:
            continue
        used_left.add(row['left_id']); used_right.add(row['right_id'])
        accepted.append({key: row[key] for key in columns})
    return pd.DataFrame(accepted, columns=columns)
