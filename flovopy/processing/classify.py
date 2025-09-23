from __future__ import annotations

import json
import numpy as np
import pandas as pd

from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import Event, Origin, Magnitude, CreationInfo, Comment

# Local imports (names based on your refactor)
from flovopy.enhanced.stream import EnhancedStream
from flovopy.enhanced.catalog import EnhancedCatalog
from flovopy.core.trace_qc import estimate_snr


# --------------------------- Feature Extraction ---------------------------

class EventFeatureExtractor:
    """
    Extracts a compact feature vector per event from an EnhancedCatalog.

    Features (when available):
      - spectral: peakf, meanf (from a preferred Z component)
      - time: duration (from trigger dict)
      - shape: skewness, kurtosis (same preferred component)
      - station-level: pgv_vec, pgd_vec, pga_vec, pap, pap_bp_1_20
      - rsam_score: optional, if rsamObj is provided (heuristic timing score)

    Notes
    -----
    * We pick a preferred vertical trace (component 'Z','1','U'… if present).
      If multiple, we pick the one with the highest SNR (std method) over the event
      window if a trigger is supplied; else we fall back to first match.
    * station-level metrics are read from `EnhancedStream.station_metrics`
      if the stream was processed with `ampengfft(..., station_level=True)`.
    """
    def __init__(self, catalog: EnhancedCatalog, rsamObj=None):
        self.catalog = catalog
        self.rsamObj = rsamObj

    # --- helpers ---

    @staticmethod
    def _get_subclass_label(event_record) -> str | None:
        for c in event_record.event.comments:
            # support both "subclass: X" and JSON dicts in comments
            txt = c.text or ""
            if "subclass:" in txt:
                return txt.split(":", 1)[-1].strip()
            # JSON blob case:
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict) and "subclass" in obj:
                    return obj.get("subclass")
            except Exception:
                pass
        return None

    @staticmethod
    def _pick_preferred_vertical(stream: Stream, trigger: dict | None):
        """
        Pick a preferred vertical component trace.
        Priority by component letter: Z > 1 > U > (fallback *Z-like*)
        If trigger exists with duration, use SNR over the event window to pick the best.
        """
        if not stream:
            return None

        # collect candidates by component
        candidates = []
        for tr in stream:
            comp = (tr.stats.channel[-1] if tr.stats.channel else "").upper()
            if comp in ("Z", "1", "U"):
                candidates.append(tr)

        if not candidates:
            # any vertical-ish fallback (e.g., 'Z' missing but single component present)
            candidates = list(stream)

        if not candidates:
            return None

        # If we can, pick by SNR within the trigger window
        if trigger and "duration" in trigger and "time" in trigger:
            t_on = trigger["time"]
            dur = float(trigger["duration"]) or 0.0
            if isinstance(t_on, UTCDateTime) and dur > 0:
                best = None
                best_snr = -np.inf
                for tr in candidates:
                    snr, *_ = estimate_snr(
                        tr, method="std", window_length=dur,
                        split_time=(t_on, t_on + dur), verbose=False
                    )
                    if np.isfinite(snr) and snr > best_snr:
                        best_snr, best = snr, tr
                if best is not None:
                    return best

        # else: simple priority order Z > 1 > U > first
        def priority(tr):
            comp = (tr.stats.channel[-1] if tr.stats.channel else "").upper()
            return {"Z": 0, "1": 1, "U": 2}.get(comp, 9)

        candidates.sort(key=priority)
        return candidates[0]

    # --- main API ---

    def extract_features(self) -> pd.DataFrame:
        rows = []

        for rec in self.catalog.records:
            feat = {
                "event_id": rec.event.resource_id.id if rec.event.resource_id else None
            }

            # preferred vertical trace for spectral/shape
            trZ = self._pick_preferred_vertical(rec.stream, rec.trigger) if rec.stream else None
            if trZ is not None and hasattr(trZ.stats, "metrics"):
                m = trZ.stats.metrics
                feat["peakf"]    = m.get("peakf")
                feat["meanf"]    = m.get("meanf")
                feat["skewness"] = m.get("skewness")
                feat["kurtosis"] = m.get("kurtosis")

            # duration (from trigger dict)
            if rec.trigger:
                feat["duration"] = rec.trigger.get("duration")

            # station-level vector PGMs & PAP (if computed)
            if rec.stream and hasattr(rec.stream, "station_metrics") and rec.stream.station_metrics is not None:
                try:
                    sm = rec.stream.station_metrics
                    # If single station per event (common), just take the first row;
                    # else: we can summarize (mean/median). For now, median across stations:
                    def med(col):
                        vals = pd.to_numeric(sm[col], errors="coerce")
                        return float(np.nanmedian(vals)) if len(vals) else np.nan
                    feat["pgv_vec"] = med("pgv_vec")
                    feat["pgd_vec"] = med("pgd_vec")
                    feat["pga_vec"] = med("pga_vec")
                    feat["pap_med"]     = med("pap")
                    feat["pap_bp_1_20"] = med("pap_bp_1_20")
                except Exception:
                    pass

            # optional RSAM timing score
            if self.rsamObj is not None:
                rscore = 0
                try:
                    for seed_id, df2 in getattr(self.rsamObj, "dataframes", {}).items():
                        if df2 is None or df2.empty or "median" not in df2.columns:
                            continue
                        maxi = int(np.nanargmax(df2["median"]))
                        r = maxi / float(len(df2))
                        if 0.3 < r < 0.7:
                            rscore += 3
                        elif r < 0.3:
                            rscore += 1
                        else:
                            rscore -= 2
                except Exception:
                    pass
                feat["rsam_score"] = rscore

            # label (if present)
            feat["subclass"] = self._get_subclass_label(rec)

            rows.append(feat)

        return pd.DataFrame(rows)


# --------------------------- Simple Classifier ---------------------------

class VolcanoEventClassifier:
    """
    Minimal rule-based classifier.
    Labels: r (rockfall), e (explosion), l (LP), h (hybrid), t (VT/tectonic)
    """
    def __init__(self):
        self.classes = ['r', 'e', 'l', 'h', 't']

    def classify(self, f: dict) -> tuple[str, dict]:
        score = {k: 0.0 for k in self.classes}

        # --- spectral cues ---
        peakf = f.get("peakf")
        meanf = f.get("meanf")
        if np.isfinite(peakf) if peakf is not None else False:
            if peakf < 1.0:
                score['l'] += 2; score['e'] += 1
            elif peakf > 5.0:
                score['t'] += 2
            else:
                score['h'] += 1
        if np.isfinite(meanf) if meanf is not None else False:
            if meanf < 1.0:
                score['l'] += 2
            elif meanf > 4.0:
                score['t'] += 2

        # --- shape cues ---
        if f.get('skewness', 0) and f['skewness'] > 2.0:
            score['r'] += 2
        if f.get('kurtosis', 0) and f['kurtosis'] > 10:
            score['t'] += 2; score['r'] += 1

        # --- duration cue ---
        d = f.get('duration')
        if np.isfinite(d) if d is not None else False:
            d = float(d)
            if d < 5:
                score['t'] += 4; score['h'] += 3; score['r'] -= 5; score['e'] -= 10
            elif d < 10:
                score['t'] += 3; score['h'] += 2; score['e'] -= 2
            elif d < 20:
                score['t'] += 2; score['h'] += 3; score['l'] += 2
            elif d < 30:
                score['h'] += 2; score['l'] += 3; score['r'] += 1
            elif d < 40:
                score['h'] += 1; score['l'] += 2; score['r'] += 2
            else:
                score['r'] += 3; score['e'] += 5

        # --- optional PGMs/PAP nudges (very light touch) ---
        # Large vector PGMs push towards 'e'/'r', very small towards 'l'
        for k in ("pgv_vec", "pga_vec", "pgd_vec", "pap_med"):
            v = f.get(k)
            if v is None or not np.isfinite(v): 
                continue
            if v > 0:
                if k in ("pgd_vec", "pgv_vec"): score['e'] += 0.5
                if k == "pga_vec": score['r'] += 0.3
                if v < 1e-6: score['l'] += 0.5  # tiny motions -> LP-ish

        # normalize to pseudo-probabilities
        total = sum(max(0.0, v) for v in score.values())
        if total > 0:
            for k in score:
                score[k] = max(0.0, score[k]) / total

        subclass = max(score, key=score.get)
        return subclass, score


# --------------------------- Classification Wrapper ---------------------------

def classify_and_add_event(
    stream: Stream | EnhancedStream,
    catalog: EnhancedCatalog,
    trigger: dict | None = None,
    *,
    save_stream: bool = False,
    classifier: VolcanoEventClassifier | None = None,
    # controls for per-trace metrics
    differentiate: bool = False,
    compute_spectral: bool = True,
    compute_ssam: bool = False,
    compute_bandratios: bool = False,
    td_band_pairs: list[tuple[float,float,float,float]] | None = None,
    # magnitudes on/off (leave False if you haven’t attached coords yet)
    compute_magnitudes: bool = False,
    magnitude_kwargs: dict | None = None,
) -> tuple[Event, str, dict]:
    """
    Compute metrics, (optionally) magnitudes, classify, and add event to catalog.

    - Runs EnhancedStream.ampengfft() once (fast path), with optional time-domain band features.
    - If compute_magnitudes=True, call stream.compute_station_magnitudes(**magnitude_kwargs).
      (Requires station coordinates on each trace + source_coords, etc.)
    - Uses the rule-based VolcanoEventClassifier to assign subclass and stores the
      score vector as a JSON comment in the Event.

    Returns
    -------
    (event, subclass, features)
    """
    if not isinstance(stream, EnhancedStream):
        stream = EnhancedStream(stream=stream)

    # 1) per-trace metrics
    stream.ampengfft(
        differentiate=differentiate,
        compute_spectral=compute_spectral,
        compute_ssam=compute_ssam,
        compute_bandratios=compute_bandratios,
        threshold=0.707, window_length=9, polyorder=2,
        td_band_pairs=td_band_pairs,
    )

    # station-level summary (vector PGMs + PAP medians)
    if not hasattr(stream, "station_metrics") or stream.station_metrics is None:
        # In your current implementation, ampengfft() already invokes this.
        # Keeping it safe here:
        try:
            stream.station_metrics = stream._station_level_metrics()
        except Exception:
            pass

    # 2) (optional) magnitudes (needs coords + source info)
    if compute_magnitudes:
        magnitude_kwargs = magnitude_kwargs or {}
        stream.compute_station_magnitudes(**magnitude_kwargs)

    # 3) extract features (single row)
    extractor = EventFeatureExtractor(
        EnhancedCatalog(events=[Event()], records=[catalog.records[0]])  # dummy catalog not used
        if len(catalog.records) == 0 else EnhancedCatalog(events=[catalog.records[-1].event],
                                                          records=[catalog.records[-1]])
    )
    # Use a tiny internal method rather than building an ad-hoc catalog wrapper:
    # We’ll mimic the extractor logic against this single stream/trigger.
    features: dict = {}
    trZ = EventFeatureExtractor._pick_preferred_vertical(stream, trigger)
    if trZ is not None and hasattr(trZ.stats, "metrics"):
        m = trZ.stats.metrics
        for k in ("peakf", "meanf", "skewness", "kurtosis"):
            if k in m:
                features[k] = m[k]
    if trigger and "duration" in trigger:
        features["duration"] = trigger["duration"]

    if hasattr(stream, "station_metrics") and stream.station_metrics is not None:
        sm = stream.station_metrics
        for col, key in (("pgv_vec", "pgv_vec"), ("pgd_vec","pgd_vec"),
                         ("pga_vec","pga_vec"), ("pap","pap_med"), ("pap_bp_1_20","pap_bp_1_20")):
            try:
                vals = pd.to_numeric(sm[col], errors="coerce")
                features[key] = float(np.nanmedian(vals)) if len(vals) else np.nan
            except Exception:
                pass

    # 4) classify
    clf = classifier or VolcanoEventClassifier()
    subclass, score = clf.classify(features)

    # 5) build Event (origin time = stream start)
    origin_time = stream[0].stats.starttime if len(stream) else UTCDateTime()
    ev = Event(
        origins=[Origin(time=origin_time)],
        creation_info=CreationInfo(author="classify_and_add_event")
    )
    # Optional: include a magnitude if compute_magnitudes=True and you computed a network one.
    # (You can also fetch from stream.magnitudes2dataframe if desired.)

    # Store classifier outputs into event comments
    ev.comments.append(Comment(text=json.dumps({"classifier": "rule_v1", "scores": score}, indent=2)))
    ev.comments.append(Comment(text=json.dumps({"mainclass": "LV", "subclass": subclass}, indent=2)))

    # 6) add to catalog (and optionally save the stream)
    catalog.addEvent(
        ev,
        stream=stream if save_stream else None,
        trigger=trigger,
        classification=subclass,
        event_type="volcanic eruption",
        mainclass="LV",
        subclass=subclass,
    )

    return ev, subclass, features


'''
ev, label, feats = classify_and_add_event(
    stream, cat, trigger=trig, save_stream=True,
    compute_magnitudes=True,
    magnitude_kwargs=dict(
        inventory=inv,
        source_coords={"latitude":lat, "longitude":lon, "depth":depth_m},
        model="body", Q=50, c_earth=2500, correction=3.7,
        a=1.6, b=-0.15, g=0,
        attach_coords=True, compute_distances=True,
    )
)

'''