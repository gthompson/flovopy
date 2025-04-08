import os
from obspy import Catalog

class EnhancedEvent:
    def __init__(self, obspy_event, metrics=None, sfile_path=None, wav_paths=None,
                 aef_path=None, trigger_window=None, average_window=None, stream=None):
        """
        Enhanced representation of a seismic event.

        Parameters
        ----------
        obspy_event : obspy.core.event.Event
            Core ObsPy Event object.
        metrics : dict, optional
            Additional metrics or classifications (e.g., peakf, energy, subclass).
        sfile_path : str, optional
            Path to original SEISAN S-file.
        wav_paths : list of str, optional
            Paths to one or two associated waveform files.
        aef_path : str, optional
            Path to AEF file (if external).
        trigger_window : float, optional
            Trigger window duration in seconds.
        average_window : float, optional
            Averaging window duration in seconds.
        stream : obspy.Stream or EnhancedStream, optional
            Associated waveform stream.
        """
        self.event = obspy_event
        self.metrics = metrics or {}
        self.sfile_path = sfile_path
        self.wav_paths = wav_paths or []
        self.aef_path = aef_path
        self.trigger_window = trigger_window
        self.average_window = average_window
        self.stream = stream

    def to_quakeml(self):
        return self.event

    def to_json(self):
        return {
            "event_id": str(self.event.resource_id),
            "sfile_path": self.sfile_path,
            "wav_paths": self.wav_paths,
            "aef_path": self.aef_path,
            "trigger_window": self.trigger_window,
            "average_window": self.average_window,
            "metrics": self.metrics,
        }

    def save(self, outdir, base_name):
        """
        Save to QuakeML and JSON.

        Parameters
        ----------
        outdir : str
            Base directory to save outputs.
        base_name : str
            Base filename (no extension).
        """


        qml_path = os.path.join(outdir, base_name + ".qml")
        json_path = os.path.join(outdir, base_name + ".json")

        Catalog(events=[self.event]).write(qml_path, format="QUAKEML")

        with open(json_path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)


