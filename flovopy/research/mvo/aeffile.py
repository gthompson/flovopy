"""
AEFfile: Parser for SEISAN amplitude-energy-frequency (AEF) files.

AEF files contain summary metrics computed by the legacy `ampengfft` C
program, widely used in the analyst workflow at the Montserrat Volcano
Observatory (MVO).

Each AEF file corresponds to a SEISAN-format event waveform file and may
contain, per station/channel:

- Maximum average amplitude
- Total signal energy
- Dominant frequency
- SSAM-style spectral percentages in predefined frequency bands

AEF information may appear either:
1. in standalone `.AEF` files, or
2. embedded as `VOLC` lines in `.S` files.

This class parses both cases into structured dictionaries.
"""

from __future__ import annotations

from pathlib import Path
import re
from pprint import pformat
from typing import Optional

from obspy import read

from flovopy.research.mvo.mvo_ids import correct_nslc_mvo, legacy_or_not
from flovopy.seisanio.core.wavfile import wavpath2datetime
from flovopy.seisanio.utils.helpers import find_matching_wavfiles, spath2datetime


class AEFfile:
    """
    Parser for standalone AEF files and embedded VOLC lines in Seisan S-files.
    """

    DEFAULT_FREQUENCY_BANDS = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                               7.0, 8.0, 9.0, 10.0, 30.0]

    def __init__(self, path: str | Path | None = None, filetime=None, verbose: bool = False):
        self.path = str(path).strip() if path is not None else None
        self.filetime = filetime
        self.verbose = verbose

        self.trigger_window: Optional[float] = None
        self.average_window: Optional[float] = None

        self.wavpath: Optional[str] = None
        self.wav_ids: list[str] = []
        self.aef_ids: list[str] = []
        self.aefrows: list[dict] = []

        self.legacy: Optional[bool] = None
        self.network: Optional[str] = None

        if not self.path:
            return

        self.legacy, self.network = legacy_or_not(self.path)
        self._infer_context()
        self._load_wav_ids()
        self._parse_file()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _infer_context(self) -> None:
        """
        Infer filetime and wavpath from the input path.
        """
        p = self.path.upper()

        if "AEF" in p:
            self.filetime = self.filetime or wavpath2datetime(self.path)
            self.wavpath = self.path.replace("AEF", "WAV").replace("aef", "")
        elif "REA" in p:
            self.filetime = self.filetime or spath2datetime(self.path)
            matching_wavfiles = find_matching_wavfiles(self.filetime, self.path, y2kfix=False)
            if matching_wavfiles:
                self.wavpath = matching_wavfiles[0]

    def _load_wav_ids(self) -> None:
        """
        Read the associated waveform file, if available, and collect fixed trace IDs.
        """
        if not self.wavpath:
            return

        wavpath = Path(self.wavpath)
        if not wavpath.is_file():
            return

        try:
            st = read(str(wavpath))
        except Exception as e:
            self._log(f"[WARN] Could not read waveform file {wavpath}: {e}")
            return

        for tr in st:
            try:
                tr.id = correct_nslc_mvo(tr.id, tr.stats.sampling_rate)
                self.wav_ids.append(tr.id)
            except Exception as e:
                self._log(f"[WARN] Could not normalize waveform id {tr.id}: {e}")

    def _parse_file(self) -> None:
        """
        Parse the AEF/S-file content.
        """
        if not self.path or not Path(self.path).exists():
            self._log(f"[WARN] {self.path} does not exist")
            return

        try:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError as e:
            self._log(f"[ERROR] Could not read {self.path}: {e}")
            return

        for raw_line in lines:
            if self.verbose:
                print(raw_line.rstrip())

            if len(raw_line) < 2:
                continue

            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            # Some embedded S-file lines are tagged with trailing '3'
            if len(line) >= 80 and line[-1] == "3":
                pass
            elif "VOLC" not in line.upper():
                continue

            lower = line.lower()

            if "trigger window" in lower:
                self.trigger_window = self._extract_window(line, "trigger window")
                continue

            if "average window" in lower:
                self.average_window = self._extract_window(line, "average window")
                continue

            stripped = line.lstrip()
            if "VOLC MAIN" in stripped.upper():
                continue
            if not stripped.startswith("VOLC"):
                continue

            row = self.parse_aefline(stripped)
            if row:
                self.aef_ids.append(row["fixed_id"])
                self.aefrows.append(row)

    def _extract_window(self, line: str, keyword: str) -> Optional[float]:
        """
        Extract a floating-point window value from a line like:
            '... trigger window = 4.0 s ...'
        """
        i_start = line.lower().find(keyword)
        if i_start < 0:
            return None

        substring = line[i_start:i_start + 40]
        eq_index = substring.find("=")
        s_index = substring.find("s")
        if eq_index == -1 or s_index == -1:
            return None

        valstr = substring[eq_index + 1:s_index].strip()
        try:
            return float(valstr)
        except ValueError:
            self._log(f"[WARN] Could not parse {keyword} in line: {line.strip()}")
            return None

    def _guess_sampling_rate(self) -> float:
        """
        Heuristic sampling rate for fixing MVO IDs when wav_ids are unavailable.
        """
        if self.legacy:
            return 100.0
        if self.filetime and self.filetime.year <= 2004:
            return 75.0
        return 100.0

    def _match_fixed_id_to_wav_ids(self, fixed_id: str) -> str:
        """
        If a normalized id can be matched to an actual waveform id, prefer that.
        """
        if not self.wav_ids:
            return fixed_id

        try:
            _, fsta, _, fchan = fixed_id.split(".")
        except ValueError:
            return fixed_id

        for wid in self.wav_ids:
            try:
                _, tsta, _, tchan = wid.split(".")
            except ValueError:
                continue
            if fsta == tsta and fchan and tchan and fchan[-1] == tchan[-1]:
                return wid

        return fixed_id

    # ------------------------------------------------------------------
    # Public parsing methods
    # ------------------------------------------------------------------

    def parse_aefline(self, line: str) -> Optional[dict]:
        """
        Parse one VOLC AEF line into a dictionary.
        """
        content = line.split("VOLC ", 1)[-1]

        try:
            station = content[0:4].strip()
            channel = content[5:9].strip()
            tail = content[9:]

            a_idx = tail.find("A")
            e_idx = tail.find("E")
            f_idx = tail.find("F")

            if min(a_idx, e_idx, f_idx) < 0:
                raise ValueError("Missing A/E/F markers")

            amplitude = float(tail[a_idx + 1:e_idx].strip())
            energy = float(tail[e_idx + 1:f_idx].strip())

            trace_id = f"MV.{station}..{channel}"
            sr = self._guess_sampling_rate()

            if self.legacy:
                fixed_id = correct_nslc_mvo(trace_id, sr, short_period=True)
            else:
                fixed_id = correct_nslc_mvo(trace_id, sr, short_period=None)

            fixed_id = self._match_fixed_id_to_wav_ids(fixed_id)

            ssam = self.parse_ssam(tail, energy, startindex=f_idx + 1)

            maxf = None
            match = re.search(r"(\d+\.\d+)\s*3?$", tail)
            if match:
                maxf = float(match.group(1))

            return {
                "station": station,
                "channel": channel,
                "id": trace_id,
                "fixed_id": fixed_id,
                "amplitude": amplitude,
                "energy": energy,
                "ssam": ssam,
                "maxf": maxf,
            }

        except Exception as e:
            self._log(f"[ERROR] Failed to parse AEF line:\n  {line.strip()}\n  {e}")
            return None

    def parse_ssam(self, line: str, energy: float, startindex: int) -> dict:
        """
        Extract SSAM percentages and convert them to energy by band.
        """
        bands = list(self.DEFAULT_FREQUENCY_BANDS)
        n_bins = len(bands) - 1

        percentages: list[int] = []
        energies: list[float] = []

        while startindex < len(line) and len(percentages) < n_bins:
            valstr = line[startindex:startindex + 3].strip()
            if "." not in valstr:
                try:
                    val = int(valstr)
                    percentages.append(val)
                    energies.append(val / 100.0 * energy)
                except ValueError:
                    break
            startindex += 3

        return {
            "frequency_bands": bands,
            "percentages": percentages,
            "energies": energies,
        }

    # ------------------------------------------------------------------
    # Serialization / display
    # ------------------------------------------------------------------

    def to_dict(self, include_wav_ids: bool = False, include_aef_ids: bool = False) -> dict:
        """
        Return a structured dictionary representation.
        """
        out = {
            "path": self.path,
            "filetime": str(self.filetime) if self.filetime else None,
            "wavpath": self.wavpath,
            "legacy": self.legacy,
            "network": self.network,
            "trigger_window": self.trigger_window,
            "average_window": self.average_window,
            "trace_metrics": self.aefrows,
        }
        if include_wav_ids:
            out["wav_ids"] = self.wav_ids
        if include_aef_ids:
            out["aef_ids"] = self.aef_ids
        return out

    def __str__(self) -> str:
        """
        Pretty-printed summary string.
        """
        header = f"AEFfile object: {self.path}\n" if self.path else "AEFfile object\n"
        try:
            return header + pformat(self.to_dict(), indent=4)
        except Exception as e:
            return header + f"[ERROR] Could not generate summary: {e}"