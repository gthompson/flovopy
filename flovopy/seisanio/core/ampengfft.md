# ampengfft.c — Summary of Function and Output

## Overview

The `ampengfft.c` program processes SEISAN-format `.wav` files for seismic events and computes amplitude, energy, and spectral metrics. It was originally known as `wav2aef` and was used at the Montserrat Volcano Observatory (MVO) to populate `.AEF` files or insert `VOLC` lines directly into `.S` files.

## Main Features

- Reads waveform data from SEISAN `.wav` files linked via `.S` files.
- Applies a gain_factor from the SEISAN header (m/s/count) to convert a raw trace to a velocity seismogram, but without applying a full instrument correction with poles and zeros
- For each station-channel:
  - Applies a sliding window to compute **maximum average amplitude**.
  - Performs an **FFT** on the detrended signal.
  - Computes **total energy** and **spectral energy distribution** in pre-defined frequency bands.
  - Identifies the **frequency of maximum amplitude**.
- Writes structured results to the event S-file or a separate `.AEF` file.

## Frequency Bands

The FFT results are divided into **11 frequency bands** using these fixed edges:

```
0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 30.0 Hz
```

This produces **11 bands** between the 12 edges:

| Band | Frequency Range (Hz) |
|------|----------------------|
| 1    | 0.1 – 1.0            |
| 2    | 1.0 – 2.0            |
| ...  | ...                  |
| 11   | 10.0 – 30.0          |

## Output Metrics

For each valid channel, the following metrics are extracted:

- **Max average amplitude** (`amp`): Sliding window maximum (in m/s)
- **Total energy** (`energy`): Time-domain or frequency-domain estimate (in J/kg)
- **Peak frequency** (`maxf`): Frequency with highest FFT amplitude (in Hz)
- **Spectral slice percentages** (`ssam`): Energy fraction in each frequency band (11 values)

## Output Formats

Results are written in one of two ways:

1. **S-File (VOLC lines)**:
   - Inserted above the phase arrival section.
   - Can be parsed using `AEFfile.from_sfile()`.

2. **.AEF Files**:
   - Saved alongside the original waveform files.
   - Each line corresponds to a trace and includes the above metrics.

## Notes

- The values in the `ssam` (spectral slice) array represent **energy fractions**, **not amplitudes**.
- The average window, pre-trigger, and post-trigger durations are read from a config file (e.g., `/live/ampengfft.d`).

## Python Reimplementation Notes

A Python version of `ampengfft.c` can be implemented using ObsPy. Key design elements:

- Uses `ObsPy.Stream` and `ObsPy.Trace` to load and process waveforms.
- Applies optional `tr.stats.calib` to scale raw data (e.g., to m/s).
- FFT is computed using NumPy and amplitudes are extracted.
- Band edges are preserved (not center frequencies), matching the original slice layout.
- Metrics like `peakamp`, `energy`, and `maxf` are stored in `tr.stats.metrics`.
- Optionally writes `.AEF`-formatted output for direct comparison with legacy files.
- Designed to be called as a method from `EnhancedStream` (e.g., `stream.write_aef(...)`).
