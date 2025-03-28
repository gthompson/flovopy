# FLOVOpy: Florida Virtual Volcano Observatory Tools

FLOVOpy is a Python package providing advanced tools for volcano observatories and seismology research. It includes waveform processing utilities, RSAM/DRS computation, spectrogram generation, and enhanced support for legacy datasets such as Seisan files.

---

## Installation (via Conda)

To install FLOVOpy in **editable mode** using a Conda environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/flovopy.git
cd flovopy

# Create and activate a Conda environment
conda create -n flovopy-env python=3.10
conda activate flovopy-env

# Install dependencies
pip install -e .
```

This registers FLOVOpy for local development and enables the `run-iceweb` command-line interface.

---

## Running the IceWeb Pipeline

The `run-iceweb` command processes RSAM, DRS, and spectrogram products from waveform archives.

```bash
run-iceweb \
  --config config/ \
  --start 2023-01-01T00:00:00 \
  --end 2023-01-02T00:00:00 \
  --subnet SHV \
  --inventory metadata/station.xml \
  --trace_ids XX.STA..BHZ XX.STA..HHZ
```

### Arguments
- `--config`: Path to the configuration directory with `.config.csv` files
- `--start`, `--end`: UTC time range in ISO format
- `--subnet`: Label for the virtual network/subnet
- `--trace_ids`: List of N.S.L.C. trace IDs (optional)
- `--inventory`: StationXML file for instrument response correction (optional)

Output products (RSAM, DRS, spectrogram PNGs) are stored under paths defined in the general config file.

---

## Project Structure

```text
flovopy/
├── core/               # Core utilities (inventory, plotting, time, simulation)
├── processing/         # Metrics, spectrograms, RSAM/DRS (SAM)
├── analysis/           # Amplitude-based source location (ASL)
├── seisanio/           # Seisan file support
├── sds/                # SDS archive reader
├── wrappers/           # CLI entry points and scripts
└── obsolete/           # Legacy code modules
```

---

## Upcoming CLI Tools

Additional scripts will be made available via CLI:
- `run-seisan2sds`: Convert Seisan waveform files to SDS format
- `run-fdsn2sds`: Download FDSN waveforms into an SDS archive
- `run-compute-sam`: Compute statistical/amplitude metrics (SAM)

---

## Requirements
- Python 3.8 or later
- Conda environment (recommended)
- Dependencies:
  - `obspy`
  - `numpy`
  - `pandas`

---

## License
FLOVOpy is released under the MIT License.

---

## Acknowledgments
FLOVOpy is developed at the University of South Florida as part of the Florida Virtual Volcano Observatory (FLO-VO) initiative, with contributions from volcano seismologists, geophysicists, and observatory partners.

