# FLOVOpy: Florida Virtual Volcano Observatory Tools

FLOVOpy is a Python package providing advanced tools for volcano observatories and seismology research. It includes waveform processing utilities, RSAM/DRS computation, spectrogram generation, amplitude source location (ASL), and enhanced support for legacy datasets such as Seisan files.

---

## 📦 Installation (via Conda)

To install FLOVOpy in **editable mode** using a Conda environment:

```bash
# Clone the repository
git clone https://github.com/gthompson/flovopy.git
cd flovopy

# Create and activate a Conda environment
conda create -n flovopy-env python=3.10
conda activate flovopy-env

# Or from YML
conda create -n flovopy-env -f flovopy_plus

# Install dependencies and the package in editable mode
pip install -e .
```

---

## 🚀 Available CLI Tools

The following command-line tools are installed with the package:

| Command            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `run-iceweb`       | Generate RSAM, DRS, and spectrograms from SDS or FDSN waveform sources      |
| `run-asl`          | Amplitude-based source location (ASL) processing of Seisan database events |
| `run-seisan2sds`   | Convert Seisan waveform files to SDS archive format                         |
| `run-fdsn2sds`     | Download FDSN waveform data into SDS archive format                         |
| `run-sds2rsam`     | Compute RSAM from SDS archive                                                |
| `run-sds2disp`     | Compute displacement streams from SDS archive                               |

Run any tool with `--help` for full usage.

---

## 🧰 Example Usage

### `run-iceweb`

```bash
run-iceweb \
  --config config/ \
  --start 2023-01-01T00:00:00 \
  --end 2023-01-02T00:00:00 \
  --subnet SHV \
  --inventory metadata/station.xml \
  --trace_ids XX.STA..BHZ XX.STA..HHZ
```

### `run-asl`

```bash
run-asl \
  --start 2001-01-01T00:00:00 \
  --end 2001-01-02T00:00:00 \
  --subnet SHV \
  --db MVOE_ \
  --seisan /data/SEISAN_DB \
  --inventory metadata/station.xml \
  --outdir ASL_DB \
  --Q 23 \
  --peakf 8.0 \
  --metric rms \
  --surfaceWaveSpeed_kms 1.5 \
  --interactive False
```

---

## 📂 Project Structure

```text
flovopy/
├── analysis/             # Amplitude-based source location (ASL)
├── core/                 # Core utilities: inventory, plotting, simulation
├── processing/           # RSAM/DRS and signal metrics (SAM)
├── seisanio/             # Seisan format support and waveform parsing
├── sds/                  # SDS archive handling
├── wrappers/             # CLI entry points and batch wrappers
├── obsolete/             # Deprecated modules
├── utils.py              # Shared utility functions
└── tests/                # Unit tests
```

---

## 📋 Requirements

- Python 3.8 or newer
- Conda environment (recommended)
- Dependencies:
  - `obspy`
  - `numpy`
  - `pandas`
  - (plus others depending on specific modules)

---

## 📖 Documentation

Full documentation (under construction) is being developed using Sphinx and will include:
- API reference
- Tutorials for real-time observatory pipelines
- Guides for legacy Seisan data conversion
- Tools for ASL modeling and seismic/infrasound visualization

---

## 📜 License

FLOVOpy is released under the MIT License.

---

## 🤝 Acknowledgments

FLOVOpy is developed at the University of South Florida as part of the Florida Virtual Volcano Observatory (FLO-VO) initiative, with contributions from volcano seismologists, geophysicists, and observatory partners across the Caribbean and the Americas.

