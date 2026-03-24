# FLOVOpy: Florida Virtual Volcano Observatory Tools

FLOVOpy is a Python package providing advanced tools for volcano observatories and seismology research. It includes waveform processing utilities, RSAM/DRS computation, spectrogram generation, amplitude source location (ASL), and enhanced support for legacy datasets such as Seisan files.

---

## 📦 Installation (via Conda) on Linux/MacOS

To install FLOVOpy and make it available, you can clone the repository, and then build the flovopy_env Conda environment, and then add to your Python path using pip:

```bash
# 1. Clone the repository
git clone https://github.com/gthompson/flovopy.git
cd flovopy

# 2. Create the conda environment from YML. 
# The flovopy_plus environment contains more packages than we need, but these are generally useful for seismology
conda env create --name flovopy_env --file environment.yml
conda activate flovopy_env

# 3. Install dependencies and the package in editable mode (make sure you are still in the flovopy top directory)
pip install .

# 4. Optionally add this to your .bashrc or .zshrc file:
conda activate flovopy_env
```

---

## 📂 Project Structure

```text
bin/                      # Bash scripts
flovopy/
├── analysis/             # Miscellaneous analysis tools
├── asl/                  # Amplitude-based source location (ASL)
├── core/                 # Core utilities: data pre-processing, robust data loading and saving, response removal, physics
├── dem/                  # DEM tools mostly for ASL on Montserrat
├── enhanced/             # Enhanced versions of Trace, Stream, Event, Catalog, sdsclient, plus new EventRate class.
├── processing/           # Seismic Amplitude Measurement, classification, detection, spectrograms
├── sds/                  # SDS archive handling
├── seisanio/             # Seisan format support and waveform parsing
├── stationmetadata/      # Tools for building Inventory objects from NRL and local files for USF or MVO equipment
├── tutorials/            # Mostly ASL tutorials
├── sds/                  # SDS archive handling
├── utils/                # Miscellaneous helper packages
├── wrappers/             # Various wrappers designed for running FLOVOPY tools on archives
tests/                    # Unit tests
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

## 📜 License

FLOVOpy is released under the MIT License.

---

## 🤝 Acknowledgments

FLOVOpy is developed by Glenn Thompson at the University of South Florida as part of the Florida Virtual Volcano Observatory (FLO-VO) initiative, based on operational codes formerly developed at the Montserrat Volcano Observatory, Alaska Volcano Observatory, Alaska Earthquake Center, and the University of Alaska Fairbanks Geophysical Institute. 

