import os
from pathlib import Path
import requests
import zipfile

NRL_ZIP_URL = "https://ds.iris.edu/files/nrl/NRLv2.zip"
NRL_ZIP_PATH = Path("/data/station_metadata/NRLv2.zip")
NRL_EXTRACT_DIR = Path("/data/station_metadata/NRLv2")

INFRABSU_XML_URL = "https://service.iris.edu/irisws/nrl/1/combine?instconfig=sensor_JeffreyBJohnson_infraBSU_LP21_SG0.000046_STairPressure&format=stationxml"
INFRABSU_XML_PATH = Path("/data/station_metadata/infraBSU_21s_0.5inch.xml")

def download_if_missing(url: str, target_path: Path):
    if not target_path.exists():
        print(f"[INFO] Downloading: {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(response.content)
        print(f"[OK] Saved to: {target_path}")
    else:
        print(f"[OK] File already exists: {target_path}")

def extract_nrl_zip(zip_path: Path, extract_to: Path):
    if not extract_to.exists():
        print(f"[INFO] Extracting {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[OK] Extracted.")
    else:
        print(f"[OK] NRL directory already exists: {extract_to}")

def initialize_station_metadata():
    # Ensure base directory exists
    Path("/data/station_metadata").mkdir(parents=True, exist_ok=True)

    # Download and extract full NRL v2 if not already present
    #download_if_missing(NRL_ZIP_URL, NRL_ZIP_PATH)
    #extract_nrl_zip(NRL_ZIP_PATH, NRL_EXTRACT_DIR)

    # Download infraBSU StationXML if not already present
    download_if_missing(INFRABSU_XML_URL, INFRABSU_XML_PATH)

# Run this when called as a script
if __name__ == "__main__":
    initialize_station_metadata()
