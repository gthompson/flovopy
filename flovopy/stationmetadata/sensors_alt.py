import os, shutil, subprocess, tempfile
import requests
from obspy import read_inventory
from obspy.core.inventory import Inventory
from obspy.io.xseed import Parser

# ---- URL resolution for Raspberry Shake SC3 XML by model/variant
# You can extend this map as needed from https://manual.raspberryshake.org/metadata.html
RS_SC3_URLS = {
    # examples – fill with the exact links you need from the table
    # ("RS1D","V7"): "https://.../RS1D_V7.sc3.xml",
    # ("RS&BOOM","V3-1s"): "https://.../RSBOOM_V3_1s.sc3.xml",
    # ("RS&BOOM","V3-20s"): "https://.../RSBOOM_V3_20s.sc3.xml",
}

def have_seiscomp_tools() -> bool:
    """Return True if SeisComP's fdsnxml2inv is on PATH."""
    return shutil.which("fdsnxml2inv") is not None

def download_sc3xml(url: str, out_path: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def substitute_placeholders_sc3xml(path: str, station_code: str):
    """
    Many RS templates ask to replace STNNM. Adjust if their template uses other tokens.
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # Minimal replacement; extend if templates include NET/LOC/CHAN placeholders
    txt = txt.replace("STNNM", station_code)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def sc3xml_to_stationxml(sc3xml_path: str, stationxml_out: str):
    """
    Convert SeisComP XML to FDSN StationXML via SeisComP tool if available.
    """
    if not have_seiscomp_tools():
        raise RuntimeError("fdsnxml2inv not found on PATH; cannot convert SeisComP XML to StationXML.")
    # fdsnxml2inv converts both directions; we want SCML -> FDSN StationXML
    # The tool autodetects by input; explicit flags vary by version. Try basic usage:
    cmd = ["fdsnxml2inv", sc3xml_path, stationxml_out]
    subprocess.run(cmd, check=True)

def build_inventory_from_sc3_template(
    device: str, variant: str, station_code: str,
    url_override: str = None
) -> Inventory:
    """
    Download a Raspberry Shake SeisComP XML template, substitute placeholders,
    convert to StationXML (if SeisComP tools are present), and return an Inventory.

    Fallback: if SeisComP tools are missing, try RESP/dataless sibling template instead.
    """
    url = url_override or RS_SC3_URLS.get((device, variant))
    if not url:
        raise ValueError(f"No SC3 XML URL configured for {(device, variant)}; provide url_override.")

    with tempfile.TemporaryDirectory() as td:
        sc3_path = os.path.join(td, f"{device}_{variant}.sc3.xml")
        download_sc3xml(url, sc3_path)
        substitute_placeholders_sc3xml(sc3_path, station_code)

        if have_seiscomp_tools():
            # Convert to StationXML via SeisComP CLI
            staxml_path = os.path.join(td, f"{device}_{variant}.xml")
            sc3xml_to_stationxml(sc3_path, staxml_path)
            return read_inventory(staxml_path)

        # ---- Fallback path: try a sibling RESP or dataless template if published ----
        # If Raspberry Shake provides a RESP/dataless link right next to the SC3 template,
        # fetch it here and parse with ObsPy.
        # For illustration, we assume a sibling RESP URL; adjust per the manual.
        resp_url = None  # put the RESP link here if you have it for this device/variant
        dless_url = None # or dataless link

        if dless_url:
            dless_path = os.path.join(td, f"{device}_{variant}.dataless")
            download_sc3xml(dless_url, dless_path)
            Parser(dless_path).write_seed(staxml_path := os.path.join(td, "tmp.xml"))
            return read_inventory(staxml_path)

        if resp_url:
            resp_path = os.path.join(td, f"{device}_{variant}.RESP")
            download_sc3xml(resp_url, resp_path)
            # Parsing RESP directly to full Inventory isn't supported; you’d need to
            # build a skeleton Inventory and attach responses per channel. Prefer dataless.
            raise RuntimeError("RESP fallback not implemented to full Inventory. Prefer dataless fallback.")

        raise RuntimeError("SeisComP converter not available and no fallback RESP/dataless URL provided.")