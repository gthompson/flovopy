#!/usr/bin/env python3
import os
import sys
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import requests

try:
    from obspy import read_inventory
    from obspy.core.inventory import Inventory
except Exception as e:
    print("[ERROR] ObsPy is required (pip install obspy):", e)
    sys.exit(1)

# ----------------------------
# Config: the instruments/URLs
# ----------------------------
INSTRUMENTS = [
    ("RS1D_v4_50Hz", "https://manual.raspberryshake.org/_downloads/28ac4957c7169e3cadd9b1fe46472d4c/raspShake-V4-racotech-gphone.tpl.xml", "RS1D0"),
    ("RS1D_v6_100Hz", "https://manual.raspberryshake.org/_downloads/c9d0c435f7cb64cf24ce3dc6b9934dea/out4.response.restored-EHZ-plus-decimation.dataless-reformatted.xml", "RS1D6"),
    ("RS3D_v5_100Hz", "https://manual.raspberryshake.org/_downloads/3234152c8f71cebc3cd0617ec6c4a968/out4.response.restored-EHZ-plus-decimation.dataless-new-reformatted.xml", "RS3D5"),
    ("RSnB_v3_100Hz_20s", "https://manual.raspberryshake.org/_downloads/e3a34ac4fe6b742fcf00ac7e3665d815/RSnBV3-1s.dataless.xml-reformatted.xml", "RSNB3"),
    ("RBOOM_v3_100Hz_20s", "https://manual.raspberryshake.org/_downloads/2236659d1aba5f12a5da823e4ad782e5/RBmV3-20s.dataless.xml-reformatted.xml", "RBOOM"),
]

# ----------------------------
# Utils
# ----------------------------
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def download(url: str, out_path: Path):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def sniff_format(path: Path) -> str:
    """
    - 'sc3xml' : SeisComP XML (<seiscomp ...>)
    - 'staxml' : FDSN StationXML (<FDSNStationXML> or stationxml namespace)
    - 'dataless' : likely binary SEED
    - 'unknown'
    """
    try:
        head = path.read_bytes()[:4096]
    except Exception:
        return "unknown"

    if b"\x00" in head[:64] and b"<" not in head[:64]:
        return "dataless"

    text = head.decode("utf-8", errors="ignore").lstrip()
    if text.startswith("<"):
        if re.search(r"<\s*seiscomp\b", text, re.IGNORECASE):
            return "sc3xml"
        if re.search(r"<\s*FDSNStationXML\b", text) or "fdsn.org/xml/station" in text or "StationXML" in text:
            return "staxml"
    return "unknown"

# ----------------------------
# SC3 template patching (minimal & safe)
# ----------------------------
def patch_sc3_template(path: Path,
                       station_code: str,
                       network_code: Optional[str] = "AM",
                       default_start: str = "1900-01-01T00:00:00.00Z",
                       default_end:   str = "2100-01-01T00:00:00.00Z"):
    """
    Minimal patching for Raspberry Shake SeisComP XML templates:
      - STNNM -> station_code
      - NETNM -> network_code
      - Replace literal 'YYYY-MM-DDT00:00:00.00Z' in element content.
        We treat <end>...</end> specially and give it default_end.
      - Compact FIR coefficient whitespace (cosmetic).
    """
    txt = path.read_text(encoding="utf-8")

    # Station/network tokens
    txt = txt.replace("STNNM", station_code)
    if network_code:
        txt = txt.replace("NETNM", network_code)

    # Mark <end> placeholders first so we can set them to default_end
    txt = re.sub(
        r'(<\s*end[^>]*>\s*)YYYY-MM-DDT00:00:00\.00Z(\s*<\s*/\s*end\s*>)',
        lambda m: f"{m.group(1)}__PLACEHOLDER_END__{m.group(2)}",
        txt,
        flags=re.IGNORECASE
    )

    # Any remaining literal placeholders become default_start
    txt = txt.replace("YYYY-MM-DDT00:00:00.00Z", default_start)

    # Now fill the end markers
    txt = txt.replace("__PLACEHOLDER_END__", default_end)

    # Compact FIR coefficients
    def _norm_fir(m):
        body = " ".join(m.group(2).split())
        return f"{m.group(1)}{body}{m.group(3)}"
    txt = re.sub(r'(<coefficients>)([^<]+)(</coefficients>)', _norm_fir, txt, flags=re.S)

    path.write_text(txt, encoding="utf-8")

# ----------------------------
# Conversions
# ----------------------------
def convert_sc3_to_stationxml(sc3_path: Path, out_xml: Path, use_seiscomp_exec: bool = True) -> str:
    """
    Convert SeisComP XML → StationXML by capturing stdout.
    Uses --relaxed-ns-check (works on your box).
    Returns stderr text for diagnostics.
    """
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    if use_seiscomp_exec and which("seiscomp"):
        cmd = ["seiscomp", "exec", "fdsnxml2inv", "--to-staxml", "--relaxed-ns-check", "-f", str(sc3_path)]
    else:
        fdsn = which("fdsnxml2inv")
        if not fdsn:
            raise RuntimeError("fdsnxml2inv not found in PATH.")
        cmd = [fdsn, "--to-staxml", "--relaxed-ns-check", "-f", str(sc3_path)]

    # Show the exact command (no output filename -> stdout)
    print("[CMD]", " ".join(cmd), ">", str(out_xml))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_xml = proc.stdout
    stderr_text = proc.stderr.decode("utf-8", "ignore")

    if proc.returncode != 0:
        # Dump head of patched input on failure
        try:
            with open(sc3_path, "r", encoding="utf-8") as fh:
                preview = "".join([next(fh) for _ in range(80)])
        except Exception:
            preview = "<unable to preview input file>"
        raise RuntimeError(
            f"fdsnxml2inv failed (rc={proc.returncode}).\n"
            f"---- STDERR ----\n{stderr_text}\n"
            f"---- PATCHED INPUT HEAD ({sc3_path}) ----\n{preview}\n"
            f"----------------"
        )

    # Basic sanity: stdout should look like StationXML and be non-trivial
    if not stdout_xml or b"<FDSNStationXML" not in stdout_xml:
        raise RuntimeError("fdsnxml2inv returned no StationXML on stdout.")

    out_xml.write_bytes(stdout_xml)
    return stderr_text
# ----------------------------
# Validation / summary
# ----------------------------
def validate_stationxml(path: Path) -> dict:
    inv: Inventory = read_inventory(str(path))
    n_networks = len(inv.networks)
    n_stations = sum(len(net.stations) for net in inv.networks)
    n_channels = sum(len(sta.channels) for net in inv.networks for sta in net.stations)
    networks = [net.code for net in inv.networks]
    return {
        "file": str(path),
        "networks": networks,
        "n_networks": n_networks,
        "n_stations": n_stations,
        "n_channels": n_channels,
    }

def summarize_outputs(out_dir: Path):
    print("\n=== StationXML Validation Summary ===")
    ok = 0
    fail = 0
    for p in sorted(out_dir.glob("*.xml")):
        try:
            info = validate_stationxml(p)
            print("[OK] {file} nets={nets} stations={ns} channels={nc}".format(
                file=os.path.basename(info["file"]),
                nets=",".join(info["networks"]) if info["networks"] else "-",
                ns=info["n_stations"],
                nc=info["n_channels"],
            ))
            ok += 1
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}")
            fail += 1
    print(f"=== Done: {ok} OK, {fail} FAIL ===")

# ----------------------------
# Main routine
# ----------------------------
def fetch_and_convert(label: str, url: str, out_dir: Path, station_code: str, network_code: str = "AM") -> Path:
    """
    Download a metadata file, patch placeholders, convert to StationXML, and return the output path.
    """
    ensure_dir(out_dir)
    tmpdir = Path(tempfile.mkdtemp(prefix=f"rs_meta_{label}_"))
    downloaded = tmpdir / Path(url).name

    print(f"[INFO] Downloading {label} from {url}")
    download(url, downloaded)

    kind = sniff_format(downloaded)
    print(f"[INFO] Detected format for {label}: {kind}")
    out_xml = out_dir / f"{label}.xml"

    if kind == "staxml":
        shutil.copy2(downloaded, out_xml)
        print(f"[OK] Saved StationXML: {out_xml}")
        return out_xml

    elif kind == "sc3xml":
        # Save a patched copy so you can inspect it on failures
        patched = downloaded.with_suffix(downloaded.suffix + ".patched")
        shutil.copy2(downloaded, patched)
        patch_sc3_template(patched, station_code=station_code, network_code=network_code)

        try:
            stderr_text = convert_sc3_to_stationxml(patched, out_xml, use_seiscomp_exec=True)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            return out_xml

        # Validate & warn if empty after conversion
        try:
            info = validate_stationxml(out_xml)
            if info["n_stations"] == 0 or info["n_channels"] == 0:
                print(f"[WARN] {label}: 0 stations/channels after conversion.")
                if stderr_text.strip():
                    print(f"[WARN] Converter stderr:\n{stderr_text.strip()}")
        except Exception as e:
            print(f"[FAIL] Could not parse converted StationXML for {label}: {e}")
            return out_xml

        print(f"[OK] Converted SC3 → StationXML: {out_xml}")
        return out_xml

    elif kind == "dataless":
        raise RuntimeError(f"{label}: binary dataless detected; please use a StationXML or SeisComP XML source.")

    else:
        # Try ObsPy parse as a fallback (some RS links are mislabeled)
        try:
            _ = read_inventory(str(downloaded))
            shutil.copy2(downloaded, out_xml)
            print(f"[OK] Parsed as StationXML via ObsPy fallback: {out_xml}")
            return out_xml
        except Exception:
            raise RuntimeError(f"{label}: Unknown format and not parsable as StationXML. URL may have changed.")

def main():
    out_dir = Path("stationxml_out")
    ensure_dir(out_dir)

    for label, url, default_sta in INSTRUMENTS:
        try:
            fetch_and_convert(label, url, out_dir, station_code=default_sta, network_code="AM")
        except Exception as e:
            print(f"[ERROR] {label}: {e}")

    summarize_outputs(out_dir)

if __name__ == "__main__":
    main()
