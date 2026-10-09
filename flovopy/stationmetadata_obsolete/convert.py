import os
from obspy import UTCDateTime
from obspy.core.inventory import read_inventory, Inventory, Network, Station, Channel, Site
from obspy.io.xseed import Parser

def inventory2dataless_and_resp(inv, output_dir=None, stationxml_seed_converter_jar=None):
    """
    Converts each channel in an ObsPy Inventory to individual StationXML, Dataless SEED, 
    and RESP files. These are saved in the specified output directory.

    Parameters:
    -----------
    inv : obspy.core.inventory.inventory.Inventory
        The input Inventory object containing full metadata.
    output_dir : str
        Directory path where output files will be written. Will be created if it doesn't exist.
    stationxml_seed_converter_jar : str
        Path to the IRIS StationXML-SEED converter JAR file (e.g. `stationxml-seed-converter.jar`).
        This JAR must be installed and available on the system with Java.

    Output:
    -------
    - One StationXML file per channel.
    - One Dataless SEED file per channel.
    - One RESP directory per channel (unzipped).
    """
    os.makedirs(output_dir, exist_ok=True)

    for net in inv:
        for sta in net:
            for cha in sta:
                try:
                    # Construct a minimal Inventory with one channel
                    new_channel = Channel(
                        code=cha.code,
                        location_code=cha.location_code,
                        latitude=cha.latitude,
                        longitude=cha.longitude,
                        elevation=cha.elevation,
                        depth=cha.depth,
                        azimuth=cha.azimuth,
                        dip=cha.dip,
                        sample_rate=cha.sample_rate,
                        start_date=cha.start_date,
                        end_date=cha.end_date,
                        response=cha.response
                    )

                    new_station = Station(
                        code=sta.code,
                        latitude=sta.latitude,
                        longitude=sta.longitude,
                        elevation=sta.elevation,
                        site=Site(name=sta.site.name if sta.site.name else ""),
                        channels=[new_channel],
                        start_date=sta.start_date,
                        end_date=sta.end_date
                    )

                    new_network = Network(code=net.code, stations=[new_station])
                    mini_inv = Inventory(networks=[new_network], source=inv.source)

                    # Create base filename using channel start/end date
                    sdt = (cha.start_date or UTCDateTime(0)).format_iris_web_service().replace(":", "-")
                    edt = (cha.end_date or UTCDateTime(2100, 1, 1)).format_iris_web_service().replace(":", "-")
                    basename = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}_{sdt}_{edt}"
                    xmlfile = os.path.join(output_dir, f"{basename}.xml")

                    # Write StationXML
                    mini_inv.write(xmlfile, format="stationxml")
                    print(f"[OK] Wrote StationXML: {xmlfile}")

                    # Convert to Dataless SEED using Java JAR
                    dataless_file = os.path.join(output_dir, f"{basename}.dseed")
                    java_cmd = f"java -jar {stationxml_seed_converter_jar} -s {xmlfile} -o {dataless_file}"
                    ret = os.system(java_cmd)
                    if ret != 0 or not os.path.exists(dataless_file):
                        raise RuntimeError(f"[WARN] Dataless SEED conversion failed for {xmlfile}")
                    print(f"[OK] Wrote Dataless SEED: {dataless_file}")

                    # Convert to RESP using ObsPy Parser
                    resp_dir = os.path.join(output_dir, f"{basename}_resp")
                    os.makedirs(resp_dir, exist_ok=True)
                    sp = Parser(dataless_file)
                    sp.write_resp(folder=resp_dir, zipped=False)
                    print(f"[OK] Wrote RESP files to: {resp_dir}")

                except Exception as e:
                    print(f"[ERROR] Failed for {net.code}.{sta.code}.{cha.location_code}.{cha.code}: {e}")


def write_inventory_as_resp(inventory, seed_tempfile, resp_outdir):
    """
    Writes RESP files for all channels in an ObsPy Inventory by:
    1. Writing a temporary Dataless SEED file from the Inventory.
    2. Using ObsPy's Parser to extract and write RESP files.

    Parameters:
    -----------
    inventory : obspy.core.inventory.inventory.Inventory
        ObsPy Inventory object to convert.
    seed_tempfile : str
        Temporary file path to write Dataless SEED output (e.g., /tmp/temp.seed).
    resp_outdir : str
        Output directory where RESP files will be saved.
    """
    inventory.write(seed_tempfile, format='SEED')
    sp = Parser(seed_tempfile)
    sp.write_resp(folder=resp_outdir, zipped=False)
    print(f"[OK] RESP files written to {resp_outdir}")


def convert_stationxml_to_resp(stationxml_path, output_dir):
    """
    Convert a StationXML file into RESP files using ObsPy's Parser.

    Parameters:
    -----------
    stationxml_path : str
        Path to input StationXML file.
    output_dir : str
        Directory to save the RESP files.
    """
    try:
        inv = read_inventory(stationxml_path)
    except Exception as e:
        print(f"[ERROR] Failed to read StationXML file: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    seed_tempfile = os.path.join(output_dir, "temp.seed")
    try:
        write_inventory_as_resp(inv, seed_tempfile, output_dir)
    except Exception as e:
        print(f"[ERROR] Failed to convert to RESP: {e}")
        return
    finally:
        if os.path.exists(seed_tempfile):
            os.remove(seed_tempfile)