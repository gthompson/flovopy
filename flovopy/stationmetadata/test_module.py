import os
import platform
from flovopy.stationmetadata.sensors import (
    countsPerMS,
    countsPerPa40,
    countsPerPa1,
    countsPerPaChap40,
    countsPerPaChap1,
)
from flovopy.stationmetadata.utils import (
    download_infraBSU_stationxml,
    get_stationXML_inventory,
    inventory2dataless_and_resp,
)

# Get user's home directory
home = os.path.expanduser("~")

# Adjust paths based on OS
if platform.system() == 'Darwin':  # macOS
    metadata_dir = os.path.join(home, 'Dropbox', 'DATA', 'station_metadata')
else:
    metadata_dir = '/data/station_metadata'
os.makedirs(metadata_dir, exist_ok=True)

xmlfile = os.path.join(metadata_dir, 'KSC.xml')
respdir = os.path.join(metadata_dir, 'RESP')
os.makedirs(respdir, exist_ok=True)

metadata_csv = os.path.join(metadata_dir, 'ksc.csv')
coord_csv = os.path.join(metadata_dir, 'ksc_coordinates_only.csv')

NRLpath = os.path.join(metadata_dir, 'NRL')
infraBSUstationXML = os.path.join(metadata_dir, 'infraBSU_21s_0.5inch.xml')

# Always use $HOME/bin for converter JAR
stationxml_seed_converter_jar = os.path.join(home, 'bin', 'stationxml-seed-converter.jar')

# Print calibration constants
print('### calibrations only ###:')
print('trillium+centaur40 = %f' % countsPerMS)        
print('infraBSU+centaur40 = %f' % countsPerPa40)        
print('infraBSU+centaur1 = %f' % countsPerPa1)        
print('chaparralM25+centaur40 = %f' % countsPerPaChap40)        
print('chaparralM25+centaur1 = %f' % countsPerPaChap1)   
print('************************************\n\n')

# Build full inventories with responses
print('### Building full inventories with responses ###')
print('First try to get combined response for infraBSU and Centaur:')

if not os.path.isfile(infraBSUstationXML):
    download_infraBSU_stationxml(save_path=infraBSUstationXML)

inv = get_stationXML_inventory(
    xmlfile=xmlfile,
    overwrite=True,
    infraBSUstationxml=infraBSUstationXML,
    metadata_csv=metadata_csv,
    coord_csv=coord_csv,
    nrl_path=NRLpath
)
print(inv)

inventory2dataless_and_resp(
    inv,
    output_dir=respdir,
    stationxml_seed_converter_jar=stationxml_seed_converter_jar
)