import os
import pandas as pd

def parse_station0hyp(station0hypfile):
    """
    Parses a SEISAN STATION0.HYP file and returns a DataFrame
    with station, latitude, longitude, and elevation.
    """
    station_locations = []

    if not os.path.exists(station0hypfile):
        raise FileNotFoundError(f"STATION0.HYP file not found: {station0hypfile}")

    with open(station0hypfile, 'r') as f:
        for line in f:
            line = line.replace('^M', '').strip()
            if 'HEAD' in line or 'TEST' in line or len(line) < 25:
                continue

            try:
                station = line[0:4].strip()
                latdeg = int(line[4:6])
                if line[8] == '.':
                    latmin = float(line[6:11])
                else:
                    latmin = float(line[6:8]) + float(line[9:11])/60
                latsign = -1 if line[11].upper() == 'S' else 1

                londeg = int(line[12:15])
                if line[17] == '.':
                    lonmin = float(line[15:20])
                else:
                    lonmin = float(line[15:17]) + float(line[18:20])/60
                lonsign = -1 if line[20].upper() == 'W' else 1

                elev = float(line[21:30].strip())

                lat = (latdeg + latmin / 60.0) * latsign
                lon = (londeg + lonmin / 60.0) * lonsign

                station_locations.append({
                    'name': station,
                    'lat': lat,
                    'lon': lon,
                    'elev': elev
                })

            except Exception as e:
                print(f"Warning: Failed to parse line: {line}\n{e}")

    return pd.DataFrame(station_locations)