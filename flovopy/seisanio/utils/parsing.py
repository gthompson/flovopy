# parsing.py


def parse_string(line, pos0, pos1, astype='float', stripstr=True, default=None):
    """Safely extract a substring from a line and convert to int/float/str."""
    _s = line[pos0:pos1]
    if stripstr:
        _s = _s.strip()
    if not _s:
        return default
    try:
        if astype == 'float':
            return float(_s)
        elif astype == 'int':
            return int(_s)
        return _s
    except ValueError:
        return default

def parse_aefline(line, correct_nslc_func):
    """Parse an AEF-format line from a SEISAN S-file."""
    aefrow = {'station': None, 'channel': None, 'amplitude': None,
              'energy': None, 'ssam': None, 'maxf': None}

    a_idx = line[15:22].find('A') + 15  # amplitude info starts here
    station = line[6:10].strip()
    channel = line[11:14].strip()

    aefrow['id'] = f".{station}..{channel}"
    Fs = 100.0
    shortperiod = station[:2] != 'MB'
    aefrow['fixed_id'] = correct_nslc_func(aefrow['id'], Fs, shortperiod=shortperiod)

    try:
        aefrow['amplitude'] = float(line[a_idx+1:a_idx+9].strip())
        aefrow['energy'] = float(line[a_idx+11:a_idx+19].strip())
        aefrow['ssam'] = parse_ssam(line, aefrow['energy'], a_idx+21)
        if a_idx < 20:
            aefrow['maxf'] = float(line[73:79].strip())
    except ValueError:
        pass

    return aefrow

def parse_ssam(line, energy, startindex):
    """Parse spectral amplitude data from an AEF line."""
    F = {
        "frequency_bands": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                             6.0, 7.0, 8.0, 9.0, 10.0, 30.0],
        "percentages": [],
        "energies": []
    }
    while startindex < 79 and len(F["percentages"]) < 12:
        ssamstr = line[startindex:startindex+3].strip()
        if "." not in ssamstr:
            try:
                val = int(ssamstr)
                F["percentages"].append(val)
                F["energies"].append(val / 100.0 * energy)
            except ValueError:
                pass
        startindex += 3
    return F



def parse_station0hyp(station0hypfile):
    """
    Parses a SEISAN STATION0.HYP file and returns a DataFrame
    with station, latitude, longitude, and elevation.
    """

    import os
    import pandas as pd
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