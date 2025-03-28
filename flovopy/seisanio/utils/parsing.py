# parsing.py

def parse_string(line, pos0, pos1, astype='float', stripstr=True):
    """Extract and convert a substring from a line."""
    _s = line[pos0:pos1]
    if stripstr:
        _s = _s.strip()
    if not _s:
        return None
    try:
        if astype == 'float':
            return float(_s)
        elif astype == 'int':
            return int(_s)
        else:
            return _s
    except ValueError:
        return None

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