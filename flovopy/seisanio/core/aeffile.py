import os
from flovopy.seisanio.utils.helpers import correct_nslc


class AEFfile:
    def __init__(self, path=None):
        self.aefrows = []  # each element is an AEFrow dict
        self.trigger_window = None
        self.average_window = None
        self.path = path.strip() if path else None

        if not os.path.exists(self.path):
            print(f"{self.path} does not exist")
            return

        try:
            with open(self.path, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            print(f"Error reading {self.path}: {e}")
            return

        for line in lines:
            if len(line) < 80:
                continue

            if line[79] == '3' or (line[79] == ' ' and line[1:5] == "VOLC"):
                if line[1:10] != "VOLC MAIN":
                    aefrow = self.parse_aefline(line)
                    if aefrow:
                        self.aefrows.append(aefrow)

                if "trigger window" in line:
                    self.trigger_window = self._extract_window(line, "trigger window")

                if "average window" in line:
                    self.average_window = self._extract_window(line, "average window")

    def _extract_window(self, line, keyword):
        i_start = line.find(keyword)
        if i_start > -1:
            substring = line[i_start:i_start + 24]
            eq_index = substring.find('=')
            s_index = substring.find('s')
            if eq_index != -1 and s_index != -1:
                value = substring[eq_index + 1:s_index].strip()
                try:
                    return float(value)
                except ValueError:
                    pass
        return None

    def __str__(self):
        info = f"aeffile: {self.path}"
        if self.trigger_window:
            info += f"\n\ttrigger window: {self.trigger_window:.2f}"
        if self.average_window:
            info += f"\n\taverage window: {self.average_window:.2f}"
        for aefrow in self.aefrows:
            info += "\n\t" + str(aefrow)
        return info

    @staticmethod
    def parse_aefline(line):
        try:
            station = line[6:10].strip()
            channel = line[11:14].strip()
            a_idx = line[15:22].find('A') + 15

            amplitude = float(line[a_idx + 1:a_idx + 9].strip())
            energy = float(line[a_idx + 11:a_idx + 19].strip())
            ssam = AEFfile.parse_F(line, energy, a_idx + 21)
            maxf = float(line[73:79].strip()) if a_idx < 20 else None

            trace_id = f".{station}..{channel}"
            fixed_id = correct_nslc(trace_id, 100.0, shortperiod=(station[:2] != 'MB'))

            return {
                'station': station,
                'channel': channel,
                'id': trace_id,
                'fixed_id': fixed_id,
                'amplitude': amplitude,
                'energy': energy,
                'ssam': ssam,
                'maxf': maxf
            }
        except Exception as e:
            print(f"Error parsing AEF line: {e}")
            return None

    @staticmethod
    def parse_F(line, energy, startindex):
        F = {
            "frequency_bands": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 30.0],
            "percentages": [],
            "energies": []
        }
        while startindex < 79 and len(F["percentages"]) < 12:
            ssamstr = line[startindex:startindex + 3].strip()
            if "." not in ssamstr:
                try:
                    val = int(ssamstr)
                    F["percentages"].append(val)
                    F["energies"].append(val / 100.0 * energy)
                except ValueError:
                    pass
            startindex += 3
        return F
