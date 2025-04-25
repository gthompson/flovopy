import os
import re
import shutil
from pathlib import Path

def move_date_folders(source_root, target_root):
    """
    Move folders from source_root that start with a date in YYYY-MM-DD or YYMM-DD format
    to a YYYY/MM/DD/ folder tree in target_root.

    :param source_root: Path to start scanning for folders.
    :param target_root: Root under which to build YYYY/MM/DD folder structure.
    """
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()

    # Regex patterns
    pattern_full = re.compile(r"^(20\d{2}|19\d{2})-(\d{2})-(\d{2})")
    pattern_short = re.compile(r"^(\d{2})(\d{2})-(\d{2})")  # YYMM-DD

    for item in source_root.iterdir():
        print(item)
        if item.is_dir():
            folder_name = item.name
            match_full = pattern_full.match(folder_name)
            match_short = pattern_short.match(folder_name)

            if match_full:
                year, month, day = match_full.groups()
            elif match_short:
                yy, mm, dd = match_short.groups()
                year = f"19{yy}"  # Assuming all YY < 2000 are in the 1900s
                month, day = mm, dd
            else:
          
                continue  # Skip non-matching folders

            # Build target path
            new_path = target_root / year / month / day
            new_path.mkdir(parents=True, exist_ok=True)

            # Move the folder
            destination = new_path / item.name
            print(f"Moving: {item} -> {destination}")
            shutil.move(str(item), str(destination))

# Example usage:
# move_date_folders("/path/to/source", "/path/to/target")

if __name__ == "__main__":
    move_date_folders('/data/b18_waveform_processing', '/data/b18_waveform_processing')
