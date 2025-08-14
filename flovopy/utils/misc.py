import os

def remove_ds_store_files_and_empty_dirs(sds_directory):
    """ Recursively removes all .DS_Store files and deletes any empty directories. """
    for root, dirs, files in os.walk(sds_directory, topdown=False):  # Bottom-up to clean empty dirs
        # Remove .DS_Store files
        for file in files:
            if ".DS_Store" in file:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        # Remove empty directories
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):  # Check if the directory is empty
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")

def commandExists(command):
    output = os.popen('which %s' % command).read()
    if output:
        return True
    else:
        print('Command %s not found.' % command)
        print('Make sure the PASSOFT tools are installed on this computer, and available on the $PATH')
        return False
    
def yes_or_no(question, default_answer='y', auto=False):
    #while "the answer is invalid":
    if auto==True:
        reply = ''
    else:
        reply = str(input(question+' (y/n): ')).lower().strip()
    if len(reply) == 0:
        reply = default_answer
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    

def yn_choice(message, default='y'):
    choices = 'Y/n' if default.lower() in ('y', 'yes') else 'y/N'
    choice = input("%s (%s) " % (message, choices))
    values = ('y', 'yes', '') if default == 'y' else ('y', 'yes')
    return True if choice.strip().lower() in values else False   


import ast
from pathlib import Path

def extract_definitions(path: Path):
    """Return a string listing the class and function names in the file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            node = ast.parse(f.read(), filename=str(path))
        classes = [n.name for n in node.body if isinstance(n, ast.ClassDef)]
        funcs = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        parts = []
        if classes:
            parts.append("classes: " + ", ".join(classes))
        if funcs:
            parts.append("functions: " + ", ".join(funcs))
        return "  # " + "; ".join(parts) if parts else ""
    except Exception as e:
        return "  # [error parsing]"

from pathlib import Path

def tree(dir_path, prefix: str = ""):
    """
    Recursively yield a visual tree of a directory. Accepts str or Path.
    """
    # Coerce to Path and normalize
    dir_path = Path(dir_path).expanduser()

    if not dir_path.exists():
        yield prefix + f"(missing) {dir_path}"
        return
    if not dir_path.is_dir():
        # If it's a file, just return the file name (plus definitions for .py)
        line = prefix + dir_path.name
        if dir_path.suffix == ".py":
            line += extract_definitions(dir_path)  # your helper
        yield line
        return

    space  = "    "
    branch = "│   "
    tee    = "├── "
    last   = "└── "

    try:
        contents = sorted(
            dir_path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        yield prefix + f"{dir_path.name}/ (permission denied)"
        return

    pointers = [tee] * (len(contents) - 1) + [last] if contents else []
    for pointer, path in zip(pointers, contents):
        line = prefix + pointer + path.name
        if path.is_file() and path.suffix == ".py":
            line += extract_definitions(path)  # your helper
        yield line
        if path.is_dir():
            extension = branch if pointer == tee else space
            yield from tree(path, prefix=prefix + extension)



if __name__ == "__main__":
    homedir = Path.home()
    flovodir = homedir / 'Developer' / 'flovopy' / 'flovopy'
    if not flovodir.exists():
        print(f"ERROR: {flovodir} does not exist")
    else:
        for line in tree(flovodir):
            print(line)
