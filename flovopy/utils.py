from pathlib import Path
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

def tree(dir_path: Path, prefix: str=''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """    

    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '
    
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)