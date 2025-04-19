# db_backup
import os
import shutil
def backup_db(DB, scriptname):
    myscript = os.path.basename(scriptname).replace('.py', '')
    DB_BACKUP = DB.replace('.sqlite', f'_before_{myscript}.sqlite')
    shutil.copy(DB, DB_BACKUP)
    if os.path.isfile(DB_BACKUP):
        print(f'backup of db made to {DB_BACKUP}')
        return True
    else:
        print(f'Could not create {DB_BACKUP}')
        return False
