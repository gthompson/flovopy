import os
from flovopy.seisanio.core.sfile import Sfile
os.system('clear')
for sfile_path in [
    '/data/SEISAN_DB/REA/MVOE_/1996/10/23-0212-03L.S199610',
    #'/data/SEISAN_DB/REA/MVOE_/2000/02/21-2205-14L.S200002',
    #'/data/SEISAN_DB/REA/MVOE_/1997/01/01-0127-52L.S199701',
    #'/data/SEISAN_DB/REA/MVOE_/1997/01/30-1625-17L.S199701',
    #'/data/SEISAN_DB/REA/MVOE_/2007/03/01-0429-34L.S200703', 
    #'/data/SEISAN_DB/REA/MVOE_/2005/04/20-1541-26R.S200504', 
    #'/data/SEISAN_DB/REA/MVOE_/1997/01/14-1803-38R.S199701' 
    ]:

    print('')
    print(f'Sfile {sfile_path} contents:')
    print('0123456789'*8)
    os.system(f'cat {sfile_path}')
    s = Sfile(sfile_path, use_mvo_parser=True, verbose=True, parse_aef=True, try_external_aeffile=True);
    print(s.eventobj)
    #print(s.to_enhancedevent())