import os
from flovopy.seisanio.core.sfile import Sfile
os.system('clear')
for sfile_path in [
    '/data/SEISAN_DB/REA/MVOE_/2007/03/01-0429-34L.S200703', 
    '/data/SEISAN_DB/REA/MVOE_/2005/04/20-1541-26R.S200504', 
    '/data/SEISAN_DB/REA/MVOE_/1997/01/14-1803-38R.S199701' 
    ]:

    print('0123456789'*8)
    print(f'Sfile {sfile_path} contents:')
    os.system(f'cat {sfile_path}')
    s = Sfile(sfile_path, use_mvo_parser=True, verbose=True, parse_aef=True, try_external_aeffile=True);
    print(s)
    print(s.eventobj)
    for p in s.eventobj.picks:
        print(p)

    print(s.to_enhancedevent())