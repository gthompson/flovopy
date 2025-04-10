import os
from flovopy.seisanio.core.sfile import Sfile
sfile_path = '/data/SEISAN_DB/REA/MVOE_/2007/03/01-0429-34L.S200703'
print('0123456789'*8)
print(f'Sfile {sfile_path} contents:')
os.system(f'cat {sfile_path}')
s = Sfile(sfile_path, use_mvo_parser=True, verbose=True, parse_aef=False)
print(' ')
print(s)
print(' ')
for k,v in s.to_dict().items():
    print(k, v)