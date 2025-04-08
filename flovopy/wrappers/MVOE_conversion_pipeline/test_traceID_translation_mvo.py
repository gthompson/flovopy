# test ID translation
from flovopy.core.mvo import correct_nslc_mvo
while True:
    original_id = input('Enter trace ID ?')
    Fs = input('Enter sampling rate, or enter for None')
    if len(Fs) == 0:
        Fs = 0
    else:
        Fs = float(Fs)
    new_id = correct_nslc_mvo(original_id, Fs)
    print(f'{original_id} -> {new_id}')