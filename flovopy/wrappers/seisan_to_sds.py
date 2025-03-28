import os
import glob
from obspy import UTCDateTime, read
from flovopy.sds.sds import SDSobj
from flovopy.core.mvo import fix_trace_mvo

SECONDS_PER_DAY = 86400

def seisan_to_sds(seisandbdir, sdsdir, startt0, endt0, net, dbout=None, round_sampling_rate=True, MBWHZ_only=False):
    sdsobj = SDSobj(sdsdir)
    startt = UTCDateTime(startt0.date)
    endt = UTCDateTime(endt0.date)
    mseeddir = 'seisan2mseed'
    os.makedirs(mseeddir, exist_ok=True)
    os.makedirs(sdsdir, exist_ok=True)

    dayt = startt
    while dayt <= endt:
        ymd = dayt.strftime("%Y%m%d")
        yyyy, mm, dd = dayt.strftime("%Y %m %d").split()
        currentdb = f"{seisandbdir}/{yyyy}/{mm}"
        prevdb = f"{seisandbdir}/{UTCDateTime(dayt - SECONDS_PER_DAY).strftime('%Y/%m')}"

        allfiles = sorted(set(
            glob.glob(f"{prevdb}/{yyyy}-{mm}-{dd}-23[45]*S.MVO___*") +
            glob.glob(f"{currentdb}/{yyyy}-{mm}-{dd}*S.MVO___*")
        ))

        if not allfiles:
            dayt += SECONDS_PER_DAY
            continue

        print(f"Processing {ymd}: {len(allfiles)} files")
        firstfile = True
        for file in allfiles:
            try:
                st = read(file, format='SEISAN')
            except Exception:
                continue
            if net=='MV':
                for tr in st:
                    fix_trace_mvo(tr, legacy=False, netcode=net)
            if MBWHZ_only:
                st = st.select(station='MBWH', component='Z')
            st.trim(dayt, dayt + SECONDS_PER_DAY)
            sdsobj.stream = st
            sdsobj.write(overwrite=firstfile)
            firstfile = False

        if dbout:
            jday = dayt.strftime('%j')
            dboutday = f"{dbout}{ymd}"
            files = glob.glob(os.path.join(sdsdir, '*', '*', '*', '*.D', f'*{jday}'))
            os.system(f"miniseed2db {' '.join(files)} {dboutday}")

        dayt += SECONDS_PER_DAY







