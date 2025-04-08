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
    print(startt, endt)
    seisandbdir = os.path.join(seisandbdir, 'WAV', 'DSNC_')

    dayt = startt
    while dayt <= endt:
        ymd = dayt.strftime("%Y%m%d")
        yyyy, mm, dd = dayt.strftime("%Y %m %d").split()
        currentdb = f"{seisandbdir}/{yyyy}/{mm}"
        prevdb = f"{seisandbdir}/{UTCDateTime(dayt - SECONDS_PER_DAY).strftime('%Y/%m')}"
        print(ymd, prevdb, currentdb)
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

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert Seisan waveform files to SDS archive format")

    parser.add_argument("--start", type=str, required=True,
                        help="Start date in UTC (e.g., 2001-01-01T00:00:00)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date in UTC (e.g., 2001-01-02T00:00:00)")
    parser.add_argument("--seisan", type=str, required=True,
                        help="Path to Seisan database root directory (e.g., /data/SEISAN_DB)")
    parser.add_argument("--sds", type=str, required=True,
                        help="Path to output SDS archive directory")
    parser.add_argument("--net", type=str, required=True,
                        help="Network code (e.g., MV)")
    parser.add_argument("--dbout", type=str,
                        help="Optional Datascope database name prefix (e.g., SDS2DB/MVOE_)")
    parser.add_argument("--round_sampling_rate", action="store_true",
                        help="Round sampling rate to nearest integer Hz")
    parser.add_argument("--MBWHZ_only", action="store_true",
                        help="Only include MBWH station Z component")

    args = parser.parse_args()

    seisan_to_sds(
        seisandbdir=args.seisan,
        sdsdir=args.sds,
        startt0=UTCDateTime(args.start),
        endt0=UTCDateTime(args.end),
        net=args.net,
        dbout=args.dbout,
        round_sampling_rate=args.round_sampling_rate,
        MBWHZ_only=args.MBWHZ_only
    )

if __name__ == "__main__":
    main()
"""
run-seisan2sds \
  --start 2001-01-01T00:00:00 \
  --end 2001-01-02T00:00:00 \
  --seisan /data/SEISAN_DB \
  --sds SDS_ARCHIVE \
  --net MV \
  --dbout SDS2DB/MVOE_ \
  --MBWHZ_only

"""





