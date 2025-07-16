from flovopy.sds.sds2 import SDSobj
import obspy
sdsin=SDSobj('/data/KSC/beforePASSCAL/CONTINUOUS')
sdsin.read(obspy.UTCDateTime(2016,2,26,18,0,0), obspy.UTCDateTime(2016,2,26,19,0,0), speed=1)
st1 = sdsin.stream.copy()
sdsin.read(obspy.UTCDateTime(2016,2,26,18,0,0), obspy.UTCDateTime(2016,2,26,19,0,0), speed=2)
st2 = sdsin.stream.copy()
print(st1)
print(st2)
st1.plot(equal_scale=False, outfile='st1.png')
st2.plot(equal_scale=False, outfile='st2.png')
