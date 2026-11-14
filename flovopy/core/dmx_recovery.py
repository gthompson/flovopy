"""Conservative diagnostic recovery: only complete structures before first defect.

No resynchronization, byte edits, clock correction, ID mapping or count offset.
Candidates are never merged into the production waveform/event archive.
"""
from pathlib import Path
import csv,hashlib,json,re,struct,tempfile
import numpy as np
from obspy import read,UTCDateTime
from obspy.io.dmx.core import descript_trace_dtypes,types


def inspect_prefix(blob):
    pos=0;traces=[];reason='end_of_file'
    while pos<len(blob):
        if pos+12>len(blob):reason='incomplete_structure_tag';break
        sync,machine,kind,length,payload=struct.unpack_from('<cchii',blob,pos)
        if sync!=b'S' or machine!=b'6' or not 0<kind<100 or min(length,payload)<0 or length+payload<=0:
            reason='invalid_structure_tag';break
        end=pos+12+length+payload
        if end>len(blob):reason='structure_exceeds_file';break
        if kind==7:
            try:
                if length!=descript_trace_dtypes.itemsize:raise ValueError('unexpected trace descriptor size')
                desc=np.frombuffer(blob,dtype=descript_trace_dtypes,count=1,offset=pos+12)[0]
                dtype=desc['datatype'].decode('ascii');n=int(desc['length']);rate=float(desc['rate']);t=float(desc['begintime'])
                if dtype not in types or n<=0 or n*np.dtype(types[dtype]).itemsize!=payload:raise ValueError('sample count/data size mismatch')
                if not np.isfinite(rate) or not 0<rate<=10000:raise ValueError('invalid sampling rate')
                stamp=UTCDateTime(t)
                if not 1900<=stamp.year<=2100:raise ValueError('implausible recorded year')
                station=desc['st_name'].decode('ascii').strip();network=desc['network'].decode('ascii').strip();component=desc['component'].decode('ascii').strip()
                if not re.fullmatch(r'[A-Za-z0-9_]{1,5}',station):raise ValueError('invalid station text')
                data=np.frombuffer(blob,dtype=types[dtype],count=n,offset=pos+12+length)
                if not np.isfinite(data).all():raise ValueError('nonfinite waveform samples')
                traces.append({'offset':pos,'station':station,'network':network,'component':component,'npts':n,'rate':rate,'recorded_epoch':t,'constant':bool(np.all(data==data[0])),'irig':station.upper().startswith('IRIG')})
            except Exception as exc:reason='invalid_trace_descriptor: '+str(exc);break
        pos=end
    return pos,traces,reason


def compare_ascii(path,stream):
    """Compare complete ASCII trace blocks; an incomplete block is not evidence."""
    if not path or not Path(path).exists():return {'status':'not_available'}
    text=Path(path).read_bytes().decode('cp1252');blocks=re.split(r'(?m)^\$\s+',text);results=[];index=0
    for block in blocks:
        if not re.match(r'7\s+\d+\s*;',block):continue
        fields={};data=[];samples=False
        for line in block.splitlines():
            if ';' in line:
                value,key=line.split(';',1);fields[key.strip()]=value.strip()
                if key.strip()=='rate correction':samples=True
            elif samples:
                try:data.extend(float(v) for v in line.split())
                except ValueError:break
        try:n=int(fields['number of samples'])
        except (KeyError,ValueError):continue
        if len(data)!=n:results.append({'trace_index':index,'status':'incomplete_ascii','samples':len(data),'declared':n});index+=1;continue
        if index>=len(stream):results.append({'trace_index':index,'status':'extra_ascii_trace'});index+=1;continue
        trace=stream[index]
        results.append({'trace_index':index,'status':'match' if fields.get('station name')==trace.stats.station and n==trace.stats.npts and np.array_equal(trace.data,np.asarray(data)) else 'mismatch','samples':n,'ascii_station':fields.get('station name'),'binary_station':trace.stats.station,'different_samples':int(np.count_nonzero(trace.data!=np.asarray(data))) if n==trace.stats.npts else None})
        index+=1
    return {'status':'compared','traces':results}



def recover_dmx_prefix(path, *, ascii_file=None, guard_last_trace=True):
    """Return (Stream, report); never scan or splice beyond the first defect.

    Without an independent ASCII comparison, discard the trace immediately
    preceding the bad boundary as an additional corruption guard.
    """
    import io
    path=Path(path);blob=path.read_bytes();stop,descriptors,reason=inspect_prefix(blob)
    report={'method':'validated_prefix','source_sha256':hashlib.sha256(blob).hexdigest(),
            'source_bytes':len(blob),'valid_prefix_bytes':stop,'stop_reason':reason,
            'partial':stop<len(blob),'discarded_guard_trace':False,'ascii_compared':False}
    if not descriptors:return read_empty(),report
    stream=read(io.BytesIO(blob[:stop]),format='DMX')
    if len(stream)!=len(descriptors):raise ValueError('Trace descriptor/readback count mismatch')
    if ascii_file is not None and Path(ascii_file).exists():
        comparison=compare_ascii(ascii_file,stream);report['ascii_comparison']=comparison
        # Require an exact independent match for every accepted trace.
        accepted=0
        for record in comparison.get('traces',[]):
            if record['trace_index']!=accepted or record['status']!='match':break
            accepted+=1
        report['ascii_compared']=True
        if accepted<len(stream):
            stop=descriptors[accepted]['offset'];stream=stream[:accepted];descriptors=descriptors[:accepted]
    elif guard_last_trace and report['partial']:
        stop=descriptors[-1]['offset'];stream=stream[:-1];descriptors=descriptors[:-1];report['discarded_guard_trace']=True
    report.update(valid_prefix_bytes=stop,accepted_traces=len(stream),partial=stop<len(blob))
    for tr,desc in zip(stream,descriptors):
        tr.stats.dmx_recovery={**report,'trace_offset':desc['offset']}
    return stream,report


def read_empty():
    from obspy import Stream
    return Stream()
