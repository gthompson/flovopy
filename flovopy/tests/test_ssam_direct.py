import json
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flovopy.legacy.bob import read_ssam, read_ssam_band_definitions, ssam_to_spectrograms
from flovopy.processing.spectrograms import BandSpectrogramResult, plot_band_spectrograms, plot_daily_band_spectrograms

BANDS=[f'BN{i:02d}' for i in range(1,11)]
BROAD=[f'BN{i:02d}' for i in range(11,17)]
LIMITS=[(i-.5,i+.5) for i in range(1,11)]+[(.7,10),(1.5,3),(3,4.5),(4.5,6),(1,5),(15,25)]
DOC='\n'.join(f'BN{i+1:02d} {lo} Hz to {hi} Hz (source description ??)' for i,(lo,hi) in enumerate(LIMITS))

def fixture(tmp_path):
 root=tmp_path/'SSAM';root.mkdir()
 (root/'BANDS.TXT').write_text(DOC)
 for name in ['STA1','STA2']:
  folder=root/name;folder.mkdir()
  lines=[]
  for minute in ['00:00','00:01','00:03']:
   lines.append('01 JUN 91 '+minute+' '+' '.join(str(i+1) for i in range(16)))
  lines.append('02 JUN 91 00:00 '+' '.join(['-998']+['20']*15))
  (folder/'01JUN91.DAT').write_text('\n'.join(lines))
 return read_ssam(root,band_definitions='documents')

def test_source_definitions_overlap_and_validation():
 defs=read_ssam_band_definitions({'a':DOC,'b':DOC})
 assert len(defs)==16 and defs[10]['low_hz']==.7 and len(defs[0]['sources'])==2
 with pytest.raises(ValueError,match='Conflicting'):
  read_ssam_band_definitions({'a':DOC,'b':'BN01 1 Hz to 2 Hz'})
 with pytest.raises(ValueError,match='Missing'):
  read_ssam_band_definitions({'a':'BN01 0.5 Hz to 1.5 Hz'})

def test_adapter_values_clock_and_missing(tmp_path):
 ds=fixture(tmp_path);before=ds.data.copy(deep=True)
 results=ssam_to_spectrograms(ds,interval_seconds=60)
 assert len(results)==2 and results[0].values.shape==(16,4)
 assert np.isnan(results[0].values[0,-1])
 assert results[0].values[10,0]==11
 assert results[0].times[0]==np.datetime64('1991-06-01')
 assert results[0].metadata['clock_correction_applied'] is False
 assert results[0].metadata['averaging_window_seconds'] is None
 assert ds.data.equals(before)
 ds.data.loc[1,'recorded_time']=ds.data.loc[0,'recorded_time']
 with pytest.raises(ValueError,match='Duplicate'):ssam_to_spectrograms(ds,interval_seconds=60)

def test_plot_gaps_and_noncontiguous_band_guard(tmp_path):
 ds=fixture(tmp_path);results=ssam_to_spectrograms(ds,interval_seconds=60)
 original=results[0].values.copy()
 fig=plot_band_spectrograms(results,spectral_bands=BANDS,broad_bands=BROAD,
   start='1991-06-01',end='1991-06-02',color_scale='shared')
 # 3 retained minute rows * 10 bands; absent 00:02 receives no colored cell.
 assert len(fig.axes[0].collections[0].get_paths())==30
 paths=fig.axes[0].collections[0].get_paths()
 assert np.isclose(paths[0].vertices[1,0]-paths[0].vertices[0,0],60/86400)
 assert fig.axes[0].collections[0].norm is fig.axes[2].collections[0].norm
 assert any(np.isnan(line.get_ydata()).any() for line in fig.axes[1].lines)
 assert np.array_equal(results[0].values,original,equal_nan=True)
 plt.close(fig)
 with pytest.raises(ValueError,match='nonoverlapping'):
  plot_band_spectrograms(results)  # all 16 would falsely overlap on frequency axis
 with pytest.raises(ValueError,match='overlap'):
  plot_band_spectrograms(ssam_to_spectrograms(ds,interval_seconds=600),spectral_bands=BANDS)

def test_batch_daily_and_hourly_bounds_and_overwrite(tmp_path):
 results=ssam_to_spectrograms(fixture(tmp_path),interval_seconds=60)
 out=tmp_path/'pages'
 paths=plot_daily_band_spectrograms(results,out,spectral_bands=BANDS,broad_bands=BROAD,
    start='1991-06-01',end='1991-06-03')
 assert len(paths)==2 and all(p.exists() for p in paths)
 manifest=json.loads((out/'ssam_pages.json').read_text())
 assert manifest['channels'][0]['band_limits_hz'][15]==[15,25]
 with pytest.raises(FileExistsError):plot_daily_band_spectrograms(results,out,spectral_bands=BANDS)
 hours=plot_daily_band_spectrograms(results,tmp_path/'hours',page='hour',spectral_bands=BANDS,
    start='1991-06-01',end='1991-06-02')
 assert len(hours)==1
 assert not plt.get_fignums()

def test_result_validation():
 with pytest.raises(ValueError,match='sorted and unique'):
  BandSpectrogramResult('x',['1991-06-01','1991-06-01'],[[1,2]],[[1,2]],['A'],60)
 with pytest.raises(ValueError,match='positive'):
  BandSpectrogramResult('x',['1991-06-01'],[[1]],[[1,2]],['A'],0)


def test_explicit_mask_policy_keeps_conflicts_in_audit(tmp_path):
    ds=fixture(tmp_path)
    ds.data.loc[1,'recorded_time']=ds.data.loc[0,'recorded_time']
    results=ssam_to_spectrograms(ds,interval_seconds=60,duplicate_policy='mask')
    first=results[0]
    assert first.values.shape==(16,3)
    assert np.isnan(first.values[:,0]).all()
    assert first.metadata['ambiguous_timestamp_count']==1
    assert len(first.metadata['duplicate_time_records'])==2
    assert len(ds.data)==8
