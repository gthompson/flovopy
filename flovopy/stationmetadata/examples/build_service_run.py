"""Run: python build_service_run.py metadata.xlsx 20260922 output_dir"""
import sys
from pathlib import Path
from flovopy.stationmetadata.deployments import get_tables,validate,stationxml_for_run,export_css
src,run,out=sys.argv[1:4]
t=get_tables(src)
issues=validate(t)
if issues: raise SystemExit('\n'.join(issues))
p=Path(out);p.mkdir(parents=True,exist_ok=True)
print(stationxml_for_run(t,run,p/f'{run}.stationxml',require_responses=False))
print(export_css(t,run,p/'css'))
print('PRELIMINARY: supply verified response templates and set require_responses=True before EPIC submission')
