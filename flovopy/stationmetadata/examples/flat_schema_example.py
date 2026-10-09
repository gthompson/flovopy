"""Inspect migration issues; no automatic archival StationXML without responses."""
from stationmetadata.flat_schema import read_flat, validate_flat
rows=read_flat('KSC_metadata_flat_v2.xlsx')
issues=validate_flat(rows,strict=False)
print(f'{len(rows)} channel epochs; {sum(i["severity"]=="error" for i in issues)} errors; {sum(i["severity"]=="warning" for i in issues)} warnings')
