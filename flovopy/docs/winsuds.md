# Win-SUDS wrappers

`flovopy.core.winsuds` wraps seven external utilities: DEMUX, IRIG, SUD2MSED, SUD2SAC, SUD2GSE, SUD2ASC and SUD2MAT. It requires only Python's standard library at import time. Actual conversions require a compatible Windows host and the Win-SUDS executable distribution; these executables are not bundled, downloaded or replaced by FLOVOpy. FLOVOpy's existing ObsPy-based `vdap.read_dmx_file` remains a separate reader.

The original command flags are preserved. Executables are resolved using PATH, or an explicit `tool_dir` which never falls back to another installation. Configure the normal Win-SUDS environment, including SUDS_ETC when required, before starting Python. There is no automatic Wine execution.

```python
from pathlib import Path
from flovopy.core import winsuds

print(winsuds.find_tools(tool_dir=r'C:\suds\bin'))
work = Path(r'C:\work')  # existing destination directory

result = winsuds.sud2msed(
    work / '91060301.DMX', '91060301.mseed',
    cwd=work, tool_dir=r'C:\suds\bin', timeout=120,
)
print(result.output_files)  # persistent absolute paths
print(result.stdout, result.stderr)
```

All relative input/output filenames are resolved against `cwd`, or the caller's working directory if omitted. Destination directories must already exist. Pass one file per call; wildcard specifications are deliberately rejected. DEMUX defaults to `<input-stem>.DMX` in `cwd`; explicitly choose another name when the input already has that destination name. `sud2asc` without an output filename returns text in `result.stdout`.

Every wrapper runs in its own temporary directory with a copied input and short local filenames. It does not change the Python process's working directory. Successful expected outputs must exist and be nonempty. Stale output files cannot masquerade as a successful conversion. SAC `.Snn`/`.sac`, GSE `.GSE` and MATLAB `.MAT` products are discovered there; a staging `input` prefix is replaced with the original input stem when publishing. Other generated names are preserved. Temporary directories are discarded after the call, including on failure. Configure Windows TEMP/TMP to a short, accessible path if your utility build has path-length restrictions; short staging filenames do not guarantee that every legacy build accepts every temporary path.

Existing destination files raise `FileExistsError` unless `overwrite=True`. Existing outputs are replaced only after conversion/output checks pass. Group collisions are checked before any files are published. Each replacement is atomic, but a multi-file publication is not a transaction: an I/O failure can leave a partially published group. Concurrent jobs targeting the same destinations are unsupported. Filesystem errors propagate with their native exception types.

Return values subclass `subprocess.CompletedProcess`, preserving `.args`, `.returncode`, `.stdout`, `.stderr`, and `.check_returncode()`, and add `.source_file` and `.output_files`. `.args` retains temporary command paths for provenance; use `.output_files` to open the published files. Missing executables, launch failures, timeouts, nonzero exit codes and missing/empty products raise `WinSUDSError`, with `.command`, `.returncode`, `.stdout` and `.stderr` where available. `encoding` optionally selects output decoding; undecodable bytes are replaced rather than aborting conversion. `timeout=None` preserves an unlimited runtime; supply a positive timeout for unattended use.

## IRIG changes the file it reads

The PC-SUDS manual describes IRIG writing TIMECORRECTION structures into its input. DEMUX subsequently applies existing timing corrections. This is why the new wrapper stages IRIG on a copy by default:

```python
result = winsuds.irig(
    work / '91060301.DMX', diagnostics=True,
    cwd=work, tool_dir=r'C:\suds\bin', timeout=120,
)
# Default corrected copy: work / '91060301.irig.DMX'
# Logs if produced: 91060301.irig.log and 91060301.irig.asc
print(result.output_files)
```

Use `output_file=` for a different corrected-copy destination. Use `inplace=True` only when you explicitly intend to replace the original after successful execution; it is mutually exclusive with `output_file`. `inplace=True` does not authorize replacing existing log files: use `overwrite=True` for those. The default no longer modifies original inputs. Per-input logs replace the original wrapper's shared, appended CWD IRIG.LOG behaviour.

A nonempty copied file and exit code zero **do not establish a valid IRIG decode or scientific timing accuracy**. Review the saved log, diagnostics and status output before accepting or applying time corrections. File presence checks likewise do not validate MiniSEED/SAC contents, waveform completeness, calibration or the absolute time standard. No new clock correction or response calibration is inferred by these wrappers.

## Migration and validation

Existing names and positional arguments are retained. New keyword options are `tool_dir`, `timeout`, `overwrite` and `encoding`; IRIG additionally accepts `output_file` and `inplace`. Overwriting is now opt-in. `suds_wrappers/winsuds.py` in the Pinatubo project forwards to this authoritative module, so existing imports can continue while new code imports FLOVOpy directly. No Pinatubo station names, network codes, data paths or scientific policies live in this module.

Tests cover all seven wrapper calls, output staging/publication, failure diagnostics, missing and empty outputs, timeout and launch failures, relative paths, paths with spaces, no global chdir, input/output identity, overwrite/collision behaviour and safe/explicit-inplace IRIG. A POSIX executable fixture also exercises the actual subprocess path; this is not the real Win-SUDS program. Run:

```
python -m unittest flovopy.tests.test_winsuds
```

A real Windows integration check remains necessary: use a disposable representative SUDS file with your executable versions, review command diagnostics and IRIG status, and read waveform products back with ObsPy. Alternate utility builds may have different implicit filenames. Unexpected names fail output checks rather than being silently accepted.

References: [Win-SUDS 2.90 documentation](https://banfill.net/suds/Win-SUDS.pdf), [PC-SUDS manual, IRIG and DEMUX chapter](https://www.geoinstr.com/pub/manuals/pc-suds.pdf). Flags for the conversion programs preserve those in the supplied wrapper; they have not been certified against installed Windows binaries here.
