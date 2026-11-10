"""File-based wrappers for external Windows Win-SUDS utilities.

Requires Win-SUDS executables on PATH or in ``tool_dir``; importing this module
requires only the standard library and works on other platforms. All public
wrappers stage a COPY of one input in a temporary working directory, verify
nonempty outputs, then publish them. No wildcard inputs, shell or global chdir.
IRIG corrects a copy by default; only ``inplace=True`` replaces its source.

Relative input/output paths are relative to ``cwd`` (the caller's working
directory when omitted). Returns a CompletedProcess-compatible WinSUDSResult
with persistent ``output_files``. Output checks establish file presence, not
scientific validity: inspect MiniSEED/SAC data and IRIG diagnostics separately.

References: https://banfill.net/suds/Win-SUDS.pdf (Win32 release/reference)
and https://www.geoinstr.com/pub/manuals/pc-suds.pdf (IRIG/DEMUX, chapter 7).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

__all__ = ['WinSUDSError', 'WinSUDSResult', 'find_tools', 'demux', 'irig',
           'sud2msed', 'sud2sac', 'sud2gse', 'sud2asc', 'sud2mat']
TOOLS = ('demux', 'irig', 'sud2msed', 'sud2sac', 'sud2gse', 'sud2asc', 'sud2mat')


class WinSUDSError(RuntimeError):
    """Invocation, timeout, nonzero exit or missing/empty output failure.

    ``command``, ``returncode``, ``stdout`` and ``stderr`` retain diagnostics.
    Failed utility runs publish no outputs; private temporary files are discarded.
    """
    def __init__(self, message, *, command=None, returncode=None, stdout='', stderr=''):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f'{message}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}')


class WinSUDSResult(subprocess.CompletedProcess):
    """CompletedProcess plus persistent absolute output paths and source path.

    ``args`` records the executed command, including temporary paths that no
    longer exist. ``output_files`` identifies the published files instead.
    """
    def __init__(self, process, source, output_files=()):
        super().__init__(process.args, process.returncode, process.stdout, process.stderr)
        self.source_file = source
        self.output_files = tuple(output_files)


def _executable(exe, tool_dir=None):
    name = exe if exe.lower().endswith('.exe') else exe + '.exe'
    if tool_dir is not None:
        path = Path(tool_dir).expanduser().resolve() / name
        return path if path.is_file() else None
    path = shutil.which(name)
    return Path(path).resolve() if path else None


def find_tools(tool_dir=None):
    """Return {tool: absolute executable Path or None}, without executing tools.

    An explicit tool_dir is authoritative: missing tools do not fall back to PATH.
    """
    return {name: _executable(name, tool_dir) for name in TOOLS}


def _run_suds_tool(exe, args, *, cwd=None, check=True, tool_dir=None,
                   timeout=None, encoding=None):
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be finite and positive, or None')
    exe_path = _executable(exe, tool_dir)
    if exe_path is None:
        raise WinSUDSError(f'{exe} not found in ' + (str(tool_dir) if tool_dir else 'PATH'))
    cmd = [str(exe_path), *map(str, args)]
    try:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=False,
            capture_output=True, text=True, encoding=encoding, errors='replace',
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(encoding or 'utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        raise WinSUDSError(f'{exe} timed out after {timeout} seconds', command=cmd,
            stdout=text(exc.stdout), stderr=text(exc.stderr)) from exc
    except OSError as exc:
        raise WinSUDSError(f'Cannot run {exe}: {exc}. Native Win-SUDS needs a compatible Windows host.', command=cmd) from exc
    if check and result.returncode:
        raise WinSUDSError(f'{exe} failed (exit code {result.returncode})', command=cmd,
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    return result


def _paths(input_file, cwd):
    root = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    if not root.is_dir():
        raise NotADirectoryError(root)
    value = str(input_file)
    if not value or any(ch in value for ch in '*?'):
        raise ValueError('Pass one input file, not a wildcard specification')
    source = Path(input_file).expanduser()
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, root


def _destination(value, root):
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _publish(files, overwrite, replace_source=None):
    # Check the entire group before publishing anything. Concurrent exporters to
    # the same targets are unsupported; I/O failures can leave a partial group.
    if len({dst for _, dst in files}) != len(files):
        raise ValueError('Multiple generated files map to the same destination')
    for src, dst in files:
        if dst.exists() and (not (overwrite or dst == replace_source) or not dst.is_file()):
            raise FileExistsError(dst)
        if not dst.parent.is_dir():
            raise NotADirectoryError(dst.parent)
    for src, dst in files:
        if overwrite or dst == replace_source:
            fd, name = tempfile.mkstemp(prefix='.winsuds-', dir=dst.parent)
            try:
                with os.fdopen(fd, 'wb') as handle, src.open('rb') as inp:
                    shutil.copyfileobj(inp, handle)
                os.replace(name, dst)
            finally:
                if os.path.exists(name): os.unlink(name)
        else:
            with dst.open('xb') as handle, src.open('rb') as inp:
                shutil.copyfileobj(inp, handle)


def _convert(tool, input_file, *, flags=(), tail=(), output_file=None,
             extension=None, implicit=None, mutate=False, inplace=False,
             diagnostics=False, cwd=None, tool_dir=None, timeout=None,
             overwrite=False, encoding=None):
    source, root = _paths(input_file, cwd)
    if inplace and not mutate:
        raise ValueError('inplace is only supported for IRIG')
    if inplace and output_file is not None:
        raise ValueError('output_file and inplace are mutually exclusive')
    target = source if inplace else (_destination(output_file, root) if output_file is not None else
             (root / (source.stem + '.irig' + source.suffix) if mutate else
              root / (source.stem + extension) if extension else None))
    if target == source and not inplace:
        raise ValueError('Output must differ from the input; IRIG requires explicit inplace=True')
    if target is not None:
        if not target.parent.is_dir(): raise NotADirectoryError(target.parent)
        if target.exists() and not (overwrite or inplace): raise FileExistsError(target)
    # Stage outside the supplied destination, using short basenames for legacy
    # utilities. Windows TEMP can still have a long path; use a short TMPDIR if needed.
    with tempfile.TemporaryDirectory(prefix='winsuds-') as name:
        stage = Path(name)
        staged_input = stage / ('input' + source.suffix)
        shutil.copyfile(source, staged_input)
        args = [*flags, staged_input.name, *tail]
        staged_output = stage / ('output' + (target.suffix if target else '.out'))
        if target is not None and not mutate: args.append(staged_output.name)
        process = _run_suds_tool(tool+'.exe', args, cwd=stage, tool_dir=tool_dir,
                                 timeout=timeout, encoding=encoding)
        if mutate:
            files = [(staged_input, target)]
            # Per-input logs replace the legacy shared CWD log to avoid mixing runs.
            for entry in stage.iterdir():
                if entry.name.lower() in ('irig.log', 'irig.asc'):
                    files.append((entry, root / (source.stem + '.irig' + entry.suffix.lower())))
            if diagnostics and not any(p.suffix.lower()=='.asc' for p,_ in files):
                raise WinSUDSError('IRIG did not create requested diagnostic IRIG.ASC', command=process.args,
                    returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        elif target is not None:
            files = [(staged_output, target)]
        elif implicit is not None:
            generated = sorted(p for p in stage.iterdir() if p.is_file() and p != staged_input and implicit(p))
            files = [(p, root / (source.stem + p.name[5:] if p.name.lower().startswith('input.') else p.name)) for p in generated]
        else:
            # SUD2ASC with no output filename deliberately returns captured stdout.
            files = []
        if (implicit is not None and not files) or any(not p.is_file() or not p.stat().st_size for p,_ in files):
            raise WinSUDSError(f'{tool} returned success without nonempty expected output', command=process.args,
                returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        _publish(files, overwrite, replace_source=source if inplace else None)
        return WinSUDSResult(process, source, [p for _,p in files])


def demux(input_file, output_file=None, *, bits=None, cwd=None, tool_dir=None,
          timeout=None, overwrite=False, encoding=None):
    """DEMUX [/12|/16] input output. Default output: cwd/<stem>.DMX.

    Applies existing SUDS timing corrections; does not infer a clock correction.
    """
    if bits is not None and (type(bits) is not int or bits not in (12,16)):
        raise ValueError('bits must be 12, 16, or None')
    return _convert('demux', input_file, flags=[f'/{bits}'] if bits else [],
        output_file=output_file, extension='.DMX', cwd=cwd, tool_dir=tool_dir,
        timeout=timeout, overwrite=overwrite, encoding=encoding)


def irig(suds_file, *, station=None, diagnostics=False, output_file=None,
         inplace=False, cwd=None, tool_dir=None, timeout=None, overwrite=False,
         encoding=None):
    """IRIG [/D] input [station]. Correct a COPY, default cwd/<stem>.irig<suffix>.

    The executable writes TIMECORRECTION structures. A successful exit does not
    prove a valid IRIG decode; review stdout and any per-input .irig.log/.irig.asc.
    Set inplace=True to explicitly replace the original after successful execution.
    """
    if station is not None and (not isinstance(station,str) or not station.strip() or station.startswith(('/', '-'))):
        raise ValueError('station must be a nonempty identifier, not a switch')
    return _convert('irig', suds_file, flags=['/D'] if diagnostics else [],
        tail=[station] if station is not None else [], mutate=True, inplace=inplace,
        diagnostics=diagnostics, output_file=output_file, cwd=cwd, tool_dir=tool_dir,
        timeout=timeout, overwrite=overwrite, encoding=encoding)


def sud2msed(suds_file, seed_file, *, cwd=None, tool_dir=None, timeout=None,
             overwrite=False, encoding=None):
    """SUD2MSED input output. Check nonempty output; no format/calibration claim."""
    return _convert('sud2msed', suds_file, output_file=seed_file, cwd=cwd,
        tool_dir=tool_dir, timeout=timeout, overwrite=overwrite, encoding=encoding)


def sud2sac(suds_file, *, little_endian=False, cwd=None, tool_dir=None,
            timeout=None, overwrite=False, encoding=None):
    """SUD2SAC [-l] input. Publish generated .Snn (hex) or .sac files."""
    return _convert('sud2sac', suds_file, flags=['-l'] if little_endian else [],
        implicit=lambda p:bool(re.fullmatch(r'\.(?:s[0-9a-f]+|sac)',p.suffix,re.I)),
        cwd=cwd, tool_dir=tool_dir, timeout=timeout, overwrite=overwrite, encoding=encoding)


def sud2gse(suds_file, *, cwd=None, tool_dir=None, timeout=None,
            overwrite=False, encoding=None):
    """SUD2GSE input. Publish generated .GSE files."""
    return _convert('sud2gse', suds_file, implicit=lambda p:p.suffix.lower()=='.gse',
        cwd=cwd, tool_dir=tool_dir, timeout=timeout, overwrite=overwrite, encoding=encoding)


def sud2asc(input_file, output_file=None, *, no_data=False, comments_only=False,
            quiet=False, date_format=None, cwd=None, tool_dir=None,
            timeout=None, overwrite=False, encoding=None):
    """SUD2ASC [switches] input [output]; no output file returns stdout.

    Retains original /N, /C, /Q and MD/J/U date-format switches.
    """
    dates={None:None,'MD':'/DM','J':'/DJ','U':'/DU'}
    if date_format not in dates:raise ValueError('date_format must be MD, J, U, or None')
    flags=[s for s,enabled in [('/N',no_data),('/C',comments_only),('/Q',quiet)] if enabled]
    if dates[date_format]:flags.append(dates[date_format])
    return _convert('sud2asc', input_file, flags=flags, output_file=output_file,
        cwd=cwd, tool_dir=tool_dir, timeout=timeout, overwrite=overwrite, encoding=encoding)


def sud2mat(suds_file, *, cwd=None, tool_dir=None, timeout=None,
            overwrite=False, encoding=None):
    """SUD2MAT input. Publish generated MATLAB v4 .MAT files."""
    return _convert('sud2mat', suds_file, implicit=lambda p:p.suffix.lower()=='.mat',
        cwd=cwd, tool_dir=tool_dir, timeout=timeout, overwrite=overwrite, encoding=encoding)
