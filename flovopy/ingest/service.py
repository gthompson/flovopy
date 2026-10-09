from __future__ import annotations
import shutil
from datetime import date
from pathlib import Path
import yaml
from .config import load_project, load_yaml, run_dir, master_sds, database_path
from flovopy.sds.merge_sds_archives import merge_sds_archives, discover_sds_files

INSTRUMENT_DIRS = ("Centaur", "Gem", "SiliconAudio", "Guralp", "SmartSolo", "Pegasus", "Q330")


def _run_id(start_date: str) -> str:
    return f"{start_date.replace('-', '')}_service"


def new_service_run(project_yaml, start_date, end_date=None, run_id=None):
    project = load_project(project_yaml)
    run_id = run_id or _run_id(str(start_date))
    root = run_dir(project, run_id)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "00_download"
    for name in INSTRUMENT_DIRS:
        (raw / name).mkdir(parents=True, exist_ok=True)
    (root / "10_conversion").mkdir(exist_ok=True)
    (root / "20_archive").mkdir(exist_ok=True)
    (root / "30_qc").mkdir(exist_ok=True)
    cfg = {
        "service_run": {"id": run_id, "start_date": str(start_date), "end_date": end_date},
        "paths": {"raw": "00_download", "conversion": "10_conversion", "archive": "20_archive", "qc": "30_qc"},
        "sources": {
            "centaur": {"enabled": True, "input": "00_download/Centaur", "mode": "fast"},
            "gem": {"enabled": True, "raw": "00_download/Gem/raw", "converted": "10_conversion/Gem/mseed"},
            "silicon_audio": {"enabled": True, "input": "00_download/SiliconAudio"},
            "guralp": {"enabled": True, "input": "00_download/Guralp"},
            "smartsolo": {"enabled": True, "input": "00_download/SmartSolo"},
            "pegasus": {"enabled": False, "input": "00_download/Pegasus"},
            "q330": {"enabled": False, "input": "00_download/Q330"},
        },
    }
    with (root / "service_run.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return root


def load_run(project_yaml, run_id):
    project = load_project(project_yaml)
    root = run_dir(project, run_id)
    cfg, _ = load_yaml(root / "service_run.yaml")
    return project, cfg, root


def ingest_centaur(project_yaml, run_id, *, mode=None, dry_run=False, source=None, progress_every=50):
    project, cfg, root = load_run(project_yaml, run_id)
    scfg = cfg.get("sources", {}).get("centaur", {})
    if not scfg.get("enabled", True):
        return {"status": "disabled", "instrument": "centaur"}
    src = Path(source).expanduser() if source else root / scfg.get("input", "00_download/Centaur")
    if source is None and not src.exists():
        # Historical consolidated service-run archives may already contain
        # Centaur, Gem, and other SDS channels; do not silently substitute them.
        pass
    if not src.exists():
        return {"status": "no_source", "instrument": "centaur", "source": str(src)}
    count = sum(1 for _ in discover_sds_files(src))
    if count == 0:
        raise ValueError(f"No SDS files found in {src}; if this Centaur was not configured for SDS, use stage-instrument centaur first")
    mode = mode or scfg.get("mode", "fast")
    target = master_sds(project)
    target.mkdir(parents=True, exist_ok=True)
    # Generic FLOVOpy merger keeps its own transaction DB/cache beside the master SDS.
    summary = merge_sds_archives(src, target, mode=mode, dry_run=dry_run, progress_every=progress_every)
    result = summary.as_dict()
    result.update({"source": str(src), "target": str(target), "instrument": "centaur", "source_files": count})
    return result


def ingest_siliconaudio(project_yaml, run_id, *, station=None, year=None, dry_run=False,
                        pattern="*.ms", strict=True, verbose=True):
    """Stage SiliconAudio minute MiniSEED as SDS; do not merge into master."""
    from .siliconaudio import convert_siliconaudio
    project, cfg, root = load_run(project_yaml, run_id)
    scfg = cfg.get("sources", {}).get("silicon_audio", {})
    if not scfg.get("enabled", True):
        return {"status": "disabled", "instrument": "silicon_audio"}
    base = root / scfg.get("input", "00_download/SiliconAudio")
    if not base.is_dir():
        raise FileNotFoundError(base)
    stations = [base / station] if station else sorted(p for p in base.iterdir() if p.is_dir())
    if not stations:
        raise ValueError(f"No SiliconAudio station directories in {base}")
    output = {}
    for station_dir in stations:
        data = station_dir / "data"
        if not data.is_dir():
            raise FileNotFoundError(data)
        # Keep output isolated from canonical SDS until reviewed and merged.
        stage = root / "20_archive" / "SiliconAudio" / "SDS"
        output[station_dir.name] = convert_siliconaudio(
            data, stage, year=year or int(str(cfg["service_run"]["start_date"])[:4]),
            pattern=pattern, dry_run=dry_run, strict=strict, verbose=verbose)
    return output


def ingest_gem(project_yaml, run_id, *, mapping=None, dry_run=False,
               pattern='*.mseed', strict=True, verbose=True):
    """Stage gemconvert MiniSEED in 20_archive/Gem/SDS, not master SDS."""
    from .gem import stage_gem_mseed
    project, cfg, root = load_run(project_yaml, run_id)
    scfg = cfg.get('sources', {}).get('gem', {})
    if not scfg.get('enabled', True):
        return {'status': 'disabled', 'instrument': 'gem'}
    source = root / scfg.get('converted', '10_conversion/Gem/mseed')
    stage = root / '20_archive' / 'Gem' / 'SDS'
    return stage_gem_mseed(source, stage, mapping=mapping, pattern=pattern,
                           dry_run=dry_run, strict=strict, verbose=verbose)


def ingest_run(project_yaml, run_id, instruments=None, *, dry_run=False, centaur_mode=None, progress_every=50):
    wanted = [x.lower() for x in (instruments or ["centaur"])]
    out = {}
    if "centaur" in wanted:
        out["centaur"] = ingest_centaur(project_yaml, run_id, mode=centaur_mode, dry_run=dry_run, progress_every=progress_every)
    if "silicon_audio" in wanted or "siliconaudio" in wanted:
        out["silicon_audio"] = ingest_siliconaudio(project_yaml, run_id, dry_run=dry_run)
    if "gem" in wanted:
        out["gem"] = ingest_gem(project_yaml, run_id, dry_run=dry_run)
    for name in ("pegasus", "q330"):
        if name in wanted:
            scfg = cfg.get("sources", {}).get(name, {})
            if not scfg.get("enabled", False):
                out[name] = {"status": "disabled"}
            else:
                from .epic import stage_miniseed
                out[name] = stage_miniseed(root / scfg.get("input", f"00_download/{name}"),
                    root / "20_archive" / name / "SDS", instrument=name, dry_run=dry_run)
    # Other adapters are deliberately deferred until their real acquisition/conversion
    # workflows are finalized. Do not pretend that generic MiniSEED ingestion is correct.
    for name in wanted:
        if name not in ("centaur", "silicon_audio", "siliconaudio", "gem", "pegasus", "q330"):
            out[name] = {"status": "not_implemented", "message": "Adapter intentionally deferred; raw download is preserved."}
    return out


def status(project_yaml, run_id):
    project, cfg, root = load_run(project_yaml, run_id)
    rows = []
    for key, scfg in cfg.get("sources", {}).items():
        rel = scfg.get("input") or scfg.get("raw")
        p = root / rel if rel else None
        rows.append({"instrument": key, "enabled": scfg.get("enabled", True), "path": str(p) if p else None, "exists": bool(p and p.exists())})
    return {"run": run_id, "run_dir": str(root), "master_sds": str(master_sds(project)), "sources": rows}


def ingest_sds(project_yaml, source, *, mode="fast", dry_run=False, progress_every=50):
    """Ingest ANY existing SDS tree (mixed instruments are allowed)."""
    project = load_project(project_yaml)
    src = Path(source).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"SDS source directory not found: {src}")
    count = sum(1 for _ in discover_sds_files(src))
    if not count:
        raise ValueError(f"No SDS files found in {src}; refusing an empty ingest")
    target = master_sds(project)
    if src.resolve() == target.resolve() or target.resolve().is_relative_to(src.resolve()):
        raise ValueError("Source must not contain the master SDS target")
    summary = merge_sds_archives(src, target, mode=mode, dry_run=dry_run, progress_every=progress_every)
    return {**summary.as_dict(), "source": str(src), "target": str(target), "source_files": count}
