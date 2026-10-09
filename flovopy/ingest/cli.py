from __future__ import annotations
import argparse, json
from .service import new_service_run, ingest_run, ingest_centaur, ingest_sds, ingest_siliconaudio, ingest_gem, status
from .config import load_project, master_sds
from flovopy.sds.merge_sds_archives import rollback_merge_session


def main(argv=None):
    p = argparse.ArgumentParser(prog="flovopy-ingest")
    p.add_argument("--config", default=None)
    sp = p.add_subparsers(dest="cmd", required=True)
    n = sp.add_parser("new-run"); n.add_argument("start_date"); n.add_argument("--end-date")
    i = sp.add_parser("ingest"); i.add_argument("run_id"); i.add_argument("--instrument", action="append"); i.add_argument("--dry-run", action="store_true"); i.add_argument("--centaur-mode", choices=["fast", "slow"]); i.add_argument("--progress-every", type=int, default=50)
    c = sp.add_parser("centaur"); c.add_argument("run_id"); c.add_argument("--source"); c.add_argument("--mode", choices=["fast", "slow"], default="fast"); c.add_argument("--dry-run", action="store_true"); c.add_argument("--progress-every", type=int, default=50)
    a = sp.add_parser("merge-sds", help="Merge any existing SDS tree, including mixed-instrument archives")
    a.add_argument("source"); a.add_argument("--mode", choices=["fast", "slow"], default="fast"); a.add_argument("--dry-run", action="store_true"); a.add_argument("--progress-every", type=int, default=50)
    sa = sp.add_parser("siliconaudio", help="Convert station minute MiniSEED to staged SDS")
    sa.add_argument("run_id"); sa.add_argument("--station"); sa.add_argument("--year", type=int)
    sa.add_argument("--pattern", default="*.ms"); sa.add_argument("--dry-run", action="store_true")
    sa.add_argument("--allow-partial-days", action="store_true")
    gm = sp.add_parser("gem", help="Stage gemconvert MiniSEED into Gem/SDS")
    gm.add_argument("run_id"); gm.add_argument("--mapping")
    gm.add_argument("--pattern", default="*.mseed")
    gm.add_argument("--dry-run", action="store_true")
    gm.add_argument("--allow-unmapped", action="store_true")
    gps = sp.add_parser("gem-gps", help="Summarize gemconvert GPS logs as a QC CSV")
    gps.add_argument("gps_dir"); gps.add_argument("output_csv"); gps.add_argument("--mapping")
    gr = sp.add_parser("gem-report", help="Summarize Gem waveforms, GPS, metadata into one CSV")
    gr.add_argument("converted_root", help="Directory containing mseed/, gps/, metadata/")
    gr.add_argument("output_csv"); gr.add_argument("--mapping", required=True)
    gr.add_argument("--daily-csv"); gr.add_argument("--deployment-start")
    gr.add_argument("--deployment-end")
    gc = sp.add_parser("gem-convert", help="Run upstream gemconvert in a prepared raw/ workspace")
    gc.add_argument("workspace"); gc.add_argument("--executable", default="gemconvert")
    r = sp.add_parser("rollback"); r.add_argument("transaction_id"); r.add_argument("--force", action="store_true")
    s = sp.add_parser("status"); s.add_argument("run_id")
    # Generic direct-path operations do not need project configuration.
    direct = sp.add_parser("stage-gem", help="Convert Gem hourly MiniSEED to staged SDS")
    direct.add_argument("input_dir"); direct.add_argument("output_sds")
    direct.add_argument("--mapping", required=True); direct.add_argument("--pattern", default="*.mseed")
    direct.add_argument("--dry-run", action="store_true"); direct.add_argument("--allow-unmapped", action="store_true")
    direct_sa = sp.add_parser("stage-siliconaudio", help="Convert a SiliconAudio station data directory to SDS")
    direct_sa.add_argument("input_dir"); direct_sa.add_argument("output_sds")
    direct_sa.add_argument("--year", type=int, required=True); direct_sa.add_argument("--pattern", default="*.ms")
    direct_sa.add_argument("--dry-run", action="store_true"); direct_sa.add_argument("--allow-partial-days", action="store_true")
    ep = sp.add_parser("epic-bud", help="Export SDS to EPIC DAYS layout")
    ep.add_argument("sds_root"); ep.add_argument("days_root")
    ep.add_argument("--execute", action="store_true"); ep.add_argument("--validate", action="store_true")
    ep.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    ep.add_argument("--overwrite-links", action="store_true"); ep.add_argument("--no-soh", action="store_true")
    ev = sp.add_parser("epic-validate", help="Validate staged SDS and create JSON/CSV reports")
    ev.add_argument("sds_root"); ev.add_argument("--json"); ev.add_argument("--csv")
    ev.add_argument("--no-header-check", action="store_true")
    es = sp.add_parser("stage-instrument", help="Stage raw Centaur/Pegasus/Q330 MiniSEED into fresh SDS")
    es.add_argument("instrument", choices=["centaur", "pegasus", "q330"])
    es.add_argument("raw_root"); es.add_argument("output_sds")
    es.add_argument("--execute", action="store_true"); es.add_argument("--allow-errors", action="store_true")
    args = p.parse_args(argv)
    if args.cmd in {"new-run", "ingest", "centaur", "merge-sds", "siliconaudio", "gem", "rollback", "status"} and not args.config:
        p.error("--config PROJECT.yaml is required for project-oriented commands")
    if args.cmd == "epic-bud":
        from .sds_to_bud import export_sds_to_bud
        result = export_sds_to_bud(args.sds_root, args.days_root, mode=args.mode,
            dry_run=not args.execute, overwrite_links=args.overwrite_links,
            include_soh=not args.no_soh, validate=args.validate)
    elif args.cmd == "epic-validate":
        from .epic import validate_sds
        result = validate_sds(args.sds_root, inspect_headers=not args.no_header_check,
            output_json=args.json, output_csv=args.csv)
        result = {k:v for k,v in result.items() if k != 'rows'}
    elif args.cmd == "stage-instrument":
        from .epic import stage_miniseed
        result = stage_miniseed(args.raw_root, args.output_sds, instrument=args.instrument,
            dry_run=not args.execute, strict=not args.allow_errors)
    elif args.cmd == "stage-gem":
        from .gem import stage_gem_mseed
        result = stage_gem_mseed(args.input_dir, args.output_sds, mapping=args.mapping,
                                 pattern=args.pattern, dry_run=args.dry_run,
                                 strict=not args.allow_unmapped)
    elif args.cmd == "stage-siliconaudio":
        from .siliconaudio import convert_siliconaudio
        result = convert_siliconaudio(args.input_dir, args.output_sds, year=args.year,
                                      pattern=args.pattern, dry_run=args.dry_run,
                                      strict=not args.allow_partial_days)
    elif args.cmd == "new-run": result = str(new_service_run(args.config, args.start_date, args.end_date))
    elif args.cmd == "ingest": result = ingest_run(args.config, args.run_id, args.instrument, dry_run=args.dry_run, centaur_mode=args.centaur_mode, progress_every=args.progress_every)
    elif args.cmd == "centaur": result = ingest_centaur(args.config, args.run_id, mode=args.mode, dry_run=args.dry_run, source=args.source, progress_every=args.progress_every)
    elif args.cmd == "merge-sds": result = ingest_sds(args.config, args.source, mode=args.mode, dry_run=args.dry_run, progress_every=args.progress_every)
    elif args.cmd == "siliconaudio":
        result = ingest_siliconaudio(args.config, args.run_id, station=args.station,
                                    year=args.year, pattern=args.pattern,
                                    dry_run=args.dry_run, strict=not args.allow_partial_days)
    elif args.cmd == "gem":
        result = ingest_gem(args.config, args.run_id, mapping=args.mapping,
                            pattern=args.pattern, dry_run=args.dry_run,
                            strict=not args.allow_unmapped)
    elif args.cmd == "gem-gps":
        from .gem import summarize_gem_gps
        df = summarize_gem_gps(args.gps_dir, args.output_csv, mapping=args.mapping)
        result = {"output_csv": args.output_csv, "stations": len(df)}
    elif args.cmd == "gem-report":
        from .gem_report import report_gem_deployment
        result = report_gem_deployment(args.converted_root, args.mapping, args.output_csv,
                                       daily_csv=args.daily_csv, deployment_start=args.deployment_start,
                                       deployment_end=args.deployment_end)
    elif args.cmd == "gem-convert":
        from .gem import run_gemconvert
        result = {"mseed_dir": str(run_gemconvert(args.workspace, executable=args.executable))}
    elif args.cmd == "rollback":
        target = master_sds(load_project(args.config))
        result = rollback_merge_session(args.transaction_id, db_path=target / ".merge_tracking.sqlite", force=args.force)
    else: result = status(args.config, args.run_id)
    print(json.dumps(result, indent=2, default=str) if not isinstance(result, str) else result)
    if args.cmd == "epic-bud" and result.get("issues"):
        return 1
    if args.cmd == "epic-validate" and result.get("errors"):
        return 1
    if args.cmd == "stage-instrument" and result.get("errors"):
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
