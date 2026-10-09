from __future__ import annotations
import os
from pathlib import Path
import yaml


def load_yaml(path):
    path = Path(path).expanduser().resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, path


def load_project(path):
    cfg, path = load_yaml(path)
    root = os.path.expandvars(str(cfg["project_root"]))
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    cfg["_config_path"] = path
    cfg["_root"] = root
    return cfg


def run_dir(project, run_id):
    return project["_root"] / run_id


def master_sds(project):
    p = Path(project.get("paths", {}).get("master_sds", "SDS"))
    return p if p.is_absolute() else project["_root"] / p


def database_path(project):
    p = Path(project.get("paths", {}).get("database", "database/archive.sqlite"))
    return p if p.is_absolute() else project["_root"] / p
