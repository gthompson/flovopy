#!/usr/bin/env python3
# normalize_and_clean.py

import argparse
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

# ----------------------------
# Normalization helpers
# ----------------------------

def normalize_name(name: str) -> str:
    s = name
    s = re.sub(r'[ \-:;,]', '_', s)   # spaces, '-', ':' -> '_'
    s = s.replace('&', '_and_')     # '&' -> '_and_'
    s = re.sub(r'_+', '_', s)       # collapse multiple underscores
    s = s.strip('_')                # strip leading/trailing underscores
    return s or name                # never return empty

def samefile_safe(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except FileNotFoundError:
        return False

def unique_path(target: Path) -> Path:
    """If target exists, append __N before the suffix to avoid collision."""
    if not target.exists():
        return target
    stem, suffix, parent = target.stem, target.suffix, target.parent
    i = 1
    while True:
        cand = parent / f"{stem}__{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def rename_path(old: Path, new_name: str, dry_run: bool = False, verbose: bool = False) -> Path:
    """Rename 'old' to sibling 'new_name', handling collisions and case-only renames."""
    if new_name == old.name:
        return old

    new = old.with_name(new_name)

    # If a different entry already exists, pick a unique name.
    if new.exists() and not samefile_safe(old, new):
        new = unique_path(new)

    if dry_run:
        print(f"[dry-run] rename {old}  ->  {new}")
        return new

    # Case-only rename fix (macOS/Windows).
    if new.exists() and samefile_safe(old, new):
        tmp = old.with_name(f"{old.name}.__tmp__{uuid.uuid4().hex[:8]}")
        if verbose:
            print(f"[case-fix] {old} -> {tmp} -> {new}")
        old.rename(tmp)
        tmp.rename(new)
    else:
        if verbose:
            print(f"rename {old} -> {new}")
        old.rename(new)

    return new

# ----------------------------
# Traversal helpers
# ----------------------------

def collect_dirs(root: Path):
    """All directories under root (excluding root), deepest-first."""
    dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        dirs.append(Path(dirpath))
    dirs = [d for d in dirs if d != root]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    return dirs

def collect_files(root: Path):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            files.append(Path(dirpath) / fn)
    return files

# ----------------------------
# Trash / deletion helpers
# ----------------------------

def ensure_trash(trash_root: Path, dry_run: bool = False):
    if dry_run:
        print(f"[dry-run] ensure trash dir {trash_root}")
        return
    trash_root.mkdir(parents=True, exist_ok=True)

def move_to_trash(root: Path, item: Path, trash_root: Path, dry_run: bool = False, verbose: bool = False):
    rel = item.relative_to(root)
    dest = trash_root / rel
    if dry_run:
        print(f"[dry-run] trash {item} -> {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"trash {item} -> {dest}")
    shutil.move(str(item), str(dest))

def delete_tree_or_file(path: Path, dry_run: bool = False, verbose: bool = False):
    if dry_run:
        print(f"[dry-run] delete {path}")
        return
    if path.is_dir():
        if verbose:
            print(f"delete dir {path}")
        shutil.rmtree(path)
    else:
        if verbose:
            print(f"delete file {path}")
        path.unlink(missing_ok=False)

# ----------------------------
# Removal phase
# ----------------------------

def remove_unwanted(root: Path,
                    dry_run: bool = False,
                    verbose: bool = False,
                    rm_dir_names = ("_vnc", "_vti_conf"),
                    remove_hidden_files: bool = True,
                    remove_hidden_dirs: bool = False,
                    remove_lnk: bool = True,
                    remove_specials: bool = True,
                    trash: Optional[Path] = None):
    """
    Remove specific directories (exact names), hidden files ('.*'),
    optional hidden directories, *.lnk files, and .DS_Store / Thumbs.db.

    If 'trash' is provided, move items there preserving relative paths.
    """
    if trash:
        ensure_trash(trash, dry_run=dry_run)

    # Walk topdown so we can prune directories from traversal.
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)

        # If we're inside the trash dir, skip processing its contents.
        if trash and current == trash:
            dirnames[:] = []
            continue

        # Prevent descending into the trash directory later.
        if trash:
            for d in list(dirnames):
                if (current / d) == trash:
                    dirnames.remove(d)

        # 1) Remove target directories by exact name, and optionally hidden dirs
        to_remove = []
        for d in list(dirnames):
            if d in rm_dir_names or (remove_hidden_dirs and d.startswith('.')):
                full = current / d
                if trash:
                    move_to_trash(root, full, trash, dry_run=dry_run, verbose=verbose)
                else:
                    delete_tree_or_file(full, dry_run=dry_run, verbose=verbose)
                to_remove.append(d)
        for d in to_remove:
            if d in dirnames:
                dirnames.remove(d)  # prune traversal

        # 2) Remove hidden files, *.lnk, and .DS_Store / Thumbs.db
        for fn in filenames:
            name_lower = fn.lower()
            is_hidden = fn.startswith('.')
            is_lnk = remove_lnk and name_lower.endswith('.lnk')
            is_special = remove_specials and (name_lower in {'.ds_store', 'thumbs.db'})
            if (remove_hidden_files and is_hidden) or is_lnk or is_special:
                full = current / fn
                if trash:
                    move_to_trash(root, full, trash, dry_run=dry_run, verbose=verbose)
                else:
                    delete_tree_or_file(full, dry_run=dry_run, verbose=verbose)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Clean & normalize a tree: remove junk, then rename dirs/files "
                    "(spaces/-/: -> _, & -> _and_, compress __)."
    )
    ap.add_argument("root", type=Path, help="Root directory to process")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without changing anything")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print each action")

    # Deletion / trash options
    ap.add_argument("--no-delete", action="store_true",
                    help="Skip deletion phase (only rename)")
    ap.add_argument("--trash", type=Path, default=None,
                    help="Move junk into this folder instead of deleting (preserve structure)")
    ap.add_argument("--rm-dirs", default="_vnc,_vti_conf",
                    help="Comma-separated exact directory names to remove (default: _vnc,_vti_conf)")
    ap.add_argument("--remove-hidden-dirs", action="store_true",
                    help="Also remove hidden directories (names starting with '.')")

    ap.add_argument("--keep-lnk", action="store_true",
                    help="Do NOT remove *.lnk files")
    ap.add_argument("--keep-specials", action="store_true",
                    help="Do NOT remove .DS_Store / Thumbs.db")

    # Renaming options
    ap.add_argument("--rename-root", action="store_true",
                    help="Also rename the root directory itself (done last)")

    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    rm_dir_names = tuple(s for s in (x.strip() for x in args.rm_dirs.split(',')) if s)

    # 0) Removal phase (optional)
    if not args.no_delete:
        remove_unwanted(
            root=root,
            dry_run=args.dry_run,
            verbose=args.verbose,
            rm_dir_names=rm_dir_names,
            remove_hidden_files=True,
            remove_hidden_dirs=args.remove_hidden_dirs,
            remove_lnk=(not args.keep_lnk),
            remove_specials=(not args.keep_specials),
            trash=(args.trash.resolve() if args.trash else None),
        )

    changed = 0

    # 1) Rename directories (deepest-first)
    for d in collect_dirs(root):
        new_name = normalize_name(d.name)
        if new_name != d.name:
            rename_path(d, new_name, dry_run=args.dry_run, verbose=args.verbose)
            changed += 1

    # 2) Rename files
    for f in collect_files(root):
        new_name = normalize_name(f.name)
        if new_name != f.name:
            rename_path(f, new_name, dry_run=args.dry_run, verbose=args.verbose)
            changed += 1

    # 3) Optionally rename the root directory itself (do this last)
    if args.rename_root:
        new_name = normalize_name(root.name)
        if new_name != root.name:
            rename_path(root, new_name, dry_run=args.dry_run, verbose=args.verbose)
            changed += 1

    if args.dry_run:
        print(f"[dry-run] Would change {changed} names.")
    else:
        print(f"Done. Changed {changed} names.")

if __name__ == "__main__":
    main()
