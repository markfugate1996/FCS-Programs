#!/usr/bin/env python3
"""
Add a "Plot defaults" tab to Preferences: residual trace style, residual
line width, tight-crop default, SVG editable text, and gridline defaults.

Touches fcs_plottools.py, fcs_prefs.py and the six modules that draw fit
residual panels.  Every anchor in every file is validated BEFORE anything
is written, so a stale file aborts the whole run rather than leaving the
suite half-patched.  Line endings are preserved per file.

Usage:  python3 patch_plot_defaults.py [path/to/FCS-Programs]   (default .)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_defaults_edits as E       # noqa: E402


def _encoder(data: bytes, name: str):
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    if crlf and lf_only:
        raise ValueError(f"{name}: mixed line endings "
                         f"({crlf} CRLF, {lf_only} LF) - refusing to guess")
    newline = b"\r\n" if crlf else b"\n"
    return newline, (lambda text: text.encode("utf-8")
                     .replace(b"\n", newline))


def _plan(path: pathlib.Path, edits):
    """Apply *edits* to the file's bytes in memory.  Raises on any mismatch."""
    data = path.read_bytes()
    newline, enc = _encoder(data, path.name)
    applied = []
    for name, old_lf, new_lf in edits:
        old, new = enc(old_lf), enc(new_lf)
        if data.count(new) and not data.count(old):
            raise ValueError(f"{path.name}: '{name}' looks already applied")
        n = data.count(old)
        if n != 1:
            raise ValueError(f"{path.name}: anchor for '{name}' matched "
                             f"{n} times, expected exactly 1")
        data = data.replace(old, new, 1)
        applied.append(name)
    return data, applied, ("CRLF" if newline == b"\r\n" else "LF")


def main(root_str: str) -> int:
    root = pathlib.Path(root_str)
    jobs = {"fcs_plottools.py": list(E.PLOTTOOLS),
            "fcs_prefs.py": list(E.PREFS_EDIT)}
    for fname, sites in E.RESID_SITES.items():
        jobs.setdefault(fname, []).extend(sites)

    missing = [f for f in jobs if not (root / f).is_file()]
    if missing:
        print(f"ERROR: not found under {root}: {', '.join(sorted(missing))}")
        return 2

    # Pass 1: plan everything, write nothing.
    planned = {}
    for fname in sorted(jobs):
        try:
            planned[fname] = _plan(root / fname, jobs[fname])
        except ValueError as exc:
            print(f"ERROR: {exc}")
            print("Nothing was written.")
            return 2
    print(f"planned {sum(len(v[1]) for v in planned.values())} edits across "
          f"{len(planned)} files; all anchors matched")

    # Pass 2: write.
    for fname in sorted(planned):
        data, applied, eol = planned[fname]
        (root / fname).write_bytes(data)
        print(f"  {fname} ({eol}, {len(applied)} edit"
              f"{'s' if len(applied) != 1 else ''})")
        for name in applied:
            print(f"      - {name}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
