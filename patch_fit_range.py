#!/usr/bin/env python3
"""
Restrict a correlation fit to a lag-time window, set in the fit dialog.

Touches fcs_fit.py only.  Every anchor is validated before anything is
written; line endings are preserved.

Usage:  python3 patch_fit_range.py [path/to/fcs_fit.py]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit_range_edits import EDITS      # noqa: E402


def _target(argv):
    p = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    if p.is_dir():
        p = p / "fcs_fit.py"
    if not p.is_file():
        print(f"ERROR: {p} not found. Run this from the FCS-Programs "
              f"directory, or pass the path to fcs_fit.py.")
        raise SystemExit(2)
    return p


def main(argv) -> int:
    path = _target(argv)
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    if crlf and lf_only:
        print(f"ERROR: mixed line endings ({crlf} CRLF, {lf_only} LF)")
        return 2
    newline = b"\r\n" if crlf else b"\n"
    print(f"line endings: {'CRLF' if crlf else 'LF'}")

    def enc(text):
        return text.encode("utf-8").replace(b"\n", newline)

    staged = data
    for name, old_lf, new_lf in EDITS:
        old, new = enc(old_lf), enc(new_lf)
        if staged.count(new) and not staged.count(old):
            print(f"ERROR: '{name}' looks already applied. Nothing written.")
            return 2
        n = staged.count(old)
        if n != 1:
            print(f"ERROR: anchor for '{name}' matched {n} times, expected 1. "
                  f"Nothing written.")
            return 2
        staged = staged.replace(old, new, 1)
        print(f"  staged: {name}")

    path.write_bytes(staged)
    print(f"wrote {path} ({len(staged)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
