#!/usr/bin/env python3
"""
fix_dialog_buttons.py
=====================
Find, and optionally fix, every dialog in the suite whose action buttons can
be clipped off the bottom of the window.

The bug
-------
A dialog does two things that combine badly:

  * it sets a hard-coded ``win.geometry("WxH")``, and
  * it creates its button row as a child frame of the Toplevel and packs
    that frame LAST, after a widget packed with ``expand=True``.

pack() hands out space in packing order, so the frame packed last is the one
that loses its space when the content outgrows W x H.  The buttons then sit
below the window edge and the only way to reach them is to drag the window
bigger.  This is what adding two models to the correlation registry did to
"Fit - select model": the list grew by two radio buttons and Next/Cancel went
over the edge.

fcs_corr._correlation_options_dialog already gets this right and says why:
the button frame is packed first, against side="bottom", so it reserves its
slot before the scrolling canvas claims the rest.  This script applies that
same pattern everywhere the ingredients for the bug are present.

What it changes, per affected dialog
------------------------------------
  [button-strip]    The button frame is re-ordered to the front of the
                    packing list and given side="bottom".  Done with
                    pack_configure(before=...) at the existing call site, so
                    no code moves and the dialog's build order is untouched.

  [size-to-content] The hard-coded geometry is replaced by a measurement made
                    after the widgets exist.  Clamped so the window is NEVER
                    smaller than the author's original W x H -- a dialog
                    given extra room on purpose keeps it -- and never larger
                    than the screen.

Both inserted blocks carry a sentinel comment, so re-running is a no-op.

Usage
-----
    python fix_dialog_buttons.py                 # scan CWD, report only
    python fix_dialog_buttons.py --diff          # report + show unified diffs
    python fix_dialog_buttons.py --apply         # write the changes
    python fix_dialog_buttons.py --apply FILE... # restrict to these files
    python fix_dialog_buttons.py --include-variants   # also scan *.<tag>.py copies

Nothing is written without --apply.  Line endings of each file are detected
and preserved.  Use git to review and revert.
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
from pathlib import Path

SENTINEL_BTN  = "# [button-strip]"
SENTINEL_SIZE = "# [size-to-content]"

RE_TOPLEVEL = re.compile(r"^\s*(\w+)\s*=\s*tk\.Toplevel\(")
RE_BUTTON   = re.compile(r"tk\.Button\(\s*(\w+)\s*,")
RE_WAIT     = re.compile(r"^\s*(\w+)\.wait_window\(\s*\)\s*$")

# Button text that marks a frame as the dialog's action row.  A frame of
# buttons that is not the action row (a toolbar inside the body) is not a
# direct child of the Toplevel and is filtered out by that test instead.
ACTION_WORDS = ("Cancel", "Close", "Next", "OK", "Back", "Apply", "Done",
                "Compute", "Plot", "Fit", "Save")


class Site:
    """One dialog function that was examined."""

    def __init__(self, path, func, lineno):
        self.path, self.func, self.lineno = path, func, lineno
        self.win = None
        self.status = "skipped"
        self.reasons = []
        self.actions = []          # (lineno0, kind, payload)

    def note(self, msg):
        self.reasons.append(msg)

    def __str__(self):
        head = f"{self.path.name}:{self.lineno}  {self.func}()"
        tag = {"fix": "WILL FIX", "fixed": "FIXED",
               "skipped": "skip", "review": "REVIEW"}[self.status]
        out = [f"  [{tag:8s}] {head}"]
        for r in self.reasons:
            out.append(f"             - {r}")
        return "\n".join(out)


def _pack_args_with_bottom(args: str) -> str:
    """Add side="bottom" to a pack() argument string, preserving the rest."""
    args = args.strip()
    if not args:
        return 'side="bottom"'
    return 'side="bottom", ' + args


def _btn_block(indent, frame, win):
    return [
        f'{indent}{SENTINEL_BTN} Claim the bottom slot before anything that can grow.\n',
        f'{indent}# pack() hands out space in packing order, so a frame packed last is\n',
        f'{indent}# the one squeezed out when the content outgrows the window -- which\n',
        f'{indent}# is how action buttons end up below the bottom edge.  Re-ordering\n',
        f'{indent}# here rather than moving the code keeps the build order intact.\n',
        f'{indent}_above = [_w for _w in {win}.pack_slaves() if _w is not {frame}]\n',
        f'{indent}if _above:\n',
        f'{indent}    {frame}.pack_configure(before=_above[0])\n',
    ]


def _size_block(indent, win, w, h):
    return [
        f'{indent}{SENTINEL_SIZE} Size to what the dialog actually needs instead of a\n',
        f'{indent}# fixed height: a hard-coded height is what pushed the buttons off\n',
        f'{indent}# the bottom when the content grew.  Never smaller than the original\n',
        f'{indent}# {w}x{h}, so a dialog deliberately given room keeps it, and never\n',
        f'{indent}# larger than the screen.\n',
        f'{indent}{win}.update_idletasks()\n',
        f'{indent}{win}.geometry("%dx%d" % (\n',
        f'{indent}    min(max({w}, {win}.winfo_reqwidth()),  {win}.winfo_screenwidth() - 80),\n',
        f'{indent}    min(max({h}, {win}.winfo_reqheight()), {win}.winfo_screenheight() - 120)))\n',
    ]


def examine(path, lines, node):
    """Classify one FunctionDef; return a Site."""
    site = Site(path, node.name, node.lineno)
    lo = node.lineno - 1
    hi = getattr(node, "end_lineno", len(lines))
    body = lines[lo:hi]

    # The Toplevel this function owns.  Functions that merely receive a parent
    # (fcs_main._add_ok_cancel) have none and are not ours to touch.
    win = None
    for ln in body:
        m = RE_TOPLEVEL.match(ln)
        if m:
            win = m.group(1)
            break
    if win is None:
        return None
    site.win = win

    if any(SENTINEL_BTN in ln or SENTINEL_SIZE in ln for ln in body):
        site.status = "fixed"
        site.note("already carries this fix")
        return site

    # ── the hard-coded geometry ──────────────────────────────────────────────
    re_geom = re.compile(
        r'^(\s*)' + re.escape(win) + r'\.geometry\(\s*["\'](\d+)x(\d+)["\']\s*\)\s*$')
    geom_i = geom_w = geom_h = None
    for i, ln in enumerate(body):
        m = re_geom.match(ln)
        if m:
            geom_i, geom_w, geom_h = lo + i, int(m.group(2)), int(m.group(3))
            break

    # ── the action-button frame: a tk.Frame child of the Toplevel that holds
    #    buttons, packed on one line without an explicit side= ───────────────
    containers = {m.group(1) for ln in body for m in [RE_BUTTON.search(ln)] if m}
    action = None
    for name in sorted(containers):
        re_make = re.compile(r'^\s*' + re.escape(name) +
                             r'\s*=\s*tk\.Frame\(\s*' + re.escape(win) + r'\s*[,)]')
        if not any(re_make.match(ln) for ln in body):
            continue                      # not a direct child of the Toplevel
        texts = [ln for ln in body if re.search(
            r'tk\.Button\(\s*' + re.escape(name) + r'\s*,', ln)]
        if not any(w in t for t in texts for w in ACTION_WORDS):
            continue                      # not the action row
        re_pack = re.compile(r'^(\s*)' + re.escape(name) +
                             r'\.pack\(((?:[^()]|\([^()]*\))*)\)\s*$')
        for i, ln in enumerate(body):
            m = re_pack.match(ln)
            if m:
                action = (name, lo + i, m.group(1), m.group(2))
                break
        else:
            site.status = "review"
            site.note(f"button frame '{name}' found, but its .pack() call is not "
                      f"a single line -- left alone")
            return site
        break

    # Buttons packed straight onto the Toplevel, with no frame to re-order:
    # tk.Button(win, ...).pack(...).  These cannot be pinned without naming
    # the widget, but sizing the window to its content still saves them.
    # Matched over the joined body, not line by line: the usual form of this
    # spans two lines -- tk.Button(win, text="Close", ...,\n  pady=4).pack(...)
    blob = "".join(body)
    re_bare = re.compile(
        r'tk\.Button\(\s*' + re.escape(win) +
        r'\s*,((?:[^()]|\([^()]*\))*)\)\s*\.pack\(', re.S)
    bare = [m.group(1) for m in re_bare.finditer(blob)
            if any(w in m.group(1) for w in ACTION_WORDS)]

    if action is None and not bare:
        return None                       # no action buttons here at all

    expands = any(".pack(" in ln and "expand=True" in ln for ln in body)

    if action is None:
        site.note("action button(s) packed directly on the Toplevel, with no "
                  "frame to re-order")
        if geom_i is None:
            site.note("no fixed geometry: the window sizes itself to its content")
            return site
        name = pack_i = indent = pack_args = None
        pinned = False
    else:
        name, pack_i, indent, pack_args = action
        pinned = "side=" in pack_args

    if pinned:
        site.note(f"'{name}' is already packed with an explicit side=, so it "
                  f"claims its slot first -- this dialog is already correct")
        return site

    # A button strip never expands.  A frame that holds buttons AND is packed
    # expand=True is the dialog's body, which happens to contain its buttons;
    # pinning that to the bottom would rearrange the dialog, not repair it.
    if action is not None and "expand=True" in pack_args:
        site.status = "review"
        site.note(f"'{name}' holds the buttons but is packed expand=True, so it "
                  f"is the dialog body rather than a button strip -- left alone")
        site.actions = []
        return site

    if geom_i is None:
        site.note("no fixed geometry: the window sizes itself to its content, "
                  "so the buttons are only at risk if the content exceeds the "
                  "screen -- left alone")
        return site

    if not expands and action is not None:
        site.note("nothing packed with expand=True to compete for the space")

    # ── from here on there IS a hard-coded geometry and an at-risk button row ─
    if action is not None:
        site.actions.append((pack_i, "pack",
                             f'{indent}{name}.pack({_pack_args_with_bottom(pack_args)})\n'))
        site.actions.append((pack_i, "pin", _btn_block(indent, name, win)))

    # where the sizing call goes
    anchor_i = None
    for i, ln in enumerate(body):
        m = RE_WAIT.match(ln)
        if m and m.group(1) == win:
            anchor_i = lo + i
            break
    if anchor_i is None:
        for i, ln in enumerate(body):
            if re.match(r'^\s*return\s+' + re.escape(win) + r'\s*$', ln):
                anchor_i = lo + i
                break
    if anchor_i is None:
        site.status = "review"
        site.note(f"fixed geometry {geom_w}x{geom_h} but no {win}.wait_window() or "
                  f"return {win} to anchor the sizing call -- left alone")
        site.actions = []
        return site

    g_indent = re_geom.match(lines[geom_i]).group(1)
    site.actions.append((geom_i, "drop", None))
    site.actions.append((anchor_i, "size",
                         _size_block(g_indent, win, geom_w, geom_h)))

    site.status = "fix"
    if action is not None:
        site.note(f"fixed geometry {geom_w}x{geom_h}; action frame '{name}' "
                  f"packed last at line {pack_i + 1}")
    else:
        site.note(f"fixed geometry {geom_w}x{geom_h}; sizing only")
    if expands:
        site.note("an expanding widget competes for the same space")
    return site


def process(path, apply_changes, show_diff):
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            src = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"  [ERROR   ] {path.name}: {exc}")
        return 0, 0

    nl = "\r\n" if "\r\n" in src else "\n"
    flat = src.replace("\r\n", "\n")
    try:
        tree = ast.parse(flat)
    except SyntaxError as exc:
        print(f"  [ERROR   ] {path.name}: does not parse ({exc})")
        return 0, 0

    lines = flat.splitlines(keepends=True)
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s = examine(path, lines, node)
            if s is not None:
                sites.append(s)
    sites.sort(key=lambda s: s.lineno)

    if not sites:
        return 0, 0

    print(f"\n{path}")
    for s in sites:
        print(s)

    todo = [s for s in sites if s.status == "fix"]
    if not todo:
        return 0, len([s for s in sites if s.status == "review"])

    out = list(lines)
    edits = [a for s in todo for a in s.actions]
    for lineno, kind, payload in sorted(edits, key=lambda e: -e[0]):
        if kind == "pack":
            out[lineno] = payload
        elif kind == "pin":
            out[lineno + 1:lineno + 1] = payload
        elif kind == "drop":
            del out[lineno]
        elif kind == "size":
            out[lineno:lineno] = payload

    new_flat = "".join(out)
    try:
        ast.parse(new_flat)
    except SyntaxError as exc:
        print(f"  [ERROR   ] {path.name}: the edit would not parse ({exc}); "
              f"nothing written for this file")
        return 0, 1

    if show_diff:
        for ln in difflib.unified_diff(lines, out, f"a/{path.name}",
                                       f"b/{path.name}", n=2):
            sys.stdout.write("    " + ln if not ln.endswith("\n")
                             else "    " + ln)
        print()

    if apply_changes:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_flat.replace("\n", nl) if nl != "\n" else new_flat)
        print(f"  -> wrote {path.name}  ({len(todo)} dialog"
              f"{'' if len(todo) == 1 else 's'} fixed)")

    return len(todo), len([s for s in sites if s.status == "review"])


def main(argv):
    # The source carries em dashes and Greek; a Windows console is cp1252 and
    # raises UnicodeEncodeError part-way through printing a diff, which aborts
    # the scan and silently leaves the rest of the suite unexamined.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    apply_changes = "--apply" in argv
    show_diff = "--diff" in argv
    include_variants = "--include-variants" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]

    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted(Path(".").glob("fcs_*.py"))
    paths = [p for p in paths if p.is_file() and p.name != Path(__file__).name]

    # fcs_lifetime_fit.fractions.py and friends are working copies sitting
    # beside the module they were cut from.  Editing both halves of a pair is
    # how the two quietly diverge, so they are listed and skipped unless asked
    # for by name or with --include-variants.
    if not args and not include_variants:
        variants = [p for p in paths if p.stem.count(".")]
        paths = [p for p in paths if not p.stem.count(".")]
    else:
        variants = []

    if not paths:
        sys.exit("No fcs_*.py files found.  Run this from the repository root, "
                 "or name the files explicitly.")

    print(f"{'APPLYING' if apply_changes else 'SCANNING (no files written)'} "
          f"- {len(paths)} file(s)")
    if variants:
        print("\nNot scanned -- the name suggests a working copy rather than "
              "the live module.\nPass it by name, or --include-variants, if it "
              "is live:")
        for v in variants:
            print(f"    {v.name}")

    fixed = review = 0
    for p in paths:
        f, r = process(p, apply_changes, show_diff)
        fixed += f
        review += r

    print(f"\n{'=' * 64}")
    print(f"dialogs {'fixed' if apply_changes else 'to fix'}: {fixed}")
    if review:
        print(f"needing manual review      : {review}")
    if not apply_changes and fixed:
        print("\nNothing was written." +
              ("" if show_diff else "  Add --diff to see the edits in full.") +
              "\nRe-run with --apply to make them, then:  "
              "python -m py_compile fcs_*.py")


if __name__ == "__main__":
    main(sys.argv)
