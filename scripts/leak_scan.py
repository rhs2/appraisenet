"""Pre-push leak scan: fails if anything private could reach the public repository.

Checks every file git would commit for generic identity leaks (email addresses,
VIN-shaped strings, phone numbers, absolute home paths) plus any project-specific
patterns supplied in `scripts/private_patterns.txt` (one regex per line, git-ignored),
so this scanner never has to name the very strings it exists to keep out.
Run it before every push; CI cannot catch what it never sees.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")),
    ("VIN-shaped string", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")),
    ("phone number", re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("home path", re.compile(r"/Users/[a-z]+/|C:\\\\Users\\\\", re.IGNORECASE)),
]
PRIVATE = ROOT / "scripts" / "private_patterns.txt"
ALLOWED_FILE = ROOT / "scripts" / "allowed_matches.txt"
ALLOW = {"scripts/leak_scan.py"}
SKIP_SUFFIX = {".png", ".joblib", ".db", ".npz", ".pdf", ".docx"}


def load_allowed() -> set[str]:
    """Exact matched strings that are published on purpose (author contact details)."""
    if not ALLOWED_FILE.exists():
        return set()
    return {line.strip() for line in ALLOWED_FILE.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def load_private() -> int:
    if not PRIVATE.exists():
        return 0
    n = 0
    for line in PRIVATE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            n += 1
            PATTERNS.append((f"private pattern {n}", re.compile(line, re.IGNORECASE)))
    return n


def main() -> int:
    n_private = load_private()
    allowed = load_allowed()
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    bad = 0
    for f in files:
        p = ROOT / f
        if f in ALLOW or not p.is_file() or p.suffix in SKIP_SUFFIX:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for label, rx in PATTERNS:
            for m in rx.finditer(text):
                if m.group(0) in allowed:
                    continue
                line = text[:m.start()].count("\n") + 1
                print(f"LEAK {f}:{line}  [{label}]  ...{text[max(0, m.start()-30):m.end()+30].strip()!r}...")
                bad += 1
    tag = f"({len(PATTERNS)} patterns, {n_private} private)"
    print(f"clean: nothing private found {tag}" if not bad else f"\n{bad} finding(s) {tag}: fix before pushing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
