"""Audit a checkout for common public-release failures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import subprocess
from pathlib import Path


TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".r",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SELF_EXCLUSIONS = {
    Path("analysis/python/release_audit.py"),
    Path("tests/test_release_qa.py"),
}
MACHINE_PATH_MARKERS = (
    "/" + "home/",
    "/" + "scratch/",
    "/" + "data/hf/",
    "C:" + "\\Users\\",
)
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
INTERNAL_LABEL = re.compile(
    r"\b(?:R1|R2[-_ ]?[ABC]|Phase 1A|Phase 1B|revision_todo)\b",
    re.IGNORECASE,
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class Finding:
    category: str
    path: Path
    detail: str


def repository_files(root: Path) -> list[Path]:
    """Return tracked files, or all non-Git files in an exported checkout."""

    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return [root / path.decode() for path in completed.stdout.split(b"\0") if path]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]


def audit_markdown_links(root: Path, path: Path, text: str) -> list[Finding]:
    findings = []
    for target in MARKDOWN_LINK.findall(text):
        target = target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        relative = target.split("#", 1)[0]
        if not (path.parent / relative).resolve().exists():
            findings.append(
                Finding("broken local link", path.relative_to(root), target)
            )
    return findings


def audit_repository(root: Path, *, large_file_mb: int = 50) -> list[Finding]:
    root = root.resolve()
    findings = []
    for path in repository_files(root):
        relative = path.relative_to(root)
        if not path.exists():
            findings.append(Finding("missing tracked file", relative, "path does not exist"))
            continue
        size = path.stat().st_size
        if size > large_file_mb * 1024 * 1024:
            findings.append(Finding("large file", relative, f"{size / 1024 / 1024:.1f} MB"))
        if path.suffix.lower() not in TEXT_SUFFIXES or relative in SELF_EXCLUSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in MACHINE_PATH_MARKERS:
            if marker in text:
                findings.append(Finding("machine-specific path", relative, marker))
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(Finding("possible secret", relative, label))
        if relative.parts[0] in {"docs", "manifest", "notebooks", "figures"} or relative.name in {
            "README.md",
            "CONTRIBUTING.md",
            "THIRD_PARTY_NOTICES.md",
        }:
            match = INTERNAL_LABEL.search(text)
            if match:
                findings.append(Finding("internal working label", relative, match.group(0)))
        if path.suffix.lower() == ".md":
            findings.extend(audit_markdown_links(root, path, text))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--large-file-mb", type=int, default=50)
    args = parser.parse_args()
    findings = audit_repository(args.root, large_file_mb=args.large_file_mb)
    if findings:
        for finding in findings:
            print(f"[{finding.category}] {finding.path}: {finding.detail}")
        raise SystemExit(f"Release audit failed with {len(findings)} finding(s)")
    print("Release audit passed")


if __name__ == "__main__":
    main()
