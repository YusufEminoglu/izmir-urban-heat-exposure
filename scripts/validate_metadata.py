"""Validate CITATION.cff, .zenodo.json, and README.md for consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_cff() -> dict:
    path = ROOT / "CITATION.cff"
    if not path.exists():
        print(f"missing: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_zenodo() -> dict:
    path = ROOT / ".zenodo.json"
    if not path.exists():
        print(f"missing: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_title_match(cff: dict, zenodo: dict) -> list[str]:
    errors: list[str] = []
    if cff.get("title") != zenodo.get("title"):
        errors.append(f"title mismatch: CFF vs Zenodo")
    return errors


def check_authors_match(cff: dict, zenodo: dict) -> list[str]:
    errors: list[str] = []
    cff_authors = cff.get("authors", [])
    zenodo_creators = zenodo.get("creators", [])
    if len(cff_authors) != len(zenodo_creators):
        errors.append(f"author count mismatch: {len(cff_authors)} vs {len(zenodo_creators)}")
    return errors


def check_orcid_present(cff: dict) -> list[str]:
    errors: list[str] = []
    for author in cff.get("authors", []):
        if "orcid" not in author:
            errors.append(f"missing ORCID for {author.get('family-names', '?')}")
    return errors


def check_license() -> list[str]:
    errors: list[str] = []
    for name in ["LICENSE", "LICENSE-docs"]:
        if not (ROOT / name).exists():
            errors.append(f"missing: {name}")
    return errors


def check_required_files() -> list[str]:
    errors: list[str] = []
    for name in ["README.md", ".gitignore", "environment.yml", "requirements.txt"]:
        if not (ROOT / name).exists():
            errors.append(f"missing: {name}")
    return errors


def main() -> None:
    all_errors: list[str] = []
    try:
        cff = load_cff()
    except Exception as exc:
        print(f"cannot parse CITATION.cff: {exc}")
        sys.exit(1)
    try:
        zenodo = load_zenodo()
    except Exception as exc:
        print(f"cannot parse .zenodo.json: {exc}")
        sys.exit(1)

    all_errors.extend(check_title_match(cff, zenodo))
    all_errors.extend(check_authors_match(cff, zenodo))
    all_errors.extend(check_orcid_present(cff))
    all_errors.extend(check_license())
    all_errors.extend(check_required_files())

    if all_errors:
        print("Metadata validation failed:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    print("Metadata validation passed.")


if __name__ == "__main__":
    main()
