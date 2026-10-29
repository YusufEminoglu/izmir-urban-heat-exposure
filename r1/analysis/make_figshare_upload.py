"""figshare upload set from the deposit built by build_figshare_deposit.py: one zip of the whole deposit, plus README,
MANIFEST and the two licence files at top level (shown by figshare without unzipping)."""
import shutil
import zipfile
from pathlib import Path

BASE = Path(r"REPO_ROOT\docs\submission_ready\2026-10-28")
DEP, UP = BASE / "figshare_deposit", BASE / "figshare_upload"
if UP.exists():
    shutil.rmtree(UP)
UP.mkdir()
files = sorted(f for f in DEP.rglob("*") if f.is_file())
with zipfile.ZipFile(UP / "izmir_lcz_heat_data_and_code.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        z.write(f, f.relative_to(DEP).as_posix())
for n in ("README.md", "MANIFEST.tsv", "LICENSE-code.txt", "LICENSE-data.txt"):
    shutil.copy2(DEP / n, UP / n)
print(len(files), "files zipped;", round((UP / "izmir_lcz_heat_data_and_code.zip").stat().st_size / 1e6, 1), "MB")
