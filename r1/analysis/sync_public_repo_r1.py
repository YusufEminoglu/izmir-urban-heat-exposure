"""Update the public companion repository (izmir-urban-heat-exposure) to the R1 revision:
R1 analysis code (the sanitised copies of the figshare deposit), the eleven manuscript figures, and model-parameter
tables regenerated from the R1 outputs (replacing the earlier placeholder tables)."""
import re
import shutil
from pathlib import Path
import pandas as pd

BB = Path(r"REPO_ROOT")
REPO = BB / "izmir-urban-heat-exposure"
DEP = BB / "docs" / "submission_ready" / "2026-10-28" / "figshare_deposit"
FIG = BB / "outputs" / "figures" / "paper1" / "final"
TEX = BB / "docs" / "manuscripts" / "paper1" / "src"
mydata = dict(l.rstrip("\n").split(",", 1) for l in (TEX / "mydata.dat").read_text(encoding="utf-8").splitlines() if "," in l)

# --- R1 analysis code
r1 = REPO / "r1"
if r1.exists():
    shutil.rmtree(r1)
shutil.copytree(DEP / "code" / "analysis", r1 / "analysis")
shutil.copytree(DEP / "code" / "mgwr_run", r1 / "mgwr_run")
shutil.copy2(DEP / "code" / "requirements.txt", r1 / "requirements.txt")
# Earth Engine project from the environment (EE_PROJECT, as in the main pipeline), not a literal
for f in (r1 / "analysis").glob("*.py"):
    t = f.read_text(encoding="utf-8")
    t2 = re.sub(r'ee\.Initialize\(project="[^"]*"\)', 'ee.Initialize(project=__import__("os").environ["EE_PROJECT"])', t)
    if t2 != t:
        f.write_text(t2, encoding="utf-8")

# --- figures: the eleven main figures in manuscript order
main_tex = "".join((TEX / "sections" / f).read_text(encoding="utf-8") for f in
                   ["introduction.tex", "study_area.tex", "methodology.tex", "results.tex"])
names = list(dict.fromkeys(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+\.png)\}", main_tex)))
assert len(names) == 11, names
png = REPO / "figures" / "png"
for f in png.glob("*.png"):
    f.unlink()
for n in names:
    shutil.copy2(FIG / n, png / n)
(REPO / "figures" / "figure_order.csv").write_text(
    "figure,file\n" + "".join(f"Figure {i},{n}\n" for i, n in enumerate(names, 1)), encoding="utf-8")

# --- model parameters from the R1 outputs
mp = REPO / "data" / "model_params"
u = pd.read_csv(DEP / "data" / "results" / "lcz_lst_uncertainty.csv")
u = u[u.lcz > 0][["lcz", "name", "n", "mean", "sd", "median", "ci_lo", "ci_hi", "supported"]]
u.columns = ["lcz_class", "lcz_label", "cell_count", "mean_lst_c", "std_lst_c", "median_lst_c", "mean_ci95_lo", "mean_ci95_hi", "supported"]
u.round(3).to_csv(mp / "lcz_lst_summary.csv", index=False)

rows = []
for line in (TEX / "tables" / "table_5_ols.tex").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^([A-Za-z][A-Za-z ]+?) & (\S+) & (\S+) & (\S+) & (\S+) \\\\", line.strip())
    if m:
        num = lambda s: None if s == "--" else float(s.replace("$-$", "-"))
        rows.append(dict(predictor=m.group(1).strip(), coef_std=num(m.group(2)), se_hc3=num(m.group(3)), t_value=num(m.group(4)), vif=num(m.group(5))))
assert len(rows) == 10, rows
pd.DataFrame(rows).to_csv(mp / "ols_coefficients.csv", index=False)

bw = pd.DataFrame([
    dict(term="intercept", bandwidth_neighbours=int(mydata["mgwr_bw_intercept"])),
    dict(term="ndvi", bandwidth_neighbours=int(mydata["mgwr_bw_ndvi"])),
    dict(term="ndbi", bandwidth_neighbours=int(mydata["mgwr_bw_ndbi"])),
    dict(term="albedo", bandwidth_neighbours=int(mydata["mgwr_bw_albedo"])),
])
bw["calibration_sample"] = int(mydata["n_mgwr_sample"])
bw.to_csv(mp / "mgwr_bandwidths.csv", index=False)
print("figures:", names)
print("ok")
