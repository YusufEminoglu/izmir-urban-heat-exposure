"""Round-5 values: range of LCZ class means and its end classes, population of the ten highest-ranked mahalle,
and the consensus units by support in the five independent MGWR samples."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

v = {}
U = pd.read_csv(TABLES / "lcz_lst_uncertainty.csv")
U = U[U.lcz != 0]
hi, lo = U.loc[U["mean"].idxmax()], U.loc[U["mean"].idxmin()]
v["lcz_range_max_min"] = fmt(hi["mean"] - lo["mean"], 2)
v["lcz_range_hot"] = f"LCZ~{LCZ_LABEL[int(hi.lcz)]} ({str(hi['name']).lower()})"
v["lcz_range_cool"] = f"LCZ~{LCZ_LABEL[int(lo.lcz)]} ({str(lo['name']).lower()})"
v["lcz_range_both_supported"] = "yes" if bool(hi.supported) and bool(lo.supported) else "no"

P = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig")
t10 = P.sort_values("rank").head(10)
v["top10_pop"] = f"{int(round(t10.pop_total.sum(), -2)):,}".replace(",", "{,}")
v["top10_lt1500"] = int((t10.pop_total < 1500).sum())

SR = pd.read_csv(TABLES / "priority_rank_by_sample.csv", index_col=0)
reps = [c for c in SR.columns if c.startswith("rep")]
k = (SR[reps] <= 80).sum(1)
C = P[P.consensus_top.astype(bool)].set_index("join_key")
kc = k.reindex(C.index).fillna(0).astype(int)
v["n_cons_k4"] = int((kc >= 4).sum())
low = C[kc < 4].assign(k=kc[kc < 4]).sort_values("k")
tr = lambda s: str(s).title().replace("İ", "İ")
v["cons_low_support"] = "; ".join(f"{j.split('|')[1].title()} ({j.split('|')[0].title()}, {int(r.k)} of 5)" for j, r in low.iterrows())  # ASCII join keys, as consensus_names
print(v)
write_values("r2_33_round5_values", v)
