"""Table 7 information criteria from one definition: AIC = -2 logL + 2k with k = coefficients + error variance
(+ lambda/rho), the parameter counts printed in Table 7, and AICc = AIC + 2k(k+1)/(n-k-1).

Rewrites the aic/aicc columns of results table model_comparison_common.csv (spreg's own AIC counts
one parameter fewer) and the Table 7 values, so that table, note and deposit agree.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

p = TABLES / "model_comparison_common.csv"
t = pd.read_csv(p)
n = 4000  # MGWR calibration sample (n_mgwr_sample)
spatial = t.model.isin(["OLS", "SEM", "SAR"])
t.loc[spatial, "aic"] = -2 * t.loc[spatial, "ll"] + 2 * t.loc[spatial, "k"]
t.loc[spatial, "aicc"] = t.loc[spatial, "aic"] + 2 * t.loc[spatial, "k"] * (t.loc[spatial, "k"] + 1) / (n - t.loc[spatial, "k"] - 1)
t.to_csv(p, index=False)
s = t.set_index("model")
v = {"ols_sample_aic": fmt(s.loc["OLS", "aic"], 1), "sem_aic": fmt(s.loc["SEM", "aic"], 1), "sar_aic": fmt(s.loc["SAR", "aic"], 1),
     "aicc_corr_max": fmt((s.loc[["OLS", "SEM", "SAR"], "aicc"] - s.loc[["OLS", "SEM", "SAR"], "aic"]).max(), 2)}
print(n, t.round(3).to_string(), v)
write_values("r2_24_table7_aic", v)
