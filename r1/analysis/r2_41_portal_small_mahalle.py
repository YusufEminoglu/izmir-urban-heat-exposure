"""How many of the mahalle without a population record (the ADNKS age table covers mahalle of about 500+ residents)
have a 2023 total population in the Izmir open-data portal (TUIK table 'Cinsiyete Gore Mahalle Nufus Bilgileri',
mahalle with more than 10 residents)? Matching on normalised district|mahalle names."""
import sys
import unicodedata
from pathlib import Path
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

PORTAL = Path(sys.argv[1])
POP = Path(r"EXTERNAL_DATA_DIR\DM\data\01_raw\nufus\mahalle_population_controls_2024.gpkg")
TR = str.maketrans({"Ç": "C", "Ğ": "G", "İ": "I", "I": "I", "Ö": "O", "Ş": "S", "Ü": "U", "Â": "A", "Î": "I", "Û": "U",
                    "ç": "C", "ğ": "G", "ı": "I", "i": "I", "ö": "O", "ş": "S", "ü": "U"})


def norm(s):
    s = str(s).strip().upper().translate(TR)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    for suf in (" MAHALLESI", " MAH.", " MAH"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.replace(".", ". ").split())


p = pd.read_csv(PORTAL, sep=";", encoding="utf-8")
for c in ("NUFUS_ERKEK_TOPLAM", "NUFUS_KADIN_TOPLAM"):
    p[c] = pd.to_numeric(p[c], errors="coerce")
p["pop2023"] = p.NUFUS_ERKEK_TOPLAM.fillna(0) + p.NUFUS_KADIN_TOPLAM.fillna(0)
p["key"] = p.ILCE_ADI.map(norm) + "|" + p.MAHALLE_ADI.map(norm)
m = gpd.read_file(POP)
m["key"] = m.district_name_geom.map(norm) + "|" + m.mahalle_name_geom.map(norm)
miss = m[m.pop_total.isna()].copy()
have = m[m.pop_total.notna()].copy()
pk = p.set_index("key")["pop2023"]
pk = pk[~pk.index.duplicated()]
miss["pop2023"] = miss.key.map(pk)
have["pop2023"] = have.key.map(pk)
print("portal rows", len(p), "| our mahalle", len(m), "| without population", len(miss))
print("missing matched in portal:", miss.pop2023.notna().sum(), "| unmatched:", miss.pop2023.isna().sum())
print("population of matched missing mahalle: total %d, median %d, max %d" % (miss.pop2023.sum(), miss.pop2023.median(), miss.pop2023.max()))
print("share of 2023 portal province population: %.2f%%" % (100 * miss.pop2023.sum() / p.pop2023.sum()))
print("check on covered mahalle: matched", have.pop2023.notna().sum(), "of", len(have),
      "| median ratio 2024 ADNKS / 2023 portal %.3f" % (have.pop_total / have.pop2023).median())
print("portal mahalle <= 500 residents:", (p.pop2023 <= 500).sum(), "| > 500:", (p.pop2023 > 500).sum())
print(miss.pop2023.describe().round(0).to_string())
u = miss[miss.pop2023.isna()][["district_name_geom", "mahalle_name_geom", "key"]]
print("unmatched sample:\n", u.head(25).to_string())
miss[["join_key", "district_name_geom", "mahalle_name_geom", "pop2023"]].to_csv(TABLES / "portal_2023_small_mahalle.csv", index=False, encoding="utf-8-sig")
