"""Parse JPX 銘柄別信用取引残高 (daily, all issues) PDF -> CSV.

usage: python3 parse_mtall.py YYYYMMDD_mtall.pdf out.csv
Columns: code (4/5-char local code), sell_sh, sell_chg, buy_sh, buy_chg, sell_listed_pct, buy_listed_pct, sell_val, buy_val
"""
import csv
import re
import subprocess
import sys

pdf, out = sys.argv[1], sys.argv[2]
txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
CODE = re.compile(r"(?<![0-9A-Z])([0-9][0-9A-Z]{2}[0-9A-Z]0)(?![0-9A-Z])")


def nums(s):
    s = s.replace("▲ ", "-").replace("▲", "-")
    toks = s.split()
    vals = []
    for t in toks:
        if t in ("*", "-"):
            vals.append(None)
            continue
        t2 = t.replace(",", "").rstrip("%")
        try:
            vals.append(float(t2))
        except ValueError:
            pass
    return vals


rows, cur, last_code = {}, None, None
for line in txt.splitlines():
    for m in CODE.finditer(line):
        last_code = m.group(1)
    if "株数 Shs." in line and last_code:
        v = nums(line.split("株数 Shs.", 1)[1])
        if len(v) >= 6:
            cur = last_code
            rows[cur] = {"code": cur[:4] if cur.endswith("0") else cur, "sell_sh": v[0], "sell_chg": v[1],
                         "sell_listed_pct": v[2], "buy_sh": v[3], "buy_chg": v[4], "buy_listed_pct": v[5]}
    elif "金額 Val." in line and cur in rows:
        v = nums(line.split("金額 Val.", 1)[1])
        if len(v) >= 4:
            rows[cur]["sell_val"], rows[cur]["buy_val"] = v[0], v[3]
        cur = None
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["code", "sell_sh", "sell_chg", "buy_sh", "buy_chg", "sell_listed_pct",
                                      "buy_listed_pct", "sell_val", "buy_val"])
    w.writeheader()
    for r in rows.values():
        w.writerow(r)
print(out, len(rows))
