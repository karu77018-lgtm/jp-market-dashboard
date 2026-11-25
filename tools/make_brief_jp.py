"""X/note images for the JP notes (3 cards, same look as the US ones).

usage: python3 tools/make_brief_jp.py data.json content.json outdir
1 今日の判定 : TOPIX連動ETF 120日＋50日線・200日線、全面安の日を薄赤
2 市場の中身 : 50日線より上の割合（流動性上位900）直近1年、20%線、全面安の日を薄赤
3 資金の流れ : 33業種の流入（RS21−RS189）上位8・下位8
Each card: headline (the day's finding in one sentence) + chart + three facts. No hype, no buttons.
Prints the numbers the headline must agree with.
"""
import html
import json
import sys
from pathlib import Path

D = json.load(open(sys.argv[1], encoding="utf-8"))
C = json.load(open(sys.argv[2], encoding="utf-8"))
outdir = Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
E = html.escape
INK, MUT, LINE, POS, NEG, POSBG, NEGBG, BLUE, OCH = "#1B1D1C", "#727569", "#D5D1C6", "#1E7A4D", "#B23A2E", "#DBEBDF", "#F1DED9", "#2F6DB5", "#8A7B3C"
CSS = """*{box-sizing:border-box;margin:0;padding:0}
:root{--ink:#1B1D1C;--mut:#727569;--line:#D5D1C6;--card:#F6F4EE;--accent:#17685C;--pos:#1E7A4D;--posbg:#DBEBDF;--neg:#B23A2E;--negbg:#F1DED9;--num:"DejaVu Sans Mono",monospace}
body{width:1080px;height:1350px;background:#E9E7DF;color:var(--ink);font-family:"Noto Sans CJK JP",sans-serif;padding:50px 52px 40px;display:flex;flex-direction:column;gap:20px}
.hd{display:flex;align-items:center;gap:14px}.hd .bar{width:7px;height:36px;background:var(--accent);border-radius:2px}
.hd .t{font-size:30px;font-weight:900}.hd .t em{font-style:normal;font-size:18px;color:var(--mut);font-weight:500;margin-left:8px}
.pg{border:2px solid var(--accent);color:var(--accent);border-radius:8px;font:700 17px var(--num);padding:0 9px}
.hd .dt{margin-left:auto;font-size:20px;color:var(--mut);font-family:var(--num)}
h1{font-size:54px;font-weight:900;line-height:1.36;letter-spacing:.01em;margin:6px 0 2px}
h1 em{font-style:normal;color:var(--neg)} h1 em.pos{color:var(--pos)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px 14px;flex:1;display:flex;flex-direction:column}
.cap{font-size:18px;font-weight:700}.cap em{font-style:normal;font-size:14px;color:var(--mut);font-weight:500;margin-left:6px}
.lg{display:flex;gap:18px;font-size:16px;color:var(--mut);margin-top:auto;padding-top:6px;flex-wrap:wrap}.lg i{display:inline-block;width:16px;height:12px;margin-right:6px;vertical-align:-1px}
svg text{font-family:"Noto Sans CJK JP"}.ax{font-size:16px;fill:var(--mut)}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.f{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px}
.f .k{font-size:17px;font-weight:700;color:#45473F}.f .v{font:700 38px var(--num);line-height:1.25}.f .v small{font-size:18px;color:var(--mut);font-family:"Noto Sans CJK JP";margin-left:3px}
.f .v.neg{color:var(--neg)}.f .v.pos{color:var(--pos)}.f .v.jp{font-family:"Noto Sans CJK JP";font-size:30px;font-weight:900}.f .n{font-size:16px;color:var(--mut)}
.chip{display:inline-block;font-size:20px;font-weight:700;border-radius:6px;padding:3px 10px;margin:6px 6px 0 0}
.chip.pos{background:var(--posbg);color:var(--pos)}.chip.neg{background:var(--negbg);color:var(--neg)}
.ft{display:flex;justify-content:space-between;align-items:center;font-size:18px;color:var(--mut)}
.ft b{color:var(--ink);font-size:21px}"""


def shell(no, title, en, body):
    who = f'<span><b>{E(C.get("name", ""))}</b>　{E(C["handle"])}</span>'
    return (f'<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><style>{CSS}</style></head><body>'
            f'<div class="hd"><div class="bar"></div><div class="t">{E(title)}<em>{E(en)}</em></div><span class="pg">{no}/3</span>'
            f'<div class="dt">日本株 {E(D["session"])} 終値</div></div>{body}'
            f'<div class="ft">{who}<span>売買の推奨ではありません</span></div></body></html>')


def facts(fs):
    return '<div class="facts">' + "".join(
        f'<div class="f"><div class="k">{E(f["k"])}</div><div class="v {f.get("tone", "")}">{E(f["v"])}<small>{E(f.get("unit", ""))}</small></div>'
        f'<div class="n">{E(f.get("note", ""))}</div></div>' for f in fs) + "</div>"


U = D["pulse"]
panic_by_date = dict(zip(U["hist"]["dates"], [(x or 0) >= 0.05 for x in U["hist"]["panic"]]))


def months(dates, X, H, every=1):
    s, last = "", None
    for i, d in enumerate(dates):
        if d[:7] != last and i > 2 and (int(d[5:7]) - 1) % every == 0:
            lab = f"{d[:4]}年" if d[5:7] == "01" else f"{int(d[5:7])}月"
            s += f'<text x="{X(i):.1f}" y="{H - 8}" class="ax">{lab}</text>'
        last = d[:7]
    return s


# ---------- 1 今日の判定
RH = D["regime_hist"]
dates, q, m50, m200 = RH["dates"], RH["close"], RH["ma50"], RH["ma200"]
N = len(q)
W, H, PL, PR, PT, PB = 954, 730, 8, 120, 16, 30
vals = [x for x in q + m50 + m200 if x is not None]
lo, hi = min(vals), max(vals)
pad = (hi - lo) * 0.07
lo, hi = lo - pad, hi + pad
X = lambda i: PL + (W - PL - PR) * i / (N - 1)
Y = lambda v: PT + (H - PT - PB) * (1 - (v - lo) / (hi - lo))
bw = (W - PL - PR) / (N - 1)
s = "".join(f'<rect x="{X(i) - bw / 2:.1f}" y="{PT}" width="{bw + .6:.1f}" height="{H - PT - PB}" fill="{NEGBG}"/>'
            for i, d in enumerate(dates) if panic_by_date.get(d))
for arr, col, wdt in ((m200, OCH, 2.2), (m50, BLUE, 2.2), (q, INK, 2.8)):
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(arr) if v is not None)
    s += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wdt}" stroke-linejoin="round"/>'
s += f'<circle cx="{X(N - 1):.1f}" cy="{Y(q[-1]):.1f}" r="6" fill="{INK}"/><text x="{X(N - 1) + 12:.1f}" y="{Y(q[-1]) + 7:.1f}" style="font:700 22px DejaVu Sans Mono;fill:{INK}">{q[-1]:.1f}</text>'
s += f'<text x="{X(N - 1) + 12:.1f}" y="{Y(m200[-1]) + 7:.1f}" style="font:700 17px Noto Sans CJK JP;fill:{OCH}">200日線</text>' if abs(Y(m200[-1]) - Y(q[-1])) > 26 else ""
s += f'<text x="{PL + 6}" y="{PT + 24}" style="font:700 19px Noto Sans CJK JP;fill:{INK}">TOPIX連動ETF（1306）</text>'
s += months(dates, X, H)
p1 = C["p1"]
pd_n = sum(1 for d in dates if panic_by_date.get(d))
body1 = (f'<h1>{p1["headline"]}</h1>'
         f'<div class="card"><div class="cap">TOPIXと全面安の日<em>直近120日</em></div><svg width="{W}" height="{H}">{s}</svg>'
         f'<div class="lg"><span><i style="background:{INK}"></i>終値</span><span><i style="background:{BLUE}"></i>50日線</span>'
         f'<span><i style="background:{OCH}"></i>200日線</span><span><i style="background:{NEGBG}"></i>全面安の日</span>'
         f'<span style="margin-left:auto">直近120日の全面安：{pd_n}日</span></div></div>' + facts(p1["facts"]))
(outdir / "card1.html").write_text(shell(1, "今日の判定", "Market Mode", body1), encoding="utf-8")

# ---------- 2 市場の中身
hd, ha = U["hist"]["dates"], [x * 100 if x is not None else None for x in U["hist"]["a50"]]
idx = [i for i, x in enumerate(ha) if x is not None]
hd, ha = [hd[i] for i in idx], [ha[i] for i in idx]
n2 = len(ha)
W2, H2, PL2, PR2, PT2, PB2 = 954, 730, 54, 96, 16, 30
lo2, hi2 = 0, max(100, max(ha))
X2 = lambda i: PL2 + (W2 - PL2 - PR2) * i / (n2 - 1)
Y2 = lambda v: PT2 + (H2 - PT2 - PB2) * (1 - (v - lo2) / (hi2 - lo2))
bw2 = (W2 - PL2 - PR2) / (n2 - 1)
s2 = "".join(f'<rect x="{X2(i) - bw2 / 2:.1f}" y="{PT2}" width="{bw2 + .6:.1f}" height="{H2 - PT2 - PB2}" fill="{NEGBG}"/>'
             for i, d in enumerate(hd) if panic_by_date.get(d))
for t in range(0, 101, 20):
    s2 += f'<line x1="{PL2}" x2="{W2 - PR2}" y1="{Y2(t):.1f}" y2="{Y2(t):.1f}" stroke="{LINE}" stroke-width="1"/>'
    s2 += f'<text x="{PL2 - 8}" y="{Y2(t) + 5:.1f}" class="ax" text-anchor="end">{t}%</text>'
s2 += f'<line x1="{PL2}" x2="{W2 - PR2}" y1="{Y2(20):.1f}" y2="{Y2(20):.1f}" stroke="{NEG}" stroke-width="2" stroke-dasharray="10 7"/>'
s2 += f'<text x="{W2 - PR2 + 10}" y="{Y2(20) + 6:.1f}" style="font:700 16px Noto Sans CJK JP;fill:{NEG}">20%</text>'
s2 += f'<polyline points="{" ".join(f"{X2(i):.1f},{Y2(v):.1f}" for i, v in enumerate(ha))}" fill="none" stroke="{INK}" stroke-width="2.6" stroke-linejoin="round"/>'
s2 += f'<circle cx="{X2(n2 - 1):.1f}" cy="{Y2(ha[-1]):.1f}" r="6" fill="{INK}"/><text x="{X2(n2 - 1) + 12:.1f}" y="{Y2(ha[-1]) + 7:.1f}" style="font:700 22px DejaVu Sans Mono;fill:{INK}">{ha[-1]:.0f}%</text>'
s2 += months(hd, X2, H2, 3)
p2 = C["p2"]
body2 = (f'<h1>{p2["headline"]}</h1>'
         f'<div class="card"><div class="cap">50日線より上の割合<em>流動性上位900銘柄・直近1年</em></div><svg width="{W2}" height="{H2}">{s2}</svg>'
         f'<div class="lg"><span><i style="background:{NEGBG}"></i>全面安の日（5%以上が25日線から−20%以下）</span>'
         f'<span><i style="border-top:3px dashed {NEG};height:0;margin-bottom:3px"></i>20%（反発サインの待機ライン）</span></div></div>' + facts(p2["facts"]))
(outdir / "card2.html").write_text(shell(2, "市場の中身", "Market Inside", body2), encoding="utf-8")

# ---------- 3 資金の流れ
fl = U["flow"]
rows = fl[:8] + [None] + fl[-8:]
W3, RH3, LBL = 954, 44, 210
H3 = RH3 * len(rows) + 28
mid = LBL + (W3 - LBL - 70) / 2
half = (W3 - LBL - 70) / 2
s3 = f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="0" y2="{H3 - 24}" stroke="{MUT}" stroke-width="1.4"/>'
for i, x in enumerate(rows):
    y = i * RH3
    if x is None:
        s3 += f'<text x="{LBL / 2:.0f}" y="{y + RH3 / 2 + 6:.1f}" class="ax" text-anchor="middle">⋮ 中位{len(fl) - 16}業種</text>'
        continue
    w = half * abs(x["flow"]) / 100
    x0 = mid if x["flow"] >= 0 else mid - w
    tone = POS if x["flow"] > 0 else NEG if x["flow"] < 0 else MUT
    s3 += (f'<text x="4" y="{y + RH3 / 2 + 7:.1f}" style="font:700 20px Noto Sans CJK JP;fill:{INK}">{E(x["name"])}</text>'
           f'<rect x="{x0:.1f}" y="{y + 7}" width="{max(w, 2):.1f}" height="{RH3 - 14}" rx="3" fill="{tone}" opacity=".85"/>'
           f'<text x="{W3 - 4}" y="{y + RH3 / 2 + 7:.1f}" text-anchor="end" style="font:700 20px DejaVu Sans Mono;fill:{tone}">{x["flow"]:+d}</text>')
s3 += f'<text x="{mid - 10:.1f}" y="{H3 - 4}" class="ax" text-anchor="end">← 流出</text><text x="{mid + 10:.1f}" y="{H3 - 4}" class="ax">流入 →</text>'
top = lambda k: [x["name"] for x in sorted(fl, key=lambda x: -x[k])[:3]]
ch = lambda names, tone: "".join('<span class="chip %s">%s</span>' % (tone, E(n)) for n in names)
p3 = C["p3"]
body3 = (f'<h1>{p3["headline"]}</h1>'
         f'<div class="card"><div class="cap">33業種の資金の流れ<em>短期RS(21日) − 長期RS(189日)・流動性上位900の等金額</em></div><svg width="{W3}" height="{H3}" style="margin-top:8px">{s3}</svg></div>'
         f'<div class="facts" style="grid-template-columns:1fr 1fr">'
         f'<div class="f"><div class="k">短期RS 上位3<span style="color:var(--mut);font-weight:500"> RS21</span></div>{ch(top("rs21"), "pos")}</div>'
         f'<div class="f"><div class="k">長期RS 上位3<span style="color:var(--mut);font-weight:500"> RS189</span></div>{ch(top("rs189"), "neg")}</div></div>')
(outdir / "card3.html").write_text(shell(3, "資金の流れ", "Sector Flow", body3), encoding="utf-8")
print(json.dumps({"mode": U["mode"]["label"], "panic_share": U["panic_share"], "panic_days_120": pd_n,
                  "a50": U["a50"], "flow_top3": [(x["name"], x["flow"]) for x in fl[:3]],
                  "flow_bottom3": [(x["name"], x["flow"]) for x in fl[-3:]]}, ensure_ascii=False))
