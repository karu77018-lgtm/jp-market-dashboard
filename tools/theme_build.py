"""Theme section of data.json: curated fine themes + point-in-time co-moving groups (連動群).

Validation (local, 2018-01..2026-07 month-ends, next 120 sessions, leader profile = 6-month top 20% and
>= 90% of the 52-week high):
  member of a top-30% 連動群 : big-winner 8.5% / big-loser 4.4% / excess +5.2% (2018-21 +1.9, 2022- +6.9)
  no co-moving group          : big-winner 6.1% / big-loser 8.6% / excess -2.3%
  33 TSE sectors, top 30%     : big-winner 6.9% / big-loser 5.8% / excess +2.3%
"""
from __future__ import annotations

import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from themes import BROAD_INDUSTRY_ONLY, THEMES, corr_clusters, major_of, theme_membership

IGN_R63, NEAR_HI, MAX_EXT, MIN_LEAD, LEAD_RS, WATCH_R63 = 0.30, 0.97, 1.50, 3, 85, 0.20


def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def group_stats(members, cs, hs, ret, rs, ma200, last, ma50):
    sub = ret[members].iloc[-300:].clip(-0.5, 1.0)
    daily = sub.mean(axis=1).where(sub.notna().sum(axis=1) >= 3).fillna(0.0)
    ix = (1 + daily).cumprod()
    hi = ix.iloc[-252:].max()
    r21, r63 = ix.iloc[-1] / ix.iloc[-22] - 1, ix.iloc[-1] / ix.iloc[-64] - 1
    r63_then = ix.iloc[-22] / ix.iloc[-85] - 1
    ext = ix.iloc[-1] / ix.iloc[-200:].mean() - 1
    fh = ix.iloc[-1] / hi - 1
    leaders = [t for t in members if rs.get(t, 0) >= LEAD_RS and last[t] > ma200.iloc[-1][t]]
    ign = ix.iloc[-1] >= NEAR_HI * hi and r63 >= IGN_R63
    if ign and len(leaders) >= MIN_LEAD and ext <= MAX_EXT:
        status = "合格"
    elif ign and len(leaders) >= MIN_LEAD:
        status = "過熱"
    elif r63 >= WATCH_R63 and fh >= -0.10 and len(leaders) >= MIN_LEAD:
        status = "監視"
    else:
        status = ""
    spark = (ix.iloc[-120:] / ix.iloc[-120]).round(4).tolist()
    return {"r21": _f(r21), "r63": _f(r63), "r63_then": _f(r63_then), "from_high": _f(fh), "ext200": _f(ext),
            "leaders": len(leaders), "status": status,
            "a50": _f(np.mean([last[t] > ma50.iloc[-1][t] for t in members])), "spark": spark}


def compute(cs, hs, ls, ret, mret, liquid, rs, ma50, ma200, dv50, uni, info, near, r126):
    last = cs.iloc[-1]
    live = [t for t in liquid[liquid].index]
    names = {t: unicodedata.normalize("NFKC", str(uni.at[t, "name"])) for t in live if t in uni.index}
    r126r = r126.iloc[-1].where(liquid).rank(pct=True)
    nearv = near.iloc[-1]
    lead_profile = (r126r >= 0.8) & (nearv >= 0.9)

    # ---- curated themes
    tm = theme_membership(live, info)
    mj = major_of()
    themes = []
    for major, fine, *_ in THEMES:
        mem = [t for t in live if fine in tm.get(t, [])]
        if len(mem) < 4:
            continue
        st = group_stats(mem, cs, hs, ret, rs, ma200, last, ma50)
        top = sorted(mem, key=lambda t: -(rs.get(t) or 0))[:5]
        themes.append({"major": major, "name": fine, "n": len(mem), **st, "top": top,
                       "lead": [t for t in mem if bool(lead_profile.get(t, False))]})
    themes.sort(key=lambda x: -(x["r63"] or -9))
    then = sorted(themes, key=lambda x: -(x["r63_then"] or -9))
    rk = {x["name"]: i + 1 for i, x in enumerate(then)}
    for i, x in enumerate(themes):
        x["rank"], x["rank_then"] = i + 1, rk[x["name"]]

    # ---- co-moving groups (point-in-time, last 120 sessions)
    cl = corr_clusters(ret, mret, live)
    s = pd.Series(cl)
    groups = []
    for k, mem in s.groupby(s).groups.items():
        mem = list(mem)
        st = group_stats(mem, cs, hs, ret, rs, ma200, last, ma50)
        th = Counter(th for t in mem for th in tm.get(t, []))
        ind = Counter((info.get(t) or {}).get("industry") for t in mem if (info.get(t) or {}).get("industry"))
        need = max(2, 0.4 * len(mem))
        narrow = [(k, v) for k, v in th.most_common() if k not in BROAD_INDUSTRY_ONLY and v >= need]
        broad = [(k, v) for k, v in th.most_common() if v >= need]
        label = narrow[0][0] if narrow else (broad[0][0] if broad else None)
        big = sorted(mem, key=lambda t: -(dv50.iloc[-1].get(t) or 0))
        rr = ret[mem].iloc[-120:].sub(mret.iloc[-120:], axis=0).fillna(0.0)
        cm = np.corrcoef(rr.to_numpy().T)
        coh = float((cm.sum() - len(mem)) / (len(mem) * (len(mem) - 1)))
        groups.append({"id": int(k), "n": len(mem), "label": label,
                       "industry": ind.most_common(1)[0][0] if ind else None,
                       "core": big[:4], "cohesion": _f(coh, 3), "members": sorted(mem, key=lambda t: -(rs.get(t) or 0)), **st,
                       "lead": [t for t in mem if bool(lead_profile.get(t, False))]})
    groups.sort(key=lambda x: -(x["r63"] or -9))
    n = len(groups)
    for i, g in enumerate(groups):
        g["rank_pct"] = _f(1 - i / max(n - 1, 1), 3)
    # ---- leader candidates: profile + member of a top-30% co-moving group
    top_ids = {g["id"] for g in groups if g["rank_pct"] is not None and g["rank_pct"] >= 0.7}
    in_group = {t: g["id"] for g in groups for t in g["members"]}
    cands, lone = [], []
    for t in live:
        if not bool(lead_profile.get(t, False)):
            continue
        gid = in_group.get(t)
        row = {"t": t, "rs": int(rs[t]) if pd.notna(rs.get(t)) else None, "near": _f(nearv[t], 3),
               "r126": _f(r126.iloc[-1][t]), "gid": gid, "themes": tm.get(t, [])[:3]}
        if gid in top_ids:
            cands.append(row)
        elif gid is None:
            lone.append(row)
    cands.sort(key=lambda r: -(r["rs"] or 0))
    lone.sort(key=lambda r: -(r["rs"] or 0))
    used = {t for x in themes for t in x["top"] + x["lead"]} | {t for g in groups for t in g["members"]}
    used |= {r["t"] for r in cands + lone}
    return {"themes": themes, "groups": groups, "cands": cands[:40], "lone": lone[:20],
            "names_extra": {t: names.get(t, "") for t in used}}
