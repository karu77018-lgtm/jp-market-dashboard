#!/usr/bin/env python3
"""Compute the JP dashboard numbers from cache/ohlcv.parquet -> site/data.json (runs locally only).

Ports of the US site logic (swing screener core rule, theme gate 3 conditions, trend template).
JP thresholds are a first translation and are NOT yet verified on Japanese data.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("JP_CACHE", Path.home() / "jp-cache"))
REFS = {"1306.T": "TOPIX連動ETF(1306)", "^N225": "日経平均", "2516.T": "グロース250ETF(2516)"}
REGIME_SYM = "1306.T"

# --- swing core rule (US: $10 / $20M -> JP: ¥200 / ¥5億, percentiles unchanged) ---
MIN_PRICE, MIN_DV = 200.0, 5e8
DV_PCT, RS189_PCT = 95, 90
MAX_VC = MAX_VDRY = 0.9
MAX_CHG = MAX_PREV_CHG = 0.03
MAX_UDVR, UDVR_LEN = 1.4, 50
STOP = 0.08
LATE_LAGS, LATE_MAX_UP = (1, 2), 0.03
STALE_DAYS = 60
PIVOT_LENGTHS = range(2, 11)
SAR_PARAMS = (0.02, 0.02, 0.08)
TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
GOOD_POS = (0.50, 0.75)

# --- sector (33業種) gate, same three conditions as the US theme gate ---
SECT_MIN_DV = 1e8
IGNITION_R63, NEAR_HIGH, MAX_EXT200 = 0.30, 0.97, 1.50
MIN_LEADERS, LEADER_RS, WATCH_R63, MIN_MEMBERS = 3, 85, 0.20, 5


def fnum(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


# ---------------------------------------------------------------- helpers ported from the US site
def psar_flags(h, l, c, a0=0.02, inc=0.02, amax=0.08):
    n = len(c)
    up, flip = np.zeros(n, bool), np.zeros(n, bool)
    if n < 3:
        return up, flip
    bull = c[1] >= c[0]
    ep, s, af = (h[0], l[0], a0) if bull else (l[0], h[0], a0)
    up[0] = bull
    for i in range(1, n):
        s = s + af * (ep - s)
        if bull:
            s = min(s, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < s:
                bull, s, ep, af = False, ep, l[i], a0
            elif h[i] > ep:
                ep, af = h[i], min(af + inc, amax)
        else:
            s = max(s, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > s:
                bull, s, ep, af = True, ep, h[i], a0
                flip[i] = True
            elif l[i] < ep:
                ep, af = l[i], min(af + inc, amax)
        up[i] = bull
    return up, flip


def weekly_sar_state(high, low, close):
    df = pd.DataFrame({"h": high, "l": low, "c": close}).dropna()
    if len(df) < 30:
        return None, None
    wk = df.index.to_period("W-FRI")
    agg = df.groupby(wk).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    if df.index[-1].weekday() != 4:
        agg = agg.iloc[:-1]
    if len(agg) < 10:
        return None, None
    up, flip = psar_flags(agg["h"].to_numpy(float), agg["l"].to_numpy(float), agg["c"].to_numpy(float), *SAR_PARAMS)
    if not up[-1]:
        return False, None
    hits = np.where(flip)[0]
    return True, (int(len(flip) - 1 - hits[-1]) if len(hits) else None)


def structure_pivot(high, low):
    n_bars = len(low)
    best = None
    for n in PIVOT_LENGTHS:
        roll = pd.Series(low).rolling(2 * n + 1, center=True).min().to_numpy()
        pivots = np.where((low == roll) & ~np.isnan(low))[0]
        confirm = {p + n: p for p in pivots if p + n < n_bars}
        prev, setup = None, None
        for t in range(n_bars):
            if setup is not None and low[t] < setup["hl"]:
                setup = None
            if t in confirm:
                p = confirm[t]
                if prev is not None and low[p] > low[prev]:
                    seg = high[prev:p + 1]
                    k = int(np.nanargmax(seg))
                    setup = {"line": float(seg[k]), "hl": float(low[p])}
                prev = p
        if setup is not None and (best is None or setup["line"] < best["line"]):
            best = setup
    return (float("nan"), float("nan")) if best is None else (best["line"], best["hl"])


def hl_side(r):
    pos = r.get("pos")
    return bool(r.get("inside")) and pos is not None and pos < 0.5


def buyable(r):
    return bool(r.get("sar_up")) and not hl_side(r)


def _good(r):
    pos = r.get("pos")
    return bool(r.get("inside")) and pos is not None and GOOD_POS[0] <= pos <= GOOD_POS[1]


def tier(r):
    up, age = r.get("sar_up"), r.get("sar_age")
    if not up:
        return "D"
    age = 99 if age is None else age
    if _good(r) and age <= 5:
        return "S"
    line, hl = r.get("pivot_line"), r.get("hl")
    if line is None or (isinstance(line, float) and math.isnan(line)) or not hl or hl >= line:
        return "D"
    if 2 <= age <= 5:
        return "A"
    if age <= 8:
        return "B"
    return "C"


def watch_bucket(r):
    t, n = tier(r), len(r.get("missing", []))
    if t in ("S", "A") and n <= 1:
        return 0
    if t == "B" and n <= 1:
        return 1
    if t in ("S", "A") and n == 2:
        return 2
    if t == "B" or n <= 1:
        return 3
    return 4


def pct_rank(values, mask):
    return values.where(mask).rank(pct=True) * 100


def rs_percentile(close, mask):
    """IBD-style weighted 3/6/9/12-month strength, 1-99, ranked within ``mask``."""
    last = close.iloc[-1]

    def ret(n):
        return last / close.iloc[-1 - n] - 1 if len(close) > n else pd.Series(np.nan, index=close.columns)

    raw = 0.4 * ret(63) + 0.2 * ret(126) + 0.2 * ret(189) + 0.2 * ret(252)
    raw = raw.where(raw.notna(), 0.4 * ret(63) + 0.6 * ret(126))
    return (raw.where(mask).rank(pct=True) * 98 + 1).round()


# ---------------------------------------------------------------- main
def main():
    df = pd.read_parquet(CACHE / "ohlcv.parquet")
    uni = pd.read_csv(CACHE / "universe.csv", dtype=str).set_index("ticker")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    P = {k: df.pivot(index="date", columns="ticker", values=k).sort_index() for k in
         ("open", "high", "low", "close", "volume")}
    # Yahoo occasionally ships a one-day bad print (e.g. 1306 on 2026-03-31 at 36.4 instead of 371.4).
    # (1306 on 2026-03-30/31 at ~36 instead of ~375: a mis-adjusted split.) A bar more than 2.5x away
    # from the centred 11-session median (i.e. a short excursion that comes back) is dropped for every field.
    cl = P["close"]
    med = cl.rolling(11, center=True, min_periods=5).median()
    ratio = cl / med
    bad = (ratio > 2.5) | (ratio < 0.4)
    print("bad prints dropped:", int(bad.sum().sum()))
    P = {k: v.mask(bad) for k, v in P.items()}
    # the session is the latest date that most stocks traded (guards against a stray partial bar)
    counts = P["close"][[t for t in P["close"].columns if t in uni.index]].notna().sum(axis=1)
    session = counts[counts >= 0.6 * counts.max()].index.max()
    P = {k: v.loc[:session] for k, v in P.items()}
    o, h, l, c, v = P["open"], P["high"], P["low"], P["close"], P["volume"]
    stocks = [t for t in c.columns if t in uni.index]
    cs, hs, ls, vs = c[stocks], h[stocks], l[stocks], v[stocks]
    out = {"session": str(session.date())}

    # ---------------- reference indices / regime
    refs = []
    for sym, name in REFS.items():
        if sym not in c.columns:
            continue
        s = c[sym].dropna()
        ma50, ma200 = s.rolling(50).mean(), s.rolling(200).mean()
        hi = h[sym].dropna().rolling(252, min_periods=200).max()
        refs.append({"sym": sym, "name": name, "close": fnum(s.iloc[-1], 2), "date": str(s.index[-1].date()),
                     "chg": fnum(s.iloc[-1] / s.iloc[-2] - 1), "r21": fnum(s.iloc[-1] / s.iloc[-22] - 1),
                     "r63": fnum(s.iloc[-1] / s.iloc[-64] - 1),
                     "ma50": fnum(ma50.iloc[-1], 2), "ma200": fnum(ma200.iloc[-1], 2),
                     "vs50": fnum(s.iloc[-1] / ma50.iloc[-1] - 1), "vs200": fnum(s.iloc[-1] / ma200.iloc[-1] - 1),
                     "ma200_up": bool(ma200.iloc[-1] > ma200.iloc[-21]),
                     "from_high": fnum(s.iloc[-1] / hi.iloc[-1] - 1)})
    reg = next((r for r in refs if r["sym"] == REGIME_SYM), None)
    regime_on = bool(reg and reg["date"] == out["session"] and reg["vs200"] is not None and reg["vs200"] > 0)
    out["refs"], out["regime"] = refs, {"on": regime_on, **(reg or {})}

    # ---------------- per-stock state
    ma10, ma50, ma150, ma200 = (cs.rolling(n).mean() for n in (10, 50, 150, 200))
    hi252 = hs.rolling(252, min_periods=200).max()
    lo252 = ls.rolling(252, min_periods=200).min()
    dv = cs * vs
    dv50 = dv.rolling(50, min_periods=40).mean()
    hist_ok = cs.notna().sum() >= 252
    last = cs.iloc[-1]
    traded = cs.iloc[-1].notna()
    rs = rs_percentile(cs, hist_ok & traded)

    # Minervini trend template (8 conditions) for the breadth / TT card
    tt8 = pd.DataFrame({
        "1": (last > ma150.iloc[-1]) & (last > ma200.iloc[-1]),
        "2": ma150.iloc[-1] > ma200.iloc[-1],
        "3": ma200.iloc[-1] > ma200.iloc[-21],
        "4": (ma50.iloc[-1] > ma150.iloc[-1]) & (ma50.iloc[-1] > ma200.iloc[-1]),
        "5": last > ma50.iloc[-1],
        "6": last >= 1.30 * lo252.iloc[-1],
        "7": last >= 0.75 * hi252.iloc[-1],
        "8": rs >= 70,
    }).fillna(False)
    tt_n = tt8.sum(axis=1)
    tt_pass = (tt_n == 8) & hist_ok & traded

    # ---------------- breadth (all + by market), plus a 120-session history
    mk = uni["market"].reindex(stocks)
    base = hist_ok & traded
    above50 = cs > ma50
    above200 = cs > ma200
    newhi = hs >= hs.rolling(252, min_periods=200).max()
    newlo = ls <= ls.rolling(252, min_periods=200).min()

    def breadth(mask):
        n = int(mask.sum())
        cols = mask[mask].index
        return {"n": n,
                "a50": fnum(above50.iloc[-1][cols].mean(), 3), "a200": fnum(above200.iloc[-1][cols].mean(), 3),
                "nh": int(newhi.iloc[-1][cols].sum()), "nl": int(newlo.iloc[-1][cols].sum()),
                "tt": int(tt_pass[cols].sum()), "tt_rate": fnum(tt_pass[cols].mean(), 3)}

    out["breadth"] = {"all": breadth(base)}
    for m, name in (("P", "プライム"), ("S", "スタンダード"), ("G", "グロース")):
        out["breadth"][m] = {"name": name, **breadth(base & (mk == m))}
    cols = base[base].index
    hist_idx = cs.index[-120:]
    tt_hist = []
    for d in hist_idx:  # simplified TT history (conditions 1-7, RS omitted for speed/look-ahead safety)
        k = cs.index.get_loc(d)
        if k < 21:
            continue
        lc = cs.iloc[k][cols]
        m200 = ma200.iloc[k][cols]
        ok = (lc > ma150.iloc[k][cols]) & (lc > m200) & (ma150.iloc[k][cols] > m200) & (m200 > ma200.iloc[k - 20][cols]) \
            & (ma50.iloc[k][cols] > ma150.iloc[k][cols]) & (lc > ma50.iloc[k][cols]) \
            & (lc >= 1.3 * lo252.iloc[k][cols]) & (lc >= 0.75 * hi252.iloc[k][cols])
        tt_hist.append(int(ok.sum()))
    out["breadth_hist"] = {
        "dates": [str(d.date()) for d in hist_idx][-len(tt_hist):],
        "a50": [fnum(x, 3) for x in above50.loc[hist_idx, cols].mean(axis=1)][-len(tt_hist):],
        "a200": [fnum(x, 3) for x in above200.loc[hist_idx, cols].mean(axis=1)][-len(tt_hist):],
        "nh": [int(x) for x in newhi.loc[hist_idx, cols].sum(axis=1)][-len(tt_hist):],
        "nl": [int(x) for x in newlo.loc[hist_idx, cols].sum(axis=1)][-len(tt_hist):],
        "tt7": tt_hist,
    }

    # ---------------- 33業種 rotation + 3-condition gate
    rets = cs.pct_change(fill_method=None).clip(-0.5, 1.0)
    sect_liq = hist_ok & traded & (dv50.iloc[-1] >= SECT_MIN_DV)
    r63s = cs.iloc[-1] / cs.iloc[-64] - 1
    hi60 = hs.rolling(60, min_periods=20).max().iloc[-1]
    day = cs.iloc[-1] / cs.iloc[-2] - 1
    sectors = []
    s33 = uni["s33"].reindex(stocks)
    for name, members in s33[sect_liq].groupby(s33[sect_liq]):
        mem = list(members.index)
        if len(mem) < MIN_MEMBERS:
            continue
        sub = rets[mem]
        daily = sub.mean(axis=1).where(sub.notna().sum(axis=1) >= MIN_MEMBERS).iloc[-300:].fillna(0.0)
        idx = (1 + daily).cumprod()
        win = idx.iloc[-252:]
        hi = win.max()
        t_r21, t_r63, t_r126 = (idx.iloc[-1] / idx.iloc[-1 - n] - 1 for n in (21, 63, 126))
        rank_then = idx.iloc[-22] / idx.iloc[-85] - 1  # 63-day return as of 21 sessions ago
        from_high = idx.iloc[-1] / hi - 1
        ext200 = idx.iloc[-1] / idx.iloc[-200:].mean() - 1
        leaders = [t for t in mem if rs.get(t, 0) >= LEADER_RS and last[t] > ma200.iloc[-1][t]]
        ignition = idx.iloc[-1] >= NEAR_HIGH * hi and t_r63 >= IGNITION_R63
        breadth_ok = len(leaders) >= MIN_LEADERS
        not_hot = ext200 <= MAX_EXT200
        if ignition and breadth_ok and not_hot:
            status = "合格"
        elif ignition and breadth_ok:
            status = "過熱"
        elif t_r63 >= WATCH_R63 and from_high >= -0.10 and breadth_ok:
            status = "監視"
        else:
            status = ""
        cands = []
        if status == "合格":
            top = r63s[mem].idxmax()
            for t in mem:
                if t == top:
                    continue
                off = last[t] / hi60[t] - 1
                if rs.get(t, 0) >= 70 and last[t] > ma50.iloc[-1][t] and -0.15 <= off <= -0.03 and day[t] < 0.03:
                    cands.append({"t": t, "rs": int(rs[t]), "off": fnum(off)})
            cands = sorted(cands, key=lambda x: -x["rs"])[:3]
        sectors.append({"name": name, "n": len(mem), "r21": fnum(t_r21), "r63": fnum(t_r63), "r126": fnum(t_r126),
                        "r63_then": fnum(rank_then), "from_high": fnum(from_high), "ext200": fnum(ext200),
                        "leaders": len(leaders), "lead": sorted(leaders, key=lambda t: -rs[t])[:4],
                        "tt_rate": fnum(tt_pass[mem].mean(), 3), "a50": fnum(above50.iloc[-1][mem].mean(), 3),
                        "status": status, "cands": cands})
    sectors.sort(key=lambda s: -(s["r63"] or -9))
    then = sorted(sectors, key=lambda s: -(s["r63_then"] or -9))
    rank_then = {s["name"]: i + 1 for i, s in enumerate(then)}
    for i, s in enumerate(sectors):
        s["rank"], s["rank_then"] = i + 1, rank_then[s["name"]]
    out["sectors"] = sectors
    out["sectors_skipped"] = sorted(set(s33.dropna()) - {s["name"] for s in sectors})

    # ---------------- RS leaders (TT 8/8, by RS)
    lead_mask = tt_pass & (dv50.iloc[-1] >= SECT_MIN_DV)
    leaders = sorted(lead_mask[lead_mask].index, key=lambda t: (-rs[t], -dv50.iloc[-1][t]))[:40]
    out["rs_leaders"] = [{"t": t, "rs": int(rs[t]), "close": fnum(last[t], 1),
                          "from_high": fnum(last[t] / hi252.iloc[-1][t] - 1), "dv50": fnum(dv50.iloc[-1][t] / 1e8, 1)}
                         for t in leaders]

    # ---------------- swing screener (core rule port)
    dr = hs / ls - 1
    vc = dr.rolling(10).mean() / dr.rolling(50).mean()
    v50 = vs.rolling(50).mean()
    vdry = vs.rolling(5).mean() / v50
    chg = (cs / cs.shift(1) - 1).round(9)
    el21 = ls.ewm(span=21, adjust=False).mean()
    up_v = vs.where(cs > cs.shift(1), 0.0).rolling(UDVR_LEN).mean()
    dn_v = vs.where(cs < cs.shift(1), 0.0).rolling(UDVR_LEN).mean()
    udvr = up_v / dn_v

    def at(k):
        lk = cs.iloc[k]
        liquid = (lk >= MIN_PRICE) & (dv50.iloc[k] >= MIN_DV)
        tt = (lk > ma50.iloc[k]) & (ma50.iloc[k] > ma150.iloc[k]) & (ma150.iloc[k] > ma200.iloc[k]) \
            & (ma200.iloc[k] > ma200.iloc[k - 20]) & (lk >= 0.75 * hi252.iloc[k])

        def ret(n):
            return lk / cs.iloc[k - n] - 1

        r189 = ret(189)
        dvp = pct_rank(dv50.iloc[k], liquid)
        rs189 = pct_rank(r189, liquid)
        checks = {"収縮": vc.iloc[k] <= MAX_VC, "出来高減": vdry.iloc[k] <= MAX_VDRY,
                  "当日+3%未満": chg.iloc[k] < MAX_CHG, "前日+3%以下": chg.iloc[k - 1] <= MAX_PREV_CHG}
        timing = pd.Series(True, index=cs.columns)
        for m in checks.values():
            timing &= m.fillna(False)
        selected = (liquid & tt & (dvp >= DV_PCT) & (rs189 >= RS189_PCT)).fillna(False)
        ud_ok = ~(udvr.iloc[k] >= MAX_UDVR)
        return {"last": lk, "liquid": liquid, "tt": tt, "dv_pct": dvp, "rs189": rs189,
                "rs21": pct_rank(ret(21), liquid), "rs63": pct_rank(ret(63), liquid), "checks": checks,
                "timing": timing, "selected": selected, "signal": selected & timing & ud_ok, "ud_ok": ud_ok,
                "udvr": udvr.iloc[k]}

    s = at(-1)
    selected, timing, ud_ok = s["selected"], s["timing"], s["ud_ok"]
    streak = {t: 0 for t in selected[selected].index}
    alive = set(streak)
    for back in range(1, STALE_DAYS + 2):
        if not alive:
            break
        sel = selected if back == 1 else at(-back)["selected"]
        for t in list(alive):
            if bool(sel.get(t, False)):
                streak[t] += 1
            else:
                alive.discard(t)

    def row(t):
        px = float(s["last"][t])
        line, hl = structure_pivot(hs[t].to_numpy(float)[-260:], ls[t].to_numpy(float)[-260:])
        inside = bool(not math.isnan(line) and px <= line)
        pos = (px - hl) / (line - hl) if inside and line > hl else None
        try:
            up, age = weekly_sar_state(hs[t], ls[t], cs[t])
        except Exception:  # noqa: BLE001
            up, age = None, None
        return {"t": t, "close": px, "rs189": int(round(s["rs189"][t])),
                "rs63": None if pd.isna(s["rs63"][t]) else int(round(s["rs63"][t])),
                "rs21": None if pd.isna(s["rs21"][t]) else int(round(s["rs21"][t])),
                "rs": None if pd.isna(rs.get(t)) else int(rs[t]),
                "pivot_line": None if math.isnan(line) else line, "hl": None if math.isnan(hl) else hl,
                "inside": inside, "pos": pos, "streak": streak.get(t), "dv": int(round(s["dv_pct"][t])),
                "vc": fnum(vc.iloc[-1][t], 3), "vdry": fnum(vdry.iloc[-1][t], 3), "chg": fnum(chg.iloc[-1][t]),
                "udvr": fnum(s["udvr"][t], 2), "el21": fnum(el21.iloc[-1][t], 1),
                "stop": px * (1 - STOP), "add": px * 1.10, "add2": px * 1.20, "sar_up": up, "sar_age": age}

    def was_buyable(t, lag):
        end = len(cs) - lag
        px = float(cs[t].iloc[end - 1])
        line, hl = structure_pivot(hs[t].to_numpy(float)[:end][-260:], ls[t].to_numpy(float)[:end][-260:])
        inside = bool(not math.isnan(line) and px <= line)
        pos = (px - hl) / (line - hl) if inside and line > hl else None
        try:
            up, _ = weekly_sar_state(hs[t].iloc[:end], ls[t].iloc[:end], cs[t].iloc[:end])
        except Exception:  # noqa: BLE001
            up = None
        return buyable({"inside": inside, "pos": pos, "sar_up": up})

    rk = lambda r: TIER_ORDER[tier(r)]  # noqa: E731
    core = [row(t) for t in selected[selected & timing & ud_ok.fillna(True)].index]
    core.sort(key=lambda r: (rk(r), not r["inside"], -r["rs189"]))
    late, seen = [], {r["t"] for r in core}
    for lag in LATE_LAGS:
        past = at(-1 - lag)
        sig = past["signal"].fillna(False)
        for t in sig[sig].index:
            if t in seen or not bool(selected.get(t, False)) or not bool(chg.iloc[-1][t] < MAX_CHG):
                continue
            base_px = float(past["last"][t])
            lows, closes = ls[t].iloc[-lag:], cs[t].iloc[-lag:]
            if (lows <= base_px * (1 - STOP)).any() or (closes < el21[t].iloc[-lag:]).any():
                continue
            if float(s["last"][t]) > base_px * (1 + LATE_MAX_UP) or not was_buyable(t, lag):
                continue
            r = row(t)
            r.update({"signal_date": str(cs.index[-1 - lag].date()), "from_signal": float(s["last"][t]) / base_px - 1,
                      "lag": lag})
            late.append(r)
            seen.add(t)
    late.sort(key=lambda r: (rk(r), r["lag"], -r["rs189"]))
    watch = []
    for t in selected[selected & ~(timing & ud_ok.fillna(True))].index:
        if t in seen:
            continue
        r = row(t)
        r["missing"] = [k for k, m in s["checks"].items() if not bool(m.get(t, False))]
        if not bool(ud_ok.fillna(True).get(t, True)):
            r["missing"].append("UD比1.4未満")
        watch.append(r)
    watch.sort(key=lambda r: (watch_bucket(r), rk(r), -r["rs189"]))
    for lst in (core, late, watch):
        for r in lst:
            r["tier"] = tier(r)
            r["buyable"] = buyable(r)
    out["swing"] = {"core": core, "late": late, "watch": watch[:20], "watch_total": len(watch),
                    "universe": int(s["liquid"].fillna(False).sum()), "selected": int(selected.sum())}

    # ---------------- names
    shown = {r["t"] for k in ("core", "late", "watch") for r in out["swing"][k]}
    shown |= {r["t"] for r in out["rs_leaders"]} | {c["t"] for sct in sectors for c in sct["cands"]}
    shown |= {t for sct in sectors for t in sct["lead"]}
    import unicodedata
    out["names"] = {t: {"name": unicodedata.normalize("NFKC", str(uni.at[t, "name"])), "s33": uni.at[t, "s33"], "m": uni.at[t, "market"]} for t in shown}
    out["params"] = {"min_price": MIN_PRICE, "min_dv": MIN_DV, "sect_min_dv": SECT_MIN_DV}

    site = ROOT

    def clean(x):
        if isinstance(x, float):
            return fnum(x, 4)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return fnum(float(x), 4)
        return x

    (site / "data.json").write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("session", out["session"], "regime", regime_on, "core", len(core), "late", len(late), "watch", len(watch),
          "liquid", out["swing"]["universe"], "sectors", len(sectors), "tt", out["breadth"]["all"]["tt"])


if __name__ == "__main__":
    main()
