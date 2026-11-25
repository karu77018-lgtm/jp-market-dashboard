"""Daily state of the rules that held up on Japanese data (2003-2026), for the 地合い tab and the notes.

Liquid = the 900 most traded stocks (50-day turnover) priced at ¥100 or more, the same universe as the studies.
History numbers come from stats.json (tools/signal_stats.py, run locally); this module only reports today's state.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LIQ_N, LIQ_MIN_PX = 900, 100
PANIC_DEV, PANIC_SHARE = -0.20, 0.05
WASH_TH, WASH_REC, WASH_HOLD, WASH_GAP = 0.20, 0.05, 20, 20
BEAR = -0.20
QUIET_TOP, QUIET_NEAR, QUIET_N = 300, 0.95, 20


def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def _zone(a):
    if a is None:
        return None
    for lo, hi, name in ((0, .2, "20%未満"), (.2, .4, "20〜40%"), (.4, .6, "40〜60%"), (.6, .8, "60〜80%"), (.8, 1.01, "80%以上")):
        if lo <= a < hi:
            return name
    return None


def compute(c: pd.DataFrame, h: pd.DataFrame, v: pd.DataFrame, uni: pd.DataFrame, ref_close: dict) -> dict:
    """c/h/v: stocks only (dates x tickers). ref_close: {sym: close Series}."""
    dates = c.index
    dv50 = (c * v).rolling(50, min_periods=40).mean()
    liq = dv50.where(c >= LIQ_MIN_PX).rank(axis=1, ascending=False) <= LIQ_N
    nliq = liq.sum(axis=1)
    ma25 = c.rolling(25, min_periods=25).mean()
    ma50 = c.rolling(50, min_periods=50).mean()
    a50 = (c > ma50).where(liq).mean(axis=1).where(nliq >= 500)
    dev = c / ma25 - 1
    share = ((dev <= PANIC_DEV) & liq).sum(axis=1) / nliq.replace(0, np.nan)
    panic = share >= PANIC_SHARE

    # washout rebound: a50 under 20% within 10 sessions, then +5 points; one signal per 20 sessions
    m10 = a50.rolling(10).min()
    sig = (m10 < WASH_TH) & (a50 - m10 >= WASH_REC)
    sig = (sig & ~sig.shift(1, fill_value=False)).fillna(False).to_numpy()
    ev, last = [], -999
    for i in np.flatnonzero(sig):
        if i - last >= WASH_GAP:
            ev.append(i)
            last = i
    n225 = ref_close.get("^N225")
    bear_now, n225_dd = None, None
    if n225 is not None:
        s = n225.reindex(dates).ffill()
        dd = s / s.rolling(252, min_periods=200).max() - 1
        n225_dd = _f(dd.iloc[-1])
        bear_now = bool(n225_dd is not None and n225_dd <= BEAR)
    n = len(dates)
    wash = {"last_signal": str(dates[ev[-1]].date()) if ev else None,
            "today": bool(ev and ev[-1] == n - 1),
            "days_since": int(n - 1 - ev[-1]) if ev else None,
            "a50_min10": _f(m10.iloc[-1], 3), "need": _f(m10.iloc[-1] + WASH_REC, 3) if pd.notna(m10.iloc[-1]) else None,
            "armed": bool(pd.notna(m10.iloc[-1]) and m10.iloc[-1] < WASH_TH)}
    # the 20-session window starts the session after the signal
    wash["active"] = bool(ev and 0 <= wash["days_since"] < WASH_HOLD and not bear_now)
    wash["left"] = WASH_HOLD - wash["days_since"] if wash["active"] else None

    # panic-day list: liquid stocks 20%+ under the 25-day line today (facts only)
    lastc, lastd = c.iloc[-1], dev.iloc[-1]
    chg = c.iloc[-1] / c.iloc[-2] - 1
    hit = lastd[(lastd <= PANIC_DEV) & liq.iloc[-1]].sort_values()
    bnf = [{"t": t, "dev": _f(lastd[t]), "close": _f(lastc[t], 1), "ma25": _f(ma25.iloc[-1][t], 1), "chg": _f(chg[t])}
           for t in hit.index[:15]]
    recent = [str(d.date()) for d in dates[panic.fillna(False).to_numpy()][-60:]]

    # quiet large caps: top 300 by turnover, within 5% of the 52-week high, lowest 60-day volatility
    top300 = dv50.iloc[-1].where(c.iloc[-1] >= LIQ_MIN_PX).rank(ascending=False) <= QUIET_TOP
    hi252 = c.rolling(252, min_periods=200).max().iloc[-1]
    near = lastc / hi252
    vol60 = c.pct_change(fill_method=None).iloc[-60:].std() * np.sqrt(245)
    pool = vol60[top300 & (near >= QUIET_NEAR) & vol60.notna()].sort_values()
    quiet = [{"t": t, "vol": _f(vol60[t], 3), "near": _f(near[t], 3), "r21": _f(lastc[t] / c[t].iloc[-22] - 1)}
             for t in pool.index[:QUIET_N]]

    # 33-sector money flow: equal-weight index of liquid members, RS21/RS189 = percentile rank among sectors
    s33 = uni["s33"].reindex(c.columns)
    ret = c.pct_change(fill_method=None).where(liq.shift(1, fill_value=False)).clip(-0.5, 1.0)
    rows = []
    for name, g in s33.groupby(s33):
        cols = list(g.index)
        if int(liq.iloc[-1][cols].sum()) < 5:
            continue
        idx = (1 + ret[cols].mean(axis=1).fillna(0)).cumprod()
        rows.append({"name": name, "n": int(liq.iloc[-1][cols].sum()),
                     "chg": idx.iloc[-1] / idx.iloc[-2] - 1, "r5": idx.iloc[-1] / idx.iloc[-6] - 1,
                     "r21": idx.iloc[-1] / idx.iloc[-22] - 1, "r189": idx.iloc[-1] / idx.iloc[-190] - 1,
                     "r21_5ago": idx.iloc[-6] / idx.iloc[-27] - 1, "r189_5ago": idx.iloc[-6] / idx.iloc[-195] - 1})
    F = pd.DataFrame(rows).set_index("name")
    for k in ("r21", "r189", "r21_5ago", "r189_5ago"):
        F["p" + k] = F[k].rank(pct=True) * 100
    F["flow"] = F["pr21"] - F["pr189"]
    F["flow_5ago"] = F["pr21_5ago"] - F["pr189_5ago"]
    F = F.sort_values("flow", ascending=False)
    flow = [{"name": k, "n": int(r.n), "rs21": int(round(r.pr21)), "rs189": int(round(r.pr189)),
             "flow": int(round(r.flow)), "flow_5ago": int(round(r.flow_5ago)),
             "chg": _f(r.chg), "r5": _f(r.r5), "r21": _f(r.r21)} for k, r in F.iterrows()]

    # mode (what the studied rules say about today; display only)
    if bear_now:
        mode = {"key": "bear", "label": "弱気相場", "text": "日経平均が52週高値から−20%超。反発サインは過去検証の除外条件"}
    elif bool(panic.iloc[-1]):
        mode = {"key": "panic", "label": "全面安の日", "text": "流動性上位900銘柄の5%以上が同時に25日線から−20%以下"}
    elif wash["active"]:
        mode = {"key": "wash", "label": "反発サイン点灯中", "text": f"サイン{wash['last_signal']}・20日の検証期間はあと{wash['left']}日"}
    else:
        mode = {"key": "normal", "label": "平常", "text": "全面安・反発サインとも出ていない"}

    tail = dates[-252:]
    st = {}
    p = ROOT / "stats.json"
    if p.exists():
        st = json.loads(p.read_text(encoding="utf-8"))
    a_now = _f(a50.iloc[-1], 3)
    return {
        "mode": mode, "bear": bear_now, "n225_from_high": n225_dd,
        "liquid_n": int(nliq.iloc[-1]),
        "a50": a_now, "a50_5ago": _f(a50.iloc[-6], 3), "a50_21ago": _f(a50.iloc[-22], 3), "zone": _zone(a_now),
        "panic_share": _f(share.iloc[-1], 4), "panic_today": bool(panic.iloc[-1]),
        "panic_days_20": int(panic.iloc[-20:].sum()), "panic_days_60": int(panic.iloc[-60:].sum()),
        "panic_recent": recent[-12:], "bnf": bnf, "bnf_count": int(len(hit)),
        "wash": wash, "quiet": quiet, "flow": flow,
        "hist": {"dates": [str(d.date()) for d in tail], "a50": [_f(x, 3) for x in a50.loc[tail]],
                 "panic": [_f(x, 4) for x in share.loc[tail]]},
        "stats": st,
    }
