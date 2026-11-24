#!/usr/bin/env python3
"""Local verification of the swing core rule on Japanese stocks (never run on GitHub).

Data: $JP_CACHE/hist10y.parquet (yfinance, split/dividend adjusted, current listings only ->
survivorship bias: delisted names are missing).

Rules (US site, "個別株だけ" test, ported):
  signal at the close -> buy at the next open, max 6 names, 1/6 of equity each,
  -8% stop from the first fill (intraday; a gap below fills at the open),
  close >= +10% / +20% from the first fill -> add the same amount at the next open (one name <= 40%),
  close under the 21-day EMA of lows -> sell at the next open,
  close 15% under the highest close since entry -> sell at the next open.
  Shares are bought in 100-share lots on a ¥10,000,000 start; 0.1% slippage per side, no commission.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build import TIER_ORDER, buyable, structure_pivot, tier, weekly_sar_state  # noqa: E402

CACHE = Path(os.environ.get("JP_CACHE", Path.home() / "jp-cache"))
REGIME_SYM = "1306.T"
START_CAPITAL, LOT, SLIP = 10_000_000, 100, 0.001
MAX_NAMES, INITIAL, STOP, CAP_NAME, TRAIL = 6, 1 / 6, 0.08, 0.40, 0.15


def load():
    df = pd.read_parquet(CACHE / "hist10y.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    P = {k: df.pivot(index="date", columns="ticker", values=k).sort_index() for k in
         ("open", "high", "low", "close", "volume")}
    cl = P["close"]
    med = cl.rolling(11, center=True, min_periods=5).median()
    bad = (cl / med > 2.5) | (cl / med < 0.4)
    P = {k: v.mask(bad) for k, v in P.items()}
    # bars with a zero/NaN open (yfinance sometimes) -> use the close as the open
    P["open"] = P["open"].where(P["open"] > 0, P["close"])
    return P


def signals(P, min_price=200.0, min_dv=5e8, dv_pct=95, rs_pct=90, udvr_max=1.4):
    uni = pd.read_csv(CACHE / "universe.csv", dtype=str)
    stocks = [t for t in P["close"].columns if t in set(uni["ticker"])]
    c, h, l, v = (P[k][stocks] for k in ("close", "high", "low", "volume"))
    ma50, ma150, ma200 = (c.rolling(n, min_periods=n).mean() for n in (50, 150, 200))
    hi252 = h.rolling(252, min_periods=200).max()
    dv50 = (c * v).rolling(50, min_periods=40).mean()
    liquid = (c >= min_price) & (dv50 >= min_dv)
    tt = (c > ma50) & (ma50 > ma150) & (ma150 > ma200) & (ma200 > ma200.shift(20)) & (c >= 0.75 * hi252)
    dvp = dv50.where(liquid).rank(axis=1, pct=True) * 100
    r189 = c / c.shift(189) - 1
    rsp = r189.where(liquid).rank(axis=1, pct=True) * 100
    selected = liquid & tt & (dvp >= dv_pct) & (rsp >= rs_pct)
    dr = h / l - 1
    vc = dr.rolling(10).mean() / dr.rolling(50).mean()
    vdry = v.rolling(5).mean() / v.rolling(50).mean()
    chg = (c / c.shift(1) - 1).round(9)
    timing = (vc <= 0.9) & (vdry <= 0.9) & (chg < 0.03) & (chg.shift(1) <= 0.03)
    up_v = v.where(c > c.shift(1), 0.0).rolling(50).mean()
    dn_v = v.where(c < c.shift(1), 0.0).rolling(50).mean()
    udvr = up_v / dn_v
    ud_ok = ~(udvr >= udvr_max)
    el21 = l.ewm(span=21, adjust=False).mean()
    return {"stocks": stocks, "selected": selected.fillna(False), "timing": timing.fillna(False),
            "ud_ok": ud_ok, "rsp": rsp, "el21": el21}


_ev_cache: dict = {}


def event_state(P, t, i):
    key = (t, i)
    if key in _ev_cache:
        return _ev_cache[key]
    h, l, c = P["high"][t].iloc[:i + 1], P["low"][t].iloc[:i + 1], P["close"][t].iloc[:i + 1]
    px = float(c.iloc[-1])
    line, hl = structure_pivot(h.to_numpy(float)[-260:], l.to_numpy(float)[-260:])
    inside = bool(not math.isnan(line) and px <= line)
    pos = (px - hl) / (line - hl) if inside and line > hl else None
    try:
        up, age = weekly_sar_state(h.dropna(), l.dropna(), c.dropna())
    except Exception:  # noqa: BLE001
        up, age = None, None
    r = {"inside": inside, "pos": pos, "sar_up": up, "sar_age": age,
         "pivot_line": None if math.isnan(line) else line, "hl": None if math.isnan(hl) else hl}
    r["tier"], r["buyable"] = tier(r), buyable(r)
    _ev_cache[key] = r
    return r


def run(P, S, start, *, use_regime=True, use_ud=True, use_buyable=True, stock_cap=None, max_names=MAX_NAMES,
        initial=INITIAL):
    dates = P["close"].index
    o, h, l, c = (P[k] for k in ("open", "high", "low", "close"))
    ref = P["close"][REGIME_SYM]
    regime = (ref > ref.rolling(200).mean()).reindex(dates).fillna(False)
    sig = S["selected"] & S["timing"] & (S["ud_ok"].fillna(True) if use_ud else True)
    el21 = S["el21"]
    i0 = dates.searchsorted(pd.Timestamp(start))
    cash, pos = float(START_CAPITAL), {}
    pending_buy, pending_add, pending_sell = [], set(), set()
    eq_curve, trades = [], []
    last_px: dict = {}

    def mark(t, i):
        x = c[t].iat[i]
        if not np.isnan(x):
            last_px[t] = float(x)
        return last_px.get(t, pos[t]["entry"])

    for i in range(i0, len(dates)):
        # ---- equity at the previous close (sizing base)
        equity_prev = cash + sum(p["sh"] * last_px.get(t, p["entry"]) for t, p in pos.items())
        # ---- opens: sells, adds, buys
        for t in list(pending_sell):
            if t in pos and not np.isnan(o[t].iat[i]):
                px = o[t].iat[i] * (1 - SLIP)
                p = pos.pop(t)
                cash += p["sh"] * px
                trades.append({"t": t, "in": str(dates[p["i"]].date()), "out": str(dates[i].date()),
                               "ret": px / p["avg"] - 1, "why": p.get("why", "")})
                pending_sell.discard(t)
        for t in list(pending_add):
            pending_add.discard(t)
            if t not in pos or np.isnan(o[t].iat[i]):
                continue
            p = pos[t]
            px = o[t].iat[i] * (1 + SLIP)
            name_val = p["sh"] * px
            budget = min(equity_prev * initial, equity_prev * CAP_NAME - name_val, cash)
            if stock_cap is not None:
                stock_val = sum(q["sh"] * last_px.get(u, q["entry"]) for u, q in pos.items())
                budget = min(budget, equity_prev * stock_cap - stock_val)
            sh = int(budget // (px * LOT)) * LOT
            if sh > 0:
                cash -= sh * px
                p["avg"] = (p["avg"] * p["sh"] + px * sh) / (p["sh"] + sh)
                p["sh"] += sh
        for t in pending_buy:
            if t in pos or len(pos) >= max_names or np.isnan(o[t].iat[i]):
                continue
            px = o[t].iat[i] * (1 + SLIP)
            budget = min(equity_prev * initial, cash)
            if stock_cap is not None:
                stock_val = sum(q["sh"] * last_px.get(u, q["entry"]) for u, q in pos.items())
                budget = min(budget, equity_prev * stock_cap - stock_val)
            sh = int(budget // (px * LOT)) * LOT
            if sh <= 0:
                continue
            cash -= sh * px
            pos[t] = {"sh": sh, "entry": px, "avg": px, "i": i, "adds": 0, "peak": px, "stop": px * (1 - STOP)}
            last_px[t] = px
        pending_buy = []
        # ---- intraday stop
        for t in list(pos):
            p = pos[t]
            lo, op = l[t].iat[i], o[t].iat[i]
            if not np.isnan(lo) and lo <= p["stop"] and t not in pending_sell:
                px = (min(op, p["stop"]) if not np.isnan(op) else p["stop"]) * (1 - SLIP)
                pos.pop(t)
                cash += p["sh"] * px
                trades.append({"t": t, "in": str(dates[p["i"]].date()), "out": str(dates[i].date()),
                               "ret": px / p["avg"] - 1, "why": "stop"})
        # ---- closes: exits / adds for tomorrow
        for t, p in pos.items():
            x = c[t].iat[i]
            if np.isnan(x):
                continue
            p["peak"] = max(p["peak"], x)
            if x < el21[t].iat[i]:
                pending_sell.add(t)
                p["why"] = "21EMA"
            elif x < p["peak"] * (1 - TRAIL):
                pending_sell.add(t)
                p["why"] = "trail"
            elif p["adds"] == 0 and x >= p["entry"] * 1.10:
                pending_add.add(t)
                p["adds"] = 1
            elif p["adds"] == 1 and x >= p["entry"] * 1.20:
                pending_add.add(t)
                p["adds"] = 2
        # ---- new signals for tomorrow's open
        if (not use_regime or regime.iat[i]) and len(pos) < max_names:
            row = sig.iloc[i]
            cands = []
            for t in row[row].index:
                if t in pos:
                    continue
                if use_buyable:
                    st = event_state(P, t, i)
                    if not st["buyable"]:
                        continue
                    rank = TIER_ORDER[st["tier"]]
                else:
                    rank = 0
                cands.append((rank, -(S["rsp"][t].iat[i] or 0), t))
            pending_buy = [t for _, _, t in sorted(cands)][: max_names - len(pos)]
        eq = cash + sum(p["sh"] * mark(t, i) for t, p in pos.items())
        eq_curve.append((dates[i], eq, len(pos)))
    curve = pd.DataFrame(eq_curve, columns=["date", "eq", "n"]).set_index("date")
    return curve, pd.DataFrame(trades)


def stats(curve, trades=None):
    eq = curve["eq"]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    out = {"cagr": round(cagr * 100, 1), "dd": round(dd * 100, 1),
           "years": {str(d.year): round(r * 100, 1) for d, r in yearly.items()}}
    if "n" in curve:
        out["exposure"] = round((curve["n"] > 0).mean() * 100, 1)
    if trades is not None and len(trades):
        out.update({"trades": int(len(trades)), "win": round((trades["ret"] > 0).mean() * 100, 1),
                    "avg_win": round(trades.loc[trades["ret"] > 0, "ret"].mean() * 100, 1),
                    "avg_loss": round(trades.loc[trades["ret"] <= 0, "ret"].mean() * 100, 1)})
    return out


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2017-06-01"
    P = load()
    S = signals(P)
    bench = P["close"][REGIME_SYM].loc[start:].dropna()
    res = {"period": f"{start}〜{P['close'].index[-1].date()}",
           "TOPIX連動ETF(1306)保有": stats(pd.DataFrame({"eq": bench}))}
    variants = {
        "本命（地合い＋UD比＋週足SAR/HL）": {},
        "地合いなし": {"use_regime": False},
        "UD比なし": {"use_ud": False},
        "全条件OKを全部（SAR/HLで絞らない）": {"use_buyable": False},
        "株の上限80%": {"stock_cap": 0.80},
    }
    for name, kw in variants.items():
        curve, trades = run(P, S, start, **kw)
        res[name] = stats(curve, trades)
        if not kw:
            trades.to_csv(CACHE / "bt_trades_base.csv", index=False)
            curve.to_csv(CACHE / "bt_curve_base.csv")
        print(name, res[name], flush=True)
    for name, kw in {"流動性 ¥3億": {"min_dv": 3e8}, "流動性 ¥10億": {"min_dv": 1e9},
                     "流動性 ¥20億": {"min_dv": 2e9}}.items():
        _ev_cache.clear()
        curve, trades = run(P, signals(P, **kw), start)
        res[name] = stats(curve, trades)
        print(name, res[name], flush=True)
    (CACHE / "bt_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
