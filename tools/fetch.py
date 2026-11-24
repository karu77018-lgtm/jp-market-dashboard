#!/usr/bin/env python3
"""Fetch daily OHLCV for all TSE domestic stocks + reference indices (runs locally, never on GitHub).

Output: cache/ohlcv.parquet (long format) and cache/universe.csv.
Gentle on Yahoo: chunks of 80 tickers, a pause between chunks, retry once on failure.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(os.environ.get("JP_CACHE", Path.home() / "jp-cache"))
MARKETS = {"プライム（内国株式）": "P", "スタンダード（内国株式）": "S", "グロース（内国株式）": "G"}
REFS = {"1306.T": "TOPIX(1306)", "^N225": "日経平均", "2516.T": "グロース250(2516)", "1321.T": "日経225(1321)"}
CHUNK, PAUSE, PERIOD = 80, 2.0, "2y"


def universe() -> pd.DataFrame:
    d = pd.read_excel(CACHE / "data_j.xlsx", dtype=str)
    d = d[d["市場・商品区分"].isin(MARKETS)].copy()
    d["market"] = d["市場・商品区分"].map(MARKETS)
    d["ticker"] = d["コード"].str.strip() + ".T"
    d = d.rename(columns={"銘柄名": "name", "33業種区分": "s33", "17業種区分": "s17", "規模区分": "size"})
    return d[["ticker", "name", "market", "s33", "s17", "size"]].reset_index(drop=True)


def grab(tickers: list[str]) -> pd.DataFrame:
    for attempt in range(2):
        try:
            raw = yf.download(tickers, period=PERIOD, interval="1d", auto_adjust=True, progress=False,
                              group_by="ticker", threads=True)
            break
        except Exception as e:  # noqa: BLE001
            print("retry", e, file=sys.stderr)
            time.sleep(15)
    else:
        return pd.DataFrame()
    rows = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        g = raw[t].dropna(subset=["Close"]).copy()
        if g.empty:
            continue
        g.columns = [c.lower() for c in g.columns]
        g["ticker"] = t
        g["date"] = g.index.tz_localize(None) if g.index.tz is not None else g.index
        rows.append(g[["ticker", "date", "open", "high", "low", "close", "volume"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def topup() -> None:
    """Re-fetch the last few sessions for tickers whose latest bar lags the newest date."""
    df = pd.read_parquet(CACHE / "ohlcv.parquet")
    last = df.groupby("ticker")["date"].max()
    newest = last.max()
    lag = list(last[last < newest].index)
    print(f"topup: {len(lag)} tickers behind {newest.date()}", flush=True)
    parts = []
    for i in range(0, len(lag), CHUNK):
        parts.append(grab_period(lag[i:i + CHUNK], "1mo"))
        print(f"{min(i + CHUNK, len(lag))}/{len(lag)}", flush=True)
        time.sleep(PAUSE)
    if parts:
        df = pd.concat([df] + parts, ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
        df.to_parquet(CACHE / "ohlcv.parquet", index=False)
    print("after:", df.groupby("date")["ticker"].count().tail(3).to_dict())


def grab_period(tickers, period):
    global PERIOD
    old, PERIOD = PERIOD, period
    try:
        return grab(tickers)
    finally:
        PERIOD = old


def main() -> None:
    global PERIOD
    if "--topup" in sys.argv:
        return topup()
    out_name = "ohlcv.parquet"
    if "--history" in sys.argv:  # long history for local verification only
        PERIOD, out_name = "10y", "hist10y.parquet"
    uni = universe()
    uni.to_csv(CACHE / "universe.csv", index=False)
    tickers = list(REFS) + uni["ticker"].tolist()
    parts = []
    for i in range(0, len(tickers), CHUNK):
        part = grab(tickers[i:i + CHUNK])
        parts.append(part)
        print(f"{min(i + CHUNK, len(tickers))}/{len(tickers)} rows={len(part)}", flush=True)
        time.sleep(PAUSE)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CACHE / out_name, index=False)
    got = df["ticker"].nunique()
    print(f"done: {got} tickers, last date {df['date'].max().date()}")


if __name__ == "__main__":
    main()
