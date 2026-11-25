#!/usr/bin/env python3
"""Fine themes for Japanese stocks.

Three layers:
  1. 細分類   - Yahoo industry (about 140 industries, finer than the 33 TSE sectors), from $JP_CACHE/info.json
  2. テーマ   - curated themes (rules on the Yahoo industry, keywords in the business summary, and anchor codes)
  3. 連動群   - point-in-time correlation clusters of market-residual daily returns (no hindsight), which
                also catch themes nobody named yet
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("JP_CACHE", Path.home() / "jp-cache"))

# (major, fine theme): industries (exact Yahoo names), summary keywords (regex, case-insensitive), anchor codes
THEMES: list[tuple[str, str, list[str], str, list[str]]] = [
    ("AI・半導体", "半導体製造装置", ["Semiconductor Equipment & Materials"], r"semiconductor (manufacturing |production )?equipment|wafer (processing|cleaning)|dicing|probe card|test(ing)? (systems|equipment) for semiconductor",
     ["8035", "6857", "6146", "6920", "7735", "6525", "6323", "6315", "6254", "7729"]),
    ("AI・半導体", "半導体材料", [], r"photoresist|silicon wafer|semiconductor materials?|polishing (slurr|pad)|high-purity (gas|chemical)|sputtering target",
     ["4063", "3436", "4186", "5384", "4369"]),
    ("AI・半導体", "半導体デバイス・メモリ", ["Semiconductors"], r"\bnand\b|\bdram\b|flash memory|microcontroller|power semiconductor|analog (ic|semiconductor)",
     ["285A", "6723", "6963", "6526"]),
    ("AI・半導体", "電線・光ファイバー", [], r"optical fib(er|re)|electric wires?|power cables?|optical (connector|transceiver|component)",
     ["5803", "5801", "5802", "5805"]),
    ("AI・半導体", "電子部品", ["Electronic Components"], r"multilayer ceramic capacitor|electronic components?",
     []),
    ("AI・半導体", "データセンター・クラウド", [], r"data cent(er|re)s?|cloud infrastructure|hosting service|liquid cooling|\bgpu\b",
     ["3778"]),
    ("AI・半導体", "生成AI・AIソフト", [], r"generative ai|artificial intelligence|\bai\b.{0,40}(solution|platform|model|engine)|machine learning|large language model",
     []),
    ("IT・ネット", "SaaS・ソフトウェア", ["Software - Application", "Software - Infrastructure"], r"\bsaas\b|software as a service|subscription.{0,30}software",
     []),
    ("IT・ネット", "サイバーセキュリティ", [], r"cyber ?security|information security|security software|identity (management|authentication)",
     []),
    ("IT・ネット", "ITサービス・DX", ["Information Technology Services"], r"digital transformation|\bdx\b|system integration",
     []),
    ("IT・ネット", "通信", ["Telecom Services", "Communication Equipment"], r"$^", []),
    ("IT・ネット", "ネット・EC・広告", ["Internet Content & Information", "Internet Retail", "Advertising Agencies"], r"e-commerce|online (shopping|advertising|marketplace)|internet advertising",
     []),
    ("IT・ネット", "ゲーム・アニメ・IP", ["Electronic Gaming & Multimedia", "Entertainment"], r"\banime\b|video games?|game (software|apps?)|\bcharacters?\b.{0,30}(licens|merchandis)|intellectual propert(y|ies)",
     []),
    ("重工・防衛・インフラ", "防衛", ["Aerospace & Defense"], r"\bdefen[cs]e\b|ministry of defen[cs]e|missiles?|firearms?|ammunition",
     ["7011", "7012", "7013", "6203", "7721", "6946", "6208"]),
    ("重工・防衛・インフラ", "原子力・電力設備", [], r"nuclear|power (plant|generation) equipment|transformers?|switchgear|substation|turbines?",
     ["5631"]),
    ("重工・防衛・インフラ", "造船・海洋", [], r"shipbuilding|ship ?repair|\bvessels?\b.{0,20}(construct|build)|marine engines?",
     ["7014", "7003", "7018", "6016"]),
    ("重工・防衛・インフラ", "宇宙", [], r"satellites?|space (transportation|business|debris|exploration)|lunar|rocket",
     ["9348"]),
    ("重工・防衛・インフラ", "設備工事（電気・空調）", [], r"electrical (construction|installation|engineering) work|electric(al)? work|air[- ]conditioning (system|equipment|facilit)|(facility|equipment) installation work",
     ["1942", "1944", "1979", "1980"]),
    ("重工・防衛・インフラ", "電力機器・重電", ["Electrical Equipment & Parts"], r"$^", ["6504", "6508", "6617"]),
    ("重工・防衛・インフラ", "建設・国土強靭化", ["Engineering & Construction", "Infrastructure Operations", "Building Materials"], r"civil engineering|disaster (prevention|recovery)|infrastructure (maintenance|repair)",
     []),
    ("重工・防衛・インフラ", "プラント・エンジニアリング", [], r"\bplants?\b.{0,40}(engineering|EPC|construction)|engineering, procurement",
     ["1963", "6330"]),
    ("機械・ロボット", "工作機械・FA・ロボット", ["Tools & Accessories"], r"machine tools?|industrial robots?|factory automation|\bcnc\b|servo motors?|reducers?",
     ["6954", "6506", "6273", "6268", "6141", "6103"]),
    ("機械・ロボット", "産業機械", ["Specialty Industrial Machinery", "Farm & Heavy Construction Machinery"], r"construction machinery|industrial machinery|compressors?|pumps?",
     []),
    ("素材・資源", "非鉄・レアメタル", ["Other Industrial Metals & Mining", "Copper", "Aluminum", "Other Precious Metals & Mining", "Gold", "Silver"], r"non-ferrous|rare (earth|metal)|copper|smelting|precious metals?",
     ["5706", "5711", "5713", "5714"]),
    ("素材・資源", "鉄鋼", ["Steel"], r"\bsteel\b.{0,20}(products|sheets|making)", []),
    ("素材・資源", "化学", ["Specialty Chemicals", "Chemicals"], r"$^", []),
    ("素材・資源", "石油・エネルギー資源", ["Oil & Gas Integrated", "Oil & Gas E&P", "Oil & Gas Refining & Marketing", "Thermal Coal", "Oil & Gas Equipment & Services"], r"crude oil|natural gas|\blng\b|coal mining",
     []),
    ("素材・資源", "繊維・新素材", ["Textile Manufacturing"], r"carbon fib(er|re)|glass cloth|textiles?|perovskite", ["3103"]),
    ("金融", "銀行", ["Banks - Regional", "Banks - Diversified"], r"$^", []),
    ("金融", "保険", ["Insurance - Life", "Insurance - Property & Casualty", "Insurance - Diversified", "Insurance Brokers"], r"$^", []),
    ("金融", "証券・資産運用", ["Capital Markets", "Asset Management", "Financial Data & Stock Exchanges"], r"$^", []),
    ("金融", "暗号資産・ビットコイン", [], r"bitcoin|crypto ?(asset|currenc)|blockchain|web3", ["3350"]),
    ("金融", "ノンバンク・フィンテック", ["Credit Services", "Financial Conglomerates", "Mortgage Finance"], r"fintech|cashless|payment (service|platform)|settlement service",
     []),
    ("商社", "総合商社", [], r"$^", ["8058", "8031", "8001", "8053", "8002", "2768", "8015"]),
    ("内需・消費", "インバウンド・ホテル・旅行", ["Lodging", "Travel Services", "Resorts & Casinos"], r"\bhotels?\b|inbound|duty[- ]free|tourists?", []),
    ("内需・消費", "外食", ["Restaurants"], r"$^", []),
    ("内需・消費", "小売", ["Specialty Retail", "Discount Stores", "Department Stores", "Grocery Stores", "Home Improvement Retail", "Apparel Retail", "Pharmaceutical Retailers"], r"$^", []),
    ("内需・消費", "食品・飲料", ["Packaged Foods", "Confectioners", "Beverages - Non-Alcoholic", "Beverages - Brewers", "Beverages - Wineries & Distilleries", "Food Distribution", "Farm Products"], r"$^", []),
    ("内需・消費", "化粧品・日用品", ["Household & Personal Products"], r"cosmetics?|skin ?care", []),
    ("内需・消費", "人材・教育", ["Staffing & Employment Services", "Education & Training Services"], r"staffing|recruit(ment|ing)|job (matching|placement)|e-learning", []),
    ("ヘルスケア", "創薬バイオ", ["Biotechnology"], r"drug discovery|regenerative medicine|\bips cells?\b|antibod(y|ies)|clinical trials?", []),
    ("ヘルスケア", "医薬品", ["Drug Manufacturers - General", "Drug Manufacturers - Specialty & Generic"], r"$^", []),
    ("ヘルスケア", "医療機器・介護", ["Medical Devices", "Medical Instruments & Supplies", "Diagnostics & Research", "Medical Care Facilities", "Health Information Services", "Medical Distribution"], r"nursing care|medical devices?", []),
    ("モビリティ", "自動車・部品", ["Auto Manufacturers", "Auto Parts", "Auto & Truck Dealerships", "Recreational Vehicles"], r"$^", []),
    ("モビリティ", "海運・物流", ["Marine Shipping", "Integrated Freight & Logistics", "Trucking"], r"$^", []),
    ("モビリティ", "鉄道・空運", ["Railroads", "Airlines", "Airports & Air Services"], r"$^", []),
    ("不動産", "不動産", ["Real Estate Services", "Real Estate - Development", "Real Estate - Diversified", "REIT - Diversified"], r"$^", []),
    ("電力・ガス", "電力・ガス", ["Utilities - Regulated Electric", "Utilities - Regulated Gas", "Utilities - Renewable", "Utilities - Independent Power Producers", "Utilities - Diversified"], r"renewable energy|solar power|wind power", []),
]

KW_CHARS = 300  # keywords only count in the opening of the business summary (the core business)
BROAD_INDUSTRY_ONLY = {"電子部品", "ITサービス・DX", "ネット・EC・広告", "産業機械", "建設・国土強靭化", "非鉄・レアメタル",
                       "鉄鋼", "石油・エネルギー資源", "化粧品・日用品", "人材・教育",
                       "医療機器・介護", "電力・ガス", "ノンバンク・フィンテック", "SaaS・ソフトウェア"}
INDUSTRY_JA = {}  # optional translation table for display; falls back to the English name


def load_info() -> dict:
    p = CACHE / "info.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def theme_membership(tickers: list[str], info: dict) -> dict[str, list[str]]:
    """ticker -> list of fine themes (a stock can sit in several)."""
    out: dict[str, list[str]] = {}
    pats = [(fine, set(ind), re.compile("$^" if fine in BROAD_INDUSTRY_ONLY else kw, re.I), set(anc))
            for _, fine, ind, kw, anc in THEMES]
    for t in tickers:
        code = t.replace(".T", "")
        rec = info.get(t) or {}
        if rec.get("themes") is not None and not rec.get("summary"):
            if rec["themes"]:
                out[t] = list(rec["themes"])
            continue
        ind = rec.get("industry") or ""
        summ = rec.get("summary") or ""
        hits = []
        for fine, inds, kw, anc in pats:
            if code in anc or (ind and ind in inds) or (summ and kw.pattern != "$^" and kw.search(summ[:KW_CHARS])):
                hits.append(fine)
        if hits:
            out[t] = hits
    return out


def major_of() -> dict[str, str]:
    return {fine: major for major, fine, *_ in THEMES}


def corr_clusters(ret: pd.DataFrame, mret: pd.Series, members: list[str], window=120, max_dist=0.75,
                  min_size=4, max_size=30) -> dict[str, int]:
    """Cluster market-residual returns over the last ``window`` sessions (average linkage on 1-corr)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    r = ret[members].iloc[-window:]
    m = mret.iloc[-window:]
    ok = r.notna().sum() >= int(window * 0.9)
    r = r.loc[:, ok].fillna(0.0)
    if r.shape[1] < 10:
        return {}
    beta = r.apply(lambda s: np.cov(s, m.fillna(0))[0, 1] / max(m.var(), 1e-12))
    resid = r - np.outer(m.fillna(0).to_numpy(), beta.to_numpy())
    cm = np.corrcoef(resid.to_numpy().T)
    cm = np.nan_to_num(cm, nan=0.0)
    d = 1 - cm
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(Z, t=max_dist, criterion="distance")
    s = pd.Series(lab, index=r.columns)
    # split oversized clusters (a broad "large caps" or "financials" blob) with tighter cuts
    nxt = int(s.max()) + 1
    for cut in (0.65, 0.55, 0.45):
        for k, sz in s.value_counts().items():
            if sz <= max_size:
                continue
            idx = np.flatnonzero((s == k).to_numpy())
            if len(idx) < 2:
                continue
            sub = linkage(squareform(d[np.ix_(idx, idx)], checks=False), method="average")
            sl = fcluster(sub, t=cut, criterion="distance")
            s.iloc[idx] = sl + nxt
            nxt += int(sl.max()) + 1
    sizes = s.value_counts()
    keep = sizes[sizes >= min_size].index
    return {t: int(k) for t, k in s.items() if k in keep}


def export_map(path: Path) -> None:
    """Write ticker -> {industry, themes} (no Yahoo text) for the committed map used by the daily update."""
    info = {}
    for q in (CACHE / "info.json", CACHE / "info2.json"):
        if q.exists():
            info.update(json.loads(q.read_text(encoding="utf-8")))
    tm = theme_membership(list(info), info)
    out = {t: {"industry": (info[t] or {}).get("industry"), "themes": tm.get(t, [])} for t in sorted(info)}
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    print("theme_map:", len(out), "tickers,", sum(1 for v in out.values() if v["themes"]), "with a theme")


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        export_map(Path(__file__).resolve().parent.parent / "theme_map.json")
