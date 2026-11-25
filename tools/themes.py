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
    # ---- AI・半導体
    ("AI・半導体", "半導体装置（全体）", ["Semiconductor Equipment & Materials"], r"semiconductor (manufacturing |production )?equipment", []),
    ("AI・半導体", "前工程装置", [], r"(deposition|etching|cleaning|coater|lithography|ion implant|vacuum) (equipment|systems?) for semiconductor|wafer (processing|cleaning|transfer)",
     ["8035", "7735", "6525", "6728", "6323", "6254", "6387"]),
    ("AI・半導体", "後工程・実装装置", [], r"dicing|grinders?|molding equipment|bonders?|back-end (process|equipment)|packaging equipment",
     ["6146", "6315", "7729"]),
    ("AI・半導体", "検査・テスト", [], r"probe cards?|test(ing)? (systems|equipment|handlers?) for semiconductor|semiconductor test|mask inspection|wafer inspection",
     ["6857", "6920", "6871", "6855"]),
    ("AI・半導体", "半導体材料・ウエハ", [], r"photoresist|silicon wafer|semiconductor materials?|polishing (slurr|pad)|high-purity (gas|chemical)|sputtering target",
     ["4063", "3436", "4186", "5384", "4369", "4203"]),
    ("AI・半導体", "パッケージ基板・電子部品", ["Electronic Components"], r"package substrates?|ic substrates?|multilayer ceramic capacitor", ["4062"]),
    ("AI・半導体", "メモリ", [], r"\bnand\b|\bdram\b|flash memory", ["285A"]),
    ("AI・半導体", "パワー半導体", [], r"power semiconductors?|power devices?|\bsic\b|silicon carbide|\bigbt\b", ["6963", "6707", "6504"]),
    ("AI・半導体", "ロジック・アナログ半導体", ["Semiconductors"], r"microcontrollers?|analog (ic|semiconductor)|custom (lsi|chips?)",
     ["6723", "6526", "6875"]),
    ("AIインフラ", "光通信・電線", [], r"optical fib(er|re)|optical (connector|transceiver|component|communication)|electric wires?|power cables?",
     ["5803", "5801", "5802", "5805", "6834", "7713"]),
    ("AIインフラ", "電源・変圧器", ["Electrical Equipment & Parts"], r"power conversion|transformers?|switchgear|substation|uninterruptible",
     ["6504", "6508", "6617", "6622", "6844"]),
    ("AIインフラ", "冷却・空調", [], r"chillers?|liquid cooling|cooling towers?|air[- ]conditioning equipment (manufactur|for buildings)",
     ["6367", "6458", "1969"]),
    ("AIインフラ", "データセンター工事", [], r"electrical (construction|installation|engineering) work|electric(al)? work|(facility|equipment) installation work",
     ["1942", "1944", "1979", "1980", "1959", "1721", "1969"]),
    ("AIインフラ", "GPUクラウド・データセンター", [], r"data cent(er|re)s?|cloud infrastructure|hosting service|\bgpu\b", ["3778", "3905", "4784"]),
    ("AIインフラ", "生成AI・AIソフト", [], r"generative ai|artificial intelligence|\bai\b.{0,40}(solution|platform|model|engine)|machine learning|large language model", []),
    # ---- IT・ネット
    ("IT・ネット", "SaaS・ソフトウェア", ["Software - Application", "Software - Infrastructure"], r"$^", []),
    ("IT・ネット", "サイバーセキュリティ", [], r"cyber ?security|information security|security software|identity (management|authentication)", []),
    ("IT・ネット", "ITサービス・DX", ["Information Technology Services"], r"$^", []),
    ("IT・ネット", "通信", ["Telecom Services", "Communication Equipment"], r"$^", []),
    ("IT・ネット", "ネット・EC・広告", ["Internet Content & Information", "Internet Retail", "Advertising Agencies"], r"$^", []),
    # ---- エンタメ
    ("エンタメ・IP", "ゲーム", ["Electronic Gaming & Multimedia"], r"video games?|game (software|apps?)",
     ["7974", "9766", "9697", "3659", "9684", "6460", "3635", "3765"]),
    ("エンタメ・IP", "アニメ・キャラクターIP", ["Entertainment"], r"\banime\b|\bcharacters?\b.{0,30}(licens|merchandis)|vtubers?|intellectual propert(y|ies)",
     ["7832", "4816", "9602", "8136", "5032", "9468", "7867", "9601"]),
    # ---- 重工・防衛
    ("重工・防衛", "重工（防衛・原子力・航空）", [], r"$^", ["7011", "7012", "7013"]),
    ("重工・防衛", "防衛装備・部品", ["Aerospace & Defense"], r"\bdefen[cs]e\b|ministry of defen[cs]e|missiles?|firearms?|ammunition",
     ["6946", "7721", "6203", "6208", "7224"]),
    ("重工・防衛", "原子力・原発再稼働", [], r"nuclear", ["5631"]),
    ("重工・防衛", "造船・舶用", [], r"shipbuilding|ship ?repair|\bvessels?\b.{0,20}(construct|build)|marine engines?", ["7014", "7003", "7018", "6016"]),
    ("重工・防衛", "宇宙", [], r"satellites?|space (transportation|business|debris|exploration)|lunar|rocket", ["9348", "186A", "464A", "290A", "402A"]),
    ("重工・防衛", "ドローン", [], r"\bdrones?\b|unmanned aerial", ["278A", "6232"]),
    # ---- インフラ・建設
    ("インフラ・建設", "建設・国土強靭化", ["Engineering & Construction", "Infrastructure Operations", "Building Materials"], r"$^", []),
    ("インフラ・建設", "プラント・エンジニアリング", [], r"\bplants?\b.{0,40}(engineering|EPC|construction)|engineering, procurement", ["1963", "6330"]),
    ("インフラ・建設", "電力会社", ["Utilities - Regulated Electric"], r"$^", ["9501", "9503", "9506", "9502", "9504", "9507", "9505", "9509"]),
    ("インフラ・建設", "ガス・再エネ", ["Utilities - Regulated Gas", "Utilities - Renewable", "Utilities - Independent Power Producers", "Utilities - Diversified"], r"$^", []),
    # ---- 機械・ロボット
    ("機械・ロボット", "ロボット部品（フィジカルAI）", [], r"humanoid|robot (reducers?|joints?|components?)|precision reducers?|linear motion", ["6324", "6268", "6481", "6506", "6954"]),
    ("機械・ロボット", "工作機械", ["Tools & Accessories"], r"machine tools?|\bcnc\b", ["6141", "6103", "6135", "6113"]),
    ("機械・ロボット", "FA・センサー", ["Scientific & Technical Instruments"], r"factory automation|pneumatic", ["6861", "6645", "6273"]),
    ("機械・ロボット", "産業機械", ["Specialty Industrial Machinery", "Farm & Heavy Construction Machinery"], r"$^", []),
    # ---- 素材・資源
    ("素材・資源", "非鉄・レアメタル", ["Other Industrial Metals & Mining", "Copper", "Aluminum", "Other Precious Metals & Mining", "Gold", "Silver"], r"$^",
     ["5706", "5711", "5713", "5714"]),
    ("素材・資源", "鉄鋼", ["Steel"], r"$^", []),
    ("素材・資源", "化学", ["Specialty Chemicals", "Chemicals"], r"$^", []),
    ("素材・資源", "石油・エネルギー資源", ["Oil & Gas Integrated", "Oil & Gas E&P", "Oil & Gas Refining & Marketing", "Thermal Coal", "Coking Coal", "Oil & Gas Equipment & Services"], r"$^", []),
    ("素材・資源", "繊維・新素材", ["Textile Manufacturing"], r"carbon fib(er|re)|glass cloth|perovskite", ["3103", "4204"]),
    # ---- 金融
    ("金融", "メガバンク", [], r"$^", ["8306", "8411", "8316", "8308"]),
    ("金融", "地方銀行", ["Banks - Regional"], r"$^", []),
    ("金融", "保険", ["Insurance - Life", "Insurance - Property & Casualty", "Insurance - Diversified", "Insurance Brokers"], r"$^", []),
    ("金融", "証券・取引所", ["Capital Markets", "Financial Data & Stock Exchanges"], r"$^", ["8473", "8628", "8698", "8697"]),
    ("金融", "暗号資産・ビットコイン", [], r"bitcoin|crypto ?(asset|currenc)|blockchain|web3", ["3350"]),
    ("金融", "ノンバンク・リース", ["Credit Services", "Financial Conglomerates", "Mortgage Finance", "Asset Management"], r"$^", []),
    ("商社", "総合商社", [], r"$^", ["8058", "8031", "8001", "8053", "8002", "2768", "8015"]),
    # ---- 内需・消費
    ("内需・消費", "百貨店（インバウンド）", [], r"$^", ["3099", "8233", "3086", "8242", "8237"]),
    ("内需・消費", "ホテル・旅行", ["Lodging", "Travel Services", "Resorts & Casinos"], r"$^", ["9603", "6191", "4681"]),
    ("内需・消費", "外食", ["Restaurants"], r"$^", []),
    ("内需・消費", "小売", ["Specialty Retail", "Discount Stores", "Grocery Stores", "Home Improvement Retail", "Apparel Retail", "Pharmaceutical Retailers"], r"$^", []),
    ("内需・消費", "食品・飲料", ["Packaged Foods", "Confectioners", "Beverages - Non-Alcoholic", "Beverages - Brewers", "Beverages - Wineries & Distilleries", "Food Distribution", "Farm Products"], r"$^", []),
    ("内需・消費", "化粧品・日用品", ["Household & Personal Products"], r"$^", []),
    ("内需・消費", "人材・教育", ["Staffing & Employment Services", "Education & Training Services"], r"$^", []),
    ("内需・消費", "不動産", ["Real Estate Services", "Real Estate - Development", "Real Estate - Diversified", "REIT - Diversified", "Residential Construction"], r"$^", []),
    # ---- ヘルスケア
    ("ヘルスケア", "創薬バイオ", ["Biotechnology"], r"drug discovery|regenerative medicine|\bips cells?\b|antibod(y|ies)", []),
    ("ヘルスケア", "大手医薬品", ["Drug Manufacturers - General", "Drug Manufacturers - Specialty & Generic"], r"$^", []),
    ("ヘルスケア", "医療機器・介護", ["Medical Devices", "Medical Instruments & Supplies", "Diagnostics & Research", "Medical Care Facilities", "Health Information Services", "Medical Distribution"], r"$^", []),
    # ---- モビリティ
    ("モビリティ", "完成車", ["Auto Manufacturers"], r"$^", []),
    ("モビリティ", "自動車部品", ["Auto Parts"], r"$^", []),
    ("モビリティ", "海運", ["Marine Shipping"], r"$^", ["9101", "9104", "9107", "9110", "9119"]),
    ("モビリティ", "陸運・物流", ["Integrated Freight & Logistics", "Trucking"], r"$^", []),
    ("モビリティ", "鉄道", ["Railroads"], r"$^", []),
    ("モビリティ", "空運・空港", ["Airlines", "Airports & Air Services"], r"$^", ["9201", "9202", "9706"]),
]

EXCLUDE = {"地方銀行": {"8306", "8411", "8316", "8308", "7182", "6178", "8309"}}  # megabanks, Japan Post, trust bank
KW_CHARS = 300  # keywords only count in the opening of the business summary (the core business)
# themes that only take members by industry/anchor (keywords would over-match); used to prefer
# specific labels when naming co-moving groups
BROAD_INDUSTRY_ONLY = {"半導体装置（全体）", "パッケージ基板・電子部品", "ITサービス・DX", "ネット・EC・広告", "産業機械",
                       "建設・国土強靭化", "非鉄・レアメタル", "鉄鋼", "化学", "石油・エネルギー資源", "化粧品・日用品",
                       "人材・教育", "医療機器・介護", "ノンバンク・リース", "SaaS・ソフトウェア", "地方銀行", "小売",
                       "食品・飲料", "不動産", "大手医薬品", "完成車", "自動車部品", "陸運・物流", "ガス・再エネ", "通信"}
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
            if code in EXCLUDE.get(fine, ()):
                continue
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
