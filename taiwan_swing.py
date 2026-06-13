# -*- coding: utf-8 -*-
"""
台股波段操作篩選 + 進出場訊號工具
================================================
一鍵執行:每天按一次按鈕,程式會
  1) 從股票池中篩選出最多 10 檔「符合波段條件」的標的
  2) 對每檔給出「可考慮進場 / 觀望」的判斷與理由
  3) 對你手上的持股,給出「續抱 / 減碼 / 出場」與停損停利建議

重要聲明:
  本工具依「歷史價量」的常見技術規律計算訊號,屬於「決策輔助」。
  它不是預測、不保證獲利,也不構成投資建議。實際下單請自行判斷並控制風險。

執行方式:
  介面版(推薦,有按鈕):  streamlit run taiwan_swing.py
  純文字版(無介面):       python taiwan_swing.py
"""

import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. 股票池(universe)
#    預設放流動性較好的台股大型 / 中型權值股,可自行增刪。
#    上市加 .TW,上櫃加 .TWO。波段操作建議只挑「成交量夠大」的股票。
# ----------------------------------------------------------------------
DEFAULT_UNIVERSE = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2303.TW",
    "3711.TW", "2412.TW", "2891.TW", "2881.TW", "2882.TW", "2886.TW",
    "2884.TW", "2885.TW", "2892.TW", "1216.TW", "1301.TW", "1303.TW",
    "1326.TW", "2002.TW", "2207.TW", "2105.TW", "2603.TW", "2609.TW",
    "2615.TW", "2618.TW", "2610.TW", "3008.TW", "3034.TW", "3045.TW",
    "3231.TW", "2357.TW", "2376.TW", "2377.TW", "2379.TW", "2395.TW",
    "4938.TW", "6505.TW", "9910.TW", "9904.TW", "1101.TW", "1102.TW",
    "2912.TW", "2801.TW", "5880.TW", "2887.TW", "2880.TW", "2883.TW",
    "3661.TW", "3017.TW", "3037.TW", "2345.TW", "3443.TW", "6415.TW",
    "8046.TW", "2360.TW", "2356.TW", "2474.TW", "4904.TW", "3481.TW",
    "2409.TW", "6669.TW", "3035.TW", "3406.TW", "1605.TW", "2027.TW",
    "1513.TW", "2371.TW", "2049.TW", "1519.TW",
]

# 股票代號 -> 中文名稱。自己若在 DEFAULT_UNIVERSE 加了新股票,記得這裡也補上。
NAME_MAP = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2382.TW": "廣達", "2303.TW": "聯電", "3711.TW": "日月光投控", "2412.TW": "中華電",
    "2891.TW": "中信金", "2881.TW": "富邦金", "2882.TW": "國泰金", "2886.TW": "兆豐金",
    "2884.TW": "玉山金", "2885.TW": "元大金", "2892.TW": "第一金", "1216.TW": "統一",
    "1301.TW": "台塑", "1303.TW": "南亞", "1326.TW": "台化", "2002.TW": "中鋼",
    "2207.TW": "和泰車", "2105.TW": "正新", "2603.TW": "長榮", "2609.TW": "陽明",
    "2615.TW": "萬海", "2618.TW": "長榮航", "2610.TW": "華航", "3008.TW": "大立光",
    "3034.TW": "聯詠", "3045.TW": "台灣大", "3231.TW": "緯創", "2357.TW": "華碩",
    "2376.TW": "技嘉", "2377.TW": "微星", "2379.TW": "瑞昱", "2395.TW": "研華",
    "4938.TW": "和碩", "6505.TW": "台塑化", "9910.TW": "豐泰", "9904.TW": "寶成",
    "1101.TW": "台泥", "1102.TW": "亞泥", "2912.TW": "統一超", "2801.TW": "彰銀",
    "5880.TW": "合庫金", "2887.TW": "台新金", "2880.TW": "華南金", "2883.TW": "開發金",
    "3661.TW": "世芯-KY", "3017.TW": "奇鋐", "3037.TW": "欣興", "2345.TW": "智邦",
    "3443.TW": "創意", "6415.TW": "矽力-KY", "8046.TW": "南電", "2360.TW": "致茂",
    "2356.TW": "英業達", "2474.TW": "可成", "4904.TW": "遠傳", "3481.TW": "群創",
    "2409.TW": "友達", "6669.TW": "緯穎", "3035.TW": "智原", "3406.TW": "玉晶光",
    "1605.TW": "華新", "2027.TW": "大成鋼", "1513.TW": "中興電", "2371.TW": "大同",
    "2049.TW": "上銀", "1519.TW": "華城",
}

BENCHMARK = "^TWII"   # 加權指數,用來算相對強弱

# 表格欄位說明(顯示在介面上)
COLUMN_HELP = """
- **代號**:股票代號(`.TW` 為上市、`.TWO` 為上櫃)。
- **股票名稱**:公司中文名稱。
- **收盤**:最近一個交易日的收盤價。
- **分數**:綜合評分(滿分約 100),分數越高代表越符合「波段做多」的條件。
  計分方式如下——
  加分項:均線多頭排列 5>20>60(+25)、RSI 落在 40–65 健康區(+15)、
  MACD 動能翻揚(+20)、股價貼近 20 日線(回檔不追高)(+15)、
  量能放大(+10)、相對大盤強勢(+15)。
  扣分項:RSI 過熱 >75(−10)、距 20 日線乖離過大 >12%(−10)。
- **進場訊號**:✅ 表示此刻同時符合「回檔到 20 日線附近 + RSI 未轉弱 +
  MACD 翻揚 + 當日不再續弱」的進場時機。空白代表分數雖高,但目前不是好的進場點,可先觀望等回檔。
- **RSI**:相對強弱指標(0–100),衡量近期漲跌動能。>70 偏過熱、<30 偏超賣,波段做多偏好 40–65。
- **距20MA%**:收盤價距離 20 日均線的百分比。正值=在均線上方;數值接近 0 表示剛回檔到均線附近,通常是較佳的布局位置。
- **量比**:當日成交量 ÷ 20 日均量。>1 代表放量,>1.2 通常代表有資金進場。
- **建議停損**:跌破此價建議出場,以最近 20 日低點與 4–8% 區間估算。
- **目標價**:以約 1:2 風險報酬比估算的參考停利價(進場價 + 2 倍風險距離)。
- **理由**:該股入選與得分的主要原因摘要。
"""


# ----------------------------------------------------------------------
# 2. 技術指標(只用 pandas/numpy,不依賴 TA-Lib)
# ----------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df 需要欄位: Close, Volume。回傳加上各指標的 df。"""
    df = df.copy()
    c = df["Close"]
    df["MA5"] = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RSI14"] = rsi(c, 14)
    _, _, df["MACD_HIST"] = macd(c)
    return df


# ----------------------------------------------------------------------
# 3. 對單一股票做分析:篩選分數 + 進場/持有訊號
# ----------------------------------------------------------------------
def analyze_stock(df: pd.DataFrame, bench_ret60: float | None = None) -> dict | None:
    """
    回傳該股的分析結果。資料不足回傳 None。
    bench_ret60: 大盤近 60 日報酬,用來算相對強弱(可為 None)。
    """
    if df is None or len(df) < 65:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    ma5, ma20, ma60 = float(last["MA5"]), float(last["MA20"]), float(last["MA60"])
    rsi_now = float(last["RSI14"])
    hist_now, hist_prev = float(last["MACD_HIST"]), float(prev["MACD_HIST"])
    vol, vol20 = float(last["Volume"]), float(last["VOL20"])

    if any(np.isnan(x) for x in (ma5, ma20, ma60, rsi_now, hist_now, vol20)):
        return None

    turnover = close * vol20  # 以均量估算每日成交金額(粗略流動性)
    ret60 = close / float(df["Close"].iloc[-61]) - 1 if len(df) >= 61 else 0.0
    rel_strength = (ret60 - bench_ret60) if bench_ret60 is not None else None
    dist_ma20 = (close - ma20) / ma20  # 距 20 日線多遠(正=在上方)

    # --- 硬性條件(全部要過,才進入排序) ---
    hard_ok = (
        close > ma60                 # 中期多頭:站上季線
        and ma20 > ma60              # 均線多頭排列(中期)
        and turnover > 50_000_000    # 流動性:估算日成交額 > 5000 萬,過濾冷門股
        and close > 10               # 過濾雞蛋水餃股
    )

    # --- 評分(在通過硬性條件的股票之間排名,分數越高越優先) ---
    score = 0.0
    reasons = []
    if ma5 > ma20 > ma60:
        score += 25; reasons.append("均線多頭排列(5>20>60)")
    if 40 <= rsi_now <= 65:
        score += 15; reasons.append(f"RSI 健康({rsi_now:.0f})")
    elif rsi_now > 75:
        score -= 10; reasons.append(f"RSI 偏高過熱({rsi_now:.0f})")
    if hist_now > 0 and hist_now > hist_prev:
        score += 20; reasons.append("MACD 動能轉強")
    if -0.05 <= dist_ma20 <= 0.04:
        score += 15; reasons.append("貼近 20 日線(回檔不追高)")
    elif dist_ma20 > 0.12:
        score -= 10; reasons.append("離 20 日線太遠(乖離過大)")
    if vol > vol20 * 1.2:
        score += 10; reasons.append("量能放大")
    if rel_strength is not None and rel_strength > 0:
        score += 15; reasons.append("相對大盤強勢")

    # --- 進場訊號:多頭中的「回檔不破、動能轉強」 ---
    entry_signal = (
        hard_ok
        and (-0.06 <= dist_ma20 <= 0.05)     # 拉回到 20 線附近
        and rsi_now >= 45                    # 動能未轉弱
        and hist_now > hist_prev             # MACD 柱狀體翻揚
        and close >= float(prev["Close"])    # 當日不續弱
    )

    # 建議停損 / 停利(以最近 20 日低點與固定 R 計算)
    swing_low = float(df["Close"].iloc[-20:].min())
    stop_pct = min(0.08, max(0.04, (close - swing_low) / close))  # 4%~8% 之間
    stop_price = round(close * (1 - stop_pct), 2)
    risk = close - stop_price
    target_price = round(close + 2 * risk, 2)  # 風險報酬比約 1:2

    return {
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "rsi": round(rsi_now, 1),
        "macd_hist": round(hist_now, 3),
        "vol_ratio": round(vol / vol20, 2) if vol20 else None,
        "turnover_est": int(turnover),
        "rel_strength": round(rel_strength, 3) if rel_strength is not None else None,
        "dist_ma20": round(dist_ma20, 3),
        "hard_ok": bool(hard_ok),
        "score": round(score, 1),
        "reasons": reasons,
        "entry_signal": bool(entry_signal),
        "stop_price": stop_price,
        "target_price": target_price,
    }


def holding_action(df: pd.DataFrame, entry_price: float) -> dict | None:
    """對持股給出 續抱 / 減碼 / 出場 建議。"""
    if df is None or len(df) < 25:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    close = float(last["Close"])
    ma20 = float(last["MA20"])
    rsi_now = float(last["RSI14"])
    pnl = close / entry_price - 1

    trailing_stop = round(ma20, 2)  # 以 20 日線當移動停利 / 停損
    action, reason = "續抱", []

    if close < ma20:
        action = "出場"
        reason.append("跌破 20 日線(波段轉弱)")
    elif pnl <= -0.08:
        action = "出場"
        reason.append(f"虧損達 {pnl*100:.1f}%,觸及停損紀律")
    elif rsi_now > 80:
        action = "減碼"
        reason.append(f"RSI 過熱({rsi_now:.0f}),可先獲利了結部分")
    else:
        reason.append(f"仍在 20 日線之上,移動停利設於 {trailing_stop}")

    return {
        "close": round(close, 2),
        "entry_price": entry_price,
        "pnl_pct": round(pnl * 100, 2),
        "rsi": round(rsi_now, 1),
        "trailing_stop": trailing_stop,
        "action": action,
        "reason": "；".join(reason),
    }


# ----------------------------------------------------------------------
# 4. 抓資料(yfinance)
# ----------------------------------------------------------------------
def fetch_history(tickers, period="9mo"):
    """回傳 {ticker: DataFrame(Close, Volume)}。"""
    import yfinance as yf
    data = {}
    raw = yf.download(tickers, period=period, interval="1d",
                      group_by="ticker", auto_adjust=True,
                      threads=True, progress=False)
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub[["Close", "Volume"]].dropna()
            if len(sub) >= 65:
                data[t] = sub
        except Exception:
            continue
    return data


def run_screen(universe=None, period="9mo"):
    """主流程:回傳 (候選清單 DataFrame, 全部分析 dict)。"""
    universe = universe or DEFAULT_UNIVERSE
    hist = fetch_history(universe + [BENCHMARK], period=period)

    bench_ret60 = None
    if BENCHMARK in hist and len(hist[BENCHMARK]) >= 61:
        b = hist[BENCHMARK]["Close"]
        bench_ret60 = float(b.iloc[-1] / b.iloc[-61] - 1)

    rows, full = [], {}
    for t in universe:
        if t not in hist:
            continue
        res = analyze_stock(hist[t], bench_ret60)
        if res is None:
            continue
        full[t] = res
        if res["hard_ok"]:
            rows.append({
                "代號": t, "股票名稱": NAME_MAP.get(t, ""),
                "收盤": res["close"], "分數": res["score"],
                "進場訊號": "✅" if res["entry_signal"] else "",
                "RSI": res["rsi"], "距20MA%": round(res["dist_ma20"] * 100, 1),
                "量比": res["vol_ratio"], "建議停損": res["stop_price"],
                "目標價": res["target_price"], "理由": "、".join(res["reasons"]),
            })
    cand = pd.DataFrame(rows).sort_values("分數", ascending=False).head(10) \
        if rows else pd.DataFrame()
    return cand, full


# ----------------------------------------------------------------------
# 5a. 純文字版(無 streamlit 也能跑)
# ----------------------------------------------------------------------
def cli_main():
    print("=" * 60)
    print(" 台股波段篩選 — 今日結果")
    print("=" * 60)
    cand, _ = run_screen()
    if cand.empty:
        print("今日沒有符合條件的標的(可能整體偏弱,空手也是一種紀律)。")
        return
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.max_colwidth", 30, "display.width", 200):
        print(cand.to_string(index=False))
    print("\n提醒:訊號僅供參考,進場前請確認停損,並控制單筆部位風險。")


# ======================================================================
# 5a-2. 法人籌碼 + 技術面選股(資料來源:FinMind)
#   實作規則中「籌碼面 + 技術面」的部分:
#     做多訊號 C1~C5、C8~C12;風險排除 B8(融資過熱)、B9(股價過熱)、B10(爆量轉弱)
#   尚未納入(需基本面或董監/處置資料,之後再加):
#     A1~A8、B1~B7、B11、C6、C7
#   注意:FinMind 免費版只能逐檔查詢且有用量上限,故股票池預設較小、並做快取。
# ======================================================================
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
CHIP_UNIVERSE = DEFAULT_UNIVERSE[:40]   # 預設掃描檔數(可在介面調整),避免超出免費用量
_FM_CACHE = {}

# 做多訊號代碼 -> 名稱
C_SIGNAL_NAMES = {
    "C1": "外資主導型", "C2": "投信主升段型", "C3": "冷門股轉強型",
    "C4": "法人共振型", "C5": "自營商領先型", "C8": "法人吃貨型",
    "C9": "軋空預備型", "C10": "軋空發動型", "C11": "最強主升段型",
    "C12": "起漲點型",
}


def fm_fetch(dataset, data_id, start_date, token, end_date=None):
    """向 FinMind v4 取單一資料表(逐檔)。含簡單快取與錯誤處理。"""
    import requests
    key = (dataset, data_id, start_date, end_date)
    if key in _FM_CACHE:
        return _FM_CACHE[key]
    params = {"dataset": dataset, "data_id": data_id, "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    if token:
        params["token"] = token          # 部分情況需用 query 參數帶 token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(FINMIND_URL, headers=headers, params=params, timeout=25)
    except Exception as e:
        raise RuntimeError(f"連線 FinMind 失敗:{e}")
    if r.status_code == 402:
        raise RuntimeError("FinMind 用量已達上限(免費版有每小時上限),請稍後再試或縮小股票池。")
    if r.status_code in (401, 403):
        raise RuntimeError("FinMind token 無效或未授權,請確認 token 是否正確。")
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("data", []))
    _FM_CACHE[key] = df
    return df


def fm_usage(token):
    """查詢 FinMind 目前用量與每小時上限,用來診斷 token 是否生效。"""
    import requests
    try:
        r = requests.get("https://api.web.finmindtrade.com/v2/user_info",
                         headers={"Authorization": f"Bearer {token}"}, timeout=15)
        j = r.json()
        return j.get("user_count"), j.get("api_request_limit")
    except Exception:
        return None, None


def _inst_net(df_inst):
    """三大法人買賣表 -> 每日淨買超(外資/投信/自營)時間序列。"""
    if df_inst is None or df_inst.empty or "name" not in df_inst:
        return None
    d = df_inst.copy()
    d["net"] = d["buy"].astype(float) - d["sell"].astype(float)
    piv = d.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0).sort_index()
    out = pd.DataFrame(index=piv.index)
    out["foreign"] = piv.get("Foreign_Investor", 0) + piv.get("Foreign_Dealer_Self", 0)
    out["trust"] = piv.get("Investment_Trust", 0)
    out["dealer"] = piv.get("Dealer_self", 0) + piv.get("Dealer_Hedging", 0)
    return out


def _consec_buy(s, n):
    """最近 n 日是否連續淨買超(>0)。"""
    if s is None or len(s) < n:
        return False
    return bool((s.iloc[-n:] > 0).all())


def _turn_to_buy(s, look=5):
    """是否『由賣轉買』:近期曾賣超、最新一日轉買超。"""
    if s is None or len(s) < look + 1:
        return False
    return bool(s.iloc[-1] > 0 and (s.iloc[-(look + 1):-1] < 0).any())


def chip_features(stock_id, token):
    """抓三張表並計算所有需要的特徵。資料不足回傳 None。"""
    import datetime as dt
    today = dt.date.today()
    p_start = (today - dt.timedelta(days=260)).isoformat()
    i_start = (today - dt.timedelta(days=70)).isoformat()
    m_start = (today - dt.timedelta(days=90)).isoformat()

    dfp = fm_fetch("TaiwanStockPrice", stock_id, p_start, token)
    if dfp is None or dfp.empty or len(dfp) < 65:
        return None
    dfp = dfp.sort_values("date").reset_index(drop=True)
    close = dfp["close"].astype(float)
    op = dfp["open"].astype(float)
    vol = dfp["Trading_Volume"].astype(float)

    def ma(n):
        return close.rolling(n).mean()
    def vma(n):
        return vol.rolling(n).mean()

    c0 = float(close.iloc[-1])
    ma5, ma10, ma20, ma60 = ma(5).iloc[-1], ma(10).iloc[-1], ma(20).iloc[-1], ma(60).iloc[-1]
    ma5_up = ma(5).iloc[-1] > ma(5).iloc[-4]
    ma10_up = ma(10).iloc[-1] > ma(10).iloc[-4]
    ma20_up = ma(20).iloc[-1] > ma(20).iloc[-6]
    vol0 = float(vol.iloc[-1]); vol20 = float(vma(20).iloc[-1])
    vol5_mean = float(vol.iloc[-5:].mean()); vol_prev = float(vol.iloc[-25:-5].mean()) if len(vol) >= 25 else vol5_mean
    ret20 = c0 / float(close.iloc[-21]) - 1 if len(close) >= 21 else 0.0
    ret60 = c0 / float(close.iloc[-61]) - 1 if len(close) >= 61 else 0.0
    breakout = c0 > float(close.iloc[-21:-1].max())          # 突破前 20 日高(整理平台/壓力)
    newhigh60 = c0 >= float(close.iloc[-60:].max())
    black = c0 < float(op.iloc[-1])
    up1 = c0 > float(close.iloc[-2])
    up3 = c0 > float(close.iloc[-2]) > float(close.iloc[-3])
    vol_spike_black = vol0 > vol20 * 3 and black
    vol60_black = vol0 >= float(vol.iloc[-60:].max()) and black
    no_big_black5 = not any(
        float(vol.iloc[-i]) > vol20 * 3 and float(close.iloc[-i]) < float(op.iloc[-i])
        for i in range(1, 6))
    cold = (vol20 * c0) < 1e8                                 # 冷門:日均成交額 < 1 億
    price_not_fall = c0 >= float(close.iloc[-2])

    # 三大法人
    inst = _inst_net(fm_fetch("TaiwanStockInstitutionalInvestorsBuySell", stock_id, i_start, token))
    f = inst["foreign"] if inst is not None else None
    t = inst["trust"] if inst is not None else None
    d = inst["dealer"] if inst is not None else None
    f_net0 = bool(f is not None and f.iloc[-1] > 0)
    t_net0 = bool(t is not None and t.iloc[-1] > 0)
    f_no_big_sell5 = bool(f is not None and len(f) >= 5 and (f.iloc[-5:] < 0).sum() <= 1)

    # 融資融券
    mg = fm_fetch("TaiwanStockMarginPurchaseShortSale", stock_id, m_start, token)
    margin_chg20 = short_rising = short_fast_drop = margin_not_up = margin_small_up = None
    if mg is not None and not mg.empty and "MarginPurchaseTodayBalance" in mg:
        mg = mg.sort_values("date")
        mbal = mg["MarginPurchaseTodayBalance"].astype(float)
        sbal = mg["ShortSaleTodayBalance"].astype(float) if "ShortSaleTodayBalance" in mg else None
        if len(mbal) >= 21:
            margin_chg20 = float(mbal.iloc[-1] / max(mbal.iloc[-21], 1) - 1)
        if len(mbal) >= 6:
            chg5 = float(mbal.iloc[-1] / max(mbal.iloc[-6], 1) - 1)
            margin_not_up = chg5 <= 0.02
            margin_small_up = 0 < chg5 < 0.10
        if sbal is not None and len(sbal) >= 6:
            short_rising = bool(sbal.iloc[-1] > sbal.iloc[-6])
            short_fast_drop = (sbal.iloc[-1] / max(sbal.iloc[-6], 1) - 1) < -0.10

    return dict(
        c0=c0, ma5=ma5, ma10=ma10, ma20=ma20, ma60=ma60,
        ma5_up=ma5_up, ma10_up=ma10_up, ma20_up=ma20_up,
        vol0=vol0, vol20=vol20, vol5_mean=vol5_mean, vol_prev=vol_prev,
        ret20=ret20, ret60=ret60, breakout=breakout, newhigh60=newhigh60,
        black=black, up1=up1, up3=up3, vol_spike_black=vol_spike_black,
        vol60_black=vol60_black, no_big_black5=no_big_black5, cold=cold,
        price_not_fall=price_not_fall,
        f=f, t=t, d=d, f_net0=f_net0, t_net0=t_net0, f_no_big_sell5=f_no_big_sell5,
        margin_chg20=margin_chg20, short_rising=short_rising,
        short_fast_drop=short_fast_drop, margin_not_up=margin_not_up,
        margin_small_up=margin_small_up,
    )


def chip_signals(f):
    """依特徵計算做多訊號(C)與風險旗標(B)。回傳 (c_list, b_list)。"""
    if f is None:
        return [], []
    c0, ma20 = f["c0"], f["ma20"]
    vol_up = f["vol0"] > f["vol20"]                  # 量 > 20日均量
    vol5_up = f["vol5_mean"] > f["vol_prev"]         # 近5日均量 > 前20日均量
    along_line = (c0 > f["ma5"] and f["ma5_up"]) or (c0 > f["ma10"] and f["ma10_up"])

    C = []
    # C1 外資主導型
    if _consec_buy(f["f"], 3) and c0 > ma20 and f["ma20_up"] and vol5_up and f["f_no_big_sell5"]:
        C.append("C1")
    # C2 投信主升段型
    if _consec_buy(f["t"], 5) and along_line and f["no_big_black5"]:
        C.append("C2")
    # C3 冷門股轉強型
    if f["cold"] and _consec_buy(f["t"], 3) and _turn_to_buy(f["f"]) and f["breakout"] and vol_up:
        C.append("C3")
    # C4 法人共振型
    if _consec_buy(f["f"], 3) and (_consec_buy(f["t"], 1) or _turn_to_buy(f["t"])) and f["breakout"] and vol_up:
        C.append("C4")
    # C5 自營商領先型
    if _consec_buy(f["d"], 3) and (f["f_net0"] or f["t_net0"]) and f["breakout"] and vol_up:
        C.append("C5")
    # C8 法人吃貨型
    if f["up1"] and (_consec_buy(f["f"], 2) or _consec_buy(f["t"], 2)) and (f["margin_not_up"] is True):
        C.append("C8")
    # C9 軋空預備型
    if (f["short_rising"] is True) and f["newhigh60"] and (f["f_net0"] or f["t_net0"]):
        C.append("C9")
    # C10 軋空發動型
    if f["up3"] and (f["short_fast_drop"] is True) and vol_up:
        C.append("C10")
    # C11 最強主升段型
    if (_consec_buy(f["t"], 5) and (_consec_buy(f["f"], 1) or _turn_to_buy(f["f"]))
            and c0 > ma20 and c0 > f["ma60"] and vol5_up
            and (f["margin_chg20"] is None or f["margin_chg20"] < 0.2)
            and (f["short_rising"] is True) and f["price_not_fall"]):
        C.append("C11")
    # C12 起漲點型
    if (_consec_buy(f["f"], 3) and _turn_to_buy(f["t"]) and f["breakout"] and vol_up
            and (f["margin_small_up"] is True) and (f["short_rising"] is True)):
        C.append("C12")

    # 風險排除旗標
    B = []
    if f["margin_chg20"] is not None and (
            f["margin_chg20"] > 0.20 or
            (f["margin_chg20"] > 0.05 and f["margin_chg20"] > f["ret20"])):
        B.append("B8 融資過熱")
    if f["ret20"] > 0.40 or f["ret60"] > 0.80:
        B.append("B9 股價過熱")
    # B10：5 倍爆量收黑，或創 60 日量新高且量 > 2 倍均量且收黑
    # （vol60_black 加「> 2 倍均量」門檻，避免微幅放量誤觸）
    if (f["vol0"] > f["vol20"] * 5 and f["black"]) or \
       (f["vol60_black"] and f["vol0"] > f["vol20"] * 2.0 and f["black"]):
        B.append("B10 爆量轉弱")
    return C, B


def run_chip_screen(universe, token, exclude_ids=None, progress=None):
    """逐檔分析,回傳結果 DataFrame。progress:可選的回呼(i, n, id)。"""
    exclude_ids = set(exclude_ids or [])
    rows = []
    n = len(universe)
    for i, tw in enumerate(universe):
        sid = tw.replace(".TWO", "").replace(".TW", "")
        if progress:
            progress(i, n, sid)
        if sid in exclude_ids:
            continue
        try:
            feats = chip_features(sid, token)
        except RuntimeError:
            raise                      # 用量/授權錯誤往上拋,讓介面顯示
        except Exception:
            continue                   # 個別股票資料問題就略過
        if feats is None:
            continue
        C, B = chip_signals(feats)
        if B:                          # 命中任一風險排除 -> 跳過
            continue
        if not C:                      # 沒有做多訊號 -> 不列入
            continue
        rows.append({
            "代號": sid, "股票名稱": NAME_MAP.get(tw, ""),
            "收盤": round(feats["c0"], 2),
            "做多訊號數": len(C),
            "做多訊號": "、".join(f"{c} {C_SIGNAL_NAMES[c]}" for c in C),
            "20日漲幅%": round(feats["ret20"] * 100, 1),
            "量比": round(feats["vol0"] / feats["vol20"], 2) if feats["vol20"] else None,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("做多訊號數", ascending=False).reset_index(drop=True)


# ======================================================================
# 5a-3. 基本面選股(資料來源:FinMind 財報三表 + 月營收)
#   納入 A1~A7、排除 B1~B6。A8/B7(董監質押)免費資料源無法取得,改為手動排除清單。
#   財報為 long 格式,以英文 type + 中文 origin_name 雙重比對,提高相容性。
# ======================================================================
def _pick(df, types=(), origin_kw=(), exclude_kw=()):
    """從 long 格式財報挑出某科目的時間序列(index=date 字串,已排序)。"""
    if df is None or df.empty or "type" not in df:
        return None
    sub = df[df["type"].isin(types)] if types else df.iloc[0:0]
    if (sub is None or sub.empty) and origin_kw:
        mask = pd.Series(False, index=df.index)
        for kw in origin_kw:
            mask = mask | df["origin_name"].astype(str).str.contains(kw, na=False)
        sub = df[mask]
    if sub is None or sub.empty:
        return None
    if exclude_kw is not None and len(exclude_kw):
        for kw in exclude_kw:
            sub = sub[~sub["origin_name"].astype(str).str.contains(kw, na=False)]
    if sub.empty:
        return None
    s = sub.groupby("date")["value"].first().astype(float).sort_index()
    return s


def _decum(s):
    """季報多為累計(YTD),轉成單季數值(同年內相減,當年第一筆即為單季)。"""
    if s is None or len(s) == 0:
        return None
    d = pd.DataFrame({"date": list(s.index), "value": list(s.values)})
    d["yr"] = d["date"].str[:4]
    d["q"] = d.groupby("yr")["value"].diff()
    d["q"] = d["q"].where(d["q"].notna(), d["value"])
    return pd.Series(d["q"].values, index=d["date"].values)


def _yoy_q(sq):
    """單季序列的同期(去年同季)年增率序列。"""
    if sq is None or len(sq) < 5:
        return None
    return sq / sq.shift(4) - 1


def fundamental_features(stock_id, token):
    import datetime as dt
    today = dt.date.today()
    fs_start = (today - dt.timedelta(days=900)).isoformat()   # 約 3 年財報
    mr_start = (today - dt.timedelta(days=500)).isoformat()   # 約 16 個月營收

    fs = fm_fetch("TaiwanStockFinancialStatements", stock_id, fs_start, token)
    bs = fm_fetch("TaiwanStockBalanceSheet", stock_id, fs_start, token)
    cf = fm_fetch("TaiwanStockCashFlowsStatement", stock_id, fs_start, token)
    mr = fm_fetch("TaiwanStockMonthRevenue", stock_id, mr_start, token)
    if (fs is None or fs.empty) and (bs is None or bs.empty):
        return None

    # --- 損益表(累計 -> 單季) ---
    eps_q = _decum(_pick(fs, types=["EPS"]))
    rev_q = _decum(_pick(fs, types=["Revenue"], origin_kw=["營業收入"]))
    gp_q = _decum(_pick(fs, types=["GrossProfit"], origin_kw=["營業毛利"]))
    oi_q = _decum(_pick(fs, types=["OperatingIncome", "OperatingProfit"], origin_kw=["營業利益"]))

    eps_last = float(eps_q.iloc[-1]) if eps_q is not None and len(eps_q) else None
    eps_ttm = float(eps_q.iloc[-4:].sum()) if eps_q is not None and len(eps_q) >= 4 else None

    def two_down(sq, rev):  # 連續兩季下滑(毛利率/營益率)
        if sq is None or rev is None or len(sq) < 3:
            return None
        m = (sq / rev.reindex(sq.index)).dropna()
        if len(m) < 3:
            return None
        return bool(m.iloc[-1] < m.iloc[-2] < m.iloc[-3])
    gm_two_down = two_down(gp_q, rev_q)
    om_two_down = two_down(oi_q, rev_q)

    # --- 月營收年增率(最近三個月平均) ---
    rev3_yoy_mean = rev3_all_neg = None
    if mr is not None and not mr.empty and "revenue" in mr:
        m = mr.sort_values("date")
        yoy = (m["revenue"].astype(float).pct_change(12)).dropna()
        if len(yoy) >= 3:
            last3 = yoy.iloc[-3:]
            rev3_yoy_mean = float(last3.mean())
            rev3_all_neg = bool((last3 < 0).all())

    # --- 現金流(營業活動,累計 -> 單季 -> 近一年 TTM) ---
    ocf_q = _decum(_pick(cf, types=["CashFlowsFromOperatingActivities",
                                    "NetCashProvidedByUsedInOperatingActivities"],
                         origin_kw=["營業活動之淨現金", "營業活動"]))
    ocf_ttm = ocf_prev = None
    if ocf_q is not None and len(ocf_q) >= 4:
        ocf_ttm = float(ocf_q.iloc[-4:].sum())
        if len(ocf_q) >= 8:
            ocf_prev = float(ocf_q.iloc[-8:-4].sum())

    # --- 資產負債表:負債比、應收、存貨 ---
    liab = _pick(bs, types=["Liabilities", "TotalLiabilities"], origin_kw=["負債總"])
    asset = _pick(bs, types=["TotalAssets", "Assets"], origin_kw=["資產總"])
    equity = _pick(bs, types=["Equity", "TotalEquity"], origin_kw=["權益總"])
    debt_ratio = None
    try:
        if liab is not None and asset is not None:
            debt_ratio = float(liab.iloc[-1] / asset.iloc[-1])
        elif equity is not None and asset is not None:
            debt_ratio = float(1 - equity.iloc[-1] / asset.iloc[-1])
    except Exception:
        debt_ratio = None

    ar = _pick(bs, types=["AccountsReceivable", "AccountsReceivableNet"],
               origin_kw=["應收帳款淨額", "應收帳款"], exclude_kw=["關係人"])
    inv = _pick(bs, types=["Inventory", "Inventories"], origin_kw=["存貨"])
    rev_yoy_q = _yoy_q(rev_q)
    rev_yoy_last = float(rev_yoy_q.iloc[-1]) if rev_yoy_q is not None and len(rev_yoy_q) else rev3_yoy_mean

    def yoy_bs(s):
        if s is None or len(s) < 5:
            return None
        return float(s.iloc[-1] / s.iloc[-5] - 1)
    ar_yoy = yoy_bs(ar)
    inv_yoy = yoy_bs(inv)

    return dict(
        eps_last=eps_last, eps_ttm=eps_ttm,
        gm_two_down=gm_two_down, om_two_down=om_two_down,
        rev3_yoy_mean=rev3_yoy_mean, rev3_all_neg=rev3_all_neg,
        ocf_ttm=ocf_ttm, ocf_prev=ocf_prev, debt_ratio=debt_ratio,
        ar_yoy=ar_yoy, inv_yoy=inv_yoy, rev_yoy_last=rev_yoy_last,
    )


def fundamental_eval(f):
    """
    回傳 (include:bool, B旗標list, 通過/未通過/無資料明細dict)。
    
    分離模式邏輯：
    - 資料不足的項目標記「無資料」，不計入通過/失敗
    - 有資料的 A 項「有一個 False 就扣分」，但不強制全過
    - 核心項（A1 EPS、A2 營收）至少一個要有資料且通過
    - 任何 B 旗標觸發就排除
    - 這樣即使部分財報欄位抓不到，也不會讓整支股票消失
    """
    if f is None:
        return False, ["資料不足"], {}
    B = []
    detail = {}      # key: 條件名稱, value: True/False/"無資料"

    # A1 / B1 獲利能力
    if f["eps_last"] is not None:
        detail["A1 最近季EPS>0"] = f["eps_last"] > 0
        if f["eps_last"] < 0:
            B.append("B1 最近季EPS<0")
    else:
        detail["A1 最近季EPS>0"] = "無資料"

    if f["eps_ttm"] is not None:
        detail["A1 近四季EPS合計>0"] = f["eps_ttm"] > 0
    else:
        detail["A1 近四季EPS合計>0"] = "無資料"

    # A2 / B2 營收成長
    if f["rev3_yoy_mean"] is not None:
        detail["A2 近三月營收YoY均>0"] = f["rev3_yoy_mean"] > 0
        if f.get("rev3_all_neg"):
            B.append("B2 近三月營收YoY皆負")
    else:
        detail["A2 近三月營收YoY均>0"] = "無資料"

    # A3 / B3 B4 獲利品質
    if f["gm_two_down"] is not None:
        detail["A3 毛利率未連兩季降"] = not f["gm_two_down"]
        if f["gm_two_down"]:
            B.append("B3 毛利率連兩季降")
    else:
        detail["A3 毛利率未連兩季降"] = "無資料"

    if f["om_two_down"] is not None:
        detail["A3 營益率未連兩季降"] = not f["om_two_down"]
        if f["om_two_down"]:
            B.append("B4 營益率連兩季降")
    else:
        detail["A3 營益率未連兩季降"] = "無資料"

    # A4 現金流
    if f["ocf_ttm"] is not None:
        ok = (f["ocf_ttm"] > 0) or (
            f["ocf_prev"] is not None and f["ocf_ttm"] >= 0.5 * f["ocf_prev"])
        detail["A4 近一年營業現金流"] = ok
    else:
        detail["A4 近一年營業現金流"] = "無資料"

    # A5 財務結構
    if f["debt_ratio"] is not None:
        detail["A5 負債比≤70%"] = f["debt_ratio"] <= 0.70
    else:
        detail["A5 負債比≤70%"] = "無資料"

    # A6 / B5 應收帳款
    if f["ar_yoy"] is not None and f["rev_yoy_last"] is not None:
        ok = f["ar_yoy"] <= f["rev_yoy_last"] + 0.20
        detail["A6 應收增率≤營收增率+20%"] = ok
        if not ok:
            B.append("B5 應收帳款異常")
    else:
        detail["A6 應收增率≤營收增率+20%"] = "無資料"

    # A7 / B6 存貨
    if f["inv_yoy"] is not None and f["rev_yoy_last"] is not None:
        ok = f["inv_yoy"] <= f["rev_yoy_last"] + 0.20
        detail["A7 存貨增率≤營收增率+20%"] = ok
        if not ok:
            B.append("B6 存貨異常")
    else:
        detail["A7 存貨增率≤營收增率+20%"] = "無資料"

    # ── 納入判斷（分離模式）──────────────────────────────────────────
    # 有資料的 A 項沒有任何 False（無資料不算失敗）
    scored = {k: v for k, v in detail.items() if v != "無資料"}
    no_false_a = all(v for v in scored.values()) if scored else False

    # 核心項：A1 EPS 或 A2 營收至少一個有資料且通過
    core_pass = any(
        v is True for k, v in detail.items()
        if k.startswith(("A1", "A2"))
    )

    # 有資料的項目數至少 3 項（防止資料完全空白的股票混入）
    enough_data = len(scored) >= 3

    include = bool(not B and no_false_a and core_pass and enough_data)
    return include, B, detail


def fundamental_debug(stock_id, token):
    """診斷:回傳某檔三表的 (type, origin_name) 清單,用來核對欄位對應。"""
    out = {}
    for label, ds in [("綜合損益表", "TaiwanStockFinancialStatements"),
                      ("資產負債表", "TaiwanStockBalanceSheet"),
                      ("現金流量表", "TaiwanStockCashFlowsStatement")]:
        try:
            df = fm_fetch(ds, stock_id, "2023-01-01", token)
            if df is not None and not df.empty and "type" in df:
                pairs = df[["type", "origin_name"]].drop_duplicates().head(80)
                out[label] = pairs
        except Exception as e:
            out[label] = str(e)
    return out


def run_fundamental_screen(universe, token, exclude_ids=None, progress=None):
    exclude_ids = set(exclude_ids or [])
    rows = []
    n = len(universe)
    for i, tw in enumerate(universe):
        sid = tw.replace(".TWO", "").replace(".TW", "")
        if progress:
            progress(i, n, sid)
        if sid in exclude_ids:
            continue
        try:
            feats = fundamental_features(sid, token)
        except RuntimeError:
            raise
        except Exception:
            continue
        include, B, detail = fundamental_eval(feats)
        if not include:
            continue
        rows.append({
            "代號": sid, "股票名稱": NAME_MAP.get(tw, ""),
            "最近季EPS": feats.get("eps_last"),
            "近四季EPS": round(feats["eps_ttm"], 2) if feats.get("eps_ttm") is not None else None,
            "近三月營收YoY%": round(feats["rev3_yoy_mean"] * 100, 1) if feats.get("rev3_yoy_mean") is not None else None,
            "負債比%": round(feats["debt_ratio"] * 100, 1) if feats.get("debt_ratio") is not None else None,
            "通過項目數": sum(1 for v in detail.values() if v),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("通過項目數", ascending=False).reset_index(drop=True)


# ======================================================================
# 5a-6. Backer 方案：一次抓全市場資料（不帶 data_id）
#   每張表只需一次請求即可取得全市場當日所有股票，大幅降低用量
# ======================================================================
def backer_fetch_allmarket(dataset, start_date, token, end_date=None):
    """
    Backer/Sponsor 專用：不帶 data_id，一次取得全市場資料。
    回傳 DataFrame（含 stock_id 欄位）。
    """
    import requests
    params = {"dataset": dataset, "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    headers = {"Authorization": f"Bearer {token}"}
    params["token"] = token
    try:
        r = requests.get(FINMIND_URL, headers=headers, params=params, timeout=60)
    except Exception as e:
        raise RuntimeError(f"連線 FinMind 失敗：{e}")
    if r.status_code == 402:
        raise RuntimeError("FinMind 用量已達上限，請稍後再試。")
    if r.status_code in (401, 403):
        raise RuntimeError("FinMind token 無效，請確認 token 與方案是否正確。")
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("data", []))
    return df


def backer_fetch_price_history(token, days=260):
    """一次取得全市場近 N 天日 K 線，回傳 {stock_id: DataFrame}。
    欄位保持小寫（close/open/max/min/Trading_Volume），
    與 chip_features 和 add_indicators 的需求一致。
    """
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = backer_fetch_allmarket("TaiwanStockPrice", start, token)
    if df.empty or "stock_id" not in df:
        return {}
    df = df.sort_values(["stock_id", "date"])
    result = {}
    for sid, grp in df.groupby("stock_id"):
        # 保持 FinMind 原始小寫欄位：close, open, max, min, Trading_Volume
        needed = [c for c in ["date", "open", "max", "min", "close", "Trading_Volume"]
                  if c in grp.columns]
        g = grp[needed].copy().reset_index(drop=True)
        g["close"] = pd.to_numeric(g["close"], errors="coerce")
        g["Trading_Volume"] = pd.to_numeric(g["Trading_Volume"], errors="coerce")
        g = g.dropna(subset=["close", "Trading_Volume"])
        if len(g) >= 65:
            result[sid] = g
    return result


def backer_fetch_institutional(token, days=70):
    """一次取得全市場三大法人近 N 天資料，回傳 {stock_id: DataFrame}。"""
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = backer_fetch_allmarket("TaiwanStockInstitutionalInvestorsBuySell", start, token)
    if df.empty or "stock_id" not in df:
        return {}
    result = {}
    for sid, grp in df.groupby("stock_id"):
        result[sid] = grp.reset_index(drop=True)
    return result


def backer_fetch_margin(token, days=90):
    """一次取得全市場融資融券近 N 天資料，回傳 {stock_id: DataFrame}。"""
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = backer_fetch_allmarket("TaiwanStockMarginPurchaseShortSale", start, token)
    if df.empty or "stock_id" not in df:
        return {}
    result = {}
    for sid, grp in df.groupby("stock_id"):
        result[sid] = grp.reset_index(drop=True)
    return result


def backer_fetch_financials(token, days=900):
    """一次取得全市場財報三表 + 月營收，回傳 {dataset: {stock_id: DataFrame}}。"""
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    mr_start = (dt.date.today() - dt.timedelta(days=500)).isoformat()
    out = {}
    for ds, s in [
        ("TaiwanStockFinancialStatements", start),
        ("TaiwanStockBalanceSheet", start),
        ("TaiwanStockCashFlowsStatement", start),
        ("TaiwanStockMonthRevenue", mr_start),
    ]:
        df = backer_fetch_allmarket(ds, s, token)
        if df.empty or "stock_id" not in df:
            out[ds] = {}
            continue
        result = {}
        for sid, grp in df.groupby("stock_id"):
            result[sid] = grp.reset_index(drop=True)
        out[ds] = result
    return out


def chip_features_from_cache(sid, price_df, inst_df, margin_df):
    """
    用預先下載好的全市場資料計算單支股票的籌碼特徵。
    price_df 是 backer_fetch_price_history 回傳的小寫欄位 DataFrame
    （close/open/max/min/Trading_Volume），剛好符合 chip_features 的需求。
    """
    if price_df is None or len(price_df) < 65:
        return None
    p = price_df.copy()
    # 確保有 open 欄位（chip_features 需要）
    if "open" not in p.columns:
        p["open"] = p["close"]
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p["Trading_Volume"] = pd.to_numeric(p["Trading_Volume"], errors="coerce")
    p = p.dropna(subset=["close", "Trading_Volume"])
    if len(p) < 65:
        return None

    import sys
    mod = sys.modules[__name__]
    _saved = mod.fm_fetch

    def _mock(dataset, data_id, start, token_, end_date=None):
        if dataset == "TaiwanStockPrice":
            return p   # 已是 chip_features 期望的小寫欄位格式
        if "Institutional" in dataset:
            return inst_df if inst_df is not None else pd.DataFrame()
        if "Margin" in dataset:
            return margin_df if margin_df is not None else pd.DataFrame()
        return pd.DataFrame()

    mod.fm_fetch = _mock
    try:
        feats = chip_features(sid, "__cache__")
    finally:
        mod.fm_fetch = _saved
    return feats


def fundamental_features_from_cache(sid, fs_dict, bs_dict, cf_dict, mr_dict):
    """用預先下載好的全市場財報資料計算單支股票的基本面特徵。"""
    import sys
    mod = sys.modules[__name__]
    _saved = mod.fm_fetch

    def _mock(dataset, data_id, start, token_, end_date=None):
        if "FinancialStatements" in dataset:
            return fs_dict.get(sid, pd.DataFrame())
        if "BalanceSheet" in dataset:
            return bs_dict.get(sid, pd.DataFrame())
        if "CashFlows" in dataset:
            return cf_dict.get(sid, pd.DataFrame())
        if "MonthRevenue" in dataset:
            return mr_dict.get(sid, pd.DataFrame())
        return pd.DataFrame()

    mod.fm_fetch = _mock
    try:
        feats = fundamental_features(sid, "__cache__")
    finally:
        mod.fm_fetch = _saved
    return feats


def run_full_analysis_backer(token, exclude_ids=None, min_turnover=5e7, progress=None):
    """
    Backer 版全面分析：一次下載全市場資料，再對每支跑基本面 + 籌碼面。
    大幅減少 API 請求次數（從數千次降到約 7 次）。
    """
    exclude_ids = set(exclude_ids or [])

    if progress: progress("下載全市場股價資料中…", 0, 6)
    price_cache = backer_fetch_price_history(token)

    if progress: progress("下載全市場三大法人資料中…", 1, 6)
    inst_cache = backer_fetch_institutional(token)

    if progress: progress("下載全市場融資融券資料中…", 2, 6)
    margin_cache = backer_fetch_margin(token)

    if progress: progress("下載全市場財報資料中（約需 1~2 分鐘）…", 3, 6)
    fin_cache = backer_fetch_financials(token)
    fs_dict = fin_cache.get("TaiwanStockFinancialStatements", {})
    bs_dict = fin_cache.get("TaiwanStockBalanceSheet", {})
    cf_dict = fin_cache.get("TaiwanStockCashFlowsStatement", {})
    mr_dict = fin_cache.get("TaiwanStockMonthRevenue", {})

    if progress: progress("技術面初篩中…", 4, 6)
    # 用下載好的股價做技術初篩
    # price_cache 的欄位是小寫（close/Trading_Volume），
    # add_indicators 需要 Close/Volume，先 rename 再算
    candidates = []
    for sid, pdf in price_cache.items():
        if sid in exclude_ids:
            continue
        try:
            df_ind = pdf.rename(columns={
                "close": "Close", "Trading_Volume": "Volume"
            })
            df_ind = add_indicators(df_ind)
            last = df_ind.iloc[-1]
            close = float(last["Close"])
            ma60 = float(last["MA60"]) if not pd.isna(last["MA60"]) else 0
            ma20 = float(last["MA20"]) if not pd.isna(last["MA20"]) else 0
            vol20 = float(last["VOL20"]) if not pd.isna(last["VOL20"]) else 0
            if (close > ma60 and ma20 > ma60
                    and close * vol20 >= min_turnover and close > 10):
                candidates.append(sid)
        except Exception:
            continue

    if progress: progress(f"分析 {len(candidates)} 支候選股票中…", 5, 6)

    rows = []
    errors = []   # 收集錯誤供診斷
    n = len(candidates)
    for i, sid in enumerate(candidates):
        if progress and i % 50 == 0:
            progress(f"全面分析中 {i}/{n}…", 5, 6)

        # 基本面
        try:
            fund_feats = fundamental_features_from_cache(
                sid, fs_dict, bs_dict, cf_dict, mr_dict)
        except Exception as e:
            errors.append(f"{sid} 基本面: {e}")
            fund_feats = None
        fund_include, fund_B, fund_detail = fundamental_eval(fund_feats)

        # 籌碼面
        try:
            chip_feats = chip_features_from_cache(
                sid,
                price_cache.get(sid),
                inst_cache.get(sid),
                margin_cache.get(sid)
            )
        except Exception as e:
            errors.append(f"{sid} 籌碼: {e}")
            chip_feats = None
        chip_C, chip_B = chip_signals(chip_feats) if chip_feats else ([], [])

        has_fund = fund_include
        # 分離模式：有 C 訊號就顯示，B 旗標標記在欄位裡而不是整個排除
        # （除非是 B9 股價嚴重過熱，仍然排除）
        severe_risk = any("B9" in b for b in chip_B)
        has_chip = bool(chip_C) and not severe_risk

        if not has_fund and not has_chip:
            continue

        tw = sid + ".TW"
        name = NAME_MAP.get(tw, NAME_MAP.get(sid + ".TWO", ""))
        close = round(chip_feats["c0"], 2) if chip_feats else None
        ret20 = round(chip_feats["ret20"] * 100, 1) if chip_feats else None
        vol_ratio = round(chip_feats["vol0"] / chip_feats["vol20"], 2) \
            if chip_feats and chip_feats.get("vol20") else None
        eps_last = round(fund_feats["eps_last"], 2) \
            if fund_feats and fund_feats.get("eps_last") is not None else None
        eps_ttm = round(fund_feats["eps_ttm"], 2) \
            if fund_feats and fund_feats.get("eps_ttm") is not None else None
        rev_yoy = round(fund_feats["rev3_yoy_mean"] * 100, 1) \
            if fund_feats and fund_feats.get("rev3_yoy_mean") is not None else None
        debt = round(fund_feats["debt_ratio"] * 100, 1) \
            if fund_feats and fund_feats.get("debt_ratio") is not None else None
        fund_pass = sum(1 for v in fund_detail.values() if v) if fund_detail else 0
        fund_total = len(fund_detail) if fund_detail else 0

        grade = ("⭐⭐ 雙面確認" if has_fund and has_chip
                 else "📊 基本面通過" if has_fund else "🏦 籌碼訊號")

        rows.append({
            "代號": sid, "股票名稱": name, "收盤": close,
            "綜合評級": grade,
            "基本面通過": "✅" if has_fund else "—",
            "籌碼訊號": "✅" if has_chip else "—",
            "做多訊號": "、".join(f"{c} {C_SIGNAL_NAMES[c]}" for c in chip_C),
            "籌碼風險": "、".join(chip_B) if chip_B else "無",
            "近季EPS": eps_last, "近四季EPS": eps_ttm,
            "近三月營收YoY%": rev_yoy, "負債比%": debt,
            "基本面通過項": f"{fund_pass}/{fund_total}" if fund_total else "—",
            "20日漲幅%": ret20, "量比": vol_ratio,
        })

    if not rows:
        return pd.DataFrame(), candidates, errors

    df = pd.DataFrame(rows)
    grade_order = {"⭐⭐ 雙面確認": 0, "📊 基本面通過": 1, "🏦 籌碼訊號": 2}
    df["_g"] = df["綜合評級"].map(grade_order)
    df["_cs"] = df["做多訊號"].str.count("、") + df["做多訊號"].str.len().gt(0).astype(int)
    return (df.sort_values(["_g", "_cs"], ascending=[True, False])
              .drop(columns=["_g", "_cs"]).reset_index(drop=True),
            candidates, errors)


# ======================================================================
# 持股池管理（分頁 1 用）：讀寫自訂股票池 Excel
# ======================================================================
POOL_COLS = ["代號", "股票名稱", "產業", "備註"]

def empty_pool() -> pd.DataFrame:
    rows = []
    for tw, name in NAME_MAP.items():
        sid = tw.replace(".TW", "").replace(".TWO", "")
        rows.append({"代號": tw, "股票名稱": name, "產業": "", "備註": ""})
    return pd.DataFrame(rows)

def read_pool_excel(file_bytes) -> pd.DataFrame:
    import io
    df = pd.read_excel(io.BytesIO(file_bytes))
    for col in POOL_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[POOL_COLS]

# ======================================================================
# 模擬投資（分頁 4 用）
# ======================================================================
SIM_COLS = ["日期", "代號", "股票名稱", "產業", "分數", "有進場訊號",
            "進場價", "停損價", "目標價", "入選理由", "狀態",
            "結算價", "損益%", "持有天數", "結算原因"]

DONE_COLS = SIM_COLS  # 相同結構，狀態欄為已完成

def empty_sim() -> pd.DataFrame:
    return pd.DataFrame(columns=SIM_COLS)

def read_sim_excel(file_bytes) -> tuple:
    import io
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    active = pd.read_excel(xls, "追蹤中") if "追蹤中" in xls.sheet_names else empty_sim()
    done = pd.read_excel(xls, "已完成") if "已完成" in xls.sheet_names else empty_sim()
    return active, done

def update_sim_prices(active: pd.DataFrame) -> pd.DataFrame:
    """抓最新價格，自動判斷是否觸發停損或目標，更新狀態。"""
    if active.empty:
        return active
    import datetime as dt
    active = active.copy()
    tickers = active["代號"].dropna().astype(str).unique().tolist()
    try:
        hist = fetch_history(tickers, period="5d")
    except Exception:
        hist = {}
    triggered = []
    for i, row in active.iterrows():
        tw = str(row["代號"])
        if tw not in hist or hist[tw].empty:
            continue
        cur_p = float(hist[tw]["Close"].iloc[-1])
        stop = float(row["停損價"]) if pd.notna(row["停損價"]) else None
        target = float(row["目標價"]) if pd.notna(row["目標價"]) else None
        entry = float(row["進場價"]) if pd.notna(row["進場價"]) else None
        if stop and cur_p <= stop:
            triggered.append((i, cur_p, "停損出場"))
        elif target and cur_p >= target:
            triggered.append((i, cur_p, "達目標出場"))
    done_rows = []
    drop_idx = []
    for i, cur_p, reason in triggered:
        row = active.loc[i].copy()
        entry = float(row["進場價"])
        pnl = round((cur_p / entry - 1) * 100, 2)
        try:
            days = (dt.date.today() - pd.to_datetime(row["日期"]).date()).days
        except Exception:
            days = 0
        row["狀態"] = "已完成"
        row["結算價"] = cur_p
        row["損益%"] = pnl
        row["持有天數"] = days
        row["結算原因"] = reason
        done_rows.append(row)
        drop_idx.append(i)
    new_active = active.drop(index=drop_idx).reset_index(drop=True)
    return new_active, pd.DataFrame(done_rows) if done_rows else pd.DataFrame()


def sim_analysis(active: pd.DataFrame, done: pd.DataFrame) -> dict:
    """計算程式能力分析的各維度統計。"""
    all_done = done.copy() if not done.empty else pd.DataFrame(columns=SIM_COLS)
    if all_done.empty:
        return {}

    pnl = pd.to_numeric(all_done["損益%"], errors="coerce").dropna()
    win = (pnl > 0).sum()
    total = len(pnl)
    stats = {
        "整體": {
            "總筆數": total,
            "勝率%": round(win / total * 100, 1) if total else 0,
            "平均報酬%": round(pnl.mean(), 2) if total else 0,
            "最大獲利%": round(pnl.max(), 2) if total else 0,
            "最大虧損%": round(pnl.min(), 2) if total else 0,
            "平均持有天數": round(pd.to_numeric(
                all_done["持有天數"], errors="coerce").mean(), 1) if total else 0,
        }
    }

    # 分數分組
    if "分數" in all_done.columns:
        all_done["_score"] = pd.to_numeric(all_done["分數"], errors="coerce")
        bins = [0, 60, 75, 90, 101]
        labels = ["60以下", "60~74", "75~89", "90+"]
        all_done["_sg"] = pd.cut(all_done["_score"], bins=bins, labels=labels, right=False)
        score_grp = []
        for label in labels:
            sub = all_done[all_done["_sg"] == label]
            sub_pnl = pd.to_numeric(sub["損益%"], errors="coerce").dropna()
            n = len(sub_pnl)
            score_grp.append({
                "分數區間": label,
                "筆數": n,
                "勝率%": round((sub_pnl > 0).sum() / n * 100, 1) if n else 0,
                "平均報酬%": round(sub_pnl.mean(), 2) if n else 0,
            })
        stats["分數分組"] = pd.DataFrame(score_grp)

    # 進場訊號效果
    if "有進場訊號" in all_done.columns:
        signal_grp = []
        for label, flag in [("有 ✅ 進場訊號", True), ("無進場訊號", False)]:
            sub = all_done[all_done["有進場訊號"].astype(str).isin(
                ["True", "✅", "1"] if flag else ["False", "—", "0", ""])]
            sub_pnl = pd.to_numeric(sub["損益%"], errors="coerce").dropna()
            n = len(sub_pnl)
            signal_grp.append({
                "類型": label, "筆數": n,
                "勝率%": round((sub_pnl > 0).sum() / n * 100, 1) if n else 0,
                "平均報酬%": round(sub_pnl.mean(), 2) if n else 0,
            })
        stats["進場訊號效果"] = pd.DataFrame(signal_grp)

    # 產業分析
    if "產業" in all_done.columns:
        ind_rows = []
        for ind, grp in all_done.groupby("產業"):
            if not ind or str(ind) == "nan":
                continue
            sub_pnl = pd.to_numeric(grp["損益%"], errors="coerce").dropna()
            n = len(sub_pnl)
            if n < 2:
                continue
            ind_rows.append({
                "產業": ind, "筆數": n,
                "勝率%": round((sub_pnl > 0).sum() / n * 100, 1),
                "平均報酬%": round(sub_pnl.mean(), 2),
            })
        if ind_rows:
            stats["產業分析"] = pd.DataFrame(ind_rows).sort_values(
                "勝率%", ascending=False)

    # 月度趨勢
    if "日期" in all_done.columns:
        all_done["_month"] = pd.to_datetime(
            all_done["日期"], errors="coerce").dt.strftime("%Y-%m")
        month_rows = []
        for m, grp in all_done.groupby("_month"):
            if not m or str(m) == "nan":
                continue
            sub_pnl = pd.to_numeric(grp["損益%"], errors="coerce").dropna()
            n = len(sub_pnl)
            month_rows.append({
                "月份": m, "推薦筆數": n,
                "勝率%": round((sub_pnl > 0).sum() / n * 100, 1) if n else 0,
                "平均報酬%": round(sub_pnl.mean(), 2) if n else 0,
            })
        if month_rows:
            stats["月度趨勢"] = pd.DataFrame(month_rows).sort_values("月份")

    return stats


# ======================================================================
# 5a-4 / 5a-5 現有函式保留（不刪，供免費版相容）
# ======================================================================

# ── 步驟 A：從 TWSE / TPEX 官方 OpenAPI 取得全市場股票清單 ──────────────
def fetch_market_universe():
    """
    呼叫 TWSE 與 TPEX 官方 OpenAPI，取得今日所有上市/上櫃股票代號。
    完全免費、無需 token、無次數限制。
    回傳 list of (stock_id, market) 如 [("2330","twse"), ("6669","tpex")]
    """
    import requests, datetime
    result = []
    headers = {"User-Agent": "Mozilla/5.0"}

    # 上市(TWSE) — 每日股價一覽表
    try:
        date_str = datetime.date.today().strftime("%Y%m%d")
        url_twse = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data&date={date_str}"
        r = requests.get(url_twse, headers=headers, timeout=20)
        if r.status_code == 200 and r.content:
            df = pd.read_csv(__import__("io").StringIO(r.text))
            if df.empty:  # 假日或收盤前可能無資料,改用不帶日期版
                r2 = requests.get(
                    "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data",
                    headers=headers, timeout=20)
                df = pd.read_csv(__import__("io").StringIO(r2.text))
            id_col = [c for c in df.columns if "代號" in c or "Code" in c or "stock" in c.lower()]
            if id_col:
                for sid in df[id_col[0]].astype(str).str.strip():
                    if sid.isdigit() and 4 <= len(sid) <= 5:
                        result.append((sid, "twse"))
    except Exception:
        pass

    # 上櫃(TPEX) — 當日行情
    try:
        import datetime as _dt
        today = _dt.date.today()
        ymd = f"{today.year-1911}/{today.month:02d}/{today.day:02d}"
        url_tpex = f"https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&d={ymd}&se=EW&s=0,asc"
        r = requests.get(url_tpex, headers=headers, timeout=20)
        if r.status_code == 200:
            j = r.json()
            for row in j.get("aaData", []):
                sid = str(row[0]).strip()
                if sid.isdigit() and 4 <= len(sid) <= 5:
                    result.append((sid, "tpex"))
    except Exception:
        pass

    return result


# ── 步驟 B：對全市場做技術初篩(沿用 analyze_stock 邏輯，只用 yfinance) ──
def tech_prefilter(universe_ids, min_turnover=5e7, progress=None):
    """
    對全市場股票做技術面初篩，只保留「有潛力做多」的候選清單。
    universe_ids: list of (stock_id, market)
    回傳 list of stock_id（不含後綴）已通過初篩
    """
    import yfinance as yf, datetime as dt

    # 批次下載：一次最多 200 支，減少請求次數
    BATCH = 150
    passed = []
    total = len(universe_ids)
    processed = 0

    tickers_all = [
        sid + (".TW" if mkt == "twse" else ".TWO")
        for sid, mkt in universe_ids
    ]

    for i in range(0, total, BATCH):
        batch = tickers_all[i:i + BATCH]
        try:
            raw = yf.download(
                batch, period="4mo", interval="1d",
                group_by="ticker", auto_adjust=True,
                threads=True, progress=False, timeout=30
            )
        except Exception:
            processed += len(batch)
            continue

        for tw in batch:
            processed += 1
            if progress:
                progress(processed, total, tw)
            try:
                sub = raw[tw] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = sub[["Close", "Volume"]].dropna()
                if len(sub) < 65:
                    continue
                sub = add_indicators(sub)
                last = sub.iloc[-1]
                close = float(last["Close"])
                ma60 = float(last["MA60"])
                ma20 = float(last["MA20"])
                vol = float(last["Volume"])
                vol20 = float(last["VOL20"])
                if any(pd.isna(x) for x in [close, ma60, ma20, vol, vol20]):
                    continue
                turnover = close * vol20
                # 初篩條件：站上季線 + 流動性 + 股價正常 + 未爆量轉弱
                if (close > ma60
                        and ma20 > ma60
                        and turnover >= min_turnover
                        and close > 10
                        and not (vol > vol20 * 3
                                 and close < float(sub.iloc[-1]["Close"]))
                ):
                    sid = tw.replace(".TWO", "").replace(".TW", "")
                    passed.append(sid)
            except Exception:
                continue

    return passed


# ── 工具：把初篩後的 stock_id 清單轉成 FinMind 查詢用的格式 ─────────────
def _sid_to_tw(sid, universe_ids):
    """把純代號轉回帶市場後綴的格式，同時更新 NAME_MAP。"""
    mkt_map = {s: m for s, m in universe_ids}
    mkt = mkt_map.get(sid, "twse")
    return sid + (".TW" if mkt == "twse" else ".TWO")


# ======================================================================
# 5a-5. 全面分析：對同一批初篩候選，同時跑基本面 + 籌碼面
#   每支股票查 5 張表（財報三張 + 法人 + 融資券），約 5 次請求
#   150 支 ≈ 750 次請求，分兩批各 75 支可在免費上限內完成
# ======================================================================
def run_full_analysis(universe, token, exclude_ids=None, progress=None):
    """
    對 universe（帶後綴的清單）同時執行基本面 + 籌碼面分析。
    回傳 DataFrame，欄位涵蓋兩個面向的關鍵指標與訊號。
    """
    exclude_ids = set(exclude_ids or [])
    rows = []
    n = len(universe)

    for i, tw in enumerate(universe):
        sid = tw.replace(".TWO", "").replace(".TW", "")
        if progress:
            progress(i, n, sid)
        if sid in exclude_ids:
            continue

        # ── 基本面 ──────────────────────────────────────────
        try:
            fund_feats = fundamental_features(sid, token)
        except RuntimeError:
            raise
        except Exception:
            fund_feats = None

        fund_include, fund_B, fund_detail = fundamental_eval(fund_feats)

        # ── 籌碼面 ──────────────────────────────────────────
        try:
            chip_feats = chip_features(sid, token)
        except RuntimeError:
            raise
        except Exception:
            chip_feats = None

        chip_C, chip_B = chip_signals(chip_feats) if chip_feats else ([], [])

        # ── 分離模式判斷 ─────────────────────────────────────────
        has_fund = fund_include
        severe_risk = any("B9" in b for b in chip_B)
        has_chip = bool(chip_C) and not severe_risk

        if not has_fund and not has_chip:
            continue

        # ── 組合欄位 ─────────────────────────────────────────
        name = NAME_MAP.get(tw, "")
        close = round(chip_feats["c0"], 2) if chip_feats else None
        ret20 = round(chip_feats["ret20"] * 100, 1) if chip_feats else None
        vol_ratio = round(chip_feats["vol0"] / chip_feats["vol20"], 2) \
            if chip_feats and chip_feats.get("vol20") else None

        # 基本面摘要
        eps_last = round(fund_feats["eps_last"], 2) \
            if fund_feats and fund_feats.get("eps_last") is not None else None
        eps_ttm = round(fund_feats["eps_ttm"], 2) \
            if fund_feats and fund_feats.get("eps_ttm") is not None else None
        rev_yoy = round(fund_feats["rev3_yoy_mean"] * 100, 1) \
            if fund_feats and fund_feats.get("rev3_yoy_mean") is not None else None
        debt = round(fund_feats["debt_ratio"] * 100, 1) \
            if fund_feats and fund_feats.get("debt_ratio") is not None else None
        fund_pass = sum(1 for v in fund_detail.values() if v) if fund_detail else 0
        fund_total = len(fund_detail) if fund_detail else 0

        # 籌碼摘要
        chip_signal_str = "、".join(
            f"{c} {C_SIGNAL_NAMES[c]}" for c in chip_C
        ) if chip_C else ""
        chip_risk_str = "、".join(chip_B) if chip_B else ""

        # 綜合評級
        if has_fund and has_chip:
            grade = "⭐⭐ 雙面確認"
        elif has_fund:
            grade = "📊 基本面通過"
        else:
            grade = "🏦 籌碼訊號"

        rows.append({
            "代號": sid,
            "股票名稱": name,
            "收盤": close,
            "綜合評級": grade,
            "基本面通過": "✅" if has_fund else "—",
            "籌碼訊號": "✅" if has_chip else "—",
            "做多訊號": chip_signal_str,
            "籌碼風險": chip_risk_str if chip_risk_str else "無",
            "近季EPS": eps_last,
            "近四季EPS": eps_ttm,
            "近三月營收YoY%": rev_yoy,
            "負債比%": debt,
            "基本面通過項": f"{fund_pass}/{fund_total}" if fund_total else "—",
            "20日漲幅%": ret20,
            "量比": vol_ratio,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 排序：雙面確認 > 基本面 > 籌碼；同級內依籌碼訊號數排
    grade_order = {"⭐⭐ 雙面確認": 0, "📊 基本面通過": 1, "🏦 籌碼訊號": 2}
    df["_g"] = df["綜合評級"].map(grade_order)
    df["_cs"] = df["做多訊號"].str.count("、") + df["做多訊號"].str.len().gt(0).astype(int)
    df = df.sort_values(["_g", "_cs"], ascending=[True, False]) \
           .drop(columns=["_g", "_cs"]) \
           .reset_index(drop=True)
    return df


# ======================================================================
# 工具函式：Excel 產生 / 持股記錄讀寫
# ======================================================================
def df_to_excel_bytes(sheets: dict) -> bytes:
    """
    把多個 DataFrame 寫進同一個 Excel 檔，回傳 bytes 供 st.download_button 使用。
    sheets: {工作表名稱: DataFrame}
    """
    import io, openpyxl
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


# ── 持股記錄的預設空白結構 ────────────────────────────────────────────
HOLDINGS_COLS = ["代號", "股票名稱", "買進日期", "買進價", "張數(千股)"]
HISTORY_COLS  = ["代號", "股票名稱", "買進日期", "買進價",
                 "賣出日期", "賣出價", "張數(千股)",
                 "實現損益(元)", "報酬率%", "持有天數"]

def empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(columns=HOLDINGS_COLS)

def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLS)

def read_holdings_excel(file_bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀取持股 Excel，回傳 (持倉 df, 交易歷史 df)。"""
    import io
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    hold = pd.read_excel(xls, sheet_name="持倉") if "持倉" in xls.sheet_names else empty_holdings()
    hist = pd.read_excel(xls, sheet_name="交易歷史") if "交易歷史" in xls.sheet_names else empty_history()
    # 日期欄轉字串（避免顯示時間戳）
    for col in ["買進日期", "賣出日期"]:
        for df in [hold, hist]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return hold, hist

def sell_holding(hold: pd.DataFrame, hist: pd.DataFrame,
                 idx: int, sell_price: float, sell_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把持倉第 idx 列移到交易歷史，計算實現損益。"""
    row = hold.iloc[idx]
    shares = float(row.get("張數(千股)", 1)) * 1000   # 換算成股
    buy_price = float(row["買進價"])
    pnl = round((sell_price - buy_price) * shares, 0)
    ret = round((sell_price / buy_price - 1) * 100, 2)
    try:
        days = (pd.to_datetime(sell_date) - pd.to_datetime(row["買進日期"])).days
    except Exception:
        days = 0

    new_row = {
        "代號": row["代號"],
        "股票名稱": row.get("股票名稱", ""),
        "買進日期": row["買進日期"],
        "買進價": buy_price,
        "賣出日期": sell_date,
        "賣出價": sell_price,
        "張數(千股)": row.get("張數(千股)", 1),
        "實現損益(元)": pnl,
        "報酬率%": ret,
        "持有天數": days,
    }
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    hold = hold.drop(index=hold.index[idx]).reset_index(drop=True)
    return hold, hist

def history_summary(hist: pd.DataFrame) -> dict:
    """計算交易歷史的統計摘要。"""
    if hist.empty or "報酬率%" not in hist.columns:
        return {"總交易": 0, "獲利": 0, "虧損": 0, "勝率%": 0,
                "平均報酬%": 0, "總實現損益": 0}
    n = len(hist)
    win = int((hist["報酬率%"] > 0).sum())
    lose = int((hist["報酬率%"] <= 0).sum())
    avg_ret = round(float(hist["報酬率%"].mean()), 2)
    total_pnl = round(float(hist["實現損益(元)"].sum()), 0) if "實現損益(元)" in hist.columns else 0
    return {"總交易": n, "獲利": win, "虧損": lose,
            "勝率%": round(win/n*100, 1) if n else 0,
            "平均報酬%": avg_ret, "總實現損益": total_pnl}


# ----------------------------------------------------------------------
# 5b. Streamlit 介面版(有「一鍵執行」按鈕)
# ----------------------------------------------------------------------
def streamlit_main():
    import streamlit as st
    st.set_page_config(page_title="台股波段操作助手", page_icon="📈", layout="wide")
    st.title("📈 台股波段操作助手")
    st.caption("⚠️ 本工具為決策輔助,依歷史價量規律計算,不保證獲利、不構成投資建議。")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 今日選股(波段)", "💼 我的持股",
        "🔬 全面分析(Backer)", "🎯 模擬投資追蹤"])

    with tab1:
        import datetime as _dt

        # ── Session state ─────────────────────────────────────────────
        if "t1_pool" not in st.session_state:
            st.session_state["t1_pool"] = empty_pool()
        if "t1_cand" not in st.session_state:
            st.session_state["t1_cand"] = pd.DataFrame()

        # ── 股票池管理 ─────────────────────────────────────────────────
        with st.expander("⚙️ 股票池管理（點開查看/新增/刪除追蹤股票）"):
            pool = st.session_state["t1_pool"]

            # 上傳自訂股票池
            up_pool = st.file_uploader("上傳自訂股票池 Excel（可選）",
                                       type=["xlsx"], key="t1_pool_up")
            if up_pool:
                try:
                    loaded = read_pool_excel(up_pool.read())
                    st.session_state["t1_pool"] = loaded
                    pool = loaded
                    st.success(f"已載入 {len(pool)} 支自訂股票池。")
                except Exception as e:
                    st.error(f"讀取失敗：{e}")

            st.caption(f"目前股票池共 **{len(pool)}** 支。可新增、刪除或下載。")
            st.dataframe(pool, use_container_width=True, hide_index=True, height=220)

            # 新增股票
            c1, c2, c3, c4 = st.columns(4)
            new_code = c1.text_input("股票代號（含後綴，如 2330.TW）", key="t1_new_code")
            new_name = c2.text_input("公司名稱", key="t1_new_name")
            new_ind  = c3.text_input("產業", key="t1_new_ind")
            new_note = c4.text_input("備註", key="t1_new_note")
            if st.button("➕ 新增到股票池", key="t1_add_pool"):
                code = new_code.strip().upper()
                if not code:
                    st.warning("請填入股票代號。")
                elif code in pool["代號"].values:
                    st.warning(f"{code} 已在股票池中。")
                else:
                    new_row = {"代號": code, "股票名稱": new_name,
                               "產業": new_ind, "備註": new_note}
                    st.session_state["t1_pool"] = pd.concat(
                        [pool, pd.DataFrame([new_row])], ignore_index=True)
                    st.success(f"已新增 {code}。")
                    pool = st.session_state["t1_pool"]

            # 刪除股票
            del_code = st.text_input("輸入要刪除的股票代號", key="t1_del_code")
            if st.button("🗑️ 從股票池刪除", key="t1_del_pool"):
                code = del_code.strip().upper()
                before = len(pool)
                st.session_state["t1_pool"] = pool[pool["代號"] != code].reset_index(drop=True)
                if len(st.session_state["t1_pool"]) < before:
                    st.success(f"已刪除 {code}。")
                else:
                    st.warning(f"找不到 {code}。")

            # 下載股票池
            pool_bytes = df_to_excel_bytes({"股票池": st.session_state["t1_pool"]})
            st.download_button("📥 下載股票池 Excel",
                               data=pool_bytes,
                               file_name="自訂股票池.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True, key="t1_dl_pool")

        # ── 執行篩選 ───────────────────────────────────────────────────
        st.write("按下按鈕，程式會掃描股票池並選出最多 10 檔符合波段條件的標的。")
        if st.button("🚀 執行今日篩選", type="primary", use_container_width=True):
            pool_tickers = st.session_state["t1_pool"]["代號"].dropna().astype(str).tolist()
            with st.spinner("抓取資料與計算中…(約 1 分鐘)"):
                cand, _ = run_screen(universe=pool_tickers)
            if cand.empty:
                st.warning("今日沒有符合條件的標的。整體偏弱時，空手也是一種紀律。")
                st.session_state["t1_cand"] = pd.DataFrame()
            else:
                st.success(f"找到 {len(cand)} 檔候選。✅ 表示同時出現進場訊號。")
                st.dataframe(cand, use_container_width=True, hide_index=True)
                with st.expander("📖 欄位說明"):
                    st.markdown(COLUMN_HELP)
                st.info("「建議停損 / 目標價」為以 1:2 風險報酬比估算的參考值。")
                st.session_state["t1_cand"] = cand
                today_str = _dt.date.today().strftime("%Y%m%d")
                excel_bytes = df_to_excel_bytes({"今日選股": cand})
                st.download_button(
                    label="📥 下載今日選股 Excel",
                    data=excel_bytes,
                    file_name=f"選股結果_{today_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, key="t1_dl_result",
                )

        # ── 加入模擬投資 ──────────────────────────────────────────────
        cand = st.session_state.get("t1_cand", pd.DataFrame())
        if not cand.empty:
            st.divider()
            st.markdown("#### 📌 加入模擬投資追蹤")
            st.caption("選好股票後按按鈕，會加入分頁 4 的模擬追蹤清單。")
            options = [
                f"{r['代號']} {r.get('股票名稱','')} (分數:{r['分數']}"
                f"{'  ✅' if r.get('進場訊號')=='✅' else ''})"
                for _, r in cand.iterrows()
            ]
            selected = st.multiselect("選擇要加入模擬的股票（可多選）",
                                      options=options,
                                      default=options,
                                      key="t1_sim_sel")
            if st.button("➕ 加入模擬投資追蹤", use_container_width=True, key="t1_add_sim"):
                sel_set = set(o.split()[0] for o in selected)
                today = _dt.date.today().isoformat()
                if "sim_active" not in st.session_state:
                    st.session_state["sim_active"] = empty_sim()
                existing_keys = set(
                    zip(st.session_state["sim_active"]["日期"],
                        st.session_state["sim_active"]["代號"])
                ) if not st.session_state["sim_active"].empty else set()
                added = 0
                pool_map = {r["代號"]: r.get("產業", "") for _, r in st.session_state["t1_pool"].iterrows()}
                for _, row in cand.iterrows():
                    sid = str(row["代號"])
                    if sid not in sel_set:
                        continue
                    if (today, sid) in existing_keys:
                        continue
                    new_sim = {
                        "日期": today, "代號": sid,
                        "股票名稱": row.get("股票名稱", ""),
                        "產業": pool_map.get(sid, ""),
                        "分數": row.get("分數", ""),
                        "有進場訊號": "✅" if row.get("進場訊號") == "✅" else "—",
                        "進場價": row.get("收盤", ""),
                        "停損價": row.get("建議停損", ""),
                        "目標價": row.get("目標價", ""),
                        "入選理由": row.get("理由", ""),
                        "狀態": "追蹤中",
                        "結算價": "", "損益%": "", "持有天數": "", "結算原因": "",
                    }
                    st.session_state["sim_active"] = pd.concat(
                        [st.session_state["sim_active"], pd.DataFrame([new_sim])],
                        ignore_index=True)
                    added += 1
                st.success(f"已加入 {added} 筆到模擬投資追蹤（分頁 4）。")

    with tab2:
        import datetime as _dt
        st.write("管理持股記錄、追蹤損益，並查看買賣歷史與統計摘要。")
        st.caption("每個人各自維護一個 Excel 檔（例如 持股記錄_Annie.xlsx），上傳後操作，完成後下載儲存。")

        # ── Session state 初始化 ─────────────────────────────────────────
        if "h2_hold" not in st.session_state:
            st.session_state["h2_hold"] = empty_holdings()
        if "h2_hist" not in st.session_state:
            st.session_state["h2_hist"] = empty_history()
        if "h2_loaded" not in st.session_state:
            st.session_state["h2_loaded"] = False

        # ── 上傳記錄 ─────────────────────────────────────────────────────
        with st.expander("📂 上傳持股記錄 Excel（首次使用或換裝置時）", expanded=not st.session_state["h2_loaded"]):
            uploaded = st.file_uploader("上傳你的持股 Excel 檔", type=["xlsx"], key="h2_upload")
            if uploaded:
                try:
                    hold, hist = read_holdings_excel(uploaded.read())
                    st.session_state["h2_hold"] = hold
                    st.session_state["h2_hist"] = hist
                    st.session_state["h2_loaded"] = True
                    st.success(f"✅ 載入完成：{len(hold)} 筆持倉、{len(hist)} 筆交易歷史。")
                except Exception as e:
                    st.error(f"讀取失敗：{e}")
            st.caption("第一次使用不需要上傳，直接在下方新增買進記錄即可。")

        hold = st.session_state["h2_hold"]
        hist = st.session_state["h2_hist"]

        # ── 新增買進 ─────────────────────────────────────────────────────
        with st.expander("➕ 新增買進記錄"):
            c1, c2, c3, c4 = st.columns(4)
            new_sid  = c1.text_input("股票代號（例如 2330.TW）", key="h2_sid")
            new_date = c2.date_input("買進日期", value=_dt.date.today(), key="h2_date")
            new_price = c3.number_input("買進價（元）", min_value=0.0, step=0.1, key="h2_price")
            new_lots  = c4.number_input("張數", min_value=0.1, step=1.0, value=1.0, key="h2_lots")
            if st.button("✅ 確認買進", use_container_width=True, key="h2_add"):
                sid = new_sid.strip().upper()
                if not sid:
                    st.warning("請輸入股票代號。")
                elif new_price <= 0:
                    st.warning("買進價必須大於 0。")
                else:
                    tw = sid if sid.endswith((".TW", ".TWO")) else sid + ".TW"
                    name = NAME_MAP.get(tw, "")
                    new_row = {
                        "代號": tw, "股票名稱": name,
                        "買進日期": str(new_date), "買進價": new_price,
                        "張數(千股)": new_lots,
                    }
                    st.session_state["h2_hold"] = pd.concat(
                        [hold, pd.DataFrame([new_row])], ignore_index=True
                    )
                    st.success(f"已新增 {tw}（{name}），買進價 {new_price}，{new_lots} 張。")
                    hold = st.session_state["h2_hold"]

        # ── 目前持倉 ─────────────────────────────────────────────────────
        st.markdown("#### 📋 目前持倉")
        if hold.empty:
            st.info("尚無持倉記錄，請新增買進或上傳 Excel 檔。")
        else:
            # 抓最新價格計算損益
            tickers = hold["代號"].dropna().astype(str).tolist()
            with st.spinner("更新現價中…"):
                try:
                    hist_prices = fetch_history(tickers, period="1mo")
                except Exception:
                    hist_prices = {}

            display_rows = []
            for i, row in hold.iterrows():
                tw = str(row["代號"])
                buy_p = float(row["買進價"])
                lots  = float(row.get("張數(千股)", 1))
                cur_p = None
                rsi_v = None
                action_v = "—"
                trailing_v = None
                if tw in hist_prices:
                    df_h = hist_prices[tw]
                    if not df_h.empty:
                        cur_p = round(float(df_h["Close"].iloc[-1]), 2)
                        ha = holding_action(df_h, buy_p)
                        if ha:
                            rsi_v = ha["rsi"]
                            action_v = ha["action"]
                            trailing_v = ha["trailing_stop"]

                pnl_pct = round((cur_p / buy_p - 1) * 100, 2) if cur_p else None
                pnl_amt = round((cur_p - buy_p) * lots * 1000, 0) if cur_p else None
                display_rows.append({
                    "序號": i,
                    "代號": tw,
                    "股票名稱": row.get("股票名稱", ""),
                    "買進日期": row.get("買進日期", ""),
                    "買進價": buy_p,
                    "張數": lots,
                    "現價": cur_p,
                    "損益%": pnl_pct,
                    "損益金額(元)": pnl_amt,
                    "RSI": rsi_v,
                    "移動停利": trailing_v,
                    "建議動作": action_v,
                })

            display_df = pd.DataFrame(display_rows)

            # 用顏色標示建議
            def color_action(val):
                if val == "出場": return "background-color:#FCEBEB"
                if val == "減碼": return "background-color:#FAEEDA"
                return ""

            st.dataframe(
                display_df.drop(columns=["序號"]).style.map(color_action, subset=["建議動作"]),
                use_container_width=True, hide_index=True
            )

            # ── 賣出操作 ──────────────────────────────────────────────
            st.markdown("**賣出持股**")
            sell_options = [
                f"{row['代號']} — 買進 {row['買進價']} 元（{row.get('張數(千股)',1)} 張）"
                for _, row in hold.iterrows()
            ]
            sel_idx = st.selectbox("選擇要賣出的持股", range(len(sell_options)),
                                   format_func=lambda i: sell_options[i], key="h2_sell_sel")
            sc1, sc2, sc3 = st.columns(3)
            sell_price = sc1.number_input("賣出價（元）", min_value=0.0, step=0.1, key="h2_sell_price")
            sell_date  = sc2.date_input("賣出日期", value=_dt.date.today(), key="h2_sell_date")
            if sc3.button("✅ 確認賣出", use_container_width=True, key="h2_sell_btn"):
                if sell_price <= 0:
                    st.warning("賣出價必須大於 0。")
                else:
                    new_hold, new_hist = sell_holding(
                        hold, hist, sel_idx, sell_price, str(sell_date)
                    )
                    st.session_state["h2_hold"] = new_hold
                    st.session_state["h2_hist"] = new_hist
                    sold = sell_options[sel_idx].split("—")[0].strip()
                    buy_p2 = float(hold.iloc[sel_idx]["買進價"])
                    ret2 = round((sell_price / buy_p2 - 1) * 100, 2)
                    emoji = "🟢" if ret2 > 0 else "🔴"
                    st.success(f"{emoji} {sold} 已賣出，報酬率 {ret2:+.2f}%。記得下載更新後的記錄！")
                    hold = st.session_state["h2_hold"]
                    hist = st.session_state["h2_hist"]

        # ── 交易歷史 ─────────────────────────────────────────────────────
        st.markdown("#### 📜 交易歷史")
        if hist.empty:
            st.info("尚無交易歷史。賣出持股後，記錄會出現在這裡。")
        else:
            def color_ret(val):
                try:
                    v = float(val)
                    if v > 0: return "color:#0F6E56; font-weight:500"
                    if v < 0: return "color:#A32D2D; font-weight:500"
                except Exception:
                    pass
                return ""

            st.dataframe(
                hist.style.map(color_ret, subset=["報酬率%"] if "報酬率%" in hist.columns else []),
                use_container_width=True, hide_index=True
            )

            # 統計摘要
            smry = history_summary(hist)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("總交易次數", smry["總交易"])
            m2.metric("勝率", f"{smry['勝率%']}%")
            m3.metric("獲利 / 虧損", f"{smry['獲利']} / {smry['虧損']}")
            m4.metric("平均報酬率", f"{smry['平均報酬%']:+.2f}%")
            m5.metric("總實現損益", f"{smry['總實現損益']:+,.0f} 元")

        # ── 下載 ─────────────────────────────────────────────────────────
        st.divider()
        excel_bytes = df_to_excel_bytes({
            "持倉": st.session_state["h2_hold"],
            "交易歷史": st.session_state["h2_hist"],
        })
        st.download_button(
            label="📥 下載持股記錄 Excel（請定期儲存，換裝置時上傳此檔）",
            data=excel_bytes,
            file_name="持股記錄.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with tab3:
        st.write("Backer 版全市場掃描：一次下載全市場資料，同時做基本面 + 籌碼面全面分析。")
        st.markdown("""
**升級優勢（Backer 方案）**
- 只需 **6~7 次** API 請求即可取得全市場 ~1,800 支股票的所有資料
- 從原本分兩批等 2 小時，縮短為 **5~10 分鐘**跑完全市場
- 不需要技術初篩步驟，直接全市場掃描
        """)
        st.caption("⚠️ A8/B7(董監質押)與 B11(處置/警示股)FinMind 無資料，請手動填入排除清單。")

        # ── Token ─────────────────────────────────────────────────────
        token_a = ""
        try:
            token_a = st.secrets.get("FINMIND_TOKEN", "")
        except Exception:
            pass
        if not token_a:
            token_a = st.text_input("FinMind API token (Backer)", type="password", key="token_a")

        col_a1, col_a2 = st.columns(2)
        min_turn_a = col_a1.number_input(
            "技術篩選：最低日均成交額（萬元）",
            min_value=1000, max_value=50000, value=5000, step=1000, key="min_turn_a"
        )
        excl_a = col_a2.text_input(
            "手動排除清單（逗號分隔代號）",
            key="excl_a", help="例如：2618,3034（董監質押高/警示/處置股）"
        )

        # ── Session state ─────────────────────────────────────────────
        for k, v in [("fa_results", pd.DataFrame()), ("fa_scanned", 0)]:
            if k not in st.session_state:
                st.session_state[k] = v

        st.divider()
        if st.button("🔬 執行全市場全面分析（Backer）",
                     type="primary", use_container_width=True, key="fa_backer_btn"):
            if not token_a:
                st.warning("請先填入 FinMind token。")
            else:
                excl_ids = [x.strip() for x in excl_a.replace("，", ",").split(",") if x.strip()]
                status_box = st.empty()
                prog = st.progress(0.0)
                def _cb(msg, step, total):
                    prog.progress(min(step / total, 1.0), text=msg)
                    status_box.info(msg)
                try:
                    res, candidates, errors = run_full_analysis_backer(
                        token_a,
                        exclude_ids=excl_ids,
                        min_turnover=min_turn_a * 10000,
                        progress=_cb
                    )
                    prog.empty(); status_box.empty()
                    st.session_state["fa_results"] = res
                    st.session_state["fa_scanned"] = len(candidates)
                    if res.empty:
                        st.warning(
                            f"技術初篩通過 {len(candidates)} 支，但全面分析後無符合條件的股票。"
                            + (f"\n\n診斷：前 5 筆錯誤：{errors[:5]}" if errors else "")
                        )
                    else:
                        st.success(f"全市場掃描完成！技術初篩 {len(candidates)} 支 → 找到 {len(res)} 檔符合條件。"
                                   + (f"（{len(errors)} 筆分析例外，可展開診斷查看）" if errors else ""))
                        if errors:
                            with st.expander(f"⚠️ 診斷：{len(errors)} 筆分析例外（點開查看）"):
                                st.text("\n".join(errors[:30]))
                except RuntimeError as e:
                    prog.empty(); status_box.empty()
                    st.error(str(e))

        fa_res = st.session_state.get("fa_results", pd.DataFrame())
        if not fa_res.empty:
            import datetime as _dt
            n_both = int((fa_res["綜合評級"] == "⭐⭐ 雙面確認").sum())
            n_fund = int((fa_res["綜合評級"] == "📊 基本面通過").sum())
            n_chip = int((fa_res["綜合評級"] == "🏦 籌碼訊號").sum())

            st.divider()
            st.markdown(f"#### 🎯 分析結果：共 {len(fa_res)} 檔")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("通過總檔數", len(fa_res))
            sm2.metric("⭐⭐ 雙面確認", n_both)
            sm3.metric("📊 僅基本面", n_fund)
            sm4.metric("🏦 僅籌碼", n_chip)

            df_both = fa_res[fa_res["綜合評級"] == "⭐⭐ 雙面確認"]
            df_fund = fa_res[fa_res["綜合評級"] == "📊 基本面通過"]
            df_chip = fa_res[fa_res["綜合評級"] == "🏦 籌碼訊號"]

            if not df_both.empty:
                st.markdown(f"**⭐⭐ 雙面確認（{len(df_both)} 檔）**")
                st.dataframe(df_both, use_container_width=True, hide_index=True)
            if not df_fund.empty:
                st.markdown(f"**📊 僅基本面通過（{len(df_fund)} 檔）**")
                st.dataframe(df_fund, use_container_width=True, hide_index=True)
            if not df_chip.empty:
                st.markdown(f"**🏦 僅有籌碼訊號（{len(df_chip)} 檔）**")
                st.dataframe(df_chip, use_container_width=True, hide_index=True)

            today_str = _dt.date.today().strftime("%Y%m%d")
            summary_df = pd.DataFrame([{
                "分析日期": today_str, "通過總檔數": len(fa_res),
                "雙面確認": n_both, "僅基本面": n_fund, "僅籌碼": n_chip,
            }])
            st.download_button(
                "📥 下載全面分析結果 Excel",
                data=df_to_excel_bytes({"篩選結果": fa_res, "掃描摘要": summary_df}),
                file_name=f"全面分析_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="fa_dl"
            )

        st.divider()
        with st.expander("🔎 診斷工具"):
            dc1, dc2 = st.columns(2)
            debug_id_a = dc1.text_input("股票代號", value="2330", key="debug_id_a")
            if dc1.button("查財報欄位", key="debug_btn_a"):
                if token_a:
                    with st.spinner("抓取中…"):
                        result = fundamental_debug(debug_id_a, token_a)
                    for label, data in result.items():
                        with st.expander(f"📋 {label}"):
                            if isinstance(data, pd.DataFrame):
                                st.dataframe(data, use_container_width=True, hide_index=True)
                            else:
                                st.error(str(data))
            if dc2.button("🔎 檢查 FinMind 用量", key="usage_btn_a"):
                if token_a:
                    used, limit = fm_usage(token_a)
                    if limit is None:
                        st.error("查不到用量，token 可能無效。")
                    elif limit >= 600:
                        st.success(f"用量 {used} / 上限 {limit} ✅")
                    else:
                        st.warning(f"用量 {used} / 上限 {limit}，請確認信箱驗證與方案狀態。")

    with tab4:
        import datetime as _dt
        st.write("記錄分頁 1 的推薦結果，追蹤模擬損益，分析程式預測能力。")
        st.caption("資料存在你的 Excel 檔案中，換裝置時上傳即可延續記錄。")

        # ── Session state ─────────────────────────────────────────────
        if "sim_active" not in st.session_state:
            st.session_state["sim_active"] = empty_sim()
        if "sim_done" not in st.session_state:
            st.session_state["sim_done"] = empty_sim()

        # ── 上傳記錄 ─────────────────────────────────────────────────
        with st.expander("📂 上傳模擬投資記錄 Excel", expanded=False):
            up_sim = st.file_uploader("上傳記錄檔", type=["xlsx"], key="sim_upload")
            if up_sim:
                try:
                    act, dn = read_sim_excel(up_sim.read())
                    st.session_state["sim_active"] = act
                    st.session_state["sim_done"] = dn
                    st.success(f"載入成功：追蹤中 {len(act)} 筆，已完成 {len(dn)} 筆。")
                except Exception as e:
                    st.error(f"讀取失敗：{e}")

        # ── 更新價格 / 自動結算 ───────────────────────────────────────
        active = st.session_state["sim_active"]
        done = st.session_state["sim_done"]

        if not active.empty:
            if st.button("🔄 更新現價並自動結算", use_container_width=True, key="sim_update"):
                with st.spinner("抓取最新價格中…"):
                    result = update_sim_prices(active)
                if isinstance(result, tuple):
                    new_active, triggered = result
                    if not triggered.empty:
                        st.session_state["sim_done"] = pd.concat(
                            [done, triggered], ignore_index=True)
                        done = st.session_state["sim_done"]
                        st.success(f"自動結算 {len(triggered)} 筆（停損或達目標）。")
                    st.session_state["sim_active"] = new_active
                    active = new_active

        # ── 追蹤中的部位 ──────────────────────────────────────────────
        st.markdown("#### 📋 追蹤中的模擬部位")
        if active.empty:
            st.info("尚無追蹤中的記錄。在分頁 1 執行篩選後，勾選股票加入此清單。")
        else:
            # 顯示附現價的追蹤表
            tickers = active["代號"].dropna().astype(str).tolist()
            try:
                hist_p = fetch_history(tickers, period="5d")
            except Exception:
                hist_p = {}
            display = active.copy()
            display["現價"] = display["代號"].apply(
                lambda t: round(float(hist_p[t]["Close"].iloc[-1]), 2)
                if t in hist_p and not hist_p[t].empty else None
            )
            display["目前損益%"] = display.apply(
                lambda r: round((r["現價"] / float(r["進場價"]) - 1) * 100, 2)
                if pd.notna(r["現價"]) and pd.notna(r["進場價"]) and float(r["進場價"]) > 0
                else None, axis=1
            )
            show_cols = ["日期", "代號", "股票名稱", "分數", "有進場訊號",
                         "進場價", "停損價", "目標價", "現價", "目前損益%", "狀態"]
            st.dataframe(display[[c for c in show_cols if c in display.columns]],
                         use_container_width=True, hide_index=True)

            # 手動結算
            with st.expander("✏️ 手動結算某筆"):
                if not active.empty:
                    opts = [f"{r['日期']} {r['代號']} {r.get('股票名稱','')} 進場:{r['進場價']}"
                            for _, r in active.iterrows()]
                    sel_i = st.selectbox("選擇要結算的筆", range(len(opts)),
                                        format_func=lambda i: opts[i], key="sim_manual_sel")
                    mc1, mc2, mc3 = st.columns(3)
                    manual_price = mc1.number_input("結算價", min_value=0.0, step=0.1, key="sim_m_price")
                    manual_reason = mc2.selectbox("結算原因", ["停損出場", "達目標出場", "主動出場"], key="sim_m_reason")
                    if mc3.button("確認結算", use_container_width=True, key="sim_m_btn"):
                        if manual_price > 0:
                            row = active.iloc[sel_i].copy()
                            entry = float(row["進場價"]) if pd.notna(row["進場價"]) else 0
                            pnl = round((manual_price / entry - 1) * 100, 2) if entry > 0 else 0
                            try:
                                days = (dt.date.today() - pd.to_datetime(row["日期"]).date()).days
                            except Exception:
                                days = 0
                            row["狀態"] = "已完成"; row["結算價"] = manual_price
                            row["損益%"] = pnl; row["持有天數"] = days
                            row["結算原因"] = manual_reason
                            st.session_state["sim_done"] = pd.concat(
                                [done, pd.DataFrame([row])], ignore_index=True)
                            st.session_state["sim_active"] = active.drop(
                                index=active.index[sel_i]).reset_index(drop=True)
                            st.success(f"已結算，損益 {pnl:+.2f}%。")

        # ── 已完成紀錄 ────────────────────────────────────────────────
        st.markdown("#### 📜 已完成紀錄")
        done = st.session_state["sim_done"]
        if done.empty:
            st.info("尚無已完成的模擬記錄。")
        else:
            def _color_pnl(val):
                try:
                    v = float(val)
                    return "color:#0F6E56;font-weight:500" if v > 0 else "color:#A32D2D;font-weight:500"
                except Exception:
                    return ""
            st.dataframe(
                done.style.map(_color_pnl, subset=["損益%"] if "損益%" in done.columns else []),
                use_container_width=True, hide_index=True
            )

        # ── 程式能力分析 ──────────────────────────────────────────────
        st.markdown("#### 📊 程式能力分析")
        done = st.session_state["sim_done"]
        if done.empty or len(done) < 3:
            st.info("需要至少 3 筆已完成記錄才能進行分析。繼續累積數據中！")
        else:
            analysis = sim_analysis(st.session_state["sim_active"], done)
            if analysis:
                # 整體統計
                smry = analysis.get("整體", {})
                if smry:
                    st.markdown("**整體統計**")
                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("總筆數", smry["總筆數"])
                    mc2.metric("勝率", f"{smry['勝率%']}%")
                    mc3.metric("平均報酬", f"{smry['平均報酬%']:+.2f}%")
                    mc4.metric("最大獲利", f"{smry['最大獲利%']:+.2f}%")
                    mc5.metric("最大虧損", f"{smry['最大虧損%']:+.2f}%")
                    mc6.metric("平均持有天", f"{smry['平均持有天數']}天")

                # 分數分組
                if "分數分組" in analysis:
                    st.markdown("**分數高低 vs 勝率（驗證高分是否更準）**")
                    st.dataframe(analysis["分數分組"], use_container_width=True, hide_index=True)

                # 進場訊號效果
                if "進場訊號效果" in analysis:
                    st.markdown("**有無 ✅ 進場訊號 vs 勝率**")
                    st.dataframe(analysis["進場訊號效果"], use_container_width=True, hide_index=True)

                # 產業分析
                if "產業分析" in analysis:
                    st.markdown("**各產業勝率排名**")
                    st.dataframe(analysis["產業分析"], use_container_width=True, hide_index=True)

                # 月度趨勢
                if "月度趨勢" in analysis:
                    st.markdown("**月度勝率趨勢（程式在不同市場環境的表現）**")
                    st.dataframe(analysis["月度趨勢"], use_container_width=True, hide_index=True)

        # ── 下載 / 上傳 ───────────────────────────────────────────────
        st.divider()
        sim_excel = df_to_excel_bytes({
            "追蹤中": st.session_state["sim_active"],
            "已完成": st.session_state["sim_done"],
        })
        st.download_button(
            "📥 下載模擬投資記錄 Excel",
            data=sim_excel,
            file_name="模擬投資記錄.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key="sim_dl"
        )

def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        streamlit_main()
    else:
        cli_main()
