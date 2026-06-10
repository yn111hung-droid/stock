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


# ----------------------------------------------------------------------
# 5b. Streamlit 介面版(有「一鍵執行」按鈕)
# ----------------------------------------------------------------------
def streamlit_main():
    import streamlit as st
    st.set_page_config(page_title="台股波段操作助手", page_icon="📈", layout="wide")
    st.title("📈 台股波段操作助手")
    st.caption("⚠️ 本工具為決策輔助,依歷史價量規律計算,不保證獲利、不構成投資建議。")

    tab1, tab2 = st.tabs(["🔍 今日選股", "💼 我的持股"])

    with tab1:
        st.write("按下按鈕,程式會掃描股票池並選出最多 10 檔符合波段條件的標的。")
        if st.button("🚀 執行今日篩選", type="primary", use_container_width=True):
            with st.spinner("抓取資料與計算中…(約 1 分鐘)"):
                cand, _ = run_screen()
            if cand.empty:
                st.warning("今日沒有符合條件的標的。整體偏弱時,空手也是一種紀律。")
            else:
                st.success(f"找到 {len(cand)} 檔候選。✅ 表示同時出現進場訊號。")
                st.dataframe(cand, use_container_width=True, hide_index=True)
                with st.expander("📖 欄位說明(點開看每個欄位的意思與「分數」怎麼算)"):
                    st.markdown(COLUMN_HELP)
                st.info("「建議停損 / 目標價」為以 1:2 風險報酬比估算的參考值,請依自身資金配置調整。")

    with tab2:
        st.write("輸入持股(代號、買進價),按按鈕取得 續抱 / 減碼 / 出場 建議。")
        default = pd.DataFrame({"代號": ["2330.TW"], "買進價": [0.0]})
        edited = st.data_editor(default, num_rows="dynamic", use_container_width=True,
                                key="holdings")
        if st.button("📊 檢視我的持股", use_container_width=True):
            holds = [(str(r["代號"]).strip(), float(r["買進價"]))
                     for _, r in edited.iterrows()
                     if str(r["代號"]).strip() and float(r["買進價"]) > 0]
            if not holds:
                st.warning("請至少輸入一筆有效的代號與買進價。")
            else:
                with st.spinner("計算中…"):
                    hist = fetch_history([h[0] for h in holds])
                    out = []
                    for t, ep in holds:
                        a = holding_action(hist.get(t), ep) if t in hist else None
                        if a:
                            out.append({"代號": t, "現價": a["close"], "買進價": ep,
                                        "損益%": a["pnl_pct"], "RSI": a["rsi"],
                                        "移動停利": a["trailing_stop"],
                                        "建議": a["action"], "理由": a["reason"]})
                if out:
                    st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)


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
