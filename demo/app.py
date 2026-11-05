import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from fredapi import Fred
from google import genai
from google.genai import types


# 頁面基本配置
st.set_page_config(page_title="總體經濟與金融即時儀表板", layout="wide")
st.title("🌐 全球總體經濟與金融指標即時儀表板")

# 側邊欄：輸入 API Key 與刷新控制
st.sidebar.header("⚙️ 設定與金鑰")

# 優先使用 Streamlit Secrets，若無則讓使用者在側邊欄輸入
if "FRED_API_KEY" in st.secrets:
    fred_key = st.secrets["FRED_API_KEY"]
    st.sidebar.success("已自動讀取 Secrets 中的 API Key！")
else:
    fred_key = st.sidebar.text_input("請輸入你的 FRED API Key:", type="password")

if st.sidebar.button("手動刷新數據"):
    st.cache_data.clear()
    st.session_state.pop("ai_macro_summary", None)


# 1. 抓取 FRED 數據（快取 1 小時）
@st.cache_data(ttl=3600)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    data = {
        "CPI 物價指數": fred.get_series("CPIAUCSL"),
        "10年期美債殖利率": fred.get_series("DGS10"),
        "2年期美債殖利率": fred.get_series("DGS2"),
        "美國失業率": fred.get_series("UNRATE"),
        "聯邦基金利率": fred.get_series("FEDFUNDS"),
        "日本10年期國債殖利率": fred.get_series("IRLTLT01JPM156N"),
    }

    raw_df = pd.DataFrame(data).sort_index()

    # 保留尚未補值的原始觀測日期，供 AI 摘要準確標示資料時間。
    latest_points = {}
    for name in (
        "CPI 物價指數",
        "美國失業率",
        "10年期美債殖利率",
        "2年期美債殖利率",
        "聯邦基金利率",
    ):
        observations = raw_df[name].dropna()
        if not observations.empty:
            latest_points[name] = {
                "value": float(observations.iloc[-1]),
                "date": observations.index[-1].strftime("%Y-%m-%d"),
            }

    paired_yields = raw_df[["10年期美債殖利率", "2年期美債殖利率"]].dropna()
    if not paired_yields.empty:
        latest_pair = paired_yields.iloc[-1]
        latest_points["10Y-2Y美債殖利率利差"] = {
            "value": float(latest_pair.iloc[0] - latest_pair.iloc[1]),
            "date": paired_yields.index[-1].strftime("%Y-%m-%d"),
        }

    df = raw_df

    # 在相同觀測日期計算利差，避免跨日期拼接造成不準確。
    # DGS10 和 DGS2 的單位均為百分比，相減後單位為「百分點」。
    spread_on_observation_dates = (
        df["10年期美債殖利率"] - df["2年期美債殖利率"]
    )

    # 與原版一致：較低頻的總體指標沿用最近一期數值，以便統一作圖。
    df = df.ffill()
    # 利差仍保留雙方都有原始觀測值的日期，不把補值日期當成新資料。
    df["10Y-2Y美債殖利率利差"] = spread_on_observation_dates

    # 原版 .tail(365) 是 365 筆觀測值；這裡改為真正的最近一年。
    one_year_ago = pd.Timestamp.today().normalize() - pd.DateOffset(years=1)
    return df.loc[df.index >= one_year_ago], latest_points


# 2. 抓取金融市場數據（快取 10 分鐘）
@st.cache_data(ttl=600)
def get_market_data():
    tickers = {
        "S&P 500": "^GSPC",
        "台灣加權指數": "^TWII",
        "布蘭特原油": "BZ=F",
        "美元/台幣匯率": "TWD=X",
        "黃金期貨價格": "GC=F",
    }
    df = yf.download(list(tickers.values()), period="1y")["Close"]
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    return df


def generate_macro_summary(api_key, latest_points):
    """以已下載的 FRED 最新值請 Gemini 產生約 200 字的繁體中文短評。"""
    labels_and_units = {
        "CPI 物價指數": "指數點（不是通膨年增率）",
        "美國失業率": "%",
        "10年期美債殖利率": "%",
        "2年期美債殖利率": "%",
        "聯邦基金利率": "%",
    }
    required = set(labels_and_units)
    missing = required.difference(latest_points)
    if missing:
        raise ValueError(f"FRED 缺少資料：{', '.join(sorted(missing))}")

    lines = []
    for name, unit in labels_and_units.items():
        item = latest_points[name]
        lines.append(f"- {name}：{item['value']:.2f} {unit}（觀測日期 {item['date']}）")

    if "10Y-2Y美債殖利率利差" in latest_points:
        item = latest_points["10Y-2Y美債殖利率利差"]
        lines.append(
            f"- 10Y－2Y 利差：{item['value']:+.2f} 個百分點"
            f"（同日觀測：{item['date']}）"
        )

    prompt = "\n".join([
        "你是一位中立、嚴謹的總體經濟分析師。",
        "請只根據下列 FRED 美國經濟指標，撰寫約 200 字（180～220 個繁體中文字）的總體經濟短評。",
        "使用臺灣繁體中文，文字自然、客觀、適合大學生的課堂簡報。",
        "要簡要解讀物價、就業、政策利率及長短期殖利率利差的意義。",
        "CPI 是物價指數水準，不得直接把它當作年增率。",
        "不同指標觀測日期不同，不能宣稱它們是同一天的即時報價。",
        "沒有提供前期比較值，不要聲稱指標升降或經濟已進入衰退。",
        "不臆測未提供的數據、不給個股或買賣投資建議。",
        "只輸出一段短評，不要標題、條列、Markdown 或資料來源清單。",
        "",
        "最新可用資料：",
        *lines,
    ])

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=1200),
    )
    summary = (response.text or "").strip()
    if not summary:
        raise RuntimeError("模型沒有回傳可顯示的文字")
    return summary


# 頁面渲染邏輯
if fred_key:
    try:
        macro_df, latest_points = get_fred_data(fred_key)
        market_df = get_market_data()

        # 第一區塊：原本的四張核心 KPI 卡片
        st.subheader("📊 關鍵總體經濟指標 (FRED)")
        col1, col2, col3, col4 = st.columns(4)

        latest_cpi = macro_df["CPI 物價指數"].dropna().iloc[-1]
        col1.metric("CPI 物價指數", f"{latest_cpi:.1f}")

        rate_values = macro_df["10年期美債殖利率"].dropna()
        latest_rate = rate_values.iloc[-1]
        rate_prev = rate_values.iloc[-2]
        col2.metric(
            "10年美債殖利率",
            f"{latest_rate:.2f}%",
            f"{latest_rate - rate_prev:+.2f}%",
        )

        latest_unemp = macro_df["美國失業率"].dropna().iloc[-1]
        col3.metric("美國失業率", f"{latest_unemp:.1f}%")

        latest_fed = macro_df["聯邦基金利率"].dropna().iloc[-1]
        col4.metric("聯邦基金利率", f"{latest_fed:.2f}%")

        # 第二區塊：10Y - 2Y 利差與倒掛警示
        st.write("---")
        st.subheader("🚦 美國公債殖利率曲線警示（10Y − 2Y）")

        spread = macro_df["10Y-2Y美債殖利率利差"].dropna()
        if spread.empty:
            st.warning("目前無法取得 10Y − 2Y 利差資料。")
        else:
            latest_spread = float(spread.iloc[-1])
            observation_date = spread.index[-1].strftime("%Y-%m-%d")

            with st.container(border=True):
                status_col, chart_col = st.columns([1, 2], gap="large")

                with status_col:
                    st.metric("最新 10Y − 2Y 利差", f"{latest_spread:+.2f} 百分點")

                    if latest_spread < 0:
                        st.error("🔴 Warning：殖利率曲線倒掛！10 年期殖利率低於 2 年期。")
                    else:
                        st.success("🟢 未倒掛：10 年期殖利率不低於 2 年期。")

                    st.caption(f"FRED 最新可用觀測日期：{observation_date}")

                with chart_col:
                    spread_chart = spread.to_frame(name="10Y − 2Y 利差")
                    fig_spread = px.line(
                        spread_chart,
                        y="10Y − 2Y 利差",
                        title="近一年美債 10Y − 2Y 殖利率利差",
                    )
                    fig_spread.update_traces(line_color="#2563EB", line_width=2.5)
                    fig_spread.add_hline(
                        y=0,
                        line_color="#DC2626",
                        line_dash="dash",
                        annotation_text="零軸（倒掛分界）",
                        annotation_position="top left",
                    )
                    fig_spread.update_layout(
                        height=330,
                        xaxis_title="日期",
                        yaxis_title="利差（百分點）",
                        showlegend=False,
                        margin=dict(l=10, r=15, t=55, b=15),
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)

        # 第三區塊：保留原本的兩個互動式圖表分頁
        st.write("---")
        st.subheader("📈 市場走勢與總體指標折線圖")

        tab1, tab2 = st.tabs(["金融市場走勢 (yfinance)", "總體經濟趨勢 (FRED)"])

        with tab1:
            market_choice = st.selectbox("選擇金融市場標的：", market_df.columns)
            fig1 = px.line(market_df, y=market_choice, title=f"{market_choice} 近一年走勢")
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            macro_choice = st.selectbox("選擇總體經濟指標：", macro_df.columns)
            fig2 = px.line(macro_df, y=macro_choice, title=f"{macro_choice} 近一年走勢")
            st.plotly_chart(fig2, use_container_width=True)
            if macro_choice == "日本10年期國債殖利率":
                st.caption("日本國債殖利率為月資料，圖表會沿用最近公布的月值。")

        # 第四區塊：按一下才呼叫 Gemini，避免每次重新整理都產生費用。
        st.write("---")
        st.subheader("🤖 AI 總體經濟摘要")
        st.caption("使用已抓取的 FRED 最新 CPI、失業率及利率資料生成約 200 字短評。")

        if st.button("產生 AI 總體經濟摘要", type="primary", use_container_width=True):
            st.session_state.pop("ai_macro_summary", None)
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
            if not gemini_key:
                st.warning("尚未設定 GEMINI_API_KEY。請至 Streamlit App Settings → Secrets 新增金鑰。")
            else:
                with st.spinner("AI 正在分析最新總體經濟數據…"):
                    try:
                        st.session_state["ai_macro_summary"] = generate_macro_summary(
                            gemini_key, latest_points
                        )
                    except Exception as exc:
                        st.error(
                            f"摘要產生失敗（{type(exc).__name__}）。"
                            "請確認 API Key、模型權限與使用額度後重試。"
                        )

        if st.session_state.get("ai_macro_summary"):
            with st.container(border=True):
                st.markdown("#### 📝 AI 總體經濟短評")
                st.write(st.session_state["ai_macro_summary"])
                st.caption("由 Gemini 根據 FRED 指標自動生成，僅供學術討論，請自行核對。")

    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")
