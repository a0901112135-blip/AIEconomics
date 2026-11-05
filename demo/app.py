import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from fredapi import Fred


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

    df = pd.DataFrame(data).sort_index()

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
    return df.loc[df.index >= one_year_ago]


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


# 頁面渲染邏輯
if fred_key:
    try:
        macro_df = get_fred_data(fred_key)
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

    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")

