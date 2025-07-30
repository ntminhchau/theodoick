# dashboard.py
import streamlit as st
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
import numpy as np
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re
from urllib.parse import quote_plus
import time
import sqlite3
from supabase import create_client

# --- CẤU HÌNH ---
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
st.set_page_config(layout="wide", page_title="Dashboard Phân tích AI")
PREDICTIONS_DB_FILE = "ai_predictions.db"

# --- KẾT NỐI SUPABASE VÀ CÁC HÀM LẤY DỮ LIỆU ---

@st.cache_resource
def init_connection():
    """Khởi tạo kết nối tới Supabase, cache lại để không tạo lại liên tục."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Lỗi kết nối Supabase: {e}. Vui lòng kiểm tra file secrets.")
        return None

supabase_client = init_connection()

@st.cache_data(ttl=86400)
def load_ticker_list():
    """Tải danh sách mã cổ phiếu từ file text."""
    try:
        with open('all_tickers.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file 'all_tickers.txt'.")
        return ['FPT', 'VNM', 'HPG', 'VCB', 'MWG']

@st.cache_data(ttl=14400)
def get_default_scan_list():
    """Lấy danh sách cổ phiếu mặc định từ file text."""
    try:
        with open('default_tickers.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.warning("Lỗi: Không tìm thấy file 'default_tickers.txt'.")
        return ['FPT','VNM','HPG', 'VCB', 'MWG']

@st.cache_data(ttl=900)
def get_stock_data(ticker, days_back=730):
    """Lấy dữ liệu lịch sử cho một mã cổ phiếu từ Supabase."""
    if supabase_client is None:
        return pd.DataFrame()
    try:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        response = supabase_client.table('historical_data').select("*") \
            .eq('ticker', ticker) \
            .gte('time', start_date) \
            .order('time', desc=False).execute()
        
        df = pd.DataFrame(response.data)

        if df.empty:
            return pd.DataFrame()
            
        df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.drop(columns=['ticker'])
        
        return df
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu từ DB cho {ticker}: {e}")
        return pd.DataFrame()

def get_last_price_info(ticker):
    """Lấy thông tin giá gần nhất từ dữ liệu lịch sử."""
    df_recent = get_stock_data(ticker, days_back=5)
    if df_recent.empty or len(df_recent) < 2: return None
    last_day, prev_day = df_recent.iloc[-1], df_recent.iloc[-2]
    price = last_day['Close']
    change = price - prev_day['Close']
    pct_change = (change / prev_day['Close']) * 100 if prev_day['Close'] > 0 else 0
    return {'price': price, 'change': change, 'pct_change': pct_change, 'open': last_day['Open'], 'high': last_day['High'], 'low': last_day['Low'], 'volume': last_day['Volume']}

@st.cache_data
def add_technical_indicators(df):
    """Thêm các chỉ báo kỹ thuật."""
    if df.empty: return df
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df['Highest_High_20'] = df['High'].rolling(20).max()
    return df

@st.cache_data(ttl=3600)
def search_google_news(ticker):
    """Tìm kiếm tin tức trên Google, giới hạn ở Vietstock và CafeF."""
    try:
        query = f'"{ticker}" site:vietstock.vn OR site:cafef.vn'
        encoded_query = quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}&tbm=nws"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        articles = []
        for g in soup.find_all('div', class_='SoaBEf'):
            a_tag = g.find('a')
            title_tag = g.find('div', role='heading')
            if a_tag and title_tag:
                link, title = a_tag.get('href'), title_tag.text.strip()
                if link and title:
                    articles.append({'title': title, 'link': link})
                    if len(articles) >= 7: break
        return articles
    except Exception as e:
        print(f"Lỗi khi tìm kiếm tin tức trên Google cho {ticker}: {e}")
        return []

def scan_alerts_for_tickers(tickers):
    alerts = []
    progress_bar = st.progress(0, text="Bắt đầu quét...")
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Đang quét: {ticker}")
        df = get_stock_data(ticker, days_back=250) 
        if df.empty or len(df) < 50:
            continue
            
        df = add_technical_indicators(df)
        
        required_cols = ['SMA_20', 'SMA_50', 'MACD_12_26_9', 'RSI_14']
        if not all(col in df.columns for col in required_cols):
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        price = last['Close']
        sma20 = last.get('SMA_20')
        sma50 = last.get('SMA_50')
        sma200 = last.get('SMA_200')
        rsi = last.get('RSI_14')
        macd_line = last.get('MACD_12_26_9')
        macd_signal = last.get('MACDs_12_26_9')
        adx = last.get('ADX_14')
        bb_lower = last.get('BBL_20_2.0')
        bb_upper = last.get('BBU_20_2.0')

        # Trend signals (based on SMA_50, SMA_200 and ADX)
        if pd.notna(adx) and pd.notna(sma50) and pd.notna(sma200):
            if adx > 25:
                if price > sma50 and sma50 > sma200:
                    alerts.append({"Mã": ticker, "Tín hiệu": "Xu hướng TĂNG", "Lý do": "Giá & MA ngắn hạn trên MA dài hạn, ADX mạnh", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
                elif price < sma50 and sma50 < sma200:
                    alerts.append({"Mã": ticker, "Tín hiệu": "Xu hướng GIẢM", "Lý do": "Giá & MA ngắn hạn dưới MA dài hạn, ADX mạnh", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            elif adx < 20:
                alerts.append({"Mã": ticker, "Tín hiệu": "Đi ngang (Sideways)", "Lý do": "ADX yếu, thị trường thiếu xu hướng rõ ràng", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})

        # BUY / SELL signals (Crossover & RSI)
        if pd.notna(sma20) and pd.notna(sma50) and pd.notna(macd_line) and pd.notna(macd_signal) and pd.notna(rsi):
            # Gold Cross (MA20 cắt lên MA50)
            gold_cross_series = crossover(df['SMA_20'], df['SMA_50'])
            is_gold_cross = gold_cross_series.iloc[-1] if isinstance(gold_cross_series, pd.Series) and not gold_cross_series.empty else gold_cross_series
            if is_gold_cross:
                alerts.append({"Mã": ticker, "Tín hiệu": "MUA", "Lý do": "Giao cắt vàng (MA20 > MA50)", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})

            # MACD cắt lên Signal
            macd_cross_up_series = crossover(df['MACD_12_26_9'], df['MACDs_12_26_9'])
            is_macd_cross_up = macd_cross_up_series.iloc[-1] if isinstance(macd_cross_up_series, pd.Series) and not macd_cross_up_series.empty else macd_cross_up_series
            if is_macd_cross_up:
                alerts.append({"Mã": ticker, "Tín hiệu": "MUA", "Lý do": "MACD cắt lên Signal", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            
            # RSI Quá bán và đang hồi phục
            if rsi < 30 and pd.notna(prev['Close']) and price > prev['Close']:
                alerts.append({"Mã": ticker, "Tín hiệu": "MUA (Hồi phục)", "Lý do": "RSI quá bán và giá đang tăng trở lại", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            
            # Death Cross (MA50 cắt lên MA20)
            death_cross_series = crossover(df['SMA_50'], df['SMA_20'])
            is_death_cross = death_cross_series.iloc[-1] if isinstance(death_cross_series, pd.Series) and not death_cross_series.empty else death_cross_series
            if is_death_cross:
                alerts.append({"Mã": ticker, "Tín hiệu": "BÁN", "Lý do": "Giao cắt tử thần (MA20 < MA50)", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            
            # MACD cắt xuống Signal
            macd_cross_down_series = crossover(df['MACDs_12_26_9'], df['MACD_12_26_9'])
            is_macd_cross_down = macd_cross_down_series.iloc[-1] if isinstance(macd_cross_down_series, pd.Series) and not macd_cross_down_series.empty else macd_cross_down_series
            if is_macd_cross_down:
                alerts.append({"Mã": ticker, "Tín hiệu": "BÁN", "Lý do": "MACD cắt xuống Signal", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            
            # RSI Quá mua và đang giảm
            if rsi > 70 and pd.notna(prev['Close']) and price < prev['Close']:
                alerts.append({"Mã": ticker, "Tín hiệu": "BÁN (Điều chỉnh)", "Lý do": "RSI quá mua và giá đang giảm", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})

        # HOLD BUY / HOLD SELL signals (based on price position relative to MA)
        if pd.notna(sma20) and pd.notna(sma50):
            if price > sma20 and price > sma50:
                alerts.append({"Mã": ticker, "Tín hiệu": "GIỮ MUA", "Lý do": "Giá duy trì trên MA20 và MA50", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            elif price < sma20 and price < sma50:
                alerts.append({"Mã": ticker, "Tín hiệu": "GIỮ BÁN", "Lý do": "Giá duy trì dưới MA20 và MA50", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})

        # Bollinger Bands Breakout/Breakdown (ensure bands exist)
        if pd.notna(bb_lower) and pd.notna(bb_upper) and pd.notna(prev['Close']):
            # Giá vượt trên dải Bollinger trên
            bb_upper_cross = crossover(df['Close'], df['BBU_20_2.0'])
            is_bb_upper_cross = bb_upper_cross.iloc[-1] if isinstance(bb_upper_cross, pd.Series) and not bb_upper_cross.empty else bb_upper_cross
            if is_bb_upper_cross:
                alerts.append({"Mã": ticker, "Tín hiệu": "MUA Mạnh", "Lý do": "Giá vượt lên Dải Bollinger trên (Breakout)", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})
            
            # Giá xuyên thủng dải Bollinger dưới
            bb_lower_cross = crossover(df['BBL_20_2.0'], df['Close'])
            is_bb_lower_cross = bb_lower_cross.iloc[-1] if isinstance(bb_lower_cross, pd.Series) and not bb_lower_cross.empty else bb_lower_cross
            if is_bb_lower_cross:
                alerts.append({"Mã": ticker, "Tín hiệu": "BÁN Mạnh", "Lý do": "Giá xuyên thủng Dải Bollinger dưới (Breakdown)", "Giá": f"{price:,.1f}", "RSI": f"{rsi:.1f}"})

    progress_bar.empty()
    if alerts:
        df_alerts = pd.DataFrame(alerts).drop_duplicates(subset=['Mã', 'Tín hiệu'])
        st.dataframe(df_alerts.sort_values(by=["Mã", "Tín hiệu"]))
    else:
        st.info("Không có tín hiệu giao dịch nổi bật cho các mã đã chọn.")

# --- BACKTESTING ---
class SmaCross(Strategy):
    def init(self): self.sma1, self.sma2 = self.data.SMA_20, self.data.SMA_50
    def next(self):
        if crossover(self.sma1, self.sma2): self.buy()
        elif crossover(self.sma2, self.sma1): self.position.close()
class RsiOscillator(Strategy):
    rsi_low, rsi_high = 30, 70
    def init(self): self.rsi = self.data.RSI_14
    def next(self):
        if crossover(self.rsi, self.rsi_high): self.position.close()
        elif crossover(self.rsi_low, self.rsi): self.buy()
class MacdCross(Strategy):
    def init(self): self.macd_line, self.signal_line = self.data.MACD_12_26_9, self.data.MACDs_12_26_9
    def next(self):
        if crossover(self.macd_line, self.signal_line): self.buy()
        elif crossover(self.signal_line, self.macd_line): self.position.close()
class Breakout(Strategy):
    def init(self): self.highest_high = self.data.Highest_High_20
    def next(self):
        if crossover(self.data.Close, self.highest_high): self.buy()
class BollingerBands(Strategy):
    def init(self): self.lower_band, self.upper_band = self.data['BBL_20_2.0'], self.data['BBU_20_2.0']
    def next(self):
        if crossover(self.data.Close, self.lower_band): self.buy()
        elif crossover(self.upper_band, self.data.Close): self.position.close()
@st.cache_data
def run_backtest(_df, strategy):
    if _df.empty or len(_df) < 50: return None
    try:
        bt = Backtest(_df, strategy, cash=100_000_000, commission=.0015)
        return bt.run()
    except Exception as e:
        print(f"--- LỖI BACKTEST CHI TIẾT ---: {e}")
        return None
def format_backtest_stats(stats):
    if stats is None: return None
    stats_copy = stats.copy()
    for idx, value in stats_copy.items():
        if isinstance(value, pd.Timedelta):
            stats_copy[idx] = str(value)
    return stats_copy
def analyze_backtest_results(stats):
    if stats is None: return ""
    explanation = "#### Diễn giải các chỉ số chính:\n- **Return [%]**: Tổng tỷ suất lợi nhuận.\n- **Win Rate [%]**: Tỷ lệ giao dịch có lãi.\n- **Max. Drawdown [%]**: Mức sụt giảm tài khoản lớn nhất (đo lường rủi ro)."
    conclusion = "#### Kết luận:\n"
    ret, win_rate, drawdown = stats.get('Return [%]', 0), stats.get('Win Rate [%]', 0), stats.get('Max. Drawdown [%]', 0)
    if ret > 10 and win_rate > 50 and drawdown > -20: conclusion += "✅ **Hiệu quả tốt:** Chiến lược tạo ra lợi nhuận tốt với rủi ro chấp nhận được."
    elif ret > 0: conclusion += "⚠️ **Có tiềm năng:** Chiến lược có lãi, nhưng cần xem xét kỹ rủi ro."
    else: conclusion += "❌ **Không hiệu quả:** Chiến lược không tạo ra lợi nhuận với mã này."
    return explanation + "\n" + conclusion

@st.cache_data(ttl=3600)
def get_all_predictions_from_db():
    """Đọc toàn bộ báo cáo dự báo từ file SQLite."""
    try:
        with sqlite3.connect(f"file:{PREDICTIONS_DB_FILE}?mode=ro", uri=True) as conn:
            df = pd.read_sql_query("SELECT * FROM predictions", conn)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file báo cáo '{PREDICTIONS_DB_FILE}': {e}")
        st.warning("Vui lòng đảm bảo bạn đã tải file `ai_predictions.db` lên GitHub.")
        return pd.DataFrame()

def get_single_prediction(df_preds, ticker):
    """Lấy dự báo cho một mã cụ thể từ DataFrame đã tải."""
    if df_preds.empty or 'MaCoPhieu' not in df_preds.columns:
        return None
    
    prediction_row = df_preds[df_preds['MaCoPhieu'] == ticker]
    if not prediction_row.empty:
        return prediction_row.iloc[0]
    return None

# --- GIAO DIỆN STREAMLIT ---
st.title("📈 Dashboard Phân tích Cổ phiếu Tích hợp AI")

# --- THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("Bảng điều khiển")
    ticker_list = load_ticker_list()
    if ticker_list:
        # Tìm index của 'FPT' một cách an toàn
        try:
            fpt_index = ticker_list.index('FPT')
        except ValueError:
            fpt_index = 0
        selected_ticker = st.selectbox("Chọn mã cổ phiếu:", ticker_list, index=fpt_index)
    else:
        selected_ticker = st.text_input("Nhập mã cổ phiếu:", 'FPT')
    
    st.divider()
    page_options = ["📊 Phân tích Kỹ thuật", "🤖 Báo cáo Dự báo AI", "📰 Tin tức Liên quan", "🔬 Backtesting", "🚨 Cảnh báo"]
    page = st.radio("Chọn chức năng:", page_options)
    st.divider()
    st.info("Dashboard được Chou xây dựng để phân tích chứng khoán.")

# Tải dữ liệu chính một lần
data = get_stock_data(selected_ticker)
data_ind = add_technical_indicators(data.copy())
df_all_predictions = get_all_predictions_from_db()

# --- HEADER THÔNG TIN CHUNG ---
st.header(f"Tổng quan: {selected_ticker}")
price_info = get_last_price_info(selected_ticker)
if price_info:
    col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
    price_str = f"{price_info['price']:,.1f}"
    change_str = f"{price_info['change']:,.1f} ({price_info['pct_change']:.2f}%)"
    col1.metric("Giá", price_str, change_str)
    col2.metric("Mở cửa", f"{price_info['open']:,.1f}")
    col3.metric("Cao/Thấp", f"{price_info['high']:,.1f} / {price_info['low']:,.1f}")
    col4.metric("KLGD", f"{price_info['volume']:,.0f}")
else:
    st.warning(f"Không thể lấy thông tin giá gần nhất cho {selected_ticker}.")

# Hiển thị dự báo AI ngay tại header
prediction_info = get_single_prediction(df_all_predictions, selected_ticker)
if prediction_info is not None:
    pred_text = prediction_info['DuBao']
    if "TĂNG" in pred_text:
        prob = prediction_info['XacSuatTang']
        st.success(f"**Dự báo AI (5 ngày tới):** 📈 {pred_text} (Xác suất: {prob}) - {prediction_info['LyGiai']}")
    else:
        prob = prediction_info['XacSuatGiam']
        st.error(f"**Dự báo AI (5 ngày tới):** 📉 {pred_text} (Xác suất: {prob}) - {prediction_info['LyGiai']}")
else:
    st.info(f"Chưa có dữ liệu dự báo AI cho {selected_ticker} trong báo cáo.")

st.divider()

# --- HIỂN THỊ NỘI DUNG TƯƠNG ỨNG VỚI LỰA CHỌN TRÊN SIDEBAR ---

if page == "📊 Phân tích Kỹ thuật":
    st.subheader("Biểu đồ giá")
    if not data_ind.empty:
        fig = go.Figure(data=[go.Candlestick(x=data_ind.index, open=data_ind['Open'], high=data_ind['High'], low=data_ind['Low'], close=data_ind['Close'], name='Giá')])
        fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind.get('SMA_20'), mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind.get('SMA_50'), mode='lines', name='MA50'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không có dữ liệu để hiển thị.")

elif page == "🤖 Báo cáo Dự báo AI":
    st.subheader("Báo cáo Dự báo Xu hướng từ AI")
    st.info("Dữ liệu do AI phân tích và dự báo, chỉ mang tính tham khảo.")
    
    if not df_all_predictions.empty:
        st.markdown("#### Bộ lọc báo cáo:")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_option = st.selectbox("Lọc theo dự báo:", ["Tất cả", "TĂNG GIÁ", "GIẢM/ĐI NGANG"])
        with col2:
            sort_option = st.selectbox("Sắp xếp theo:", ["Mã Cổ phiếu", "Xác suất Tăng cao nhất", "Xác suất Giảm cao nhất"])

        df_filtered = df_all_predictions
        if filter_option == "TĂNG GIÁ":
            df_filtered = df_all_predictions[df_all_predictions['DuBao'] == 'TĂNG GIÁ']
        elif filter_option == "GIẢM/ĐI NGANG":
            df_filtered = df_all_predictions[df_all_predictions['DuBao'] == 'GIẢM/ĐI NGANG']

        if sort_option == "Xác suất Tăng cao nhất":
            df_sorted = df_filtered.sort_values(by='XacSuatTang', ascending=False)
        elif sort_option == "Xác suất Giảm cao nhất":
            df_sorted = df_filtered.sort_values(by='XacSuatGiam', ascending=False)
        else:
            df_sorted = df_filtered.sort_values(by='MaCoPhieu')
        
        st.dataframe(df_sorted)
    else:
        st.warning("Không tìm thấy file báo cáo. Vui lòng chạy `prediction_reporter.py` trước và tải file lên GitHub.")

elif page == "📰 Tin tức Liên quan":
    st.subheader(f"Tin tức Liên quan đến {selected_ticker}")
    articles = search_google_news(selected_ticker)
    if articles:
        for article in articles:
            st.markdown(f"- [{article['title']}]({article['link']})")
    else:
        st.info("Không tìm thấy tin tức cho mã này.")

elif page == "🔬 Backtesting":
    st.subheader("Backtesting Đa Chiến lược")
    st.write(f"Kiểm thử các chiến lược giao dịch cho mã: **{selected_ticker}**")

    strategies = {
        "Giao cắt MA (SmaCross)": SmaCross,
        "Dao động RSI (RsiOscillator)": RsiOscillator,
        "Giao cắt MACD (MacdCross)": MacdCross,
        "Phá vỡ nền giá (Breakout)": Breakout,
        "Dải Bollinger (BollingerBands)": BollingerBands
    }
    strategy_name = st.selectbox("Chọn chiến lược để kiểm thử:", list(strategies.keys()))
    
    if st.button("Chạy Backtest"):
        with st.spinner(f"Đang chạy backtest với chiến lược {strategy_name}..."):
            # Lấy lại data_ind để chắc chắn nó khớp với mã đang chọn
            backtest_data = add_technical_indicators(get_stock_data(selected_ticker))
            if not backtest_data.empty:
                stats = run_backtest(backtest_data, strategies[strategy_name])
                if stats is not None:
                    st.text("Kết quả Backtest:")
                    formatted_stats = format_backtest_stats(stats)
                    st.write(formatted_stats)
                    st.markdown(analyze_backtest_results(stats))
                else:
                    st.error("Không thể chạy backtest. Mã này có thể có quá ít dữ liệu lịch sử.")
            else:
                st.error(f"Không có đủ dữ liệu cho {selected_ticker} để chạy backtest.")

elif page == "🚨 Cảnh báo":
    st.subheader("Cảnh báo Tín hiệu Giao dịch Ngắn hạn")
    
    with st.expander("👉 Giải thích các loại cảnh báo (Nhấn để mở rộng)"):
        st.markdown("""
        Các cảnh báo sau đây dựa trên phân tích kỹ thuật của các chỉ báo phổ biến. Đây không phải lời khuyên đầu tư.
        - **Xu hướng TĂNG/GIẢM**: Dựa vào vị trí của giá so với các đường MA và chỉ số sức mạnh xu hướng ADX.
        - **MUA/BÁN**: Dựa vào các điểm giao cắt của đường MA, MACD, hoặc các ngưỡng quá mua/quá bán của RSI.
        - **GIỮ MUA/BÁN**: Dựa trên việc giá duy trì ổn định trên hoặc dưới các đường MA quan trọng.
        - **MUA/BÁN Mạnh**: Dựa trên tín hiệu phá vỡ các Dải Bollinger.
        """)

    st.markdown("#### Quét các rổ chỉ số chính")
    st.write("Chọn rổ chỉ số bạn muốn quét để tìm kiếm các tín hiệu giao dịch.")

    col_vn30, col_vn100 = st.columns(2) 

    with col_vn30:
        if st.button("Quét VN30"):
            st.info("Đang quét các mã trong rổ VN30...")
            try:
                vn_tickers = get_default_scan_list()
                scan_alerts_for_tickers(vn_tickers)
            except FileNotFoundError:
                st.error("Không tìm thấy file default_tickers.txt")

    with col_vn100:
        if st.button("Quét VN100"):
            st.warning("Quét VN100 có thể mất nhiều thời gian hơn.")
            st.info("Đang quét các mã trong rổ VN100...")
            try:
                vn100_tickers = get_default_scan_list()
                scan_alerts_for_tickers(vn100_tickers)
            except FileNotFoundError:
                st.error("Không tìm thấy file default_tickers.txt")

    st.divider()

    st.markdown("#### Quét các mã tự chọn")
    custom_alert_tickers = st.multiselect(
        "Chọn (hoặc gõ để tìm) các mã bạn muốn theo dõi:",
        ticker_list,
        default=['FPT', 'HPG', 'VCB']
    )

    if st.button("Quét các mã đã chọn"):
        if custom_alert_tickers:
            st.info(f"Đang quét các mã tự chọn: {', '.join(custom_alert_tickers)}...")
            scan_alerts_for_tickers(custom_alert_tickers)
        else:
            st.warning("Vui lòng chọn ít nhất một mã cổ phiếu để quét.")
