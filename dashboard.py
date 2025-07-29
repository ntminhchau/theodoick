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
from vnstock import Listing, Quote
import numpy as np
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re
from urllib.parse import quote_plus
import time # Import thư viện time
from requests.exceptions import ConnectionError, Timeout # Import các loại lỗi cụ thể

# --- CẤU HÌNH ---
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
st.set_page_config(layout="wide", page_title="Dashboard Phân tích AI")

# --- CÁC HÀM TIỆN ÍCH VÀ LẤY DỮ LIỆỆU ---

@st.cache_data(ttl=86400) # Cache 1 ngày
def load_ticker_list():
    """Tải danh sách mã cổ phiếu từ tất cả các sàn."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            listing = Listing()
            hose_symbols = listing.symbols_by_group('HOSE').tolist()
            hnx_symbols = listing.symbols_by_group('HNX').tolist()
            upcom_symbols = listing.symbols_by_group('UPCOM').tolist()
            return sorted(list(set(hose_symbols + hnx_symbols + upcom_symbols)))
        except (ConnectionError, Timeout) as e:
            st.warning(f"Lỗi kết nối khi tải danh sách mã cổ phiếu (Thử lại {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Độ trễ lũy thừa: 1s, 2s, 4s
            else:
                st.error(f"Thử lại thất bại: Lỗi tải danh sách mã cổ phiếu: {e}")
                return ['FPT', 'VNM', 'HPG', 'VCB', 'MWG'] # Trả về danh sách mặc định nếu thất bại
        except Exception as e:
            st.error(f"Lỗi tải danh sách mã cổ phiếu: {e}")
            return ['FPT', 'VNM', 'HPG', 'VCB', 'MWG']

@st.cache_data(ttl=900) # Cache 15 phút
def get_stock_data(ticker, days_back=730):
    """Lấy dữ liệu lịch sử cho một mã cổ phiếu hoặc chỉ số."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            quote = Quote(symbol=ticker)
            df = quote.history(
                start=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d'),
                resolution='1D'
            )
            if df.empty: return pd.DataFrame() # Vẫn trả về DataFrame rỗng nếu không có dữ liệu
            
            df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['Open', 'Close'], inplace=True)
            return df
        except (ConnectionError, Timeout) as e:
            st.warning(f"Lỗi kết nối khi tải dữ liệu cho {ticker} (Thử lại {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Độ trễ lũy thừa
            else:
                st.error(f"Thử lại thất bại: Lỗi khi tải dữ liệu cho {ticker}: {e}")
                return pd.DataFrame() # Trả về DataFrame rỗng nếu thất bại
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu cho {ticker}: {e}")
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

# --- CÁC HÀM AI VÀ PHÂN TÍCH ---

@st.cache_data
def get_market_condition():
    """Phân tích và trả về xu hướng thị trường chung."""
    df_vni = get_stock_data('VNINDEX', days_back=365)
    if df_vni.empty or len(df_vni) < 200:
        return "Không đủ dữ liệu", "gray"
    df_vni = add_technical_indicators(df_vni)
    last = df_vni.iloc[-1]
    price, sma50, sma200, adx = last['Close'], last['SMA_50'], last['SMA_200'], last['ADX_14']
    if adx > 25 and price > sma50 > sma200: return "Tăng mạnh", "green"
    elif adx > 25 and price < sma50 < sma200: return "Giảm mạnh", "red"
    elif adx < 20: return "Sideways / Đi ngang", "orange"
    elif price > sma50 and price > sma200: return "Tăng yếu", "#26A69A"
    elif price < sma50 and price < sma200: return "Giảm yếu", "#FFB74D"
    else: return "Không xác định", "gray"

@st.cache_resource
def load_sentiment_model():
    """Tải mô hình AI phân tích cảm xúc."""
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        st.error(f"Lỗi tải mô hình AI: {e}")
        return None

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

def analyze_sentiment(articles, model):
    """Phân tích cảm xúc của các tiêu đề bài báo."""
    sentiments = []
    for article in articles:
        try:
            result = model(article['title'])[0]
            score = int(result['label'].split()[0])
            sentiment = 'Tích cực' if score >= 4 else 'Tiêu cực' if score <= 2 else 'Trung tính'
            sentiment_result = {'title': article['title'], 'link': article['link'], 'sentiment': sentiment}
            if 'ticker' in article:
                sentiment_result['ticker'] = article['ticker']
            sentiments.append(sentiment_result)
        except Exception: continue
    return sentiments

@st.cache_data
def detect_anomalies(_df):
    """AI Phát hiện giao dịch bất thường."""
    if _df.empty or len(_df) < 50: return None
    df = _df.copy()
    df['Price_Change'] = df['Close'].pct_change().abs() * 100
    df.dropna(inplace=True)
    features = ['Volume', 'Price_Change']
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(df[features])
    return df[df['Anomaly'] == -1]

def scan_alerts_for_tickers(tickers):
    alerts = []
    progress_bar = st.progress(0, text="Bắt đầu quét...")
    for i, ticker in enumerate(tickers):
        df = get_stock_data(ticker, days_back=100)
        df = add_technical_indicators(df)
        progress_bar.progress((i + 1) / len(tickers), text=f"Đang quét: {ticker}")
        if df.empty or len(df) < 51 or 'MACD_12_26_9' not in df.columns: continue
        last = df.iloc[-1]
        if crossover(df['SMA_20'], df['SMA_50']):
            alerts.append({"Mã": ticker, "Tín hiệu": "MUA", "Lý do": "Giao cắt vàng (MA20 > MA50)", "Giá": f"{last['Close']:,.1f}", "RSI": f"{last['RSI_14']:.1f}"})
        elif crossover(df['MACD_12_26_9'], df['MACDs_12_26_9']):
            alerts.append({"Mã": ticker, "Tín hiệu": "MUA", "Lý do": "MACD cắt lên Signal", "Giá": f"{last['Close']:,.1f}", "RSI": f"{last['RSI_14']:.1f}"})
        elif last['RSI_14'] < 30:
            alerts.append({"Mã": ticker, "Tín hiệu": "MUA", "Lý do": "RSI Quá bán (< 30)", "Giá": f"{last['Close']:,.1f}", "RSI": f"{last['RSI_14']:.1f}"})
        elif crossover(df['SMA_50'], df['SMA_20']):
            alerts.append({"Mã": ticker, "Tín hiệu": "BÁN", "Lý do": "Giao cắt tử thần (MA20 < MA50)", "Giá": f"{last['Close']:,.1f}", "RSI": f"{last['RSI_14']:.1f}"})
        elif crossover(df['MACDs_12_26_9'], df['MACD_12_26_9']):
            alerts.append({"Mã": ticker, "Tín hiệu": "BÁN", "Lý do": "MACD cắt xuống Signal", "Giá": f"{last['Close']:,.1f}", "RSI": f"{last['RSI_14']:.1f}"})
        elif last['RSI_14'] > 70:
            alerts.append({"Mã": ticker, "Tín hiệu": "BÁN", "Lý do": "RSI Quá mua (> 70)", "Giá": f"{last['Close']:,.1f}", "RSI": f"{last['RSI_14']:.1f}"})
    progress_bar.empty()
    if alerts:
        st.dataframe(pd.DataFrame(alerts))
    else:
        st.info("Không có tín hiệu giao dịch ngắn hạn nổi bật cho các mã đã chọn.")

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

# --- GIAO DIỆN STREAMLIT ---
st.title("📈 Dashboard Phân tích Cổ phiếu Tích hợp AI")

# --- THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("Bảng điều khiển")
    ticker_list = load_ticker_list()
    selected_ticker = st.selectbox("Chọn mã cổ phiếu:", ticker_list, index=ticker_list.index('FPT') if 'FPT' in ticker_list else 0)
    st.divider()
    # ĐÃ XÓA: Chức năng AI Dự báo
    page_options = ["📊 Phân tích Kỹ thuật", "📰 Tin tức Liên quan", "🌐 Tổng quan Tin tức Thị trường", "🔬 Backtesting", "🚨 Cảnh báo"]
    page = st.radio("Chọn chức năng:", page_options)
    st.divider()
    st.info("Dashboard được xây dựng để phân tích chứng khoán Việt Nam.")

# Tải dữ liệu chính một lần
data = get_stock_data(selected_ticker)
data_ind = add_technical_indicators(data.copy())

# --- HEADER THÔNG TIN CHUNG ---
st.header(f"Tổng quan: {selected_ticker}")
price_info = get_last_price_info(selected_ticker)
if price_info:
    col1, col2, col3, col4 = st.columns(4)
    price_str = f"{price_info['price']:,.1f}"
    change_str = f"{price_info['change']:,.1f} ({price_info['pct_change']:.2f}%)"
    col1.metric("Giá gần nhất (k VNĐ)", price_str, change_str)
    col2.metric("Mở cửa", f"{price_info['open']:,.1f}")
    col3.metric("Cao/Thấp", f"{price_info['high']:,.1f} / {price_info['low']:,.1f}")
    col4.metric("KLGD", f"{price_info['volume']:,.0f}")
else: 
    st.warning("Không thể lấy thông tin giá gần nhất.")

st.divider()
market_status, status_color = get_market_condition()
st.markdown(f"**Xu hướng Thị trường Chung (VN-Index): <span style='color:{status_color};'> {market_status}</span>**", unsafe_allow_html=True)
st.divider()

# --- HIỂN THỊ NỘI DUNG TƯƠNG ỨNG VỚI LỰA CHỌN TRÊN SIDEBAR ---

if page == "📊 Phân tích Kỹ thuật":
    st.subheader("Biểu đồ giá")
    if not data_ind.empty:
        fig = go.Figure(data=[go.Candlestick(x=data_ind.index, open=data_ind['Open'], high=data_ind['High'], low=data_ind['Low'], close=data_ind['Close'], name='Giá')])
        fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind['SMA_20'], mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind['SMA_50'], mode='lines', name='MA50'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Phân tích Giao dịch Bất thường")
        with st.spinner("AI đang phân tích các giao dịch bất thường..."):
            anomalies = detect_anomalies(data.copy())
        if anomalies is not None and not anomalies.empty:
            st.warning(f"Phát hiện {len(anomalies)} phiên giao dịch có dấu hiệu bất thường (KLGD hoặc biên độ giá đột biến):")
            st.dataframe(anomalies[['Volume', 'Price_Change']])
        else:
            st.success("Không phát hiện giao dịch bất thường đáng chú ý.")
    else:
        st.warning("Không có dữ liệu để hiển thị.")

elif page == "📰 Tin tức Liên quan":
    st.subheader(f"Tin tức Liên quan đến {selected_ticker}")
    articles = search_google_news(selected_ticker)
    if articles:
        for article in articles:
            st.markdown(f"- [{article['title']}]({article['link']})")
    else:
        st.info("Không tìm thấy tin tức cho mã này.")

elif page == "🌐 Tổng quan Tin tức Thị trường":
    st.subheader("Tổng quan Tin tức các Cổ phiếu Hàng đầu")
    default_list = get_default_scan_list()
    
    st.markdown(f"**Tổng hợp tin tức mới nhất từ {len(default_list)} cổ phiếu trong rổ VN30 & VN100.**")
    
    if st.button("Bắt đầu quét tin tức thị trường"):
        with st.spinner("Đang quét tin tức..."):
            market_news = []
            scanned_tickers_for_news = set() 
            progress_bar = st.progress(0, text="Bắt đầu quét...")
            for i, ticker in enumerate(default_list):
                progress_bar.progress((i + 1) / len(default_list), text=f"Đang quét tin tức: {ticker}")
                articles = search_google_news(ticker)
                if articles:
                    if ticker not in scanned_tickers_for_news:
                        latest_news = articles[0]
                        market_news.append({'ticker': ticker, 'title': latest_news['title'], 'link': latest_news['link']})
                        scanned_tickers_for_news.add(ticker)
                
                time.sleep(2) # Đảm bảo có độ trễ giữa các yêu cầu Google News

            progress_bar.empty()
            st.session_state['market_news_overview'] = market_news

    if 'market_news_overview' in st.session_state:
        for news in st.session_state['market_news_overview']:
            st.markdown(f"- **{news['ticker']}**: [{news['title']}]({news['link']})")

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
            selected_strategy = strategies[strategy_name]
            stats = run_backtest(data_ind.copy(), selected_strategy)
            if stats is not None:
                st.text("Kết quả Backtest:")
                formatted_stats = format_backtest_stats(stats)
                st.write(formatted_stats)
                st.markdown(analyze_backtest_results(stats))
            else:
                st.error("Không thể chạy backtest. Mã này có thể có quá ít dữ liệu lịch sử.")

elif page == "🚨 Cảnh báo":
    st.subheader("Cảnh báo Tín hiệu Giao dịch Ngắn hạn")
    
    default_list = get_default_scan_list()

    st.markdown("#### Quét các rổ chỉ số chính")
    st.write("Nhấn nút để quét toàn bộ cổ phiếu trong rổ VN30 và VN100.")
    if st.button("Quét VN30 & VN100"):
        scan_alerts_for_tickers(default_list)

    st.divider()

    st.markdown("#### Quét tùy chọn")
    custom_alert_tickers = st.multiselect("Chọn các mã bạn muốn theo dõi:", ticker_list, default=['FPT', 'HPG', 'VCB'])
    if st.button("Quét các mã đã chọn"):
        scan_alerts_for_tickers(custom_alert_tickers)
