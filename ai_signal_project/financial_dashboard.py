import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

#import trained model
try:
    from train_model import FinancialSentimentPredictor, Config
except:
    st.error("Model import error. Please make sure train_model.py is in same folder.")


class ConfigData:
    STOCKS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA"]

#Sample Data for Charts
class SampleData:
    def __init__(self):
        self.base = {
            "AAPL": 180, "GOOGL": 140, "MSFT": 330,
            "TSLA": 240, "AMZN": 150, "META": 350,
            "NFLX": 500, "NVDA": 450
        }

    def make_chart_data(self, symbol, days=30):
        base_price = self.base.get(symbol, 100)
        end = datetime.now()
        dates = [end - timedelta(days=i) for i in range(days, 0, -1)]

        np.random.seed(42)
        changes = np.random.normal(0.001, 0.02, days)
        prices = [base_price]
        for c in changes[1:]:
            new_p = prices[-1] * (1 + c)
            prices.append(max(new_p, base_price * 0.5))

        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        vols = [int(np.random.randint(1e6, 1e7)) for _ in range(days)]

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "prices": [round(p, 2) for p in prices],
            "highs": [round(h, 2) for h in highs],
            "lows": [round(l, 2) for l in lows],
            "volumes": vols
        }

# Market Data (real + sample fallback) 
class MarketData:
    def __init__(self):
        self.sample = SampleData()

    def get_data(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1mo")

            current = info.get("currentPrice", info.get("regularMarketPrice", 100))
            prev = info.get("previousClose", current)
            change = current - prev
            per = (change / prev) * 100

            if not hist.empty and len(hist) > 5:
                hist_dict = {
                    "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                    "prices": hist["Close"].round(2).tolist(),
                    "highs": hist["High"].round(2).tolist(),
                    "lows": hist["Low"].round(2).tolist(),
                    "volumes": hist["Volume"].fillna(0).astype(int).tolist()
                }
            else:
                hist_dict = self.sample.make_chart_data(symbol)
                current = hist_dict["prices"][-1]
                change = current * 0.01
                per = 1.0

        except Exception as e:
            st.warning(f"Using sample data for {symbol}: {e}")
            hist_dict = self.sample.make_chart_data(symbol)
            current = hist_dict["prices"][-1]
            change = current * 0.01
            per = 1.0

        price = hist_dict["prices"][-1]
        support = [round(price * 0.95, 2), round(price * 0.9, 2)]
        resist = [round(price * 1.05, 2), round(price * 1.1, 2)]

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change": round(change, 2),
            "percent": round(per, 2),
            "history": hist_dict,
            "support": support,
            "resist": resist
        }

#Chart Drawin
class ChartMaker:
    def draw_chart(self, symbol, data):
        try:
            hist = data["history"]
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in hist["dates"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=hist["prices"],
                                     mode="lines", name=f"{symbol} Price",
                                     line=dict(color="blue", width=2)))

            # current price
            fig.add_trace(go.Scatter(
                x=[dates[-1]], y=[data["price"]],
                mode="markers+text", text=[f"${data['price']:.2f}"],
                textposition="top center", marker=dict(size=10, color="red"),
                showlegend=False
            ))

            # support/resistance lines
            for s in data["support"]:
                fig.add_hline(y=s, line_dash="dash", line_color="green",
                              annotation_text=f"Support ${s}")
            for r in data["resist"]:
                fig.add_hline(y=r, line_dash="dash", line_color="red",
                              annotation_text=f"Resistance ${r}")

            fig.update_layout(title=f"{symbol} Price Chart",
                              xaxis_title="Date", yaxis_title="Price ($)",
                              height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart Error: {e}")

#  Simple Demo Model (Backup)
class DemoModel:
    def predict_sentiment(self, text):
        return {"sentiment": "POSITIVE", "confidence": 0.8,
                "scores": {"positive": 0.7, "negative": 0.2, "neutral": 0.1}}

#Main Dashboard
class FinAIDashboard:
    def __init__(self):
        self.cfg = ConfigData()
        self.market = MarketData()
        self.chart = ChartMaker()
        self.model = None
        self.setup()
        self.load_model()

    def setup(self):
        st.set_page_config(page_title="FinAI Dashboard", page_icon="📈", layout="wide")

    def load_model(self):
        try:
            cfg = Config()
            cfg.output_dir = "./trained_financial_model"   # same folder as this file
            self.model = FinancialSentimentPredictor(cfg)
            if self.model.load_model():
                st.sidebar.success("Model Loaded Successfully")
            else:
                self.model = DemoModel()
                st.sidebar.info("Using Demo Model")
        except Exception as e:
            self.model = DemoModel()
            st.sidebar.info("Using Demo Model (Error loading model)")

    def run(self):
        st.sidebar.title("Controls")
        symbol = st.sidebar.selectbox("Select Stock", self.cfg.STOCKS)
        news = st.sidebar.text_area("Paste Financial News", height=120,
                                    placeholder="Enter news here...")

        if st.sidebar.button("Refresh"):
            st.rerun()

        st.title("FinAI - Financial Dashboard")
        st.markdown("Live market data and AI-based sentiment analysis")
        st.markdown("---")

        data = self.market.get_data(symbol)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Live Price", f"${data['price']:.2f}",
                      f"{data['change']:.2f} ({data['percent']:.2f}%)")
        with c2:
            st.metric("Change (%)", f"{data['percent']:.2f}%")
        with c3:
            st.metric("Support", f"${data['support'][0]}")
        with c4:
            st.metric("Resistance", f"${data['resist'][0]}")

        st.subheader("Price Chart")
        self.chart.draw_chart(symbol, data)

        # Sentiment section
        if news:
            st.subheader("Sentiment Analysis Result")
            try:
                result = self.model.predict_sentiment(news)
                sent = result["sentiment"]
                conf = result["confidence"]
                st.write(f"**Sentiment:** {sent}")
                st.write(f"**Confidence:** {conf:.2f}")
                st.write("**Scores:**", result["scores"])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Paste financial news in the sidebar to analyze sentiment.")

        st.markdown("---")
        st.caption("FinAI Dashboard - Developed for FYP 2025")

#  Run App
if __name__ == "__main__":
    try:
        app = FinAIDashboard()
        app.run()
    except Exception as e:
        st.error(f"App Error: {e}")
