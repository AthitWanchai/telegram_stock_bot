import os
import logging
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

# ===============================
# 🔧 CONFIG
# ===============================
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# เก็บรายการติดตามของผู้ใช้แต่ละคน
user_watchlist = {}  # {user_id: {"EURUSD=X": "last_signal"}}

# ===============================
# 📊 INDICATORS
# ===============================
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def find_support_resistance(df, window=20):
    support = df['Low'].tail(window).min()
    resistance = df['High'].tail(window).max()
    return support, resistance

def detect_patterns(df):
    patterns = []
    if len(df) < 2:
        return patterns
    latest, prev = df.iloc[-1], df.iloc[-2]
    # Engulfing
    if latest['Close'] > latest['Open'] and prev['Close'] < prev['Open'] and latest['Open'] <= prev['Close'] and latest['Close'] >= prev['Open']:
        patterns.append("🟢 Bullish Engulfing")
    elif latest['Close'] < latest['Open'] and prev['Close'] > prev['Open'] and latest['Open'] >= prev['Close'] and latest['Close'] <= prev['Open']:
        patterns.append("🔴 Bearish Engulfing")
    # Doji
    body = abs(latest['Close'] - latest['Open'])
    rng = latest['High'] - latest['Low']
    if rng > 0 and body < rng * 0.1:
        patterns.append("⭐ Doji (Indecision)")
    return patterns

# ===============================
# ⚙️ ANALYSIS
# ===============================
def analyze_trading_signal(df):
    # คำนวณ indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df['Close'])
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['SMA_20'] - (df['BB_std'] * 2)
    
    # ลบ NaN
    df = df.dropna()
    
    if len(df) < 2:
        return "NEUTRAL", 0, 0, ["ข้อมูลไม่เพียงพอ"], df

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    buy, sell, signals = 0, 0, []

    # แปลงเป็น scalar values
    rsi_val = float(latest['RSI'])
    macd_val = float(latest['MACD'])
    signal_val = float(latest['Signal'])
    prev_macd = float(prev['MACD'])
    prev_signal = float(prev['Signal'])
    close_val = float(latest['Close'])
    prev_close = float(prev['Close'])
    ema9_val = float(latest['EMA_9'])
    ema20_val = float(latest['EMA_20'])
    ema50_val = float(latest['EMA_50'])
    bb_upper_val = float(latest['BB_upper'])
    bb_lower_val = float(latest['BB_lower'])
    sma20_val = float(latest['SMA_20'])

    # 1. RSI Analysis
    if rsi_val < 30:
        buy += 3
        signals.append(f"✅ RSI Oversold: {rsi_val:.1f}")
    elif rsi_val > 70:
        sell += 3
        signals.append(f"⚠️ RSI Overbought: {rsi_val:.1f}")
    elif 40 < rsi_val < 60:
        signals.append(f"⚪ RSI Neutral: {rsi_val:.1f}")
    else:
        signals.append(f"📊 RSI: {rsi_val:.1f}")

    # 2. MACD Crossover
    if macd_val > signal_val and prev_macd <= prev_signal:
        buy += 4
        signals.append("✅ MACD Bullish Cross")
    elif macd_val < signal_val and prev_macd >= prev_signal:
        sell += 4
        signals.append("⚠️ MACD Bearish Cross")
    elif macd_val > signal_val:
        buy += 1
        signals.append("🟢 MACD Above Signal")
    else:
        sell += 1
        signals.append("🔴 MACD Below Signal")

    # 3. EMA Trend
    if close_val > ema9_val > ema20_val > ema50_val:
        buy += 4
        signals.append("✅✅ Strong Uptrend (EMA)")
    elif close_val < ema9_val < ema20_val < ema50_val:
        sell += 4
        signals.append("⚠️⚠️ Strong Downtrend (EMA)")
    elif close_val > ema20_val:
        buy += 2
        signals.append("🟢 Price Above EMA20")
    else:
        sell += 2
        signals.append("🔴 Price Below EMA20")

    # 4. Bollinger Bands
    if close_val < bb_lower_val:
        buy += 2
        signals.append("✅ Price Below BB Lower")
    elif close_val > bb_upper_val:
        sell += 2
        signals.append("⚠️ Price Above BB Upper")
    else:
        bb_position = ((close_val - bb_lower_val) / (bb_upper_val - bb_lower_val)) * 100
        signals.append(f"📊 BB Position: {bb_position:.0f}%")

    # 5. Price Action
    if close_val > prev_close:
        buy += 1
        signals.append("📈 Bullish Candle")
    else:
        sell += 1
        signals.append("📉 Bearish Candle")

    # 6. Momentum
    price_change = ((close_val - prev_close) / prev_close) * 100
    if abs(price_change) > 0.5:
        if price_change > 0:
            buy += 1
            signals.append(f"🚀 Strong Momentum: +{price_change:.2f}%")
        else:
            sell += 1
            signals.append(f"⬇️ Strong Momentum: {price_change:.2f}%")

    # สรุปสัญญาณ
    if buy > sell and buy >= 5:
        signal_type = "STRONG BUY" if buy >= 10 else "BUY"
    elif sell > buy and sell >= 5:
        signal_type = "STRONG SELL" if sell >= 10 else "SELL"
    else:
        signal_type = "NEUTRAL"

    return signal_type, buy, sell, signals, df

# ===============================
# 🧩 CHART
# ===============================
def create_chart(df, symbol, signal_type):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Price', color='black')
    ax.plot(df.index, df['EMA_20'], label='EMA20', color='orange')
    ax.plot(df.index, df['EMA_50'], label='EMA50', color='purple')
    ax.fill_between(df.index, df['BB_lower'], df['BB_upper'], color='gray', alpha=0.1)
    ax.set_title(f"{symbol} - {signal_type}", color='green' if "BUY" in signal_type else 'red' if "SELL" in signal_type else 'gray')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# ===============================
# 💹 FOREX ANALYSIS
# ===============================
async def analyze_forex(symbol: str):
    try:
        df = yf.download(symbol, period='3mo', interval='1h', auto_adjust=True, progress=False)
        
        # แก้ไข MultiIndex ถ้ามี
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            return None, "❌ ไม่พบข้อมูล", "NEUTRAL"

        signal_type, buy, sell, signals, df = analyze_trading_signal(df)
        price = df.iloc[-1]['Close']

        # คำแนะนำอัตโนมัติ
        if signal_type.startswith("STRONG BUY"):
            advice = "📈 แนวโน้มแข็งแรง เหมาะสำหรับเปิด Buy หรือทยอยสะสม"
        elif signal_type == "BUY":
            advice = "🟢 แนวโน้มเริ่มขาขึ้น ควรจับตาหาจังหวะเข้า Buy"
        elif signal_type.startswith("STRONG SELL"):
            advice = "📉 แนวโน้มลงแรง เหมาะสำหรับเปิด Sell หรือปิด Buy"
        elif signal_type == "SELL":
            advice = "🔴 มีแรงขาย ควรระวังการย่อราคา"
        else:
            advice = "⚪️ แนวโน้มยังไม่ชัดเจน รอดูสัญญาณเพิ่มเติม"

        msg = (
            f"💹 {symbol} - TRADING SIGNAL\n"
            f"{'='*35}\n"
            f"📊 สัญญาณ: {signal_type}\n"
            f"📈 คะแนน: BUY {buy} | SELL {sell}\n"
            f"💰 ราคา: {price:.5f}\n"
            f"{'='*35}\n\n"
            f"🔍 SIGNALS:\n"
        )
        for i, sig in enumerate(signals[:6], 1):
            msg += f"{i}. {sig}\n"

        msg += f"\n💬 คำแนะนำ:\n{advice}\n"
        msg += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        chart = create_chart(df.tail(100), symbol, signal_type)
        return chart, msg, signal_type
    except Exception as e:
        logger.error(f"Error: {e}")
        return None, f"❌ Error: {e}", "NEUTRAL"

# ===============================
# 🤖 COMMANDS
# ===============================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Forex Trading Bot v2.0\n\n"
        "📊 พิมพ์ชื่อคู่เงิน เช่น EURUSD=X เพื่อดูสัญญาณ\n"
        "🧭 คำสั่ง:\n"
        "/add EURUSD=X ➕ เพิ่มติดตาม\n"
        "/remove EURUSD=X ➖ ลบออก\n"
        "/list 📋 ดูรายการติดตาม"
    )

async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("⚠️ ใช้เช่น /add EURUSD=X")
        return
    symbol = context.args[0].upper()
    user_watchlist.setdefault(user_id, {})[symbol] = "NONE"
    await update.message.reply_text(f"✅ เพิ่ม {symbol} เข้ารายการติดตามแล้ว")

async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("⚠️ ใช้เช่น /remove EURUSD=X")
        return
    symbol = context.args[0].upper()
    if user_id in user_watchlist and symbol in user_watchlist[user_id]:
        del user_watchlist[user_id][symbol]
        await update.message.reply_text(f"🗑 ลบ {symbol} ออกแล้ว")
    else:
        await update.message.reply_text("❌ ไม่พบคู่เงินนี้ในรายการ")

async def list_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in user_watchlist or not user_watchlist[user_id]:
        await update.message.reply_text("📭 ยังไม่มีคู่เงินในรายการติดตาม")
        return
    symbols = "\n".join(user_watchlist[user_id].keys())
    await update.message.reply_text(f"📋 รายการติดตามของคุณ:\n{symbols}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    if '=' not in symbol and len(symbol) == 6:
        symbol += '=X'
    wait_msg = await update.message.reply_text(f"⏳ กำลังวิเคราะห์ {symbol}...")
    chart, msg, _ = await analyze_forex(symbol)
    await wait_msg.delete()
    if chart:
        await update.message.reply_photo(photo=chart, caption=msg)
    else:
        await update.message.reply_text(msg)

# ===============================
# ⏰ AUTO CHECK
# ===============================
async def auto_check(context: ContextTypes.DEFAULT_TYPE):
    for user_id, symbols in user_watchlist.items():
        for symbol, last_signal in symbols.items():
            chart, msg, signal_type = await analyze_forex(symbol)
            if not chart:
                continue
            if signal_type != last_signal:
                user_watchlist[user_id][symbol] = signal_type
                if signal_type in ["BUY", "STRONG BUY"]:
                    await context.bot.send_photo(chat_id=user_id, photo=chart,
                        caption=f"📈 {symbol} ถึงเวลาเข้าซื้อ!\n\n{msg}")
                elif signal_type in ["SELL", "STRONG SELL"]:
                    await context.bot.send_photo(chat_id=user_id, photo=chart,
                        caption=f"📉 {symbol} ถึงเวลาขาย!\n\n{msg}")

# ===============================
# 🚀 MAIN
# ===============================
def main():
    # ใส่ Token ของคุณที่นี่
    TOKEN = "8500948741:AAG_tkexujcGY5Pig6ta3KFwbvT7mQ6zpXs"
    
    # สร้าง Application
    application = Application.builder().token(TOKEN).build()
    
    # เพิ่ม handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add", add_symbol))
    application.add_handler(CommandHandler("remove", remove_symbol))
    application.add_handler(CommandHandler("list", list_symbol))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # ตั้งค่าการแจ้งเตือนอัตโนมัติ (ทุก 15 นาที)
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(auto_check, interval=900, first=10)
        logger.info("✅ Auto-check enabled (every 15 minutes)")
    
    # เริ่มต้น bot
    logger.info("🚀 Forex Signal Bot started...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
