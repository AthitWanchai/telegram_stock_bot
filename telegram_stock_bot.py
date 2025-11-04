import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

# ตั้งค่า logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ฟังก์ชันตรวจสอบประเภทสินทรัพย์
def get_asset_type(symbol):
    symbol_upper = symbol.upper()
    
    # Forex / สกุลเงิน
    if 'USD' in symbol_upper and ('THB' in symbol_upper or 'JPY' in symbol_upper or 'EUR' in symbol_upper or 'GBP' in symbol_upper or 'CNY' in symbol_upper or 'AUD' in symbol_upper or 'CAD' in symbol_upper or 'CHF' in symbol_upper):
        return "💱 อัตราแลกเปลี่ยน"
    if symbol_upper.startswith('DX-Y'):
        return "💵 ดัชนีดอลลาร์"
    
    # Commodities / สินค้าโภคภัณฑ์
    if symbol_upper in ['GC=F', 'GOLD', 'XAU=F']:
        return "🪙 ทองคำ"
    if symbol_upper in ['SI=F', 'SILVER', 'XAG=F']:
        return "⚪ เงิน"
    if symbol_upper in ['CL=F', 'BZ=F']:
        return "🛢️ น้ำมันดิบ"
    if symbol_upper in ['NG=F']:
        return "🔥 ก๊าซธรรมชาติ"
    if '=F' in symbol_upper:
        return "📦 สินค้าฟิวเจอร์ส"
    
    # Crypto
    if 'BTC' in symbol_upper or 'ETH' in symbol_upper or 'DOGE' in symbol_upper or 'ADA' in symbol_upper or '-USD' in symbol_upper:
        return "₿ คริปโต"
    
    # Index / ดัชนี
    if symbol_upper.startswith('^'):
        if 'SET' in symbol_upper:
            return "📊 ดัชนีตลาดหุ้นไทย"
        return "📊 ดัชนีตลาดหุ้น"
    
    # Thai Stock / หุ้นไทย
    if symbol_upper.endswith('.BK'):
        return "🇹🇭 หุ้นไทย"
    
    # ETF / กองทุน
    if symbol_upper in ['SPY', 'QQQ', 'VOO', 'VTI', 'IVV', 'DIA', 'EEM', 'GLD', 'SLV', 'TLT', 'AGG']:
        return "📈 กองทุน ETF"
    
    # Default: หุ้น
    return "📈 หุ้น"

# ฟังก์ชันคำนวณ RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ฟังก์ชันคำนวณ MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ฟังก์ชันคำนวณ Bollinger Bands
def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

# ฟังก์ชันสร้างกราฟพร้อมแนวรับ-แนวต้าน
def create_chart(df, symbol, support, resistance):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # วาดกราฟราคา
    ax.plot(df.index, df['Close'], linewidth=2, color='#4A90E2', label='ราคา')
    ax.fill_between(df.index, df['Close'].min() * 0.99, df['Close'], alpha=0.2, color='#4A90E2')
    
    # เส้นแนวรับ (Support) - สีเขียว
    ax.axhline(y=support, color='#2ECC71', linestyle='--', linewidth=2, label=f'แนวรับ: {support:.2f}', alpha=0.8)
    ax.fill_between(df.index, support * 0.995, support * 1.005, color='#2ECC71', alpha=0.1)
    
    # เส้นแนวต้าน (Resistance) - สีแดง
    ax.axhline(y=resistance, color='#E74C3C', linestyle='--', linewidth=2, label=f'แนวต้าน: {resistance:.2f}', alpha=0.8)
    ax.fill_between(df.index, resistance * 0.995, resistance * 1.005, color='#E74C3C', alpha=0.1)
    
    # เส้น EMA 20 (สีส้ม)
    if 'EMA_20' in df.columns:
        ax.plot(df.index, df['EMA_20'], linewidth=1.5, color='#F39C12', label='EMA 20', alpha=0.7, linestyle='-.')
    
    # ตกแต่งกราฟ
    ax.set_title(f'{symbol} - Avg Price + Support/Resistance (last 60 days)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', color='gray')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    
    # ปรับ y-axis ให้เห็นแนวรับ-แนวต้านชัดเจน
    y_min = min(df['Close'].min(), support) * 0.98
    y_max = max(df['Close'].max(), resistance) * 1.02
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf

# ฟังก์ชันวิเคราะห์หุ้น
async def analyze_stock(symbol: str):
    try:
        # ดึงข้อมูลหุ้น
        stock = yf.Ticker(symbol)
        df = stock.history(period='6mo')
        
        if df.empty:
            return None, "ไม่พบข้อมูลหุ้น กรุณาตรวจสอบชื่อหุ้นอีกครั้ง"
        
        # คำนวณ indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = calculate_macd(df['Close'])
        df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = calculate_bollinger_bands(df['Close'])
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # ข้อมูลล่าสุด
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # ตรวจสอบประเภทสินทรัพย์
        asset_type = get_asset_type(symbol)
        
        # คำนวณ Bollinger Bands 20 วัน
        bb_20_high = df['Upper_BB'].iloc[-1]
        bb_20_low = df['Lower_BB'].iloc[-1]
        
        # คำนวณราคาเฉลี่ย 5 วัน
        avg_5d = df['Close'].tail(5).mean()
        
        # วิเคราะห์แนวโน้ม
        trend_emoji = "📈" if latest['Close'] > prev['Close'] else "📉"
        rsi_status = "กลาง" if 30 < latest['RSI'] < 70 else ("สัญญาณขาย" if latest['RSI'] > 70 else "สัญญาณซื้อ")
        rsi_emoji = "🟢" if latest['RSI'] < 30 else ("🔴" if latest['RSI'] > 70 else "🟡")
        
        macd_status = "แนวโน้มลง" if latest['MACD'] < latest['Signal'] else "แนวโน้มขึ้น"
        macd_emoji = "🟢" if latest['MACD'] > latest['Signal'] else "🔴"
        
        price_status = "สูง" if latest['Close'] > avg_5d else ("ต่ำ" if latest['Close'] < avg_5d else "ปกติ")
        price_emoji = "🔴" if latest['Close'] > avg_5d else "🟢"
        
        ema_20_50 = "ขาขึ้น" if latest['EMA_20'] > latest['EMA_50'] else "ขาลง"
        ema_20_50_emoji = "🟢" if latest['EMA_20'] > latest['EMA_50'] else "🔴"
        
        ema_50_200 = "โกลเด้นครอส" if latest['EMA_50'] > latest['EMA_200'] else "เดธครอส"
        ema_50_200_emoji = "🟢" if latest['EMA_50'] > latest['EMA_200'] else "🔴"
        
        obv_trend = "เพิ่มขึ้น" if latest['Volume'] > df['Volume'].tail(5).mean() else "ลดลง"
        obv_emoji = "🟢" if latest['Volume'] > df['Volume'].tail(5).mean() else "📉"
        
    
        message = (
            f"{asset_type}: {symbol}\n"
            f"─────────────────────────────\n"
            f"📈 แนวโน้มโดยรวม: {'ขาขึ้น' if latest['Close'] > prev['Close'] else 'ขาลง'} {trend_emoji}\n"
            f"─────────────────────────────\n"
            f"📊 RSI: {rsi_status} {rsi_emoji}\n"
            f"📉 MACD: {macd_status} {macd_emoji}\n"
            f"💰 ราคาเฉลี่ย 5 วัน: {avg_5d:.2f} ({price_status}) {price_emoji}\n"
            f"📎 โบลลิงเจอร์ (20): {bb_20_low:.2f} - {bb_20_high:.2f}\n"
            f"📈 EMA 20/50: {ema_20_50} {ema_20_50_emoji}\n"
            f"📊 EMA 50/200: {ema_50_200} {ema_50_200_emoji}\n"
            f"📦 ปริมาณ (OBV): {obv_trend} {obv_emoji}\n"
            f"💚 แนวรับ: {df['Low'].tail(20).min():.2f}\n"
            f"❤️ แนวต้าน: {df['High'].tail(20).max():.2f}\n"
            f"─────────────────────────────\n"
            f"*เพื่อเป็นข้อมูลทั่วไป ไม่ใช่คำแนะนำด้านการลงทุน*\n"
            f"พิมพ์ /english สำหรับสรุปภาษาอังกฤษ\n"
            f"─────────────────────────────\n"
        )
        trend_text = "ขาขึ้น" if latest['Close'] > prev['Close'] else "ขาลง"
        rsi_text = (
            "อยู่ในระดับกลาง แรงซื้อขายสมดุล"
            if 30 < latest['RSI'] < 70
            else ("อยู่ในโซนขายมากเกินไป มีโอกาสกลับตัว" if latest['RSI'] < 30 else "อยู่ในโซนซื้อมากเกินไป มีความเสี่ยงปรับฐาน")
        )
        macd_text = "แสดงโมเมนตัมเป็นบวก อาจมีแรงขึ้นต่อ" if latest['MACD'] > latest['Signal'] else "แสดงโมเมนตัมอ่อนแรง อาจปรับตัวลง"

        summary = (
            f"📝 **สรุปภาพรวม ({symbol})**\n\n"
            f"แนวโน้มปัจจุบันอยู่ในภาวะ *{trend_text}* โดยมีสัญญาณจากอินดิเคเตอร์หลักดังนี้:\n\n"
            f"• **RSI** — {rsi_text}\n"
            f"• **MACD** — {macd_text}\n"
            f"• **EMA 20/50** — แนวโน้มระยะสั้นเป็น *{ema_20_50}*\n"
            f"• **EMA 50/200** — แนวโน้มระยะยาวเป็น *{ema_50_200}*\n"
            f"• **OBV** — ปริมาณการซื้อขาย {obv_trend} "
            f"{'สนับสนุน' if obv_trend == 'เพิ่มขึ้น' else 'ไม่สนับสนุน'}แนวโน้ม\n\n"
            f"💡 ราคาปัจจุบันอยู่{'สูงกว่า' if latest['Close'] > avg_5d else 'ต่ำกว่า'}ค่าเฉลี่ย 5 วัน "
            f"และอยู่ในช่วง{'บน' if latest['Close'] > df['Middle_BB'].iloc[-1] else 'ล่าง'}ของกรอบ Bollinger Bands\n\n"
            f"📊 สรุปโดยรวม: แนวโน้มยัง{'แข็งแกร่ง' if latest['Close'] > prev['Close'] else 'อ่อนแรง'} "
            f"แต่ควรจับตาการเปลี่ยนทิศของ MACD และแรงซื้อขายในระยะสั้น."
        )
        message += summary
        
        # คำนวณแนวรับ-แนวต้าน
        support = df['Low'].tail(20).min()
        resistance = df['High'].tail(20).max()
        
        # สร้างกราฟ
        chart = create_chart(df.tail(60), symbol, support, resistance)
        
        return chart, message
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None, f"เกิดข้อผิดพลาด: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "สวัสดีครับ! 👋\n\n"
        "ผมเป็น Stock Analysis Bot 📊\n"
        "ส่งชื่อหุ้นมาได้เลย เช่น:\n"
        "- AAPL (Apple)\n"
        "- GC=F (Gold Futures)\n"
        "- IONQ\n"
        "- ^SET (ตลาดหุ้นไทย)\n\n"
        "คำสั่ง:\n"
        "/start - เริ่มต้นใช้งาน\n"
        "/help - ดูวิธีใช้งาน"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 วิธีใช้งาน:\n\n"
        "1. ส่งชื่อหุ้นที่ต้องการวิเคราะห์\n"
        "2. รอสักครู่ ระบบจะวิเคราะห์และส่งผลลัพธ์กลับมา\n\n"
        "ตัวอย่างชื่อหุ้น:\n"
        "- หุ้นอเมริกา: AAPL, TSLA, MSFT\n"
        "- Futures: GC=F (ทองคำ), CL=F (น้ำมัน)\n"
        "- Crypto: BTC-USD, ETH-USD\n"
        "- หุ้นไทย: PTT.BK, KBANK.BK"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    
    # ส่งข้อความรอ
    wait_msg = await update.message.reply_text(f"🔍 กำลังวิเคราะห์ {symbol}...")
    
    # วิเคราะห์หุ้น
    chart, message = await analyze_stock(symbol)
    
    # ลบข้อความรอ
    await wait_msg.delete()
    
    if chart:
        # ส่งกราฟและข้อความ
        await update.message.reply_photo(photo=chart, caption=message)
    else:
        await update.message.reply_text(message)

def main():
    # ใส่ Token ของคุณที่นี่
    TOKEN = "8500948741:AAG_tkexujcGY5Pig6ta3KFwbvT7mQ6zpXs"
    
    # สร้าง Application
    application = Application.builder().token(TOKEN).build()
    
    # เพิ่ม handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # เริ่มต้น bot
    logger.info("Bot started...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
