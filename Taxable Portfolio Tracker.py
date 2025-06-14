import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import gspread
from gspread_dataframe import set_with_dataframe
import os
import sys
from gspread_formatting import *

# --- CONFIGURATION ---
CREDENTIALS_FILE = r"C:\Users\jisbe\Documents\Data\Projects\Taxable-Portfolio\gspread_credentials.json"
SHEET_NAME = "Financial Details"
WORKSHEET_NAME = "Signals"
TICKERS = ['PEP', 'KO', 'MSFT', 'AAPL', 'GOOGL', 'MCD', 'CL', 'KMB', 'PG', 'SCHD', 'DGRO', 'SCHY', 'VIGI', 'VTI']

# --- AUTHENTICATE GOOGLE SHEETS ---
if not os.path.exists(CREDENTIALS_FILE):
    sys.exit("❌ Missing gspread_credentials.json file.")

gc = gspread.service_account(filename=CREDENTIALS_FILE)

try:
    sheet = gc.open(SHEET_NAME)
except gspread.SpreadsheetNotFound:
    sys.exit(f"❌ Google Sheet '{SHEET_NAME}' not found. Please create it and share with your service account.")

try:
    worksheet = sheet.worksheet(WORKSHEET_NAME)
except gspread.WorksheetNotFound:
    worksheet = sheet.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="20")

# --- SCREENING FUNCTION ---
def check_buy_signal(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist_full = stock.history(period='1y')
        if len(hist_full) < 200:
            return None  # Not enough data for MA200

        price_col = 'Adj Close' if 'Adj Close' in hist_full.columns else 'Close'
        hist_recent = hist_full.tail(126)  # ~6 months

        current_price = hist_full[price_col].iloc[-1]
        ma50 = hist_full[price_col].rolling(window=50).mean().iloc[-1]
        ma200 = hist_full[price_col].rolling(window=200).mean().iloc[-1]

        # RSI Calculation (14-day)
        delta = hist_full[price_col].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14).mean().iloc[-1]

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = round(100 - (100 / (1 + rs)), 2)

        info = stock.info or {}
        raw_yield = info.get("dividendYield")
        if raw_yield and raw_yield > 20:
            print(f"⚠️ Suspiciously high dividend yield for {ticker}: {raw_yield}")
        pe_ratio = round(info.get("trailingPE", 0) or 0, 2)
        payout_ratio = round((info.get("payoutRatio", 0) or 0) * 100, 2)
        eps = round(info.get("trailingEps", 0) or 0, 2)

        price_change_pct_ma50 = round(((current_price - ma50) / ma50) * 100, 2)
        price_change_pct_ma200 = round(((current_price - ma200) / ma200) * 100, 2)

        hist_recent['Daily Return'] = hist_recent[price_col].pct_change()
        avg_daily_return = round(hist_recent['Daily Return'].mean() * 100, 3)

        hist_5yr = stock.history(period='5y')
        if len(hist_5yr) < 2:
            cagr_5yr = np.nan
        else:
            price_col_5yr = 'Adj Close' if 'Adj Close' in hist_5yr.columns else 'Close'
            start_price = hist_5yr[price_col_5yr].iloc[0]
            end_price = hist_5yr[price_col_5yr].iloc[-1]
            n_days = (hist_5yr.index[-1] - hist_5yr.index[0]).days
            cagr_5yr = round(((end_price / start_price) ** (365 / n_days) - 1) * 100, 2)

        # Signal logic
        if (price_change_pct_ma50 <= -5 or price_change_pct_ma200 <= -5) and rsi <= 30:
            signal = "BUY"
        elif current_price < ma50 or current_price < ma200 or rsi < 40:
            signal = "WATCH"
        else:
            signal = "HOLD"

        return {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker,
            "Current Price": round(current_price, 2),
            "50-Day MA": round(ma50, 2),
            "200-Day MA": round(ma200, 2),
            "Dividend Yield (%)": raw_yield, #round(raw_yield * 100, 2) if raw_yield and raw_yield < 1.5 else round(raw_yield, 2) if raw_yield else np.nan,
            "P/E Ratio": pe_ratio,
            "Payout Ratio (%)": payout_ratio,
            "EPS": eps,
            "% vs MA50": price_change_pct_ma50,
            "% vs MA200": price_change_pct_ma200,
            "RSI (14d)": rsi,
            "Avg Daily Return (6mo %)": avg_daily_return,
            "CAGR (5yr %)": cagr_5yr,
            "Signal": signal
        }
    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")
        return None



# --- COLLECT & COMBINE DATA ---
results = [check_buy_signal(t) for t in TICKERS]
results = [r for r in results if r is not None]
df_new = pd.DataFrame(results)

existing_data = worksheet.get_all_records()
df_existing = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()

# Combine and sort
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'], errors='coerce')

# Keep only the latest record per ticker
df_combined.sort_values(by='Timestamp', ascending=False, inplace=True)
df_combined = df_combined.drop_duplicates(subset='Ticker', keep='first')

# Enforce column order
columns = [
    "Timestamp", "Ticker", "Current Price", "50-Day MA", "200-Day MA",
    "% vs MA50", "% vs MA200", "Dividend Yield (%)", "P/E Ratio", "Payout Ratio (%)", "EPS",
    "RSI (14d)", "Avg Daily Return (6mo %)", "CAGR (5yr %)", "Signal"
]
for col in columns:
    if col not in df_combined.columns:
        df_combined[col] = np.nan

df_combined = df_combined[columns]


# --- EXPORT TO GOOGLE SHEET ---
worksheet.clear()
set_with_dataframe(worksheet, df_combined)

# --- SHEET FORMATTING ---
fmt_header = cellFormat(
    backgroundColor=color(0.9, 0.9, 0.9),
    textFormat=textFormat(bold=True),
    horizontalAlignment='CENTER'
)

# Fix header range from 'A1:L1' to include all 15 columns ('A1:O1')
format_cell_range(worksheet, 'A1:O1', fmt_header)
set_frozen(worksheet, rows=1)

# Auto-resize columns
worksheet.resize(rows=df_combined.shape[0] + 10, cols=len(columns))

# --- CONDITIONAL FORMATTING ---

# Signal column
signal_col_index = columns.index("Signal") + 1
watch_rule = ConditionalFormatRule(
    ranges=[GridRange(sheetId=worksheet._properties['sheetId'], startRowIndex=1, endRowIndex=1000,
                      startColumnIndex=signal_col_index - 1, endColumnIndex=signal_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('TEXT_EQ', ['WATCH']),
        format=CellFormat(backgroundColor=color(0.85, 1.0, 0.85))
    )
)
buy_rule = ConditionalFormatRule(
    ranges=[GridRange(sheetId=worksheet._properties['sheetId'], startRowIndex=1, endRowIndex=1000,
                      startColumnIndex=signal_col_index - 1, endColumnIndex=signal_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('TEXT_EQ', ['BUY']),
        format=CellFormat(backgroundColor=color(0.7, 1.0, 0.7), textFormat=textFormat(bold=True))
    )
)


# RSI formatting if RSI < 30
rsi_col_index = columns.index("RSI (14d)") + 1
rsi_rule = ConditionalFormatRule(
    ranges=[GridRange(sheetId=worksheet._properties['sheetId'], startRowIndex=1, endRowIndex=1000,
                      startColumnIndex=rsi_col_index - 1, endColumnIndex=rsi_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('NUMBER_LESS', ['30']),
        format=CellFormat(backgroundColor=color(1.0, 0.85, 0.85))
    )
)

# P/E formatting if P/E > 30
pe_col_index = columns.index("P/E Ratio") + 1
pe_rule = ConditionalFormatRule(
    ranges=[GridRange(sheetId=worksheet._properties['sheetId'], startRowIndex=1, endRowIndex=1000,
                      startColumnIndex=pe_col_index - 1, endColumnIndex=pe_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('NUMBER_GREATER', ['30']),
        format=CellFormat(backgroundColor=color(1.0, 0.9, 0.6))
    )
)


# --- Corrected Conditional formatting for % vs MA50 and % vs MA200 ---

ma50_col_index = columns.index("% vs MA50") + 1
ma200_col_index = columns.index("% vs MA200") + 1

# BRIGHT RED: value <= -5
ma50_bright_red = ConditionalFormatRule(
    ranges=[GridRange(
        sheetId=worksheet._properties['sheetId'],
        startRowIndex=1, endRowIndex=1000,
        startColumnIndex=ma50_col_index - 1, endColumnIndex=ma50_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('NUMBER_LESS_THAN_EQ', ['-5']),
        format=CellFormat(backgroundColor=color(1.0, 0.6, 0.6), textFormat=textFormat(bold=True))
    )
)

ma200_bright_red = ConditionalFormatRule(
    ranges=[GridRange(
        sheetId=worksheet._properties['sheetId'],
        startRowIndex=1, endRowIndex=1000,
        startColumnIndex=ma200_col_index - 1, endColumnIndex=ma200_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('NUMBER_LESS_THAN_EQ', ['-5']),
        format=CellFormat(backgroundColor=color(1.0, 0.6, 0.6), textFormat=textFormat(bold=True))
    )
)

# LIGHT RED: value < 0
ma50_light_red = ConditionalFormatRule(
    ranges=[GridRange(
        sheetId=worksheet._properties['sheetId'],
        startRowIndex=1, endRowIndex=1000,
        startColumnIndex=ma50_col_index - 1, endColumnIndex=ma50_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('NUMBER_LESS', ['0']),
        format=CellFormat(backgroundColor=color(1.0, 0.9, 0.9))
    )
)

ma200_light_red = ConditionalFormatRule(
    ranges=[GridRange(
        sheetId=worksheet._properties['sheetId'],
        startRowIndex=1, endRowIndex=1000,
        startColumnIndex=ma200_col_index - 1, endColumnIndex=ma200_col_index)],
    booleanRule=BooleanRule(
        condition=BooleanCondition('NUMBER_LESS', ['0']),
        format=CellFormat(backgroundColor=color(1.0, 0.9, 0.9))
    )
)

# Apply all formatting rules
rules = get_conditional_format_rules(worksheet)
rules.clear()
rules.append(watch_rule)
rules.append(buy_rule)
rules.append(rsi_rule)
rules.append(pe_rule)
rules.append(ma50_bright_red)
rules.append(ma50_light_red)
rules.append(ma200_bright_red)
rules.append(ma200_light_red)
rules.save()


print("✅ Google Sheet updated and formatted successfully.")

