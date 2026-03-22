#!/usr/bin/env python3
"""
Download diverse stock data for Kronos pretraining.

Downloads daily OHLCV data for ~200 tickers across sectors,
preprocesses into Kronos CSV format, and saves to data/pretrain/.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "pretrain")

# Diverse tickers: large-cap, mid-cap, across sectors + some ETFs
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSM", "AVGO", "ORCL", "ADBE",
    "CRM", "CSCO", "ACN", "INTC", "AMD", "TXN", "QCOM", "IBM", "NOW", "INTU",
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "FTNT", "PANW", "CRWD",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "BK", "STT", "FITB", "HBAN", "KEY", "CFG", "RF",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "MDT", "ISRG", "SYK", "BSX", "VRTX", "REGN", "ZTS", "CI",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "HD", "TJX", "ORLY", "AZO", "ROST", "DG", "DLTR", "YUM", "CMG", "DHI",
    # Industrial
    "CAT", "DE", "HON", "UNP", "UPS", "RTX", "BA", "GE", "LMT", "MMM",
    "EMR", "ETN", "ITW", "PH", "ROK", "CMI", "SWK", "GD", "NOC", "TDG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
    "DVN", "HES", "FANG", "HAL", "BKR", "TRGP", "WMB", "KMI", "OKE", "ET",
    # Materials / Utilities / Real Estate
    "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "DOW", "ALB", "PPG",
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ES", "WEC",
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB",
    # Telecom / Media
    "GOOG", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD", "PARA",
    # ETFs (broad market, international, sectors)
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "FXI", "GLD", "SLV",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC",
]

# Remove duplicates
TICKERS = list(dict.fromkeys(TICKERS))

START_DATE = "2004-01-01"
END_DATE = "2025-12-31"


def download_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    total_rows = 0
    success = 0
    failed = []

    print(f"Downloading daily OHLCV for {len(TICKERS)} tickers...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Save dir: {DATA_DIR}\n")

    for i, ticker in enumerate(TICKERS):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True)

            if df.empty or len(df) < 200:
                print(f"  [{i+1}/{len(TICKERS)}] {ticker}: skipped (only {len(df)} rows)")
                failed.append(ticker)
                continue

            # Rename to Kronos format
            out = pd.DataFrame()
            out['timestamps'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
            out['open'] = df['Open'].values
            out['high'] = df['High'].values
            out['low'] = df['Low'].values
            out['close'] = df['Close'].values
            out['volume'] = df['Volume'].values.astype(float)
            out['amount'] = (df['Volume'] * df['Close']).values  # approx dollar volume

            # Drop any NaN rows
            out = out.dropna().reset_index(drop=True)

            if len(out) < 200:
                print(f"  [{i+1}/{len(TICKERS)}] {ticker}: skipped after cleanup ({len(out)} rows)")
                failed.append(ticker)
                continue

            save_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            out.to_csv(save_path, index=False)
            total_rows += len(out)
            success += 1

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(TICKERS)}] {ticker}: {len(out)} rows  (total so far: {total_rows:,})")

        except Exception as e:
            print(f"  [{i+1}/{len(TICKERS)}] {ticker}: FAILED ({e})")
            failed.append(ticker)
            continue

    print(f"\nDone. {success}/{len(TICKERS)} tickers downloaded.")
    print(f"Total rows: {total_rows:,}")
    print(f"Failed: {len(failed)} tickers")
    if failed:
        print(f"  {failed[:20]}{'...' if len(failed) > 20 else ''}")

    # Write manifest
    manifest = {
        "total_tickers": success,
        "total_rows": total_rows,
        "date_range": f"{START_DATE} to {END_DATE}",
        "failed": failed,
    }
    import json
    with open(os.path.join(DATA_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    return total_rows


if __name__ == "__main__":
    download_and_save()
