#!/usr/bin/env python3
"""
Download popular hourly ticker data for Kronos pretraining.

Fetches hourly Yahoo Finance history for a curated set of liquid tickers,
keeping the full available lookback for each ticker and stopping once the
combined corpus reaches the requested row budget.
"""

import glob
import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "pretrain")

SOURCE_NAME = os.environ.get("PRETRAIN_SOURCE_NAME", "Yahoo Finance hourly bars")
FREQUENCY = os.environ.get("PRETRAIN_FREQUENCY", "60m").strip()
LOOKBACK_PERIOD = os.environ.get("PRETRAIN_LOOKBACK", "730d").strip()
YFINANCE_CACHE_DIR = os.environ.get("PRETRAIN_CACHE_DIR", os.path.join(ROOT_DIR, ".yfinance_cache"))

# Keep the default list to simple, liquid symbols with reliable hourly coverage.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "JPM", "BAC", "WFC", "GS", "UNH", "JNJ", "PFE", "LLY",
    "WMT", "COST", "HD", "MCD", "XOM", "CVX", "CAT", "GE",
    "QQQ", "SPY", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV",
]


def read_positive_int_env(name, default):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


TARGET_TOTAL_ROWS = read_positive_int_env("PRETRAIN_TOTAL_ROWS", 100_000)
MIN_ROWS_PER_TICKER = read_positive_int_env("PRETRAIN_MIN_ROWS_PER_TICKER", 1_000)


def candidate_tickers():
    override = os.environ.get("PRETRAIN_TICKERS", "").strip()
    raw_items = override.split(",") if override else DEFAULT_TICKERS
    tickers = []
    for ticker in raw_items:
        ticker = ticker.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def safe_filename_for_ticker(ticker):
    safe = ticker.replace("^", "index_").replace("/", "_").replace("=", "_")
    safe = safe.replace("-", "_").replace(".", "_")
    return f"{safe.lower()}.csv"


def remove_existing_pretrain_artifacts():
    for csv_path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        os.remove(csv_path)
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)


def extract_timestamp_values(df):
    for col in ("datetime", "time", "timestamp", "date", "datetime64", "日期", "时间"):
        if col in df.columns:
            return df[col]
    if not isinstance(df.index, pd.RangeIndex):
        return pd.Series(df.index, index=df.index)
    raise ValueError("Could not find timestamps in index or columns.")


def normalize_ohlcv(raw_df):
    if raw_df is None or raw_df.empty:
        raise ValueError("Provider returned no rows.")

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(col).strip().lower() for col in df.columns]

    timestamps = pd.to_datetime(extract_timestamp_values(df), errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)

    required_cols = ("open", "high", "low", "close")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    open_col = pd.to_numeric(df["open"], errors="coerce")
    high_col = pd.to_numeric(df["high"], errors="coerce")
    low_col = pd.to_numeric(df["low"], errors="coerce")
    close_col = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        volume_col = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        volume_col = pd.Series(0.0, index=df.index)

    if "amount" in df.columns:
        amount_col = pd.to_numeric(df["amount"], errors="coerce")
    else:
        amount_col = volume_col * close_col

    out = pd.DataFrame(
        {
            "timestamps": timestamps,
            "open": open_col,
            "high": high_col,
            "low": low_col,
            "close": close_col,
            "volume": volume_col,
            "amount": amount_col,
        }
    )
    out = out.dropna(subset=["timestamps", "open", "high", "low", "close"])
    out = out.sort_values("timestamps")
    out = out.drop_duplicates(subset=["timestamps"], keep="last").reset_index(drop=True)

    if len(out) < MIN_ROWS_PER_TICKER:
        raise ValueError(
            f"Only {len(out):,} clean rows available; expected at least {MIN_ROWS_PER_TICKER:,}."
        )

    return out


def fetch_ticker_history(ticker):
    raw_df = yf.Ticker(ticker).history(
        period=LOOKBACK_PERIOD,
        interval=FREQUENCY,
        auto_adjust=True,
    )
    return normalize_ohlcv(raw_df)


def collect_ticker_data():
    os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
    yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

    selected = {}
    failures = []
    total_rows = 0
    tickers = candidate_tickers()

    for idx, ticker in enumerate(tickers, start=1):
        try:
            df = fetch_ticker_history(ticker)
            selected[ticker] = df
            total_rows += len(df)
            print(
                f"  [{idx}/{len(tickers)}] {ticker}: {len(df):,} rows "
                f"(total so far: {total_rows:,})"
            )
            if total_rows >= TARGET_TOTAL_ROWS:
                break
        except Exception as exc:
            print(f"  [{idx}/{len(tickers)}] {ticker}: FAILED ({exc})")
            failures.append(f"{ticker}: {exc}")

    if total_rows < TARGET_TOTAL_ROWS:
        raise RuntimeError(
            f"Only collected {total_rows:,} rows across {len(selected)} tickers; "
            f"requested {TARGET_TOTAL_ROWS:,}.\n" + "\n".join(failures)
        )

    return selected, failures, total_rows


def write_manifest(selected, failures, total_rows):
    ordered_tickers = list(selected.keys())
    first_timestamp = min(df["timestamps"].iloc[0] for df in selected.values())
    last_timestamp = max(df["timestamps"].iloc[-1] for df in selected.values())
    manifest = {
        "source_name": SOURCE_NAME,
        "frequency": FREQUENCY,
        "lookback_period": LOOKBACK_PERIOD,
        "requested_total_rows": TARGET_TOTAL_ROWS,
        "total_tickers": len(selected),
        "total_rows": int(total_rows),
        "selected_tickers": ordered_tickers,
        "rows_by_ticker": {ticker: int(len(df)) for ticker, df in selected.items()},
        "failed_tickers": failures,
        "first_timestamp": first_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp": last_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def save_selected_tickers(selected, failures, total_rows):
    remove_existing_pretrain_artifacts()
    for ticker, df in selected.items():
        save_path = os.path.join(DATA_DIR, safe_filename_for_ticker(ticker))
        save_df = df.copy()
        save_df["timestamps"] = save_df["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")
        save_df.to_csv(save_path, index=False)
    write_manifest(selected, failures, total_rows)


def download_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading {SOURCE_NAME}...")
    print(f"Frequency: {FREQUENCY}")
    print(f"Lookback: {LOOKBACK_PERIOD}")
    print(f"Target rows: {TARGET_TOTAL_ROWS:,}")
    print(f"Save dir: {DATA_DIR}\n")

    selected, failures, total_rows = collect_ticker_data()
    save_selected_tickers(selected, failures, total_rows)

    print(f"\nDone. {len(selected)}/{len(candidate_tickers())} tickers downloaded.")
    print(f"Total rows: {total_rows:,}")
    return total_rows


if __name__ == "__main__":
    download_and_save()
