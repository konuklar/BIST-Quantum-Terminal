# ============================================================
# 📊 BIST Risk Budgeting System - CORRECTED TICKER FORMATS
# All tickers must be in format: SYMBOL.IS (e.g., AKBNK.IS)
# No ticker modifications - exact as Yahoo Finance expects
# ============================================================

from __future__ import annotations

import json
import math
import re
import time
import hashlib
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(
    page_title="BIST Risk Budgeting", 
    page_icon="📊", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

SIGNATURE = "The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"

# ------------------------------------------------------------
# CORRECT BIST TICKERS - MUST HAVE .IS SUFFIX FOR YAHOO FINANCE
# ------------------------------------------------------------
BIST50_TICKERS = [
    "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS",
    "BIMAS.IS", "CCOLA.IS", "DOAS.IS", "EGEEN.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "MGROS.IS",
    "ODAS.IS", "OTKAR.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "SOKM.IS", "TCELL.IS", "THYAO.IS",
    "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS",
    "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YKBNK.IS",
    "KONTR.IS", "KLSER.IS", "CIMSA.IS", "KOZAA.IS", "ENJSA.IS",
    "BRSAN.IS", "BAGFS.IS", "KMPUR.IS", "AKSEN.IS", "AEFES.IS"
]

# Special tickers that don't need .IS suffix
SPECIAL_TICKERS = {
    "^XU100", "XU100.IS", "^XU050", "XU050.IS",  # Indices
    "TRY=X", "USDTRY=X", "EURTRY=X",  # Currencies
    "GC=F", "CL=F",  # Commodities
    "^TNX", "TR10YT=RR"  # Rates
}

# Explicit exclusions
EXCLUDED = {"KOZAL.IS", "TRALTIN.IS"}

# Clean BIST50 - remove excluded tickers
BIST50_TICKERS = [t for t in BIST50_TICKERS if t not in EXCLUDED]

# Ensure ASTOR.IS is included
if "ASTOR.IS" not in BIST50_TICKERS:
    BIST50_TICKERS.insert(0, "ASTOR.IS")

# Benchmark candidates (exact Yahoo Finance formats)
BENCHMARK_CANDIDATES = ["XU100.IS", "^XU100"]  # XU100.IS works better
FX_USDTRY_CANDIDATES = ["TRY=X", "USDTRY=X"]
RATE_CANDIDATES = ["^TNX", "TR10YT=RR"]

# Sector mapping (unchanged)
SECTOR_MAP = {
    "AKBNK.IS":"Banking","GARAN.IS":"Banking","ISCTR.IS":"Banking","VAKBN.IS":"Banking","YKBNK.IS":"Banking","TSKB.IS":"Banking",
    "ARCLK.IS":"Industrial","ALARK.IS":"Holding","ENKAI.IS":"Holding","KCHOL.IS":"Holding","SAHOL.IS":"Holding",
    "ASELS.IS":"Defense","BIMAS.IS":"Retail","MGROS.IS":"Retail","SOKM.IS":"Retail","ULKER.IS":"Consumer","CCOLA.IS":"Consumer","AEFES.IS":"Consumer",
    "EKGYO.IS":"Real Estate",
    "EREGL.IS":"Iron & Steel","KRDMD.IS":"Iron & Steel","BRSAN.IS":"Iron & Steel",
    "FROTO.IS":"Automotive","TOASO.IS":"Automotive","DOAS.IS":"Automotive","OTKAR.IS":"Automotive","TTRAK.IS":"Automotive",
    "PETKM.IS":"Petrochemical","TUPRS.IS":"Energy",
    "PGSUS.IS":"Aviation","THYAO.IS":"Aviation",
    "SASA.IS":"Chemicals","CIMSA.IS":"Cement","SISE.IS":"Materials",
    "TTKOM.IS":"Telecom","TCELL.IS":"Telecom",
    "ASTOR.IS":"Industrial","AKSEN.IS":"Energy","ENJSA.IS":"Energy",
    "HEKTS.IS":"Chemicals","GUBRF.IS":"Chemicals",
    "EGEEN.IS":"Industrial","TKFEN.IS":"Industrial",
    "KOZAA.IS":"Mining",
    "ODAS.IS":"Energy",
    "KONTR.IS":"Industrial","KLSER.IS":"Industrial","KMPUR.IS":"Industrial","BAGFS.IS":"Industrial"
}

# ------------------------------------------------------------
# CRITICAL FIX: Ticker validation function
# Ensures all BIST stocks have .IS suffix
# ------------------------------------------------------------
def validate_bist_ticker(ticker: str) -> str:
    """
    Validate and correct BIST ticker format for Yahoo Finance.
    BIST stocks MUST end with .IS (e.g., AKBNK.IS)
    Special tickers (indices, FX, commodities) are left as-is.
    """
    if not ticker or pd.isna(ticker):
        return ""
    
    ticker = str(ticker).strip().upper()
    
    # Remove any whitespace
    ticker = re.sub(r'\s+', '', ticker)
    
    # Check if it's a special ticker (indices, FX, commodities)
    if any(ticker.startswith(prefix) for prefix in ['^', '=']) or ticker in SPECIAL_TICKERS:
        return ticker
    
    # If it already has .IS, ensure it's uppercase
    if ticker.endswith('.IS'):
        return ticker
    
    # If it has another suffix, remove it and add .IS
    if '.' in ticker:
        ticker = ticker.split('.')[0]
    
    # Add .IS suffix
    return f"{ticker}.IS"

def validate_ticker_list(tickers: List[str]) -> List[str]:
    """Validate and correct a list of tickers"""
    validated = []
    for t in tickers:
        vt = validate_bist_ticker(t)
        if vt and vt not in validated:
            validated.append(vt)
    return validated

# ------------------------------------------------------------
# Persistent Cache System
# ------------------------------------------------------------
class PersistentCache:
    """Filesystem-based cache with TTL to reduce API calls"""
    
    def __init__(self, cache_dir: str = "bist_cache", ttl_hours: int = 12):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 60 * 60
    
    def _get_cache_key(self, tickers: List[str], start: str, end: str) -> str:
        """Generate unique cache key from sorted tickers"""
        # Sort and ensure consistent format
        sorted_tickers = sorted([validate_bist_ticker(t) for t in tickers if t])
        key_str = f"{sorted_tickers}|{start}|{end}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """Retrieve from cache if not expired"""
        cache_key = self._get_cache_key(tickers, start, end)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            mod_time = cache_file.stat().st_mtime
            age = time.time() - mod_time
            
            if age < self.ttl_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
        return None
    
    def set(self, tickers: List[str], start: str, end: str, data: pd.DataFrame):
        """Save to cache"""
        cache_key = self._get_cache_key(tickers, start, end)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

# ------------------------------------------------------------
# CRITICAL FIX: Yahoo Finance Fetcher with exact ticker formats
# ------------------------------------------------------------
@dataclass
class FetchReport:
    mode: str
    requested: List[str]
    received_cols: List[str]
    dropped_raw: List[str]
    dropped_clean: List[str]
    notes: List[str]
    fetch_time: float = 0.0
    cache_hit: bool = False
    invalid_tickers: List[str] = field(default_factory=list)

class YahooFinanceFetcher:
    """Yahoo Finance fetcher with exact ticker formats"""
    
    def __init__(self):
        self.cache = PersistentCache()
    
    def _extract_close_prices(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Extract close prices from Yahoo Finance download"""
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Single ticker case
        if len(tickers) == 1:
            ticker = tickers[0]
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    return pd.DataFrame({ticker: data['Close']})
                elif 'Adj Close' in data.columns:
                    return pd.DataFrame({ticker: data['Adj Close']})
                # Try to find any price column
                price_cols = [c for c in data.columns if c in ['Close', 'Adj Close', 'Price']]
                if price_cols:
                    return pd.DataFrame({ticker: data[price_cols[0]]})
            return pd.DataFrame()
        
        # Multi-ticker case with MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            # Try to get Close prices
            if 'Close' in data.columns.get_level_values(0):
                closes = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                closes = data['Adj Close']
            else:
                # Try to find any price level
                for level in data.columns.get_level_values(0).unique():
                    if level in ['Close', 'Adj Close', 'Price']:
                        closes = data[level]
                        break
                else:
                    return pd.DataFrame()
            
            # Convert to DataFrame if it's a Series
            if isinstance(closes, pd.Series):
                closes = closes.to_frame()
            return closes
        
        return pd.DataFrame()
    
    def fetch_batch(self, tickers: List[str], start: str, end: str, max_retries: int = 2) -> pd.DataFrame:
        """Fetch a batch of tickers"""
        if not tickers:
            return pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                    timeout=20
                )
                
                closes = self._extract_close_prices(data, tickers)
                if closes is not None and not closes.empty:
                    return closes
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    st.caption(f"Batch fetch error for {len(tickers)} tickers: {str(e)[:50]}")
                time.sleep(1)
        
        return pd.DataFrame()
    
    def fetch_single(self, ticker: str, start: str, end: str) -> Optional[pd.Series]:
        """Fetch a single ticker"""
        try:
            ticker_data = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            
            if ticker_data is not None and not ticker_data.empty:
                if 'Close' in ticker_data.columns:
                    return ticker_data['Close']
                elif 'Adj Close' in ticker_data.columns:
                    return ticker_data['Adj Close']
        except Exception:
            pass
        return None
    
    def fetch_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, FetchReport]:
        """
        Main fetch method with exact ticker formats
        """
        start_time = time.time()
        
        # Validate all tickers first
        validated_tickers = []
        invalid_tickers = []
        
        for t in tickers:
            vt = validate_bist_ticker(t)
            if vt:
                if vt not in validated_tickers:
                    validated_tickers.append(vt)
            else:
                invalid_tickers.append(t)
        
        report = FetchReport(
            mode="unknown",
            requested=tickers,
            received_cols=[],
            dropped_raw=[],
            dropped_clean=[],
            notes=[],
            invalid_tickers=invalid_tickers
        )
        
        if not validated_tickers:
            report.notes.append("No valid tickers after validation")
            report.fetch_time = time.time() - start_time
            return pd.DataFrame(), report
        
        # Try cache first
        if use_cache:
            cached = self.cache.get(validated_tickers, start, end)
            if cached is not None and not cached.empty:
                report.mode = "cache"
                report.cache_hit = True
                report.received_cols = cached.columns.tolist()
                report.dropped_raw = [t for t in validated_tickers if t not in cached.columns]
                report.fetch_time = time.time() - start_time
                return cached, report
        
        # Strategy 1: Try batch download with smaller chunks
        # Yahoo works better with chunks of 10-15 tickers
        chunk_size = 12
        all_data = []
        successful_tickers = []
        
        for i in range(0, len(validated_tickers), chunk_size):
            chunk = validated_tickers[i:i + chunk_size]
            chunk_data = self.fetch_batch(chunk, start, end)
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
                successful_tickers.extend(chunk_data.columns.tolist())
            
            # Small delay between chunks
            time.sleep(0.5)
        
        if all_data:
            # Combine all chunks
            combined = pd.concat(all_data, axis=1)
            combined = combined.sort_index()
            
            # Remove duplicate columns
            combined = combined.loc[:, ~combined.columns.duplicated()]
            
            report.mode = "chunked"
            report.received_cols = combined.columns.tolist()
            report.dropped_raw = [t for t in validated_tickers if t not in combined.columns]
            
            # Cache the result
            if use_cache and not combined.empty:
                self.cache.set(validated_tickers, start, end, combined)
            
            report.fetch_time = time.time() - start_time
            return combined, report
        
        # Strategy 2: Fallback to per-ticker fetch
        report.mode = "sequential"
        sequential_data = {}
        
        # Try to fetch each ticker individually
        for ticker in validated_tickers:
            series = self.fetch_single(ticker, start, end)
            if series is not None:
                sequential_data[ticker] = series
            time.sleep(0.2)  # Rate limiting
        
        if sequential_data:
            combined = pd.DataFrame(sequential_data)
            combined = combined.sort_index()
            
            report.received_cols = combined.columns.tolist()
            report.dropped_raw = [t for t in validated_tickers if t not in combined.columns]
            
            # Cache the result
            if use_cache and not combined.empty:
                self.cache.set(validated_tickers, start, end, combined)
        
        report.fetch_time = time.time() - start_time
        return combined if sequential_data else pd.DataFrame(), report

# ------------------------------------------------------------
# Data cleaning functions
# ------------------------------------------------------------
def clean_prices(
    prices: pd.DataFrame,
    ffill_limit: int = 5,
    min_obs: int = 80,
    max_missing_pct: float = 0.25,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Clean price data with forward fill only"""
    if prices is None or prices.empty:
        return pd.DataFrame(), [], []
    
    df = prices.copy()
    df = df.dropna(axis=1, how="all")
    df = df.ffill(limit=ffill_limit)
    
    missing_pct = df.isna().mean()
    drop_missing = missing_pct[missing_pct > max_missing_pct].index.tolist()
    
    obs = df.notna().sum()
    drop_minobs = obs[obs < min_obs].index.tolist()
    
    drop_cols = sorted(set(drop_missing + drop_minobs))
    cleaned = df.drop(columns=drop_cols, errors="ignore")
    cleaned = cleaned.dropna(how="all")
    
    return cleaned, drop_missing, drop_minobs

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from prices"""
    r = prices.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan)
    r = r.dropna(how="any")
    return r

# ------------------------------------------------------------
# Risk calculation functions
# ------------------------------------------------------------
def portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return pd.Series(returns.values @ w, index=returns.index, name="Portfolio")

def risk_contributions(returns: pd.DataFrame, w: np.ndarray) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Calculate risk contributions"""
    tickers = returns.columns.tolist()
    n = len(tickers)
    
    cov = returns.cov().values * 252
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()
    
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 1e-12)))
    
    indiv_vol = np.sqrt(np.clip(np.diag(cov), 0, None))
    
    mrc = (cov @ w) / port_vol
    crc = w * mrc
    pct = (crc / port_vol) * 100.0
    
    df = pd.DataFrame({
        "Symbol": tickers,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in tickers],
        "Weight": w,
        "Individual_Vol": indiv_vol,
        "Risk_Contribution_%": pct,
    }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    
    metrics = {
        "volatility": port_vol,
        "diversification": float(np.sum(w * indiv_vol)) / port_vol,
        "n_assets": n,
    }
    
    return df, metrics, pd.DataFrame(cov, index=tickers, columns=tickers)

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    st.markdown("# 📊 BIST Risk Budgeting System")
    st.markdown(f"*{SIGNATURE}*")
    st.markdown("---")
    
    # Initialize fetcher
    fetcher = YahooFinanceFetcher()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📅 Date Range")
        today = datetime.now().date()
        start_date = st.date_input("Start", value=date(2020, 1, 1), max_value=today - timedelta(days=30))
        end_date = st.date_input("End", value=today, max_value=today)
        
        st.markdown("### 📈 Universe")
        universe_option = st.radio("Select universe:", ["BIST50", "Custom"])
        
        if universe_option == "BIST50":
            tickers = BIST50_TICKERS
            st.caption(f"{len(tickers)} tickers")
        else:
            custom = st.text_area(
                "Enter tickers (one per line or comma-separated)",
                "AKBNK.IS\nGARAN.IS\nTHYAO.IS"
            )
            # Parse and validate tickers
            raw_tickers = re.split(r'[,\n\s]+', custom)
            tickers = validate_ticker_list([t for t in raw_tickers if t])
            st.caption(f"{len(tickers)} valid tickers")
        
        st.markdown("### ⚙️ Settings")
        ffill_limit = st.slider("Forward-fill limit", 1, 10, 5)
        min_obs = st.slider("Min observations", 30, 300, 120)
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if not run_button:
        st.info("👈 Configure settings and click 'Run Analysis'")
        return
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return
    
    # Progress
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Step 1: Validate tickers
    status.text("✅ Validating tickers...")
    validated_tickers = validate_ticker_list(tickers)
    invalid_tickers = [t for t in tickers if validate_bist_ticker(t) not in validated_tickers]
    
    if invalid_tickers:
        st.warning(f"Invalid tickers ignored: {invalid_tickers}")
    
    progress_bar.progress(10)
    
    # Step 2: Fetch benchmark
    status.text("📡 Checking benchmark...")
    benchmark_ticker = None
    for b in BENCHMARK_CANDIDATES:
        try:
            test = yf.download(b, start=start_date, end=end_date, progress=False)
            if not test.empty:
                benchmark_ticker = b
                break
        except:
            continue
    
    progress_bar.progress(20)
    
    # Step 3: Fetch prices
    status.text("📥 Fetching price data...")
    
    # Add benchmark to fetch list if found
    fetch_tickers = validated_tickers.copy()
    if benchmark_ticker:
        fetch_tickers.append(benchmark_ticker)
    
    prices_raw, report = fetcher.fetch_prices(fetch_tickers, str(start_date), str(end_date))
    
    if prices_raw.empty:
        st.error("❌ No data received from Yahoo Finance")
        with st.expander("Diagnostics"):
            st.json({
                "mode": report.mode,
                "cache_hit": report.cache_hit,
                "invalid_tickers": report.invalid_tickers,
                "notes": report.notes
            })
        return
    
    progress_bar.progress(50)
    
    # Step 4: Separate benchmark
    bench_prices = None
    asset_prices_raw = prices_raw.copy()
    
    if benchmark_ticker and benchmark_ticker in prices_raw.columns:
        bench_prices = prices_raw[benchmark_ticker].copy()
        asset_prices_raw = prices_raw.drop(columns=[benchmark_ticker])
    
    progress_bar.progress(60)
    
    # Step 5: Clean prices
    status.text("🧹 Cleaning data...")
    asset_prices, drop_missing, drop_minobs = clean_prices(
        asset_prices_raw,
        ffill_limit=ffill_limit,
        min_obs=min_obs
    )
    
    if asset_prices.shape[1] < 2:
        st.error("❌ Insufficient assets after cleaning")
        return
    
    progress_bar.progress(80)
    
    # Step 6: Calculate returns
    returns = compute_returns(asset_prices)
    
    # Step 7: Equal weight portfolio (simplified for demo)
    w = np.ones(returns.shape[1]) / returns.shape[1]
    
    # Step 8: Calculate risk metrics
    risk_df, metrics, cov_df = risk_contributions(returns, w)
    port_returns = portfolio_returns(returns, w)
    
    progress_bar.progress(100)
    status.text("✅ Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status.empty()
    
    # Display results
    st.markdown("## 📊 Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Assets", returns.shape[1])
    with col2:
        st.metric("Days", returns.shape[0])
    with col3:
        st.metric("Volatility", f"{metrics['volatility']:.2%}")
    with col4:
        st.metric("Div Ratio", f"{metrics['diversification']:.2f}")
    
    # Risk contribution chart
    st.markdown("### 🎯 Risk Contributions")
    fig = go.Figure()
    sorted_df = risk_df.sort_values("Risk_Contribution_%", ascending=True)
    fig.add_trace(go.Bar(
        y=sorted_df["Symbol"],
        x=sorted_df["Risk_Contribution_%"],
        orientation='h',
        marker_color=sorted_df["Risk_Contribution_%"],
        text=sorted_df["Risk_Contribution_%"].round(1).astype(str) + '%',
        textposition='outside'
    ))
    fig.update_layout(height=600, title="Risk Contribution by Asset")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### 📋 Asset Details")
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # Diagnostics
    with st.expander("🔍 Fetch Diagnostics"):
        st.json({
            "fetch_mode": report.mode,
            "cache_hit": report.cache_hit,
            "fetch_time": f"{report.fetch_time:.2f}s",
            "requested": len(report.requested),
            "received": len(report.received_cols),
            "dropped_raw": len(report.dropped_raw),
            "dropped_clean": len(drop_missing) + len(drop_minobs),
            "benchmark": benchmark_ticker
        })

if __name__ == "__main__":
    main()
