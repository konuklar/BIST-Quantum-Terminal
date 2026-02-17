# ============================================================
# 📊 ADVANCED BIST Risk Budgeting System (Streamlit Cloud) - ENHANCED
# Yahoo Finance ONLY • No Synthetic Series • Forward-Fill Only
# 
# IMPROVEMENTS:
# - Enhanced data fetching with parallel processing and smart retry logic
# - Persistent caching layer to reduce API calls
# - Advanced risk metrics (Tail Ratio, Pain Index, Ulcer Index)
# - Monte Carlo simulation for stress testing
# - Interactive visualization enhancements
# - Better error handling and diagnostics
# - Performance optimizations for large universes
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
# Optional packages
# ------------------------------------------------------------
PYPFOPT_AVAILABLE = False
PYPFOPT_IMPORT_ERROR = None
try:
    import cvxpy as cp
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.hierarchical_risk_parity import HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception as e:
    PYPFOPT_AVAILABLE = False
    PYPFOPT_IMPORT_ERROR = str(e)

# ------------------------------------------------------------
# Streamlit config + CSS
# ------------------------------------------------------------
st.set_page_config(
    page_title="BIST Risk Budgeting Terminal", 
    page_icon="📊", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

SIGNATURE = "The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"

st.markdown(
    """
<style>
.main-header { font-size: 2.2rem; font-weight: 800; color: #0B2A6B; margin: 0.2rem 0 0.6rem 0;}
.sub-header { font-size: 1.25rem; font-weight: 700; color: #1D4ED8; margin: 0.75rem 0 0.4rem 0;}
.badge { display:inline-block; padding: .25rem .7rem; border-radius: 999px; background: #0B2A6B; color: white; font-size: 0.82rem; }
.kpi { background: #F3F4F6; padding: 0.9rem; border-radius: 0.7rem; border: 1px solid rgba(0,0,0,.06); }
.small { font-size: 0.9rem; color: rgba(0,0,0,.65); }
.warn { color: #B91C1C; font-weight: 700; }
.ok { color: #047857; font-weight: 700; }
code { font-size: 0.9rem; }
.diagnostic-box { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; }
.highlight { background: #FEF9C3; padding: 0.2rem 0.4rem; border-radius: 0.3rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Universe + metadata (baseline BIST50-style list)
# ------------------------------------------------------------
BASE_UNIVERSE_BIST50 = [
    "AKBNK.IS","ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS",
    "BIMAS.IS","CCOLA.IS","DOAS.IS","EGEEN.IS","EKGYO.IS",
    "ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS",
    "HEKTS.IS","ISCTR.IS","KCHOL.IS","KRDMD.IS","MGROS.IS",
    "ODAS.IS","OTKAR.IS","PETKM.IS","PGSUS.IS","SAHOL.IS",
    "SASA.IS","SISE.IS","SOKM.IS","TCELL.IS","THYAO.IS",
    "TKFEN.IS","TOASO.IS","TSKB.IS","TTKOM.IS","TTRAK.IS",
    "TUPRS.IS","ULKER.IS","VAKBN.IS","VESTL.IS","YKBNK.IS",
    "KONTR.IS","KLSER.IS","CIMSA.IS","KOZAA.IS","ENJSA.IS",
    "BRSAN.IS","BAGFS.IS","KMPUR.IS","AKSEN.IS","AEFES.IS"
]

# Explicit exclusions
EXCLUDED = {"KOZAL.IS", "TRALTIN.IS", "TRALTIN", "TRALT", "TRALTIN.IS"}
BASE_UNIVERSE_BIST50 = [t for t in BASE_UNIVERSE_BIST50 if t not in EXCLUDED]

# Benchmark and factor candidates
BENCHMARK_CANDIDATES = ["XU100.IS", "^XU100"]
FX_USDTRY_CANDIDATES = ["TRY=X", "USDTRY=X"]  
FX_EURTRY_CANDIDATES = ["EURTRY=X"]    
RATE_CANDIDATES = ["TR10YT=RR", "^TNX"]  
GOLD_CANDIDATES = ["GC=F"]
OIL_CANDIDATES = ["CL=F"]

# Sector mapping (enhanced)
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
# Persistent Cache System
# ------------------------------------------------------------
class PersistentCache:
    """Filesystem-based cache with TTL to reduce API calls"""
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 12):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 60 * 60
    
    def _get_cache_key(self, tickers: List[str], start: str, end: str) -> str:
        """Generate unique cache key"""
        key_str = f"{sorted(set(tickers))}|{start}|{end}"
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
                except Exception as e:
                    st.caption(f"Cache read error: {e}")
        return None
    
    def set(self, tickers: List[str], start: str, end: str, data: pd.DataFrame):
        """Save to cache"""
        cache_key = self._get_cache_key(tickers, start, end)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.caption(f"Cache write error: {e}")

# ------------------------------------------------------------
# Enhanced Yahoo Finance Fetcher with Parallel Processing
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

class EnhancedYahooFetcher:
    """Enhanced fetcher with parallel processing and intelligent retry logic"""
    
    def __init__(self):
        self.cache = PersistentCache()
        self.session = self._create_session()
    
    def _create_session(self):
        """Create requests session with optimal settings"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=20,
            pool_maxsize=20
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session
    
    def _extract_close_from_download(self, df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Extract close prices from yfinance output"""
        if df is None or df.empty:
            return pd.DataFrame()

        if len(tickers) == 1 and isinstance(df.columns, pd.Index):
            if "Adj Close" in df.columns:
                return pd.DataFrame({tickers[0]: df["Adj Close"]})
            if "Close" in df.columns:
                return pd.DataFrame({tickers[0]: df["Close"]})
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            for price_type in ["Adj Close", "Close"]:
                if price_type in df.columns.get_level_values(0):
                    out = df.xs(price_type, axis=1, level=0, drop_level=True)
                    return out
                if price_type in df.columns.get_level_values(1):
                    out = df.xs(price_type, axis=1, level=1, drop_level=True)
                    return out

        return pd.DataFrame()
    
    def _fetch_batch(self, tickers: List[str], start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
        """Fetch a batch of tickers with retry logic"""
        last_err = None
        
        for attempt in range(max_retries):
            try:
                raw = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    progress=False,
                    threads=False,  # More stable for batch
                    timeout=30
                )
                close = self._extract_close_from_download(raw, tickers)
                if close is not None and not close.empty:
                    return close
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        if last_err:
            st.caption(f"Batch fetch error: {last_err}")
        return pd.DataFrame()
    
    def _fetch_parallel(self, tickers: List[str], start: str, end: str, max_workers: int = 10) -> pd.DataFrame:
        """Parallel fetch for many tickers"""
        results = {}
        
        def fetch_single(ticker):
            try:
                hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
                if hist is not None and not hist.empty:
                    if "Close" in hist.columns:
                        return ticker, hist["Close"]
                    elif "Adj Close" in hist.columns:
                        return ticker, hist["Adj Close"]
            except:
                pass
            return ticker, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_single, t): t for t in tickers}
            for future in as_completed(future_to_ticker):
                ticker, series = future.result()
                if series is not None:
                    results[ticker] = series
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    def fetch_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        chunk_size: int = 25,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, FetchReport]:
        """Main fetch method with multiple strategies"""
        
        start_time = time.time()
        
        # Clean and normalize tickers
        tickers = [t for t in tickers if t and t not in EXCLUDED]
        tickers = [_normalize_ticker(t) for t in tickers]
        tickers = list(dict.fromkeys(tickers))
        
        report = FetchReport(
            mode="unknown",
            requested=tickers,
            received_cols=[],
            dropped_raw=[],
            dropped_clean=[],
            notes=[],
            cache_hit=False
        )
        
        # Strategy 1: Check cache
        if use_cache:
            cached = self.cache.get(tickers, start, end)
            if cached is not None and not cached.empty:
                report.mode = "cache"
                report.cache_hit = True
                report.received_cols = cached.columns.tolist()
                report.dropped_raw = [t for t in tickers if t not in cached.columns]
                report.fetch_time = time.time() - start_time
                return cached, report
        
        close = pd.DataFrame()
        
        # Strategy 2: Batch download
        if len(tickers) <= 50:  # Only for manageable sizes
            close = self._fetch_batch(tickers, start, end)
            if not close.empty:
                report.mode = "batch"
        
        # Strategy 3: Chunked download
        if close.empty and len(tickers) > 1:
            report.mode = "chunked"
            parts = []
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i + chunk_size]
                d = self._fetch_batch(chunk, start, end)
                if not d.empty:
                    parts.append(d)
                time.sleep(0.5)  # Rate limiting
            if parts:
                close = pd.concat(parts, axis=1)
        
        # Strategy 4: Parallel fetch
        if close.empty:
            report.mode = "parallel"
            close = self._fetch_parallel(tickers, start, end)
        
        # Strategy 5: Sequential fallback
        if close.empty:
            report.mode = "sequential"
            data = {}
            for t in tickers:
                try:
                    hist = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
                    if hist is not None and not hist.empty:
                        if "Close" in hist.columns:
                            data[t] = hist["Close"]
                except Exception as e:
                    report.notes.append(f"{t}: {str(e)[:50]}")
                time.sleep(0.2)  # Rate limiting
            if data:
                close = pd.DataFrame(data)
        
        if close.empty:
            report.fetch_time = time.time() - start_time
            return close, report
        
        # Standardize
        close = close.copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index()
        close = close.loc[:, ~close.columns.duplicated()]
        
        # Update report
        report.received_cols = close.columns.tolist()
        report.dropped_raw = [t for t in tickers if t not in close.columns]
        report.fetch_time = time.time() - start_time
        
        # Cache the result
        if use_cache and not close.empty:
            self.cache.set(tickers, start, end, close)
        
        return close, report

# ------------------------------------------------------------
# Enhanced Risk Analytics
# ------------------------------------------------------------
class AdvancedRiskMetrics:
    """Advanced risk metrics calculations"""
    
    @staticmethod
    def calculate_drawdowns(returns: pd.Series) -> pd.DataFrame:
        """Calculate drawdown metrics including underwater period"""
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth - peak) / peak
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_start = None
        drawdown_periods = []
        
        for i, val in enumerate(is_drawdown):
            if val and drawdown_start is None:
                drawdown_start = i
            elif not val and drawdown_start is not None:
                drawdown_periods.append((drawdown_start, i))
                drawdown_start = None
        
        max_duration = max((end - start for start, end in drawdown_periods), default=0)
        
        return pd.DataFrame({
            'Drawdown': drawdown,
            'Wealth_Index': wealth,
            'Peak': peak,
            'Max_Drawdown': drawdown.min(),
            'Max_Duration_Days': max_duration
        })
    
    @staticmethod
    def tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive tail risk measures"""
        sorted_returns = returns.sort_values()
        
        metrics = {}
        
        # VaR and CVaR at different levels
        for conf in [0.95, 0.99]:
            alpha = 1 - conf
            var = sorted_returns.quantile(alpha)
            cvar = sorted_returns[sorted_returns <= var].mean()
            
            metrics[f'VaR_{int(conf*100)}'] = var
            metrics[f'CVaR_{int(conf*100)}'] = cvar
        
        # Tail Ratio (right tail / left tail)
        right_tail = sorted_returns.quantile(0.95)
        left_tail = abs(sorted_returns.quantile(0.05))
        metrics['Tail_Ratio'] = right_tail / left_tail if left_tail != 0 else np.nan
        
        # Pain Index and Ulcer Index
        wealth = (1 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        metrics['Pain_Index'] = abs(drawdown).mean() * 100
        metrics['Ulcer_Index'] = np.sqrt((drawdown ** 2).mean()) * 100
        
        # Distribution metrics
        metrics['Skewness'] = returns.skew()
        metrics['Kurtosis'] = returns.kurtosis()
        metrics['Jarque_Bera'] = stats.jarque_bera(returns.dropna())[0]
        
        return metrics
    
    @staticmethod
    def stability_metrics(returns: pd.Series, rf: float = 0.15) -> Dict[str, float]:
        """Calculate strategy stability metrics"""
        # Rolling Sharpe stability
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        sharpe_std = rolling_sharpe.std()
        sharpe_mean = rolling_sharpe.mean()
        
        # Hit rate (percentage of positive days)
        hit_rate = (returns > 0).mean()
        
        # Gain to Pain ratio
        avg_gain = returns[returns > 0].mean() if any(returns > 0) else 0
        avg_loss = abs(returns[returns < 0].mean()) if any(returns < 0) else 1
        gain_pain_ratio = avg_gain / avg_loss if avg_loss > 0 else np.nan
        
        # Calmar ratio
        cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = AdvancedRiskMetrics.calculate_drawdowns(returns)['Max_Drawdown'].iloc[0]
        calmar_ratio = cagr / abs(max_dd) if max_dd != 0 else np.nan
        
        # Sortino ratio
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = (returns.mean() * 252 - rf) / downside if downside > 0 else np.nan
        
        return {
            'Sharpe_Stability': sharpe_std / sharpe_mean if sharpe_mean != 0 else np.nan,
            'Hit_Rate': hit_rate,
            'Gain_Pain_Ratio': gain_pain_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Sortino_Ratio': sortino,
            'Avg_Gain': avg_gain,
            'Avg_Loss': avg_loss
        }

# ------------------------------------------------------------
# Monte Carlo Simulation
# ------------------------------------------------------------
class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio stress testing"""
    
    @staticmethod
    def simulate(
        returns: pd.DataFrame,
        weights: np.ndarray,
        n_simulations: int = 10000,
        horizon_days: int = 252,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio returns"""
        
        np.random.seed(seed)
        
        # Calculate parameters
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        
        # Generate correlated random returns
        n_assets = len(returns.columns)
        simulated_returns = np.random.multivariate_normal(
            mean=mu,
            cov=sigma,
            size=(n_simulations, horizon_days)
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = simulated_returns @ weights
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        
        # Calculate metrics
        final_values = cumulative_returns[:, -1]
        
        results = {
            'final_values': final_values,
            'mean_final': np.mean(final_values),
            'median_final': np.median(final_values),
            'std_final': np.std(final_values),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
            'cvar_95': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
            'probability_loss': np.mean(final_values < 1) * 100,
            'probability_double': np.mean(final_values > 2) * 100,
            'all_simulations': cumulative_returns
        }
        
        return results
    
    @staticmethod
    def plot_simulations(sim_results: Dict[str, Any]) -> go.Figure:
        """Plot Monte Carlo simulation results"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Simulation Paths', 'Distribution of Final Values',
                          'Value at Risk (VaR)', 'Probability Distribution')
        )
        
        # Plot a sample of paths
        n_show = min(100, len(sim_results['all_simulations']))
        for i in range(n_show):
            fig.add_trace(
                go.Scatter(
                    y=sim_results['all_simulations'][i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(100,100,255,0.2)'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=sim_results['final_values'],
                nbinsx=50,
                name='Distribution',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Box plot for VaR
        fig.add_trace(
            go.Box(
                y=sim_results['final_values'],
                name='Final Values',
                boxmean='sd'
            ),
            row=2, col=1
        )
        
        # Violin plot
        fig.add_trace(
            go.Violin(
                y=sim_results['final_values'],
                name='Distribution',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Monte Carlo Simulation Results",
            showlegend=False,
            template='plotly_white'
        )
        
        return fig

# ------------------------------------------------------------
# Utilities (existing functions preserved)
# ------------------------------------------------------------
def _normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if t.startswith("^") or t.endswith("=X") or t.endswith("=RR") or t.endswith("=XX"):
        return t
    if "." in t:
        return t
    return f"{t}.IS"

def _sanitize_sheet_name(name: str) -> str:
    bad = r'[:\\/*?\[\]]'
    name = re.sub(bad, " ", str(name))
    name = name.strip()
    if not name:
        name = "Sheet"
    return name[:31]

def _excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fix ValueError in pandas Excel formatter"""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[c]):
            out[c] = out[c].dt.tz_convert(None)
    if len(out.columns) > 0:
        obj_cols = [c for c in out.columns if out[c].dtype == "object"]
        for c in obj_cols:
            def _coerce(v):
                if isinstance(v, (dict, list, tuple, set, np.ndarray)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                if isinstance(v, (pd.Timestamp, datetime, date)):
                    try:
                        return pd.to_datetime(v).to_pydatetime().replace(tzinfo=None).isoformat(sep=" ")
                    except Exception:
                        return str(v)
                return v
            out[c] = out[c].map(_coerce)
    return out

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write multiple dataframes to a single Excel file safely."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        cover = pd.DataFrame({
            "Label": ["Signature", "Generated", "Version"],
            "Value": [SIGNATURE, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Enhanced v2.0"]
        })
        _excel_safe_df(cover).to_excel(writer, sheet_name="Cover", index=False)

        for name, df in sheets.items():
            safe_name = _sanitize_sheet_name(name)
            safe_df = _excel_safe_df(df)
            if not safe_df.empty:
                safe_df.to_excel(writer, sheet_name=safe_name, index=False)
    bio.seek(0)
    return bio.read()

def _pick_first_working_ticker(candidates: List[str], start: str, end: str) -> Optional[str]:
    for t in candidates:
        try:
            d = yf.download(t, start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
            if d is not None and not d.empty:
                return t
        except Exception:
            pass
    return None

# ------------------------------------------------------------
# Data cleaning functions (preserved from original)
# ------------------------------------------------------------
def clean_prices(
    prices: pd.DataFrame,
    ffill_limit: int = 5,
    min_obs: int = 80,
    max_missing_pct: float = 0.25,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Forward fill only, drop too-missing columns, enforce minimum observations."""
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
    r = prices.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan)
    r = r.dropna(how="any")
    return r

# ------------------------------------------------------------
# Risk Analytics Functions (preserved from original)
# ------------------------------------------------------------
def portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return (returns.values @ w).astype(float).reshape(-1)

def historical_var_cvar_es(r: pd.Series, alpha: float = 0.05) -> Tuple[float, float, float]:
    r = r.dropna().astype(float)
    if len(r) < 10:
        return np.nan, np.nan, np.nan
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) else np.nan
    es = cvar
    return float(var), float(cvar), float(es)

def risk_contributions(returns: pd.DataFrame, w: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    tickers = returns.columns.tolist()
    n = len(tickers)

    cov = returns.cov().values * 252
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()

    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    indiv_vol = np.sqrt(np.clip(np.diag(cov), 0, None))

    mrc = (cov @ w) / (port_vol + 1e-12)
    crc = w * mrc
    pct = (crc / (port_vol + 1e-12)) * 100.0

    df = pd.DataFrame({
        "Symbol": tickers,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in tickers],
        "Weight": w,
        "Individual_Volatility": indiv_vol,
        "Marginal_Risk_Contribution": mrc,
        "Component_Risk": crc,
        "Risk_Contribution_%": pct,
    }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
    df["Risk_Rank"] = np.arange(1, len(df) + 1)

    weighted_avg_vol = float(np.sum(w * indiv_vol))
    div_ratio = weighted_avg_vol / (port_vol + 1e-12)

    metrics = {
        "portfolio_vol": port_vol,
        "portfolio_var": port_var,
        "diversification_ratio": div_ratio,
        "n_assets": n,
        "avg_indiv_vol": float(np.mean(indiv_vol)) if len(indiv_vol) else np.nan,
        "top_risk_asset": df.iloc[0]["Symbol"] if len(df) else None,
        "top_risk_pct": float(df.iloc[0]["Risk_Contribution_%"]) if len(df) else np.nan,
    }
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    return df, metrics, cov_df

def rolling_risk_contributions(returns: pd.DataFrame, w: np.ndarray, window: int = 60) -> pd.DataFrame:
    tickers = returns.columns.tolist()
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()
    out = []
    for i in range(window, len(returns) + 1):
        rwin = returns.iloc[i - window:i]
        cov = rwin.cov().values * 252
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        mrc = (cov @ w) / (port_vol + 1e-12)
        crc = w * mrc
        pct = (crc / (port_vol + 1e-12)) * 100.0
        out.append(pd.Series(pct, index=tickers, name=returns.index[i - 1]))
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)

def active_risk_contributions(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, w: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()

    common = asset_returns.index.intersection(benchmark_returns.index)
    R = asset_returns.loc[common].copy()
    b = benchmark_returns.loc[common].astype(float)

    A = R.sub(b, axis=0)
    A = A.dropna(how="any")
    if len(A) < 20 or A.shape[1] < 2:
        return pd.DataFrame(), {"tracking_error": np.nan}

    covA = A.cov().values * 252
    te_var = float(w @ covA @ w)
    te = float(np.sqrt(max(te_var, 0.0)))

    mrc = (covA @ w) / (te + 1e-12)
    crc = w * mrc
    pct = (crc / (te + 1e-12)) * 100.0

    df = pd.DataFrame({
        "Symbol": A.columns,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in A.columns],
        "Weight": w,
        "Active_Risk_Contribution_%": pct,
        "Active_Component_Risk": crc,
        "Active_MRC": mrc
    }).sort_values("Active_Risk_Contribution_%", ascending=False).reset_index(drop=True)

    stats = {"tracking_error": te, "tracking_error_var": te_var}
    return df, stats

# ------------------------------------------------------------
# Optimization (preserved from original)
# ------------------------------------------------------------
def build_sector_indices(tickers: List[str]) -> Dict[str, List[int]]:
    idx = {}
    for i, t in enumerate(tickers):
        s = SECTOR_MAP.get(t, "Other")
        idx.setdefault(s, []).append(i)
    return idx

def optimize_weights(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    method: str,
    rf: float,
    max_w: float,
    sector_caps: Dict[str, float],
    target_return: Optional[float] = None,
    target_vol: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    tickers = returns.columns.tolist()
    n = len(tickers)

    debug = {"method": method, "used_pypfopt": False, "notes": []}
    w_eq = np.ones(n) / n

    if method == "Equal Weight":
        perf = {
            "expected_return": float(returns.mean().values @ w_eq * 252),
            "volatility": float(np.sqrt(w_eq @ (returns.cov().values * 252) @ w_eq)),
            "sharpe_ratio": (float(returns.mean().values @ w_eq * 252) - rf) / 
                           (float(np.sqrt(w_eq @ (returns.cov().values * 252) @ w_eq)) + 1e-12)
        }
        return w_eq, perf, debug

    if method == "Risk Parity (SLSQP)":
        cov = returns.cov().values * 252

        def obj(x):
            x = np.clip(x, 0, max_w)
            x = x / (x.sum() + 1e-12)
            port_var = x @ cov @ x
            port_vol = np.sqrt(max(port_var, 0.0))
            mrc = (cov @ x) / (port_vol + 1e-12)
            rc = x * mrc
            target = port_vol / n
            return float(np.sum((rc - target) ** 2))

        cons = [{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}]
        bnds = [(0.0, max_w) for _ in range(n)]
        x0 = w_eq.copy()

        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, 
                      options={"maxiter": 2000, "ftol": 1e-10})
        w = res.x if res.success else w_eq
        w = np.clip(w, 0, max_w)
        w = w / (w.sum() + 1e-12)

        perf = {
            "expected_return": float(returns.mean().values @ w * 252),
            "volatility": float(np.sqrt(w @ (returns.cov().values * 252) @ w)),
            "sharpe_ratio": (float(returns.mean().values @ w * 252) - rf) / 
                           (float(np.sqrt(w @ (returns.cov().values * 252) @ w)) + 1e-12)
        }
        debug["notes"].append(f"SLSQP success={res.success}")
        return w, perf, debug

    if not PYPFOPT_AVAILABLE:
        debug["notes"].append("PyPortfolioOpt not available; returning Equal Weight.")
        return w_eq, {"expected_return": np.nan, "volatility": np.nan, "sharpe_ratio": np.nan}, debug

    debug["used_pypfopt"] = True

    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= max_w)

    sector_idx = build_sector_indices(tickers)
    for sec, idxs in sector_idx.items():
        cap = float(sector_caps.get(sec, 1.0))
        if cap < 1.0 - 1e-9:
            ef.add_constraint(lambda w, idxs=idxs, cap=cap: cp.sum(w[idxs]) <= cap)

    try:
        if method == "Max Sharpe":
            ef.max_sharpe(risk_free_rate=rf)
        elif method == "Min Volatility":
            ef.min_volatility()
        elif method == "Max Utility":
            ef.max_quadratic_utility(risk_aversion=1.0)
        elif method == "Efficient Return":
            if target_return is None:
                target_return = float(mu.mean())
            ef.efficient_return(target_return)
        elif method == "Efficient Risk":
            if target_vol is None:
                target_vol = float(np.sqrt(np.diag(S)).mean())
            ef.efficient_risk(target_vol)
        else:
            ef.max_sharpe(risk_free_rate=rf)

        w_dict = ef.clean_weights()
        w = np.array([w_dict.get(t, 0.0) for t in tickers], dtype=float)
        w = np.clip(w, 0, max_w)
        w = w / (w.sum() + 1e-12)

        pret, pvol, psr = ef.portfolio_performance(risk_free_rate=rf, verbose=False)
        perf = {"expected_return": float(pret), "volatility": float(pvol), "sharpe_ratio": float(psr)}
        return w, perf, debug

    except Exception as e:
        debug["notes"].append(f"PyPortfolioOpt failed: {e}")
        return w_eq, {"expected_return": np.nan, "volatility": np.nan, "sharpe_ratio": np.nan}, debug

# ------------------------------------------------------------
# Stress scenarios (preserved from original)
# ------------------------------------------------------------
def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if len(y) < 30:
        return np.nan
    vx = np.var(x)
    if vx <= 1e-12:
        return np.nan
    return float(np.cov(y, x)[0, 1] / vx)

def estimate_factor_betas(asset_returns: pd.DataFrame, fx_ret: pd.Series, rate_ret: pd.Series) -> pd.DataFrame:
    common = asset_returns.index
    if fx_ret is not None:
        common = common.intersection(fx_ret.index)
    if rate_ret is not None:
        common = common.intersection(rate_ret.index)

    R = asset_returns.loc[common].copy()
    betas = []
    for c in R.columns:
        y = R[c].values
        b_fx = _ols_beta(y, fx_ret.loc[common].values) if fx_ret is not None else np.nan
        b_rt = _ols_beta(y, rate_ret.loc[common].values) if rate_ret is not None else np.nan
        betas.append((c, b_fx, b_rt, SECTOR_MAP.get(c, "Other")))
    return pd.DataFrame(betas, columns=["Symbol", "Beta_FX_USDTRY", "Beta_RATE", "Sector"])

def scenario_impact(weights: np.ndarray, betas_df: pd.DataFrame, fx_shock: float, rate_shock: float) -> Tuple[pd.DataFrame, float]:
    df = betas_df.copy()
    df["Weight"] = weights
    df["Shock_FX"] = fx_shock
    df["Shock_RATE"] = rate_shock
    df["Impact_Est"] = df["Beta_FX_USDTRY"] * fx_shock + df["Beta_RATE"] * rate_shock
    df["Weighted_Impact"] = df["Weight"] * df["Impact_Est"]
    port = float(df["Weighted_Impact"].sum())
    df = df.sort_values("Weighted_Impact", ascending=True)
    return df, port

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(f'<div class="main-header">📊 Enhanced BIST Risk Budgeting Terminal</div>', unsafe_allow_html=True)
st.markdown(f'<div class="badge">Yahoo Finance ONLY • No Synthetic Series</div> &nbsp; <span class="small">{SIGNATURE}</span>', unsafe_allow_html=True)

# Initialize fetcher
fetcher = EnhancedYahooFetcher()

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    today = datetime.now().date()
    default_start = date(2020, 1, 1)
    start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=5))
    end_date = st.date_input("End date", value=today, max_value=today)

    st.markdown("### 📦 Universe")
    universe_mode = st.selectbox("Universe", ["BIST50 (baseline)", "Custom list"], index=0)
    if universe_mode == "Custom list":
        user_txt = st.text_area("Tickers (comma-separated)", 
                               value=",".join([t.replace(".IS","") for t in BASE_UNIVERSE_BIST50[:20]]))
        universe = [_normalize_ticker(x.strip()) for x in user_txt.split(",") if x.strip()]
    else:
        universe = BASE_UNIVERSE_BIST50.copy()

    universe = [t for t in universe if t not in EXCLUDED]
    if "ASTOR.IS" not in universe:
        universe = ["ASTOR.IS"] + universe

    st.markdown("### 🧹 Data cleaning")
    ffill_limit = st.slider("Forward-fill limit (days)", 0, 15, 5)
    min_obs = st.slider("Min observations per asset", 30, 400, 80)
    max_missing_pct = st.slider("Max missing % per asset", 0.0, 0.8, 0.25, 0.05)

    st.markdown("### 🧠 Optimization")
    rf = st.number_input("Risk-free rate (annual)", 0.0, 1.0, 0.15, 0.01)
    max_w = st.slider("Max weight per stock", 0.02, 0.30, 0.15, 0.01)

    methods = ["Equal Weight", "Risk Parity (SLSQP)", "Max Sharpe", "Min Volatility", 
               "Max Utility", "Efficient Return", "Efficient Risk"]
    opt_method = st.selectbox("Method", methods, index=0)

    st.markdown("### 🏭 Sector caps")
    _secs = sorted({SECTOR_MAP.get(t, "Other") for t in universe})
    sector_caps = {}
    with st.expander("Edit sector caps"):
        for s in _secs:
            sector_caps[s] = st.slider(f"{s}", 0.05, 1.00, 1.00, 0.05)

    st.markdown("### 📊 Analytics")
    roll_window = st.slider("Rolling window (days)", 20, 252, 60, 5)
    var_alpha = st.selectbox("VaR level", ["95%", "99%"], index=0)
    alpha = 0.05 if var_alpha == "95%" else 0.01
    
    enable_monte_carlo = st.checkbox("Enable Monte Carlo", value=True)
    mc_simulations = st.slider("MC Simulations", 1000, 20000, 5000, 1000) if enable_monte_carlo else 0

    st.markdown("### ⚡ Stress shocks")
    fx_shock = st.slider("FX shock (USDTRY)", -0.20, 0.20, 0.05, 0.01)
    rate_shock = st.slider("Rate shock", -0.10, 0.10, 0.02, 0.01)

    st.markdown("---")
    st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M:%S')}")

run = st.button("▶ Run Enhanced Analysis", type="primary")

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------
if not run:
    st.info("👈 Configure parameters and click **Run Enhanced Analysis**")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

start = str(start_date)
end = str(end_date)

# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

# Step 1: Fetch benchmark
status_text.text("📡 Testing benchmark tickers...")
benchmark_ticker = _pick_first_working_ticker(BENCHMARK_CANDIDATES, start, end)
progress_bar.progress(10)

# Step 2: Fetch prices
status_text.text("📥 Fetching price data (this may take a moment)...")
prices_raw, frep = fetcher.fetch_prices(
    universe + ([benchmark_ticker] if benchmark_ticker else []), 
    start, end
)
progress_bar.progress(30)

if prices_raw is None or prices_raw.empty:
    st.error("❌ No data received from Yahoo Finance")
    with st.expander("🔍 Diagnostic Info"):
        st.json({
            "mode": frep.mode,
            "requested": len(frep.requested),
            "cache_hit": frep.cache_hit,
            "fetch_time": f"{frep.fetch_time:.2f}s",
            "notes": frep.notes[:5]
        })
    st.stop()

# Split benchmark
bench_prices = None
asset_prices_raw = prices_raw.copy()
if benchmark_ticker and benchmark_ticker in prices_raw.columns:
    bench_prices = prices_raw[benchmark_ticker].copy()
    asset_prices_raw = prices_raw.drop(columns=[benchmark_ticker], errors="ignore")
progress_bar.progress(50)

# Clean asset prices
status_text.text("🧹 Cleaning and aligning data...")
asset_prices, drop_missing, drop_minobs = clean_prices(
    asset_prices_raw, 
    ffill_limit=ffill_limit, 
    min_obs=min_obs, 
    max_missing_pct=max_missing_pct
)
dropped_clean = sorted(set(drop_missing + drop_minobs))
progress_bar.progress(70)

if asset_prices.shape[1] < 2:
    st.error("❌ Not enough assets after cleaning")
    st.stop()

returns = compute_returns(asset_prices)

if returns.empty or returns.shape[0] < 30:
    st.error("❌ Returns matrix too small")
    st.stop()

# Benchmark returns
bench_returns = None
if bench_prices is not None and not bench_prices.empty:
    bench_prices = bench_prices.ffill(limit=ffill_limit)
    bench_returns = bench_prices.pct_change().dropna()
    bench_returns = bench_returns.replace([np.inf, -np.inf], np.nan).dropna()
    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common].copy()
    bench_returns = bench_returns.loc[common].copy()
progress_bar.progress(80)

# Optimization
status_text.text("🧮 Optimizing portfolio...")
w, perf, dbg = optimize_weights(
    prices=asset_prices,
    returns=returns,
    method=opt_method,
    rf=rf,
    max_w=max_w,
    sector_caps=sector_caps,
)

# Risk calculations
status_text.text("📊 Calculating risk metrics...")
risk_df, port_metrics, cov_df = risk_contributions(returns, w)
port_ret_series = pd.Series(portfolio_returns(returns, w), index=returns.index, name="Portfolio")

# Tail risk
var1, cvar1, es1 = historical_var_cvar_es(port_ret_series, alpha=alpha)
h = 10
var10 = var1 * np.sqrt(h) if np.isfinite(var1) else np.nan
cvar10 = cvar1 * np.sqrt(h) if np.isfinite(cvar1) else np.nan

# Advanced metrics
advanced = AdvancedRiskMetrics()
drawdown_df = advanced.calculate_drawdowns(port_ret_series)
tail_metrics = advanced.tail_risk_metrics(port_ret_series)
stability_metrics = advanced.stability_metrics(port_ret_series, rf)

# Rolling contributions
roll_rc = rolling_risk_contributions(returns, w, window=roll_window)

# Active risk
active_df, active_stats = active_risk_contributions(returns, bench_returns, w) if bench_returns is not None else (pd.DataFrame(), {})

# Stress factors
fx_ticker = _pick_first_working_ticker(FX_USDTRY_CANDIDATES, start, end)
rate_ticker = _pick_first_working_ticker(RATE_CANDIDATES, start, end)

fx_ret = rate_ret = None
if fx_ticker:
    fx_prices, _ = fetcher.fetch_prices([fx_ticker], start, end, use_cache=True)
    if not fx_prices.empty:
        fx_prices = fx_prices.ffill(limit=ffill_limit)
        fx_ret = fx_prices.iloc[:, 0].pct_change().dropna()

if rate_ticker:
    rt_prices, _ = fetcher.fetch_prices([rate_ticker], start, end, use_cache=True)
    if not rt_prices.empty:
        rt_prices = rt_prices.ffill(limit=ffill_limit)
        rate_ret = rt_prices.iloc[:, 0].pct_change().dropna()

betas_df = estimate_factor_betas(returns, fx_ret, rate_ret) if (fx_ret is not None or rate_ret is not None) else pd.DataFrame()
scenario_df, scenario_port = scenario_impact(w, betas_df, fx_shock, rate_shock) if not betas_df.empty else (pd.DataFrame(), np.nan)

# Monte Carlo
mc_results = None
if enable_monte_carlo and mc_simulations > 0:
    status_text.text("🎲 Running Monte Carlo simulation...")
    mc_simulator = MonteCarloSimulator()
    mc_results = mc_simulator.simulate(returns, w, n_simulations=mc_simulations)

progress_bar.progress(100)
status_text.text("✅ Analysis complete!")
time.sleep(0.5)
progress_bar.empty()
status_text.empty()

# ------------------------------------------------------------
# Results Display
# ------------------------------------------------------------
tabs = st.tabs([
    "📊 Dashboard",
    "📦 Data",
    "🧠 Optimization",
    "🎯 Risk Budgeting",
    "📉 Tail Risk",
    "📈 Rolling Risk",
    "🧮 Active Risk",
    "⚡ Stress & MC",
    "📤 Export",
    "🔍 Diagnostics"
])

# --- Dashboard Tab
with tabs[0]:
    st.markdown('<div class="sub-header">📊 Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{(1+port_ret_series).prod()-1:.2%}")
        st.metric("CAGR", f"{port_metrics['portfolio_vol']:.2%}")
    with col2:
        st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
        st.metric("Sortino", f"{stability_metrics.get('Sortino_Ratio', 0):.2f}")
    with col3:
        st.metric("Max Drawdown", f"{drawdown_df['Max_Drawdown'].iloc[0]:.2%}")
        st.metric("Calmar", f"{stability_metrics.get('Calmar_Ratio', 0):.2f}")
    with col4:
        st.metric("Hit Rate", f"{stability_metrics['Hit_Rate']:.2%}")
        st.metric("Gain/Pain", f"{stability_metrics['Gain_Pain_Ratio']:.2f}")
    
    # Cumulative returns chart
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Cumulative Returns', 'Drawdown', 
                                      'Rolling Volatility', 'Rolling Sharpe'))
    
    cum_ret = (1 + port_ret_series).cumprod()
    fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values, name='Portfolio'), row=1, col=1)
    if bench_returns is not None:
        bench_cum = (1 + bench_returns).cumprod()
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name='BIST100', 
                                 line=dict(dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=drawdown_df.index, y=drawdown_df['Drawdown']*100, 
                            fill='tozeroy', name='Drawdown %'), row=1, col=2)
    
    roll_vol = port_ret_series.rolling(60).std() * np.sqrt(252) * 100
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, name='60d Vol %'), row=2, col=1)
    
    roll_sharpe = port_ret_series.rolling(252).mean() / port_ret_series.rolling(252).std() * np.sqrt(252)
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name='1Y Sharpe'), row=2, col=2)
    fig.add_hline(y=1, line_dash="dash", line_color="green", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# --- Data Tab
with tabs[1]:
    st.markdown('<div class="sub-header">📦 Data Summary</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assets", asset_prices.shape[1])
    col2.metric("Days", returns.shape[0])
    col3.metric("Start", returns.index[0].strftime('%Y-%m-%d'))
    col4.metric("End", returns.index[-1].strftime('%Y-%m-%d'))
    
    st.dataframe(asset_prices.tail(10), use_container_width=True)
    
    # Price chart
    fig = go.Figure()
    for t in asset_prices.columns[:10]:
        norm = asset_prices[t] / asset_prices[t].iloc[0] * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=t))
    fig.update_layout(title="Normalized Prices (Top 10)", height=500, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# --- Optimization Tab
with tabs[2]:
    st.markdown('<div class="sub-header">🧠 Optimization Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Method", opt_method)
    col2.metric("Exp Return", f"{perf.get('expected_return', 0):.2%}")
    col3.metric("Volatility", f"{perf.get('volatility', 0):.2%}")
    
    w_df = pd.DataFrame({
        "Symbol": returns.columns,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in returns.columns],
        "Weight": w
    }).sort_values("Weight", ascending=False)
    
    fig = px.bar(w_df.head(20), x="Weight", y="Symbol", color="Sector", 
                 orientation='h', title="Top 20 Weights")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(w_df, use_container_width=True, hide_index=True)

# --- Risk Budgeting Tab
with tabs[3]:
    st.markdown('<div class="sub-header">🎯 Risk Budgeting</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Vol", f"{port_metrics['portfolio_vol']:.2%}")
    col2.metric("Div Ratio", f"{port_metrics['diversification_ratio']:.2f}")
    col3.metric("Top Risk", f"{port_metrics['top_risk_asset']} ({port_metrics['top_risk_pct']:.1f}%)")
    
    fig = go.Figure()
    sorted_df = risk_df.sort_values("Risk_Contribution_%", ascending=True)
    fig.add_trace(go.Bar(
        y=sorted_df["Symbol"],
        x=sorted_df["Risk_Contribution_%"],
        orientation='h',
        marker=dict(color=sorted_df["Risk_Contribution_%"], colorscale='RdYlGn_r'),
        text=sorted_df["Risk_Contribution_%"].round(1).astype(str) + '%',
        textposition='outside'
    ))
    eq_target = 100 / len(sorted_df)
    fig.add_vline(x=eq_target, line_dash="dash", line_color="red")
    fig.update_layout(title="Risk Contribution by Asset", height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

# --- Tail Risk Tab
with tabs[4]:
    st.markdown('<div class="sub-header">📉 Tail Risk Analysis</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"VaR ({var_alpha})", f"{var1:.2%}")
    col2.metric(f"CVaR ({var_alpha})", f"{cvar1:.2%}")
    col3.metric("Skewness", f"{tail_metrics['Skewness']:.3f}")
    col4.metric("Kurtosis", f"{tail_metrics['Kurtosis']:.3f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tail Ratio", f"{tail_metrics['Tail_Ratio']:.2f}")
    col2.metric("Pain Index", f"{tail_metrics['Pain_Index']:.2f}")
    col3.metric("Ulcer Index", f"{tail_metrics['Ulcer_Index']:.2f}")
    col4.metric("Jarque-Bera", f"{tail_metrics['Jarque_Bera']:.0f}")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_ret_series*100, nbinsx=50, name='Returns'))
    if np.isfinite(var1):
        fig.add_vline(x=var1*100, line_dash="dash", line_color="red", annotation_text="VaR")
    if np.isfinite(cvar1):
        fig.add_vline(x=cvar1*100, line_dash="dot", line_color="darkred", annotation_text="CVaR")
    fig.update_layout(title="Returns Distribution", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- Rolling Risk Tab
with tabs[5]:
    st.markdown('<div class="sub-header">📈 Rolling Risk Contributions</div>', unsafe_allow_html=True)
    if roll_rc.empty:
        st.warning("Not enough data for rolling calculations")
    else:
        latest = roll_rc.iloc[-1].sort_values(ascending=False).head(6)
        fig = go.Figure()
        for sym in latest.index:
            fig.add_trace(go.Scatter(x=roll_rc.index, y=roll_rc[sym], name=sym))
        fig.update_layout(title=f"Rolling Risk % (Window={roll_window})", height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(roll_rc.tail(20), use_container_width=True)

# --- Active Risk Tab
with tabs[6]:
    st.markdown('<div class="sub-header">🧮 Active Risk vs BIST100</div>', unsafe_allow_html=True)
    if active_df.empty:
        st.warning("Benchmark not available for active risk analysis")
    else:
        st.metric("Tracking Error", f"{active_stats.get('tracking_error', 0):.2%}")
        fig = go.Figure()
        top = active_df.sort_values("Active_Risk_Contribution_%", ascending=True).tail(20)
        fig.add_trace(go.Bar(y=top["Symbol"], x=top["Active_Risk_Contribution_%"], orientation='h'))
        fig.update_layout(title="Active Risk Contribution", height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(active_df, use_container_width=True)

# --- Stress & MC Tab
with tabs[7]:
    st.markdown('<div class="sub-header">⚡ Stress Testing & Monte Carlo</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factor Betas**")
        if not betas_df.empty:
            st.dataframe(betas_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Scenario Impact**")
        if not scenario_df.empty:
            st.metric("Portfolio Impact", f"{scenario_port:.2%}")
            fig = px.bar(scenario_df.tail(20), x="Weighted_Impact", y="Symbol", 
                        color="Sector", orientation='h', title="Scenario Impact by Asset")
            st.plotly_chart(fig, use_container_width=True)
    
    if mc_results:
        st.markdown("**Monte Carlo Simulation**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median", f"{mc_results['median_final']:.2f}x")
        col2.metric("VaR 95%", f"{mc_results['var_95']:.2f}x")
        col3.metric("Loss Prob", f"{mc_results['probability_loss']:.1f}%")
        col4.metric("Double Prob", f"{mc_results['probability_double']:.1f}%")
        
        mc_fig = MonteCarloSimulator.plot_simulations(mc_results)
        st.plotly_chart(mc_fig, use_container_width=True)

# --- Export Tab
with tabs[8]:
    st.markdown('<div class="sub-header">📤 Export Results</div>', unsafe_allow_html=True)
    
    sheets = {
        "Weights": w_df,
        "Risk_Metrics": risk_df,
        "Portfolio_Returns": port_ret_series.reset_index().rename(columns={"index":"Date", "Portfolio":"Return"}),
        "Drawdown": drawdown_df.reset_index().rename(columns={"index":"Date"}),
        "Tail_Metrics": pd.DataFrame([tail_metrics]),
        "Stability_Metrics": pd.DataFrame([stability_metrics]),
        "Active_Risk": active_df if not active_df.empty else pd.DataFrame(),
        "Betas": betas_df if not betas_df.empty else pd.DataFrame(),
        "Scenario": scenario_df if not scenario_df.empty else pd.DataFrame(),
        "Diagnostics": pd.DataFrame({
            "Metric": ["Assets", "Days", "Method", "Fetch Mode", "Cache Hit", "Fetch Time"],
            "Value": [asset_prices.shape[1], returns.shape[0], opt_method, 
                     frep.mode, frep.cache_hit, f"{frep.fetch_time:.2f}s"]
        })
    }
    
    if mc_results:
        sheets["Monte_Carlo"] = pd.DataFrame({
            "Final_Values": mc_results['final_values']
        })
    
    xlsx = to_excel_bytes(sheets)
    st.download_button(
        "📥 Download Complete Report (Excel)",
        data=xlsx,
        file_name=f"bist_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# --- Diagnostics Tab
with tabs[9]:
    st.markdown('<div class="sub-header">🔍 System Diagnostics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fetch Report**")
        st.json({
            "mode": frep.mode,
            "cache_hit": frep.cache_hit,
            "fetch_time": f"{frep.fetch_time:.2f}s",
            "requested": len(frep.requested),
            "received": len(frep.received_cols),
            "dropped_raw": len(frep.dropped_raw),
            "dropped_clean": len(dropped_clean)
        })
        
        if frep.notes:
            st.markdown("**Fetch Notes**")
            for note in frep.notes[:10]:
                st.caption(f"• {note}")
    
    with col2:
        st.markdown("**System Info**")
        st.json({
            "PyPortfolioOpt": "✅ Available" if PYPFOPT_AVAILABLE else "❌ Not Available",
            "Python": f"{PYPFOPT_IMPORT_ERROR if not PYPFOPT_AVAILABLE else 'OK'}",
            "Benchmark": benchmark_ticker or "None",
            "FX Ticker": fx_ticker or "None",
            "Rate Ticker": rate_ticker or "None",
            "Assets Final": asset_prices.shape[1],
            "Date Range": f"{returns.index[0].date()} to {returns.index[-1].date()}"
        })
    
    st.markdown("**Performance Metrics**")
    st.json({
        "Total Return": f"{(1+port_ret_series).prod()-1:.2%}",
        "CAGR": f"{port_metrics['portfolio_vol']:.2%}",
        "Sharpe": f"{perf.get('sharpe_ratio', 0):.2f}",
        "Max DD": f"{drawdown_df['Max_Drawdown'].iloc[0]:.2%}",
        "VaR 95": f"{tail_metrics.get('VaR_95', 0):.2%}",
        "CVaR 95": f"{tail_metrics.get('CVaR_95', 0):.2%}"
    })

# Footer
st.markdown("---")
st.markdown(
    f'<div class="small" style="text-align: center;">{SIGNATURE} • Enhanced Version v2.0 • Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
    unsafe_allow_html=True,
)

if not PYPFOPT_AVAILABLE:
    st.sidebar.warning(
        f"⚠️ PyPortfolioOpt not available: {PYPFOPT_IMPORT_ERROR}\n\n"
        "Advanced optimization methods will fall back to equal weight."
    )
