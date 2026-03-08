from __future__ import annotations
from typing import Dict, List, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import yfinance as yf
from IPython.display import display

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAVE_PPO = True
except ImportError:
    HAVE_PPO = False

def rsi(series: pd.Series, period: int = 20) -> pd.Series:
    """Relative Strength Index (EMA version)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger_z(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    z = (series - ma) / (sd + 1e-12)
    return z

def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr / (close.replace(0, np.nan))

def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman–Klass volatility estimator (daily). Returns sigma (not annualized)."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    return np.sqrt(0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2))

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out

def adj_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Return Adj Close table (tickers as columns) from yfinance download.

    yfinance returns a MultiIndex on the columns with level 0 being fields
    (e.g., 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume') and level 1
    the ticker symbols. This peels 'Adj Close' and returns a tidy (Date x Ticker)
    frame.
    """
    if isinstance(prices.columns, pd.MultiIndex):
        out = prices.xs("Adj Close", axis=1, level=0, drop_level=True)
    else:
        # Single‑ticker case: make it consistent
        out = prices[["Adj Close"]].rename(columns={"Adj Close": list(prices.columns)[0]})
    out = ensure_datetime_index(out)
    return out

def ohlcv_block(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract an OHLCV field ('Open','High','Low','Close','Volume') as a (Date x Ticker) table."""
    assert field in {"Open", "High", "Low", "Close", "Volume"}
    if isinstance(prices.columns, pd.MultiIndex):
        out = prices.xs(field, axis=1, level=0, drop_level=True)
    else:
        out = prices[[field]].rename(columns={field: list(prices.columns)[0]})
    return ensure_datetime_index(out)

def build_daily_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily per‑ticker features. Returns a MultiIndex index (Date, Ticker)."""
    adj = adj_table(prices)
    open_ = ohlcv_block(prices, "Open")
    high = ohlcv_block(prices, "High")
    low = ohlcv_block(prices, "Low")
    close = ohlcv_block(prices, "Close")
    vol = ohlcv_block(prices, "Volume")

    n = len(adj.columns)
    print(f"Building daily features for {n} tickers...")
    frames = []
    for i, tkr in enumerate(adj.columns, 1):
        s_adj = adj[tkr]
        s_open, s_high, s_low, s_close, s_vol = (
            open_[tkr], high[tkr], low[tkr], close[tkr], vol[tkr]
        )

        feats = pd.DataFrame({
            "rsi": rsi(s_adj),
            "bb_z": bollinger_z(s_adj),
            "atr_pct": atr_percent(s_high, s_low, s_close),
            "macd_hist": macd_hist(s_adj),
            "gk_vol": garman_klass_vol(s_open, s_high, s_low, s_close),
            "dollar_vol": (s_close * s_vol),
        })
        feats["Ticker"] = tkr
        frames.append(feats)
        if i % 50 == 0 or i == n:
            print(f"  [{i}/{n}] daily features computed...")

    out = pd.concat(frames).dropna(how="all")
    out.index.name = "Date"
    out = out.reset_index().set_index(["Date", "Ticker"]).sort_index()
    return out

def to_monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(level="Ticker").apply(
            lambda x: x.droplevel("Ticker").resample("ME").last()
        ).swaplevel().sort_index()
    )

def to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(level="Ticker").apply(
            lambda x: x.droplevel("Ticker").resample("ME").mean()
        ).swaplevel().sort_index()
    )

def trailing_months_mean(s: pd.Series, months: int) -> pd.Series:
    return (
        s.groupby(level="Ticker")
        .apply(lambda x: x.droplevel("Ticker").rolling(months).mean())
        .swaplevel()
        .sort_index()
    )

def to_weekly_last(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(level="Ticker").apply(
            lambda x: x.droplevel("Ticker").resample("W-FRI").last()
        ).swaplevel().sort_index()
    )

def to_weekly_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(level="Ticker").apply(
            lambda x: x.droplevel("Ticker").resample("W-FRI").mean()
        ).swaplevel().sort_index()
    )

def trailing_weeks_mean(s: pd.Series, weeks: int) -> pd.Series:
    return (
        s.groupby(level="Ticker")
        .apply(lambda x: x.droplevel("Ticker").rolling(weeks).mean())
        .swaplevel()
        .sort_index()
    )

def build_monthly_feature_table(daily_feats: pd.DataFrame) -> pd.DataFrame:
    # Take last value per month for oscillators; mean for dollar_vol (liquidity)
    osc_cols = ["rsi", "bb_z", "atr_pct", "macd_hist", "gk_vol"]
    osc = to_monthly_last(daily_feats[osc_cols])

    dv = to_monthly_mean(daily_feats[["dollar_vol"]])
    dv60 = trailing_months_mean(dv["dollar_vol"], 60).rename("dv60m_avg")

    out = osc.join(dv).join(dv60)
    return out

def build_weekly_feature_table(daily_feats: pd.DataFrame) -> pd.DataFrame:
    # Take last value per week for oscillators; mean for dollar_vol (liquidity)
    print("Resampling to weekly oscillators...")
    osc_cols = ["rsi", "bb_z", "atr_pct", "macd_hist", "gk_vol"]
    osc = to_weekly_last(daily_feats[osc_cols])

    print("Computing weekly dollar volume and 260-week trailing average...")
    dv = to_weekly_mean(daily_feats[["dollar_vol"]])
    # 260-week (~5yr) trailing average for liquidity ranking
    dv260 = trailing_weeks_mean(dv["dollar_vol"], 260).rename("dv60w_avg")

    out = osc.join(dv).join(dv260)
    print(f"Weekly feature table ready: {out.shape[0]} rows, {out.shape[1]} columns.")
    return out

def liquidity_filter(monthly: pd.DataFrame, topn: int = 150) -> pd.Series:
    """Return a boolean Series indexed by (Date, Ticker) marking the liquid set.

    Uses dv60w_avg (weekly) or dv60m_avg (monthly) when available;
    otherwise falls back to current period dollar_vol.
    """
    idx = monthly.index
    if "dv60w_avg" in monthly.columns:
        trailing = monthly["dv60w_avg"]
    elif "dv60m_avg" in monthly.columns:
        trailing = monthly["dv60m_avg"]
    else:
        trailing = monthly["dollar_vol"]
    liq = trailing.fillna(monthly["dollar_vol"])

    # Rank within each month and keep top N
    ranks = liq.groupby(level=0).rank(ascending=False, method="first")
    keep = ranks <= topn
    keep.name = "liquid"
    print(f"Liquidity filter: keeping top {topn} tickers per period.")
    return keep

def select_cluster_high_rsi(
        month_df: pd.DataFrame,
        k: int = 4,
        seed: int = 0,
        use_robust_scaler: bool = True,
        min_cluster_size: int = 3,
        selection: str = "best_by",
        best_by: str = "rsi",
        agg: "median" | "mean" = "mean",
        ascending: bool = False,
        select_clusters: Optional[List[int]] = None,
        centroids_scaled: bool = False,
        freeze_centroids: bool = False,
        init_centroids_df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Configurable cluster selector.

    * Cleans/filters features
    * Scales (StandardScaler or RobustScaler)
    * Clusters with KMeans
    - If `init_centroids` provided, uses them as KMeans init (n_init=1)
    - If `freeze_centroids=True`, assigns each point to nearest provided centroid (no updates)
    * Selection modes:
    - selection="best_by": choose cluster with best (mean/median) of `best_by`
    - selection="clusters": choose specific cluster label(s) via `select_clusters`

    Parameters
    ----------
    month_df : DataFrame indexed by (Date, Ticker) for a single month.
    features : list of feature column names; defaults to ['rsi','bb_z','atr_pct','macd_hist','gk_vol']
    init_centroids : DataFrame/array of shape (n_clusters, n_features). If a DataFrame,
        its columns must match `features` (after dropping near-constant cols). If in original
        feature space, set centroids_scaled=False so they are transformed by the scaler.
    return_labels : If True, returns a tuple (mask, labels) where labels aligns to month_df rows
                    that survived cleaning; other rows are NaN in labels.
    """
    feats = ["rsi", "bb_z", "atr_pct", "macd_hist", "gk_vol"]

    # 1) Clean and keep only rows with all features present
    have = [c for c in feats if c in month_df.columns]
    if not have: return out
    F = (month_df[have].apply(pd.to_numeric, errors="coerce")
                        .replace([np.inf,-np.inf], np.nan)
                        .dropna(how="any"))

    out = pd.Series(False, index=month_df.index, name="selected")
    if F.shape[0] < 2:
        return out

    # Optionally drop near-constant columns for stability
    var = F.var()
    keep_cols = var[var > 1e-12].index.tolist()
    F = F[keep_cols]
    if F.shape[1] == 0:
        return out

    # 2) Scale
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    Z = scaler.fit_transform(F.values)

    # 3) Determine clustering setup
    if init_centroids_df is not None:
        C = init_centroids_df
        # convert to numpy array with same column order as F
        if isinstance(C, pd.DataFrame):
            missing = [c for c in F.columns if c not in C.columns]
            if missing:
                raise ValueError(f"init_centroids missing columns: {missing}. Expected at least {list(F.columns)}")
            C = C[F.columns].values
        else:
            C = np.asarray(C, dtype=float)
            if C.shape[1] != F.shape[1]:
                raise ValueError(f"init_centroids has {C.shape[1]} features but cleaned data has {F.shape[1]}")

        # transform to scaled space if needed
        if not centroids_scaled:
            C_scaled = scaler.transform(C)
        else:
            C_scaled = C

        n_clusters = C_scaled.shape[0]
        # KMeans requires n_samples >= n_clusters
        n_samples = F.shape[0]
        if n_clusters > n_samples:
            C_scaled = C_scaled[:n_samples, :]
            n_clusters = n_samples
        if n_clusters < 1:
            return out

        if freeze_centroids:
            # Pure nearest-centroid assignment (no updates)
            # compute squared distances to each centroid
            # Z: (n_samples, n_features), C_scaled: (n_clusters, n_features)
            d2 = ((Z[:, None, :] - C_scaled[None, :, :]) ** 2).sum(axis=2)  # (n_samples, n_clusters)
            labels_clean = d2.argmin(axis=1)
            # synthesize a km-like object for compatibility (not returned)
            km = None
        else:
            km = KMeans(n_clusters=n_clusters, init=C_scaled, n_init=1, random_state=seed)
            labels_clean = km.fit_predict(Z)
    else:
        # Safe k (<= samples and <= unique rows)
        n_samples = F.shape[0]
        n_unique = pd.DataFrame(Z).drop_duplicates().shape[0]
        k_safe = max(1, min(k, n_samples, n_unique))
        if k_safe < 2:
            return out
        print(f"  Running KMeans with k={k_safe} clusters on {n_samples} tickers...")
        km = KMeans(n_clusters=k_safe, n_init="auto", random_state=seed)
        labels_clean = km.fit_predict(Z)

    labels = pd.Series(labels_clean, index=F.index, name="cluster")

    # 4) Choose clusters to keep
    if selection == "clusters":
        if not select_clusters:
            chosen_clusters = []
        else:
            chosen_clusters = list(select_clusters)
    else:
        # default: best_by metric
        metric_col = best_by if best_by in month_df.columns else "rsi"
        series_metric = month_df.loc[F.index, metric_col]
        if agg == "median":
            agg_series = (pd.concat([labels, series_metric.rename("metric")], axis=1)
                            .groupby("cluster")["metric"].median())
        else:
            agg_series = (pd.concat([labels, series_metric.rename("metric")], axis=1)
                            .groupby("cluster")["metric"].mean())

        sizes = labels.value_counts().reindex(agg_series.index).fillna(0).astype(int)
        # sort clusters by metric then size (descending by default)
        order = agg_series.sort_values(ascending=ascending).index.tolist()
        # choose best then enforce min_cluster_size if possible
        best_cluster = order[0]
        if sizes.loc[best_cluster] < min_cluster_size and len(order) > 1:
            for c in order:
                if sizes.loc[c] >= min_cluster_size:
                    best_cluster = c
                    break
        chosen_clusters = [best_cluster]

    # 5) Build mask mapped back to full index
    keep_idx = labels.index[labels.isin(chosen_clusters)]
    out.loc[keep_idx] = True
    print(f"  Selected {keep_idx.shape[0]} tickers from cluster(s) {chosen_clusters}.")
    return out

def equal_weight(tickers: Sequence[str]) -> pd.Series:
    w = pd.Series(1.0 / max(1, len(tickers)), index=list(tickers), dtype=float)
    return w

def optimize_weights(prices: pd.DataFrame, lower_bound: float = 0.0, upper_bound: float = 1.0) -> pd.Series:
    """Return weight vector for the given (Date x Ticker) price table.

    - If PyPortfolioOpt is installed, do max‑Sharpe with non‑negative weights.
    - Else, solve w ∝ Σ^{-1} μ with ridge regularization, clip to [0,1], renorm.
    """
    print("Optimizing weights...")
    cols = list(prices.columns)
    if len(cols) == 0:
        return equal_weight(cols)

    # Use daily simple returns
    rets = prices.pct_change().dropna(how="any")
    if rets.empty:
        return equal_weight(cols)

    if HAVE_PPO:
        print("Running PPO...")
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(lower_bound, upper_bound))
            ef.max_sharpe()
            w = pd.Series(ef.clean_weights())
            w = w.reindex(cols).fillna(0.0)
            if w.sum() <= 0:
                return equal_weight(cols)
            return w / w.sum()
        except Exception:
            pass

    # Fallback closed‑form with ridge
    mu = rets.mean().values  # (n,)
    S = rets.cov().values
    n = S.shape[0]
    ridge = 1e-4 * np.eye(n)
    try:
        inv = np.linalg.pinv(S + ridge)
        raw = inv @ mu
        w = pd.Series(raw, index=cols)
        w = w.clip(lower=lower_bound, upper=upper_bound)
        if w.sum() <= 0:
            return equal_weight(cols)
        return w / w.sum()
    except Exception:
        return equal_weight(cols)

def month_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(ts.year, ts.month, 1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end

def week_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (Monday, Friday) for the week ending on the given Friday anchor."""
    # W-FRI resampling anchors on Friday; week starts 4 days prior (Monday)
    end = ts.normalize()
    start = end - pd.Timedelta(days=4)
    return start, end

# -----------------------------
# Universe & data download
# -----------------------------

def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 members from Wikipedia.

    NOTE: Using this for past backtests introduces survivorship bias.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            storage_options={"User-Agent": "Mozilla/5.0"})
        df = tables[0]
        tickers = (
            df["Symbol"].astype(str).str.replace("\u200b", "", regex=False).str.strip().tolist()
        )
        # yfinance needs BRK.B as BRK-B, etc.
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers
    except Exception as e:
        # minimal fallback
        print("Warning: failed to fetch S&P 500 tickers from Wikipedia: %s.", e)
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "SPY"]


def download_ohlcv(tickers: Sequence[str], start: str, end: str,
                   retries: int = 3, retry_delay: float = 10.0) -> pd.DataFrame:
    """Download OHLCV data for the given tickers and date range via yfinance.

    Retries up to `retries` times with exponential backoff on failure or empty result.
    Raises RuntimeError if all attempts fail.
    """
    end_dt = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers=list(tickers),
                start=start,
                end=end_dt,
                auto_adjust=False,
                group_by="column",
                threads=True,
                progress=False,
            )
            # yfinance returns a Series for single-ticker; coerce to DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            data = ensure_datetime_index(data)
            if data.empty:
                raise ValueError("yfinance returned an empty DataFrame")
            return data
        except Exception as e:
            last_exc = e
            if attempt < retries:
                wait = retry_delay * attempt
                print(f"Download attempt {attempt}/{retries} failed: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to download OHLCV data after {retries} attempts: {last_exc}")

def precompute_indicators(universe: List[str]):
    # Start far enough back to warm up the 260-week trailing liquidity average
    start = "2015-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"Downloading OHLCV for {len(universe)} tickers ({start} to {end})...")
    # Raises RuntimeError if all download attempts fail
    prices = download_ohlcv(universe, start, end)
    print(f"Download complete: {prices.shape[0]} trading days.")
    daily_feats = build_daily_features(prices)
    print("Building weekly feature table...")
    weekly_feats = build_weekly_feature_table(daily_feats)
    print("Applying liquidity filter...")
    liq_mask = liquidity_filter(weekly_feats, topn=150)
    print("Indicator precomputation done.")
    return prices, weekly_feats, liq_mask

def run_one_config(
    prices: pd.DataFrame,
    monthly_features: pd.DataFrame,
    liquid_mask: pd.Series,
    lookback_days: int = 252,
    seed: int = 0,
    k: int = 4,
    use_robust_scaler: bool = True,
    min_cluster_size: int = 3,
    selection: str = "best_by",
    best_by: str = "rsi",
    agg: "median" | "mean" = "mean",
    ascending: bool = False,
    select_clusters: Optional[List[int]] = None,
    centroids_scaled: bool = False,
    freeze_centroids: bool = False,
    init_centroids_df: Optional[pd.DataFrame] = None
    
) -> pd.Series | None   :
    
    adj = adj_table(prices)
    rets = adj.pct_change().dropna()
    
    m = sorted(monthly_features.index.get_level_values(0).unique())[-1]
    print(f"Working with month: {m}...")
    month_slice = monthly_features.xs(m, level=0, drop_level=False)
    liquid_mi = liquid_mask.xs(m, level=0, drop_level=False)
    mask = liquid_mi.reindex(month_slice.index, fill_value=False).astype(bool)
    eligible = month_slice.index[mask].get_level_values("Ticker")

    if len(eligible) == 0:
        print(f"Month {m.date()}: no liquid tickers, skipping.")
        return
    
    # Select cluster with highest mean RSI
    selected_mask = select_cluster_high_rsi(month_df=month_slice,
        k=k,
        seed=seed,
        use_robust_scaler=use_robust_scaler,
        min_cluster_size=min_cluster_size,
        selection=selection,
        best_by=best_by,
        agg=agg,
        ascending=ascending,
        select_clusters=select_clusters,
        centroids_scaled=centroids_scaled,
        freeze_centroids=freeze_centroids,
        init_centroids_df=init_centroids_df
    )
    picks = month_slice.index[selected_mask].get_level_values("Ticker")
    # Intersect with liquid set
    picks = [t for t in picks if t in eligible and t in adj.columns]
    print(f"  {len(picks)} picks after intersecting with liquid set.")
    if len(picks) == 0:
        return
    m_start, m_end = week_bounds(pd.Timestamp(m))
    opt_end = m_start - pd.Timedelta(days=1)
    opt_start = opt_end - pd.tseries.offsets.BDay(lookback_days)
    opt_prices = adj.loc[adj.index[(adj.index >= opt_start) & (adj.index <= opt_end)], picks]
    opt_prices = opt_prices.dropna(how="any")
    if opt_prices.shape[1] == 0 or opt_prices.shape[0] < 30:
        print(f"Month {m.date()}: insufficient data for optimization, using equal weights.")
        w = equal_weight(picks)
    else:
        w = optimize_weights(opt_prices)
    result = w[w > 0.0]
    print(f"  Final portfolio: {len(result)} positions.")
    return result

def export_picks_json(weights: pd.Series, path: str) -> None:
    """Write portfolio weights to a JSON file for downstream broker consumption."""
    import json as _json
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "picks": [
            {"ticker": str(t), "weight": round(float(w), 6)}
            for t, w in weights.items()
        ],
    }
    with open(path, "w") as fh:
        _json.dump(payload, fh, indent=2)

if __name__ == "__main__":
    import json as _json, os, pathlib as _pathlib

    _configs_path = _pathlib.Path(__file__).parent / "configs.json"
    with open(_configs_path) as _fh:
        _configs = _json.load(_fh)

    print(f"Loaded {len(_configs)} configs from {_configs_path}.")
    print("Fetching S&P 500 tickers...")
    universe = get_sp500_tickers()
    print(f"Universe: {len(universe)} tickers.")
    prices, weekly_feats, liq_mask = precompute_indicators(universe)

    _all_results = []
    for _i, _cfg in enumerate(_configs, 1):
        _name = _cfg.pop("name", None)
        print(f"\n[{_i}/{len(_configs)}] Running config: {_name!r}...")
        # init_centroids_df: convert list-of-records to DataFrame if provided
        _icd = _cfg.get("init_centroids_df")
        if isinstance(_icd, list):
            _cfg["init_centroids_df"] = pd.DataFrame(_icd)
        w = run_one_config(prices=prices, monthly_features=weekly_feats, liquid_mask=liq_mask, **_cfg)
        _entry = {
            "name": _name,
            "picks": (
                [{"ticker": str(t), "weight": round(float(v), 6)} for t, v in w.items()]
                if w is not None and not w.empty else []
            ),
        }
        _all_results.append(_entry)
        print(f"  Config {_name!r} done: {len(_entry['picks'])} picks.")
        display(w)

    print(f"\nAll {len(_configs)} configs complete.")
    _payload = _json.dumps({
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "results": _all_results,
    }, indent=2)

    _bucket = os.environ.get("PICKS_BUCKET")
    if _bucket:
        import boto3 as _boto3
        _s3 = _boto3.client("s3")
        _s3.put_object(Bucket=_bucket, Key="picks/latest.json",
                       Body=_payload.encode(), ContentType="application/json")
        print(f"Picks uploaded to s3://{_bucket}/picks/latest.json")
    else:
        with open("picks.json", "w") as _fh:
            _fh.write(_payload)
        print("Picks written to picks.json")