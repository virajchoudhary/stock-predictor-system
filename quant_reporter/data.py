import pandas as pd
import yfinance as yf


def get_data(tickers, start_date, end_date):
    """
    Fetches adjusted close data for a list of tickers, handling
    mismatched trading days and missing data.

    Always downloads fresh data from yfinance — no local cache is used.
    """
    print(f"Fetching data for {', '.join(tickers)}...")
    try:
        # threads=False to avoid "OperationalError: unable to open database file" with yfinance cache
        # progress=False to suppress download progress output
        all_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            threads=False,
            progress=False,
        )

        if all_data is None or all_data.empty:
            raise ValueError("No data downloaded. Check tickers and date range.")

        if 'Close' not in all_data:
            raise ValueError("Downloaded data does not contain 'Close' prices.")

        data = all_data['Close']

        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        # Ensure column names are strings (yfinance >=0.2.40 may return MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in data.columns]
        else:
            data.columns = [str(c) for c in data.columns]

        # Forward-fill then back-fill to handle holidays/mismatched trading calendars
        data_filled = data.ffill().bfill()

        # Drop any rows still NaN (assets that didn't exist at the period start)
        data_filled = data_filled.dropna()

        return data_filled

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def get_benchmark_data(ticker: str, start_date: str, end_date: str,
                       reference_index: pd.DatetimeIndex = None) -> pd.Series:
    """
    Fetches a single benchmark ticker's daily Close prices as a fresh,
    aligned Series.  Never uses a stale cache.

    Args:
        ticker:          The benchmark symbol (e.g. "SPY", "QQQ", "^NSEI").
        start_date:      ISO date string "YYYY-MM-DD".
        end_date:        ISO date string "YYYY-MM-DD".
        reference_index: Optional DatetimeIndex from the portfolio DataFrame.
                         If provided, the benchmark is re-indexed to match it,
                         with forward-fill then back-fill applied.

    Returns:
        pd.Series of daily Close prices named after *ticker*.

    Raises:
        ValueError: if the download produces an empty result.
    """
    ticker_upper = ticker.strip().upper()
    print(f"[benchmark] Fetching fresh data for {ticker_upper} ({start_date} → {end_date})")

    raw = yf.download(
        ticker_upper,
        start=start_date,
        end=end_date,
        threads=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"Benchmark ticker '{ticker_upper}' returned no price data "
            f"for the requested period ({start_date} to {end_date})."
        )

    # Extract the Close column robustly (handles MultiIndex from yfinance >=0.2.40)
    if 'Close' in raw.columns:
        series = raw['Close']
    elif isinstance(raw.columns, pd.MultiIndex):
        close_cols = [c for c in raw.columns if c[0] == 'Close' or c[1] == 'Close']
        if not close_cols:
            raise ValueError(f"No 'Close' column found in benchmark data for '{ticker_upper}'.")
        series = raw[close_cols[0]]
    else:
        raise ValueError(f"Cannot extract Close price from benchmark data for '{ticker_upper}'.")

    # Flatten to 1-D Series
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.to_numeric(series, errors='coerce')
    series.name = ticker_upper

    # Align to portfolio date index when provided
    if reference_index is not None:
        series = series.reindex(reference_index, method=None)  # align first
        series = series.ffill().bfill()                        # fill trading-day gaps

    # Drop any residual NaN (asset not yet listed at the very start)
    series = series.dropna()

    if series.empty:
        raise ValueError(
            f"Benchmark '{ticker_upper}' has no valid data after alignment. "
            "Check that the ticker existed during the requested date range."
        )

    return series