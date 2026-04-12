import pandas as pd


def normalize_requested_tickers(tickers):
    seen = set()
    normalized = []
    for ticker in tickers or []:
        symbol = str(ticker).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    return normalized


def resolve_report_universe(input_tickers, allocation_weights, *dataframes):
    requested_tickers = normalize_requested_tickers(input_tickers)
    common_symbols = set(requested_tickers)

    for frame in dataframes:
        if frame is None or getattr(frame, "empty", True):
            common_symbols = set()
            break
        common_symbols &= set(frame.columns)

    analysis_tickers = [ticker for ticker in requested_tickers if ticker in common_symbols]
    weights = pd.Series(allocation_weights or {}, dtype=float).reindex(analysis_tickers).fillna(0.0)
    weights = weights.clip(lower=0.0)

    if analysis_tickers:
        total = float(weights.sum())
        if total <= 1e-10:
            weights = pd.Series(
                {ticker: 1.0 / len(analysis_tickers) for ticker in analysis_tickers},
                dtype=float,
            )
        else:
            weights = weights / total

    return {
        "requested_tickers": requested_tickers,
        "analysis_tickers": analysis_tickers,
        "allocation_weights": {
            ticker: round(float(weights.get(ticker, 0.0)), 6)
            for ticker in analysis_tickers
        },
        "dropped_tickers": [ticker for ticker in requested_tickers if ticker not in analysis_tickers],
    }
