import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# Import the core math function this file needs
from .opt_core import get_portfolio_stats

# ─── Attractive fintech color palette ────────────────────────────────────────
# Used for pie slices, bar segments, and scatter traces
PALETTE = [
    "#38bdf8",  # sky blue
    "#a78bfa",  # soft violet
    "#34d399",  # emerald
    "#fb923c",  # vivid orange
    "#f472b6",  # rose pink
    "#facc15",  # golden yellow
    "#60a5fa",  # royal blue
    "#4ade80",  # lime green
    "#f87171",  # coral red
]

STRATEGY_COLORS = {
    "Equal Weight (Baseline)":       "#38bdf8",
    "Minimum Volatility":            "#a78bfa",
    "Balanced (40% Cap)":            "#34d399",
    "Max Sharpe (Unconstrained)":    "#fb923c",
    "Black-Litterman":               "#f472b6",
    "Sector Balanced":               "#facc15",
    "User Portfolio":                "#60a5fa",
}

def _strategy_color(name, idx):
    return STRATEGY_COLORS.get(name, PALETTE[idx % len(PALETTE)])

def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"

def _base_layout(**extra):
    """Shared layout defaults for every chart."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#e2e8f0",
            size=13,
        ),
        legend=dict(
            bgcolor="rgba(30,41,59,0.7)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(color="#f1f5f9", size=13),
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            linecolor="rgba(148,163,184,0.3)",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#cbd5e1"),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            linecolor="rgba(148,163,184,0.3)",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#cbd5e1"),
        ),
        title_font=dict(color="#f1f5f9", size=16, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    layout.update(extra)
    return layout

# --- All Optimization Plotting Functions ---

def plot_efficient_frontier(mean_returns, cov_matrix, optimal_portfolios, frontier_curve, risk_free_rate=0.02):
    """
    Generates a Plotly scatter plot of the efficient frontier.
    """
    print("Plotting Efficient Frontier...")
    
    num_ports = 2500
    all_weights = np.zeros((num_ports, len(mean_returns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    # Vectorized generation of random portfolios
    weights = np.random.random((num_ports, len(mean_returns)))
    weights /= weights.sum(axis=1)[:, np.newaxis]
    
    # Vectorized return and volatility calculation
    ret_arr = np.dot(weights, mean_returns)
    
    # Volatility is a bit trickier to vectorize fully without a loop for the quadratic form, 
    # but we can use einsum for speed: diag(w @ Sigma @ w.T)
    # However, a simple way that is still fast is:
    # vol = sqrt( sum( (w @ Sigma) * w, axis=1 ) )
    vol_arr = np.sqrt(np.sum(np.dot(weights, cov_matrix) * weights, axis=1))
    
    sharpe_arr = (ret_arr - risk_free_rate) / vol_arr
    sharpe_arr[vol_arr <= 0.00001] = 0 # Handle division by zero case

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vol_arr, y=ret_arr, mode='markers',
        marker=dict(
            color=sharpe_arr,
            colorscale='Turbo',
            showscale=True,
            size=5,
            opacity=0.4,
            colorbar=dict(
                title="Sharpe Ratio"
            ),
        ),
        name='Random Portfolios', text=[f"Sharpe: {s:.2f}" for s in sharpe_arr]
    ))
    
    fig.add_trace(go.Scatter(
        x=frontier_curve['Volatility'], y=frontier_curve['Return'], mode='lines',
        name='Efficient Frontier', line=dict(color='#38bdf8', width=3)
    ))
    
    for idx, (name, data) in enumerate(optimal_portfolios.items()):
        port_ret, port_vol, _ = get_portfolio_stats(data['weights_arr'], mean_returns, cov_matrix, risk_free_rate)
        fig.add_trace(go.Scatter(
            x=[port_vol], y=[port_ret], mode='markers', 
            marker=dict(
                color=_strategy_color(name, idx),
                size=14,
                symbol='star',
                line=dict(width=1.5, color='#0f172a'),
            ),
            name=name
        ))
    
    if "Max Sharpe (Unconstrained)" in optimal_portfolios:
        max_sharpe_data = optimal_portfolios["Max Sharpe (Unconstrained)"]
        msr_ret, msr_vol, _ = get_portfolio_stats(max_sharpe_data['weights_arr'], mean_returns, cov_matrix, risk_free_rate)
        cml_x = [0, msr_vol * 1.5]
        cml_y = [risk_free_rate, risk_free_rate + (msr_ret - risk_free_rate) / msr_vol * (msr_vol * 1.5)]
        
        fig.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode='lines',
            name='Capital Market Line (CML)', line=dict(color='#facc15', dash='dash', width=2)
        ))
    
    fig.update_layout(
        title='Efficient Frontier & Optimal Portfolios',
        xaxis_title='Annualized Volatility (Risk)', yaxis_title='Annualized Return',
        hovermode='closest',
        **_base_layout(),
    )
    fig.update_layout(legend=dict(bgcolor="rgba(15,23,42,0.85)", bordercolor="rgba(148,163,184,0.25)", borderwidth=1))
    return fig

def plot_correlation_heatmap(log_returns):
    """
    Generates a Plotly heatmap of the asset correlation matrix.
    """
    print("Plotting Correlation Heatmap...")
    corr_matrix = log_returns.corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale='RdYlGn', title='Asset Correlation Heatmap'
    )
    fig.update_layout(**_base_layout())
    fig.update_traces(textfont=dict(color="#0f172a", size=12))
    return fig

def plot_cumulative_comparison(cumulative_returns_df, benchmark_ticker):
    """
    Plots the "Growth of $1" for all optimized portfolios.
    """
    print("Plotting Strategy Cumulative Returns...")
    fig = go.Figure()
    
    for idx, col in enumerate(cumulative_returns_df.columns):
        color = _strategy_color(col, idx)
        fig.add_trace(go.Scatter(
            x=cumulative_returns_df.index, y=cumulative_returns_df[col], name=col,
            mode='lines',
            line=dict(width=2.5 if col != benchmark_ticker else 1.5,
                      dash='dot' if col == benchmark_ticker else 'solid',
                      color=color),
        ))
    
    # Bug 8: Context for out of sample returns
    if not cumulative_returns_df.empty:
        oos_start = cumulative_returns_df.index[0]
        fig.add_vline(x=oos_start, line_dash="dash", line_color="rgba(148,163,184,0.4)")
        fig.add_annotation(
            x=oos_start, y=1.05,
            text="Out-of-sample period →", showarrow=False, xanchor="left",
            bgcolor="rgba(30, 45, 61, 0.9)", bordercolor="#f87171", borderwidth=1,
            font=dict(size=11, color="white")
        )

    fig.update_layout(
        title='Strategy Performance: Cumulative Returns (Growth of $1)',
        xaxis_title='Date', yaxis_title='Cumulative Growth',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_drawdown_comparison(drawdown_df, benchmark_ticker):
    """
    Plots the drawdown "underwater" curves for all portfolios.
    """
    print("Plotting Strategy Drawdown...")
    fig = go.Figure()
    
    portfolios_to_plot = [col for col in drawdown_df.columns if col != benchmark_ticker]
    
    for idx, col in enumerate(portfolios_to_plot):
        color = _strategy_color(col, idx)
        fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=drawdown_df[col], name=col,
            mode='lines',
            line=dict(width=2, color=color),
            fill='tozeroy',
            fillcolor=_hex_to_rgba(color, 0.12) if color.startswith('#') else color,
        ))
    
    # Bug 8: Context for drawdown
    if not drawdown_df.empty:
        # Find the absolute deepest point
        min_dd = drawdown_df.min().min()
        col_min = drawdown_df.min().idxmin()
        idx_min = drawdown_df[col_min].idxmin()
        
        fig.add_annotation(
            x=idx_min, y=min_dd,
            text="Market correction period", showarrow=True, arrowhead=1, ax=0, ay=30,
            bgcolor="rgba(30, 45, 61, 0.9)", bordercolor="#f87171", borderwidth=1,
            font=dict(size=11, color="white")
        )

    fig.update_layout(
        title='Strategy Performance: Drawdown',
        xaxis_title='Date', yaxis_title='Drawdown',
        yaxis_tickformat='.1%', hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_rolling_sharpe(rolling_sharpe_df, benchmark_ticker):
    """
    Plots the 60-day rolling sharpe ratio for all strategies.
    """
    print("Plotting Rolling Sharpe...")
    fig = go.Figure()
    
    portfolios_to_plot = [col for col in rolling_sharpe_df.columns if col != benchmark_ticker]
    
    for idx, col in enumerate(portfolios_to_plot):
        color = _strategy_color(col, idx)
        fig.add_trace(go.Scatter(
            x=rolling_sharpe_df.index, y=rolling_sharpe_df[col], name=col,
            mode='lines', line=dict(width=2, color=color)
        ))
    
    fig.update_layout(
        title='Strategy Performance: 60-Day Rolling Sharpe Ratio',
        xaxis_title='Date', yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_composition_pies(optimal_portfolios):
    """
    Plots side-by-side pie charts of portfolio weights.
    """
    print("Plotting Composition Pies...")
    
    port_names = list(optimal_portfolios.keys())
    
    short_names = {
        "Equal Weight (Baseline)": "Equal Wt",
        "Minimum Volatility": "Min Vol",
        "Balanced (40% Cap)": "Balanced (Asset Cap)",
        "Max Sharpe (Unconstrained)": "Max Sharpe",
        "Sector Balanced": "Balanced (Sector Cap)",
        "User Portfolio": "User"
    }
    
    fig = make_subplots(
        rows=1, cols=len(port_names),
        specs=[[{'type':'domain'}] * len(port_names)],
        subplot_titles=[short_names.get(name, name) for name in port_names]
    )
    
    for i, name in enumerate(port_names):
        weights_dict = optimal_portfolios[name]['weights_dict']
        labels = [ticker for ticker, weight in weights_dict.items() if weight > 0.001]
        values = [weight for ticker, weight in weights_dict.items() if weight > 0.001]
        
        fig.add_trace(go.Pie(
            labels=labels, values=values, name=name, hole=0.4,
            marker=dict(
                colors=PALETTE[:len(labels)],
                line=dict(color='rgba(15,23,42,0.8)', width=2),
            ),
            textfont=dict(color='#f1f5f9', size=12),
            hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        ), row=1, col=i+1)

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title_text='Portfolio Strategy Compositions (by Asset)',
        **_base_layout(),
    )
    # Style subplot titles in white
    for annotation in fig.layout.annotations:
        annotation.font.color = "#cbd5e1"
        annotation.font.size = 12
    return fig

def plot_risk_contribution(optimal_portfolios, mean_returns, cov_matrix, tickers, risk_free_rate=0.02):
    """
    Plots a 100% stacked bar chart of portfolio risk contribution.
    """
    print("Plotting Risk Contribution...")
    
    risk_data = []
    port_names = list(optimal_portfolios.keys())
    
    for name in port_names:
        data = optimal_portfolios[name]
        weights = data['weights_arr']
        port_ret, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        port_variance = port_vol**2
        marginal_contrib = np.dot(cov_matrix, weights)
        
        for i, ticker in enumerate(tickers):
            contrib = weights[i] * marginal_contrib[i]
            pct_contrib = contrib / port_variance if port_variance > 0 else 0
            risk_data.append({'Portfolio': name, 'Ticker': ticker, 'Risk Contribution': pct_contrib})
            
    risk_df = pd.DataFrame(risk_data)
    
    fig = px.bar(
        risk_df, x='Portfolio', y='Risk Contribution', color='Ticker',
        title='Portfolio Risk Contribution (by Asset)',
        # Re-order x-axis to match the legend of other plots
        category_orders={"Portfolio": port_names},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        yaxis_tickformat='.0%', yaxis_title='Percent of Total Risk',
        xaxis_title=None, barmode='stack',
        **_base_layout(),
    )
    fig.update_traces(marker_line_color='rgba(15,23,42,0.5)', marker_line_width=0.5)
    return fig

def plot_monthly_heatmaps(eval_data, benchmark_ticker):
    """
    Plots a heatmap of monthly returns for each strategy.
    """
    print("Plotting Monthly Returns Heatmaps...")
    
    port_names = [col for col in eval_data.columns if col != benchmark_ticker]
    
    daily_returns_df = eval_data.pct_change().dropna()
    monthly_returns = daily_returns_df.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.strftime('%b')
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=len(port_names), cols=1,
        subplot_titles=port_names, vertical_spacing=0.1
    )

    for i, name in enumerate(port_names):
        pivot = monthly_returns.pivot_table(index='Year', columns='Month', values=name)
        pivot = pivot.reindex(columns=month_order)
        
        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale='RdYlGn', zmid=0,
            text=pivot.map(lambda x: f"{x:.1%}" if not pd.isna(x) else ""),
            texttemplate="%{text}", name=name
        ), row=i+1, col=1)

    fig.update_layout(
        title_text='Strategy Monthly Returns Heatmap',
        height=300 * len(port_names),
        **_base_layout(),
    )
    for annotation in fig.layout.annotations:
        annotation.font.color = "#cbd5e1"
    return fig

def plot_portfolio_vs_constituents(all_cumulative_returns):
    """
    Plots the cumulative returns for the portfolio and all its underlying assets.
    """
    print("Plotting: Portfolio vs. Constituents")
    fig = go.Figure()
    
    for idx, col in enumerate(all_cumulative_returns.columns):
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=all_cumulative_returns.index,
            y=all_cumulative_returns[col],
            name=col,
            mode='lines',
            line=dict(width=2, color=color),
        ))
    
    fig.update_layout(
        title='Portfolio vs. Constituent Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Growth',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_sector_allocation_pies(optimal_portfolios, friendly_sector_map):
    """
    Plots side-by-side pie charts of portfolio weights aggregated by sector.
    """
    print("Plotting Sector Allocation Pies...")
    if not friendly_sector_map:
        return go.Figure().update_layout(title="Sector Allocation (No sector_map provided)")

    port_names = list(optimal_portfolios.keys())
    
    short_names = {
        "Equal Weight (Baseline)": "Equal Wt", "Minimum Volatility": "Min Vol",
        "Balanced (40% Cap)": "Balanced (Asset Cap)", "Max Sharpe (Unconstrained)": "Max Sharpe",
        "Sector Balanced": "Balanced (Sector Cap)", "User Portfolio": "User"
    }
    
    fig = make_subplots(
        rows=1, cols=len(port_names),
        specs=[[{'type':'domain'}] * len(port_names)],
        subplot_titles=[short_names.get(name, name) for name in port_names]
    )
    
    for i, name in enumerate(port_names):
        weights_dict = optimal_portfolios[name]['weights_dict']
        
        # Aggregate weights by sector
        sector_weights = {}
        for ticker, weight in weights_dict.items():
            if weight > 0.001:
                sector = friendly_sector_map.get(ticker, "Other")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        labels = list(sector_weights.keys())
        values = list(sector_weights.values())
        
        fig.add_trace(go.Pie(
            labels=labels, values=values, name=name, hole=0.4,
            marker=dict(
                colors=PALETTE[:len(labels)],
                line=dict(color='rgba(15,23,42,0.8)', width=2),
            ),
            textfont=dict(color='#f1f5f9', size=12),
            hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        ), row=1, col=i+1)

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title_text='Portfolio Strategy Compositions (by Sector)',
        **_base_layout(),
    )
    for annotation in fig.layout.annotations:
        annotation.font.color = "#cbd5e1"
        annotation.font.size = 12
    return fig

def plot_sector_risk_contribution(optimal_portfolios, mean_returns, cov_matrix, tickers, friendly_sector_map, risk_free_rate=0.02):
    """
    Plots a 100% stacked bar chart of portfolio risk contribution by sector.
    """
    print("Plotting Sector Risk Contribution...")
    if not friendly_sector_map:
        return go.Figure().update_layout(title="Sector Risk Contribution (No sector_map provided)")

    risk_data = []
    port_names = list(optimal_portfolios.keys())
    
    for name in port_names:
        data = optimal_portfolios[name]
        weights = data['weights_arr']
        port_ret, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        port_variance = port_vol**2
        marginal_contrib = np.dot(cov_matrix, weights)
        
        for i, ticker in enumerate(tickers):
            contrib = weights[i] * marginal_contrib[i]
            pct_contrib = contrib / port_variance if port_variance > 0 else 0
            risk_data.append({
                'Portfolio': name,
                'Ticker': ticker,
                'Risk Contribution': pct_contrib
            })
            
    risk_df = pd.DataFrame(risk_data)
    
    # Map tickers to sectors
    risk_df['Sector'] = risk_df['Ticker'].map(friendly_sector_map).fillna("Other")
    
    # Aggregate by sector
    sector_risk_df = risk_df.groupby(['Portfolio', 'Sector'])['Risk Contribution'].sum().reset_index()

    fig = px.bar(
        sector_risk_df,
        x='Portfolio',
        y='Risk Contribution',
        color='Sector',
        title='Portfolio Risk Contribution (by Sector)',
        category_orders={"Portfolio": port_names}, # Match other plots
        color_discrete_sequence=PALETTE,
    )
    
    fig.update_layout(
        yaxis_tickformat='.0%',
        yaxis_title='Percent of Total Risk',
        xaxis_title=None,
        barmode='stack',
        **_base_layout(),
    )
    fig.update_traces(marker_line_color='rgba(15,23,42,0.5)', marker_line_width=0.5)
    return fig