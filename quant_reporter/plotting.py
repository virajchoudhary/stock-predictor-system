import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np

# ─── Attractive fintech palette (matches opt_plotting.py) ───────────────────
PALETTE = [
    "#38bdf8",  # sky blue
    "#a78bfa",  # soft violet
    "#34d399",  # emerald
    "#fb923c",  # vivid orange
    "#f472b6",  # rose pink
    "#facc15",  # golden yellow
    "#60a5fa",  # royal blue
    "#4ade80",  # lime green
]

def _base_layout(**extra):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#e2e8f0",
            size=13,
        ),
        legend=dict(
            bgcolor="rgba(30,41,59,0.75)",
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


def plot_cumulative_returns(plot_data, asset_ticker, benchmark_ticker):
    cumulative_returns = plot_data['cumulative_returns']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns['Asset'],
        name=f'{asset_ticker} Cumulative Returns',
        mode='lines',
        line=dict(color=PALETTE[0], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns['Benchmark'],
        name=f'{benchmark_ticker} Cumulative Returns',
        mode='lines',
        line=dict(color=PALETTE[1], width=2, dash='dot'),
    ))
    fig.update_layout(
        title='Performance vs. Benchmark',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_rolling_volatility(plot_data, asset_ticker, benchmark_ticker):
    daily_returns = plot_data['daily_returns']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_returns.index,
        y=(daily_returns['Asset'].rolling(30).std() * 252**0.5),
        name=f'{asset_ticker} 30-Day Rolling Vol',
        mode='lines',
        line=dict(color=PALETTE[0], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=daily_returns.index,
        y=(daily_returns['Benchmark'].rolling(30).std() * 252**0.5),
        name=f'{benchmark_ticker} 30-Day Rolling Vol',
        mode='lines',
        line=dict(color=PALETTE[1], width=2, dash='dot'),
    ))
    fig.update_layout(
        title='Rolling Volatility (Annualized)',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_regression(plot_data, metrics, asset_ticker, benchmark_ticker):
    daily_returns = plot_data['daily_returns']
    fig = px.scatter(
        daily_returns,
        x='Benchmark',
        y='Asset',
        title=f'{asset_ticker} vs {benchmark_ticker} Daily Returns (Beta)',
        opacity=0.5,
        trendline='ols',
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.update_layout(**_base_layout())
    fig.update_traces(marker_color=PALETTE[0], selector=dict(mode='markers'))
    return fig

def plot_rolling_sharpe(plot_data, asset_ticker):
    rolling_sharpe = plot_data['rolling_sharpe']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        name='Rolling Sharpe',
        mode='lines',
        line=dict(color=PALETTE[2], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(52,211,153,0.08)',
    ))
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.3)", line_dash="dash")
    fig.update_layout(
        title=f'{asset_ticker} 60-Day Rolling Sharpe Ratio',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig

def plot_monthly_distribution(plot_data, asset_ticker):
    monthly_returns = plot_data['monthly_returns'].dropna()
    if monthly_returns.empty:
        return go.Figure().update_layout(title=f'{asset_ticker} Monthly Returns Distribution (No Data)')
    
    # Clip outliers for better visualization
    q_01 = monthly_returns.quantile(0.01)
    q_99 = monthly_returns.quantile(0.99)
    plot_data_filtered = monthly_returns[(monthly_returns >= q_01) & (monthly_returns <= q_99)]
        
    fig = px.histogram(
        plot_data_filtered,
        x=plot_data_filtered,
        title=f'{asset_ticker} Monthly Returns Distribution (Outliers Clipped)',
        color_discrete_sequence=[PALETTE[0]],
    )
    
    mean_val = monthly_returns.mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f'Mean: {mean_val:.2%}',
                  annotation_position="top right",
                  annotation_font_color=PALETTE[3])
    
    fig.update_traces(marker_line_color='rgba(15,23,42,0.5)', marker_line_width=0.5, opacity=0.8)
    fig.update_layout(
        xaxis_title='Monthly Return',
        yaxis_title='Frequency (Count of Months)',
        **_base_layout(),
    )
    return fig

def plot_yearly_returns(plot_data, asset_ticker, benchmark_ticker):
    yearly_returns = plot_data['yearly_returns']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_returns.index.year,
        y=yearly_returns['Asset'],
        name=asset_ticker,
        marker_color=PALETTE[0],
        marker_line_color='rgba(15,23,42,0.5)',
        marker_line_width=0.5,
    ))
    fig.add_trace(go.Bar(
        x=yearly_returns.index.year,
        y=yearly_returns['Benchmark'],
        name=benchmark_ticker,
        marker_color=PALETTE[1],
        marker_line_color='rgba(15,23,42,0.5)',
        marker_line_width=0.5,
    ))
    fig.update_layout(
        title='Annual Returns Comparison',
        barmode='group',
        hovermode='x unified',
        **_base_layout(),
    )
    return fig