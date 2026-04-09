import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Evolutionary HPO", layout="wide")

st.title("Evolutionary HPO for Algorithmic Trading")
st.markdown("Use a Genetic Algorithm to evolve the optimal 'brain' (LSTM Hyperparameters) for our **Market Trend Predictor**. Instead of guessing how the AI should be structured, we simulate thousands of trading scenarios acting on historical market data to find the hyperparameters that maximize our **Directional Trading Accuracy** (correctly predicting if a stock will go UP or DOWN).")

# --- Live HPO Status ---
symbol_default = "AAPL"
status_placeholder = st.empty()

def render_status(sym):
    try:
        r = requests.get(f"{API_URL}/hpo-status/?symbol={sym}", timeout=5)
        d = r.json()
        if d.get('optimized'):
            from datetime import datetime, timezone
            cached_at = datetime.fromisoformat(d['cached_at'])
            hours_ago = (datetime.now(timezone.utc) - cached_at).seconds // 3600
            status_placeholder.success(
                f"Optimized model active for **{sym}** — "
                f"Directional Accuracy: **{d['directional_accuracy']:.1f}%**, "
                f"Val MSE: {d['val_mse']:.6f} — cached {hours_ago}h ago. "
                f"Running again will overwrite."
            )
        else:
            status_placeholder.warning(
                f"No optimized model found for **{sym}**. "
                f"Predictions are using default hyperparameters."
            )
    except Exception:
        status_placeholder.info("Could not reach backend to check HPO status.")

render_status(symbol_default)

# Sidebar for Evolution Parameters
with st.sidebar:
    st.header("Genetic Algorithm Params")
    symbol = st.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
    pop_size = st.slider("Population Size", min_value=3, max_value=20, value=5)
    generations = st.slider("Number of Generations", min_value=1, max_value=20, value=5)
    mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    start_btn = st.button("Start Evolution", use_container_width=True)

    st.markdown("---")
    with st.expander("How the Genetic Algorithm Works"):
        st.markdown(
            "A population of LSTM configurations (chromosomes) is evolved over generations. "
            "Each configuration is trained on 2 years of historical data and scored on a "
            "composite fitness combining validation MSE and directional trading accuracy. "
            "The best configurations survive, crossover, and mutate — converging toward an "
            "optimal model architecture for the specific stock."
        )

if start_btn:
    # Update status for selected symbol
    render_status(symbol)

    st.subheader(f"Evolving Architecture for {symbol}...")
    
    # Progress and Status Setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Grid for metrics
    col1, col2 = st.columns(2)
    with col1:
        metrics_placeholder = st.empty()
    with col2:
        best_chrom_placeholder = st.empty()
        
    st.divider()
    
    # Data visualization placeholders
    chart_placeholder = st.empty()
    pop_table_placeholder = st.empty()
    
    history_df = pd.DataFrame(columns=["Generation", "Best Loss (MSE)"])
    all_population_data = []

    try:
        url = f"{API_URL}/evolution/?symbol={symbol}&pop_size={pop_size}&generations={generations}&mutation_rate={mutation_rate}"
        response = requests.get(url, stream=True)
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8').strip()
                if not decoded:
                    continue
                
                try:
                    data = json.loads(decoded)
                except json.JSONDecodeError:
                    continue
                
                if "error" in data:
                    st.error(f"Error from server: {data['error']}")
                    break
                    
                if "status" in data and data["status"] == "completed":
                    status_text.success(f"Evolution Completed! Best Overall MSE: {data['best_overall_loss']:.6f} | Accuracy: {data.get('best_overall_accuracy', 0):.1f}%")
                    # Final update to best chromosome box
                    with best_chrom_placeholder.container():
                        st.markdown("**Ultimate Fittest Chromosome:**")
                        st.json(data["best_overall_chromosome"])
                    # Refresh status to confirm DB cache was saved
                    render_status(symbol)
                    break
                    
                # Standard generation payload
                gen = data["generation"]
                progress_bar.progress(gen / generations)
                status_text.info(f"Processing Generation {gen} / {generations}... Please wait...")
                
                # Top Metrics
                with metrics_placeholder.container():
                    st.metric(label=f"Generation {gen} Best Loss (MSE)", value=f"{data['best_loss']:.6f}")
                    acc_color = "normal" if data.get('best_accuracy', 0) >= 50 else "inverse"
                    st.metric(label=f"Generation {gen} Directional Accuracy (%)", value=f"{data.get('best_accuracy', 0):.2f}%", delta="Trading Edge Found!" if data.get('best_accuracy', 0) > 50 else "No Trading Edge", delta_color=acc_color)
                    
                with best_chrom_placeholder.container():
                    st.markdown(f"**Best Chromosome (Gen {gen}):**")
                    st.json(data["best_chromosome"])
                    
                # Charting history
                new_row = pd.DataFrame([{
                    "Generation": gen,
                    "Best Loss (MSE)": data["best_loss"],
                    "Directional Accuracy (%)": data.get('best_accuracy', 0.0)
                }])
                history_df = pd.concat([history_df, new_row], ignore_index=True)
                
                # Create two charts horizontally: Loss and Accuracy
                fig_loss = px.line(
                    history_df, 
                    x="Generation", 
                    y="Best Loss (MSE)", 
                    title="Fitness (Validation Loss)", 
                    markers=True
                )
                fig_loss.update_layout(yaxis_title="Validation MSE (Lower is Better)", title_x=0.5)

                fig_acc = px.line(
                    history_df, 
                    x="Generation", 
                    y="Directional Accuracy (%)", 
                    title="Trading Direction Accuracy", 
                    markers=True
                )
                fig_acc.update_layout(yaxis_title="Accuracy % (Higher is Better)", yaxis_range=[0, 100], title_x=0.5)
                # Add a horizontal line at 50% marking the "profitable" barrier
                fig_acc.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Profitable Edge (>50%)")

                # Put charts side-by-side using Streamlit columns dynamically inside placeholder
                with chart_placeholder.container():
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(fig_loss, use_container_width=True)
                    c2.plotly_chart(fig_acc, use_container_width=True)

                # Population Table for current generation
                pop_df = []
                for idx, ind in enumerate(data["population"]):
                    c = ind["chromosome"]
                    pop_df.append({
                        "Rank": idx + 1,
                        "Hidden Size": c[0],
                        "Num Layers": c[1],
                        "Learning Rate": c[2],
                        "Epochs": c[3],
                        "Loss (MSE)": ind["loss"],
                        "Accuracy (%)": ind.get("accuracy", 0.0)
                    })
                pop_df_display = pd.DataFrame(pop_df)
                pop_table_placeholder.dataframe(pop_df_display, use_container_width=True)
                
                # Accumulate for parallel coordinates
                for ind in data["population"]:
                    c = ind["chromosome"]
                    all_population_data.append({
                        "Generation": gen, 
                        "Hidden Size": c[0], 
                        "Num Layers": c[1], 
                        "LR": c[2], 
                        "Epochs": c[3], 
                        "Loss": ind["loss"],
                        "Accuracy": ind.get("accuracy", 0.0)
                    })
                
    except requests.exceptions.RequestException as e:
         st.error(f"Failed to connect to backend: {e}")
         st.warning("Make sure the Django backend is running (`python manage.py runserver`).")
        
    # Render Parallel Coordinates if data exists after run
    if all_population_data:
        st.divider()
        st.subheader("Visualizing the Search Space (All Candidates)")
        st.markdown("This Parallel Coordinates plot shows every hyperparameter combination our genetic algorithm evaluated. Tracing the lines to the **darkest/lowest Loss** values reveals which specific parameter ranges are most effective.")
        df_all = pd.DataFrame(all_population_data)
        
        fig_pc = go.Figure(data=
            go.Parcoords(
                line=dict(color=df_all['Loss'], colorscale='Plasma_r', showscale=True, cmin=df_all['Loss'].min(), cmax=df_all['Loss'].max()),
                dimensions=list([
                    dict(range=[df_all['Hidden Size'].min(), df_all['Hidden Size'].max()], label='Hidden Size', values=df_all['Hidden Size']),
                    dict(range=[df_all['Num Layers'].min(), df_all['Num Layers'].max()], label='Num Layers', values=df_all['Num Layers']),
                    dict(range=[df_all['LR'].min(), df_all['LR'].max()], label='Learning Rate', values=df_all['LR']),
                    dict(range=[df_all['Epochs'].min(), df_all['Epochs'].max()], label='Epochs', values=df_all['Epochs']),
                    dict(range=[df_all['Loss'].min(), df_all['Loss'].max()], label='Validation Loss', values=df_all['Loss']),
                    dict(range=[df_all['Accuracy'].min(), df_all['Accuracy'].max()], label='Trading Accuracy %', values=df_all['Accuracy'])
                ])
            )
        )
        st.plotly_chart(fig_pc, use_container_width=True)
