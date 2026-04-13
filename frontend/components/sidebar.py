import streamlit as st
from datetime import datetime

def render_backtest_sidebar():
    """Renders the backtesting controls in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-family:'Roboto Mono',monospace;font-size:0.58rem;letter-spacing:0.2em;
    color:#7A9BB5;text-transform:uppercase;padding:0.25rem 0;">Simulation & Backtesting</div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("⚙️ Backtest Settings", expanded=False):
        # Default to today if not set
        if "target_date" not in st.session_state:
            st.session_state.target_date = datetime.now()
            
        selected_date = st.date_input(
            "Evaluation Date",
            value=st.session_state.target_date,
            max_value=datetime.now(),
            help="Set the 'Current Day' for the system. All data and predictions will respect this cutoff."
        )
        
        # Convert to datetime if it's just a date
        if isinstance(selected_date, (datetime, type(datetime.now().date()))):
            st.session_state.target_date = selected_date
            
        # Display warning if backtesting is active
        is_backtesting = selected_date < datetime.now().date()
        if is_backtesting:
            st.warning(f"Backtesting Active: Using data up to {selected_date}")
            if st.button("Reset to Today"):
                st.session_state.target_date = datetime.now()
                st.rerun()
                
    return st.session_state.target_date.strftime("%Y-%m-%d") if "target_date" in st.session_state else None
