import streamlit as st

# Set page config at the very top
st.set_page_config(page_title="OptiMind Studio", layout="wide", page_icon="🧠")

from modules.profiler import render_profiler
from modules.baseline import render_baseline
from modules.optimizer import render_optimizer
from modules.analyzer import render_analyzer
from modules.explainability import render_explainability
from modules.tracker import render_tracker

def main():
    st.title("🧠 OptiMind Studio")
    st.markdown("### A Performance-Aware ML Optimization Platform")
    st.markdown("""
    OptiMind Studio transforms the standard "black-box" hyperparameter tuning process into an observable, scalable engineering workflow. 
    Progress through the steps below to systematically audit and optimize your models for both **Accuracy** and **Computational Efficiency**.
    """)
    st.markdown("---")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Studio Configuration")
        
        st.subheader("Model Selection")
        selected_models = st.multiselect(
            "Select models to compare:",
            ["Logistic Regression", "Random Forest"],
            default=["Logistic Regression"]
        )
        st.session_state['selected_models'] = selected_models
        
        st.subheader("Optimization Settings")
        perf_mode = st.radio(
            "Performance Mode:",
            ["Fast Mode (10 Trials)", "Deep Optimization Mode (50 Trials)"]
        )
        n_trials = 10 if "Fast" in perf_mode else 50
        st.session_state['n_trials'] = n_trials
        
        st.markdown("---")
        st.info("Built with ❤️ for the Hackathon.")

    # Define the UI Structure using horizontal tabs
    tab_titles = [
        "1️⃣ Dataset Profiler", 
        "2️⃣ Baseline Trainer", 
        "3️⃣ Optimization Engine", 
        "4️⃣ Performance Intelligence", 
        "5️⃣ Explainability", 
        "6️⃣ Report Generator"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        render_profiler()

    with tabs[1]:
        render_baseline()

    with tabs[2]:
        render_optimizer()

    with tabs[3]:
        render_analyzer()

    with tabs[4]:
        render_explainability()

    with tabs[5]:
        render_tracker()

if __name__ == "__main__":
    main()