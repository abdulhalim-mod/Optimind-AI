import streamlit as st
import pandas as pd
import plotly.express as px

def render_analyzer():
    st.header("📈 Performance Intelligence Dashboard")
    
    if 'baseline_metrics' not in st.session_state or 'opt_metrics' not in st.session_state:
        st.warning("Please complete both Baseline Training and Optimization to view insights.")
        return None
        
    baseline_all = st.session_state['baseline_metrics']
    opt_all = st.session_state['opt_metrics']

    # Find the Recommended Model based on max Efficiency Score
    recommended_model = None
    max_eff = -1
    for model_name, metrics in opt_all.items():
        if metrics['eff_score'] > max_eff:
            max_eff = metrics['eff_score']
            recommended_model = model_name

    if recommended_model:
        st.success(f"🏆 **Recommended Pipeline:** {recommended_model} (Highest Efficiency Score: {max_eff:.2f})")

    st.markdown("---")
    
    for model_name in opt_all.keys():
        if model_name not in baseline_all:
            continue
            
        base = baseline_all[model_name]
        opt = opt_all[model_name]

        # Computations
        acc_improvement = (opt['test_acc'] - base['test_acc']) / base['test_acc'] * 100
        eff_improvement = (opt['eff_score'] - base['eff_score']) / base['eff_score'] * 100
        time_diff = opt['training_time'] - base['training_time']

        st.subheader(f"Optimization Impact: {model_name}")
        
        # Render Metrics Side-by-Side
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy Improvement", f"{opt['test_acc']:.4f}", f"{acc_improvement:+.2f}%")
        c2.metric("Efficiency Score", f"{opt['eff_score']:.2f}", f"{eff_improvement:+.2f}%")
        c3.metric("Training Time", f"{opt['training_time']:.4f} s", f"{time_diff:+.4f} s", delta_color="inverse")
        c4.metric("Test F1 Score", f"{opt['test_f1']:.4f}", f"{(opt['test_f1'] - base['test_f1']):+.4f}")
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Performance Trade-off Analysis")
    
    # Aggregate data for scatter plot
    scatter_data = []
    for model_name in opt_all.keys():
        if model_name in baseline_all:
            scatter_data.append({
                "Model": model_name,
                "State": "Baseline",
                "Accuracy": baseline_all[model_name]['test_acc'],
                "Training Time (s)": baseline_all[model_name]['training_time'],
                "Efficiency Score": baseline_all[model_name]['eff_score']
            })
            scatter_data.append({
                "Model": model_name,
                "State": "Optimized",
                "Accuracy": opt_all[model_name]['test_acc'],
                "Training Time (s)": opt_all[model_name]['training_time'],
                "Efficiency Score": opt_all[model_name]['eff_score']
            })
            
    comp_df = pd.DataFrame(scatter_data)
    
    fig_scatter = px.scatter(
        comp_df, x="Training Time (s)", y="Accuracy",
        color="Model", symbol="State", size="Efficiency Score",
        hover_name="Model",
        title="Accuracy vs. Training Time (Top-Left is Ideal)"
    )
    fig_scatter.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    st.plotly_chart(fig_scatter, use_container_width=True)
