import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

def render_explainability():
    st.header("🔍 Explainability & Visualization")
    
    if 'opt_models' not in st.session_state or 'opt_studies' not in st.session_state:
        st.warning("Please complete Optimization to view explainability charts.")
        return None

    selected_models = st.session_state['selected_models']
    
    # Establish sub-tabs if multiple models are selected
    if len(selected_models) > 1:
        model_tabs = st.tabs(selected_models)
    else:
        model_tabs = [st.container()]
        st.subheader(f"Analyzing: {selected_models[0]}")

    for idx, model_choice in enumerate(selected_models):
        if model_choice not in st.session_state['opt_models']:
            continue
            
        with model_tabs[idx]:
            opt_model = st.session_state['opt_models'][model_choice]
            study = st.session_state['opt_studies'][model_choice]
            feature_names = st.session_state['feature_names']
            
            tab1, tab2, tab3, tab4 = st.tabs(["Optimization History", "Parallel Coordinates", "Hyperparameter Impact", "Feature Importance"])

            with tab1:
                st.markdown("### Objective Value Evolution")
                st.markdown("Tracks how the model's accuracy improved over successive Optuna trials.")
                fig_history = plot_optimization_history(study)
                st.plotly_chart(fig_history, use_container_width=True)

            with tab2:
                st.markdown("### Hyperparameter Relationships (Parallel Coordinates)")
                st.markdown("Visualizes high-dimensional relationships between hyperparameters and the objective value. Trace the highest accuracy lines to see correlated parameter groupings.")
                try:
                    fig_parallel = plot_parallel_coordinate(study)
                    st.plotly_chart(fig_parallel, use_container_width=True)
                except Exception as e:
                    st.warning("Not enough variance or trials to plot parallel coordinates.")

            with tab3:
                st.markdown("### Hyperparameter Importance")
                st.markdown("Highlights the specific hyperparameters that mathematically drove the most performance variation.")
                try:
                    fig_importances = plot_param_importances(study)
                    st.plotly_chart(fig_importances, use_container_width=True)
                except Exception as e:
                    st.warning("Not enough variance across trials to plot parameter importances.")

            with tab4:
                st.markdown("### Model Feature Importance")
                st.markdown(f"Highlights the top 10 most influential features the optimized **{model_choice}** uses to make predictions.")
                
                try:
                    if model_choice == "Logistic Regression":
                        importance = np.abs(opt_model.coef_[0])
                    elif model_choice == "Random Forest":
                        importance = opt_model.feature_importances_
                    else:
                        importance = []
                        
                    if len(importance) > 0:
                        feat_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importance
                        }).sort_values(by="Importance", ascending=False).head(10)
                        
                        fig_feat = px.bar(
                            feat_df.sort_values(by="Importance", ascending=True), 
                            x="Importance", y="Feature", 
                            orientation='h', 
                            title=f"Top 10 Drivers of Classification ({model_choice})",
                            color="Importance",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_feat, use_container_width=True)
                    else:
                        st.info("Feature importance extraction not supported for this architecture.")
                except Exception as e:
                    st.error(f"Could not extract feature importances: {str(e)}")
