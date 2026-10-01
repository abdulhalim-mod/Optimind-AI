import streamlit as st
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from modules.baseline import evaluate_model

def get_objective(model_choice, X_train, y_train, X_test, y_test):
    def objective(trial):
        if model_choice == "Logistic Regression":
            C = trial.suggest_float("C", 0.001, 10.0, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
        elif model_choice == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model: {model_choice}")
        
        # Fit and return test accuracy for Optuna
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test, preds)
        
    return objective

def render_optimizer():
    st.header("⚙️ Optimization Engine")
    
    if 'train_test_split' not in st.session_state or 'selected_models' not in st.session_state:
        st.warning("Please configure the studio in the Sidebar and complete Profiling/Baseline.")
        return None

    X_train, X_test, y_train, y_test = st.session_state['train_test_split']
    selected_models = st.session_state['selected_models']
    n_trials = st.session_state.get('n_trials', 20)

    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
        return None

    st.write(f"**Optimization Mode:** {'Deep (50 Trials)' if n_trials >= 50 else 'Fast (10 Trials)'}")

    if st.button("🚀 Run OptiMind Engine", type="primary"):
        opt_studies = {}
        opt_best_params = {}
        opt_models = {}
        opt_metrics = {}

        for model_choice in selected_models:
            with st.spinner(f"Running Optuna ({n_trials} trials) for {model_choice}..."):
                study = optuna.create_study(direction="maximize")
                objective = get_objective(model_choice, X_train, y_train, X_test, y_test)
                study.optimize(objective, n_trials=n_trials)
                
                best_params = study.best_params
                
                # Train model with best params
                if model_choice == "Logistic Regression":
                    opt_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
                elif model_choice == "Random Forest":
                    opt_model = RandomForestClassifier(**best_params, random_state=42)
                    
                trained_opt_model, metrics = evaluate_model(opt_model, X_train, y_train, X_test, y_test)
                
                opt_studies[model_choice] = study
                opt_best_params[model_choice] = best_params
                opt_models[model_choice] = trained_opt_model
                opt_metrics[model_choice] = metrics

        # Save optimization results to session state
        st.session_state['opt_studies'] = opt_studies
        st.session_state['opt_best_params'] = opt_best_params
        st.session_state['opt_models'] = opt_models
        st.session_state['opt_metrics'] = opt_metrics
        
        st.success("Optimization Complete! Move to Performance Intelligence down below.")
            
    # Display Optimized Settings if available
    if 'opt_best_params' in st.session_state:
        for model_choice, params in st.session_state['opt_best_params'].items():
            st.subheader(f"Best Hyperparameters: {model_choice}")
            st.json(params)
