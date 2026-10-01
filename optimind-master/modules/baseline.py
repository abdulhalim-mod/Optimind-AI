import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def split_data(X, y):
    # Retrieve or create train/test split in session state
    if 'train_test_split' not in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state['train_test_split'] = (X_train, X_test, y_train, y_test)
    return st.session_state['train_test_split']

def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    
    # Train
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Test Evaluation
    preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds, average='weighted')
    
    # Train Evaluation (for overfitting detection)
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    
    eff_score = test_acc / (training_time + 1e-6)
    
    metrics = {
        'training_time': training_time,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'train_acc': train_acc,
        'eff_score': eff_score,
        'gap': train_acc - test_acc
    }
    
    return model, metrics

def render_baseline():
    st.header("⚙️ Baseline Training & Overfitting Detection")
    
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("Please configure the dataset in the Profiler tab first.")
        return None

    X = st.session_state['X']
    y = st.session_state['y']
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    selected_models = st.session_state.get('selected_models', [])

    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
        return None

    if st.button("Train Baselines for Selected Models"):
        baseline_models = {}
        baseline_metrics = {}
        
        for model_choice in selected_models:
            # Initialize Baseline Model
            if model_choice == "Logistic Regression":
                base_model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "Random Forest":
                base_model = RandomForestClassifier(random_state=42)
            else:
                continue

            # Train and Evaluate Baseline
            with st.spinner(f"Training Baseline {model_choice}..."):
                trained_model, base_metrics = evaluate_model(base_model, X_train, y_train, X_test, y_test)
                baseline_models[model_choice] = trained_model
                baseline_metrics[model_choice] = base_metrics
                
        # Save baselines to session state
        st.session_state['baseline_models'] = baseline_models
        st.session_state['baseline_metrics'] = baseline_metrics
        st.success("Baseline training complete! Proceed to Optimization Engine.")

    if 'baseline_metrics' in st.session_state:
        baseline_metrics = st.session_state['baseline_metrics']
        
        for model_choice, metrics in baseline_metrics.items():
            st.markdown(f"### Baseline: {model_choice}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", f"{metrics['test_acc']:.4f}")
            c2.metric("Test F1 Score", f"{metrics['test_f1']:.4f}")
            c3.metric("Training Time", f"{metrics['training_time']:.4f} s")
            c4.metric("Efficiency Score", f"{metrics['eff_score']:.2f}")

            # Overfitting Detection
            st.markdown("#### ⚠️ Overfitting Detection")
            train_acc = metrics['train_acc']
            test_acc = metrics['test_acc']
            variance = metrics['gap']
            
            st.write(f"**Training Accuracy:** {train_acc:.4f} &nbsp;|&nbsp; **Testing Accuracy:** {test_acc:.4f} &nbsp;|&nbsp; **Generalization Gap:** {variance:.4f}")
            
            if variance > 0.05:
                st.error(f"Potential Overfitting Detected! The model performs {variance*100:.1f}% better on training data than unseen data. Use Optimization to add regularization or constraints.")
            elif variance < -0.05:
                st.warning("Underfitting or Data distribution mismatch detected.")
            else:
                st.success("Model generalizing well. Variance is within acceptable bounds (< 5%).")
            st.markdown("---")

        return st.session_state['baseline_metrics']
