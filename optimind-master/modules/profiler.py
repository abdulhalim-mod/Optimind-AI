import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer

def render_profiler():
    st.header("📊 Dataset Profiler")
    st.write("Upload a CSV dataset or use the default Breast Cancer dataset to begin optimization.")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Custom dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None, None
    else:
        # Fallback to default
        st.info("No file uploaded. Using default Breast Cancer dataset.")
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

    # Determine target column (assume last column if not explicitly named 'target')
    if 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]
        st.warning(f"No column named 'target' found. Assuming '{target_col}' is the target variable.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    # Display Profiling Stats
    st.subheader("Dataset Statistics")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", X.shape[0])
    c2.metric("Total Features", X.shape[1])
    c3.metric("Missing Values", X.isnull().sum().sum())
    
    # Class distribution (assuming classification for now)
    num_classes = y.nunique()
    c4.metric("Target Classes", num_classes)

    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    with st.expander("Class Distribution Chart"):
        class_dist = y.value_counts().reset_index()
        class_dist.columns = ['Class', 'Count']
        st.bar_chart(class_dist.set_index('Class'))

    # Store in session state for other modules
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['feature_names'] = feature_names
    
    return X, y, feature_names
