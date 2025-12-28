import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add root directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.feature_engineering import engineer_features
from src.models.train_model import train_model, save_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict_single, explain_prediction, load_trained_model

# Page Config
st.set_page_config(
    page_title="Student Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Wow" factor and High Contrast
st.markdown("""
    <style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Background for Main Area */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }
    
    /* Sidebar Text Optimization */
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #f8fafc !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        margin-top: 0 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-size: 1.05rem !important;
        padding: 10px 5px;
    }

    /* Sidebar Radio Button Hover & Selection */
    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #1e293b;
        border-radius: 8px;
    }

    /* Main Area Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }

    /* Cards/Metric Containers */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        font-weight: 600;
        border: none;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.5);
        transform: scale(1.02);
    }

    /* Success/Error/Toast */
    .stToast {
        background-color: #1e293b;
        color: white;
        border: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHING FOR OPTIMIZATION ---
@st.cache_data
def get_data(path):
    return load_data(path)

# --- HISTORICAL STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3406/3406986.png", width=60)
    st.markdown("## Student Performance\n## Prediction System")
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    options = st.radio("MAIN MENU", 
        ["Dashboard", 
         "Data Studio", 
         "Model Training", 
         "Prediction Engine"],
        index=0
    )
    
    st.markdown("---")
    
    # History Sidebar Section
    if st.session_state.history:
        st.markdown("### 🕒 Recent Predictions")
        with st.container():
            for i, item in enumerate(reversed(st.session_state.history[-5:])): # Show last 5
                icon = "🟢" if item['result'] == "PASS" else "🔴"
                st.markdown(f"**{icon} {item['result']}** ({item['conf']:.0%})")
                st.caption(f"Att: {item['att']}% | Marks: {item['marks']}")
                st.markdown("---")
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    st.caption("v2.2.0 | Stable Build")

# --- GLOBAL STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'model' not in st.session_state:
    st.session_state.model = None

# --- SECTION 1: DASHBOARD (EDA) ---
# Moved EDA to first tab as "Dashboard" for better UX
if options == "Dashboard":
    col_t1, col_t2 = st.columns([4,1])
    with col_t1:
        st.title("Executive Dashboard")
        st.markdown("Overview of student performance metrics and logic.")
    with col_t2:
        st.markdown("### 🟢 Online")

    if st.session_state.df is None:
        st.markdown("""
            <div style='background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px dashed #475569; text-align: center; margin-bottom: 20px;'>
                <h3 style='color: #f8fafc; margin-bottom: 10px;'>👋 Welcome! Start by loading your data.</h3>
                <p style='color: #94a3b8;'>Upload your CSV file below to unlock the dashboard and prediction engine.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Add Uploader directly here for convenience
        col_up1, col_up2 = st.columns([1, 1], gap="large")
        with col_up1:
            uploaded_file = st.file_uploader("Upload CSV", type='csv', key="dash_uploader")
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.rerun()
        with col_up2:
            st.markdown("Top Features:")
            st.markdown("✅ **One-click Analysis**")
            st.markdown("✅ **AI Prediction**")
            if st.button("Load Sample Data", key="dash_sample"):
                try:
                    # Use cached data loading for sample
                    st.session_state.df = get_data("data/raw/student_data_raw.csv")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        df = st.session_state.df
        
        # High-level Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", df.shape[0])
        col2.metric("Avg Attendance", f"{df['Attendance (%)'].mean():.1f}%", "+2.4%")
        col3.metric("Avg Score", f"{df['Internal Marks (100)'].mean():.1f}", "+1.2")
        pass_rate = (df['Final Result (Pass/Fail)'].value_counts(normalize=True).get(1, 0)*100) if 'Final Result (Pass/Fail)' in df.columns else 0
        col4.metric("Pass Rate", f"{pass_rate:.1f}%", "-0.8%")
        
        st.markdown("---")
        
        # Row 1: Distributions
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Attendance Landscape")
            fig = px.histogram(df, x="Attendance (%)", nbins=20, color_discrete_sequence=['#6366f1'])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Score Distribution")
            fig = px.histogram(df, x="Internal Marks (100)", nbins=20, color_discrete_sequence=['#10b981'])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: Advanced Analysis
        c3, c4 = st.columns([1, 2])
        with c3:
            st.subheader("Outcome Ratio")
            if 'Final Result (Pass/Fail)' in df.columns:
                counts = df['Final Result (Pass/Fail)'].value_counts()
                # Fixed: px.donut does not exist, using px.pie with hole argument
                fig = px.pie(names=counts.index, values=counts.values, hole=0.6, color_discrete_sequence=['#ef4444', '#22c55e'])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
                fig.add_annotation(text=f"{pass_rate:.0f}% Pass", showarrow=False, font=dict(color="white", size=14))
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.subheader("Feature Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

# --- SECTION 2: DATA MANAGEMENT ---
elif options == "Data Studio":
    st.title("Data Studio")
    
    col_input, col_view = st.columns([1, 2], gap="large")
    
    with col_input:
        with st.container():
            st.markdown("#### 📥 Ingest")
            uploaded_file = st.file_uploader("Upload CSV", type='csv', help="Upload your student record CSV here")
            
            st.markdown("Or")
            if st.button("Load Sample Data", use_container_width=True):
                try:
                    st.session_state.df = get_data("data/raw/student_data_raw.csv")
                    st.toast("Sample data loaded successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")
                    
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.toast("File uploaded!", icon="✅")
        
        st.divider()
        
        st.markdown("#### 🧬 Operations")
        if st.session_state.df is not None:
            if st.button("Run Text-Cleaning Pipeline", use_container_width=True):
                 with st.spinner("Processing..."):
                        try:
                            # Re-run pipeline
                            cleaned_df = clean_data(st.session_state.df)
                            processed_df, feature_cols = engineer_features(cleaned_df)
                            st.session_state.df_processed = processed_df
                            st.session_state.feature_cols = feature_cols
                            st.success("Pipeline Completed")
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")
    
    with col_view:
        st.markdown("#### 🔭 Data Viewer")
        if st.session_state.df is not None:
             tab_raw, tab_proc = st.tabs(["Raw Source", "Engineered Features"])
             
             with tab_raw:
                 st.dataframe(st.session_state.df, height=400, use_container_width=True)
                 
             with tab_proc:
                 if st.session_state.df_processed is not None:
                     st.dataframe(st.session_state.df_processed, height=400, use_container_width=True)
                 else:
                     st.info("No processed data available. Run the pipeline.")

# --- SECTION 3: TRAINING ---
elif options == "Model Training":
    st.title("Model Training Lab")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ No processed data found. Please visit Data Studio.")
    else:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### Hyperparameters")
            with st.form("train_conf"):
                model_type = st.selectbox("Algorithm", ["Logistic Regression", "Decision Tree"])
                split = st.slider("Test Split Ratio", 0.1, 0.4, 0.2)
                
                st.markdown("---")
                run_train = st.form_submit_button("Start Experiment")
        
        with c2:
            st.markdown("### Experiment Results")
            if run_train:
                with st.spinner("Training model..."):
                    try:
                        df_proc = st.session_state.df_processed
                        X = df_proc[['Attendance (%)', 'Internal Marks (100)', 'Activities_Encoded']]
                        y = df_proc['Target']
                        
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
                        
                        m_key = 'logistic' if model_type == 'Logistic Regression' else 'decision_tree'
                        model = train_model(X_train, y_train, m_key)
                        st.session_state.model = model
                        
                        metrics = evaluate_model(model, X_test, y_test)
                        save_model(model, "models/student_model.pkl")
                        
                        # Results
                        r1, r2 = st.columns(2)
                        r1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        r2.metric("Precision (Pass)", f"{metrics['classification_report']['1']['precision']:.2f}")
                        
                        st.success("Model saved to registry.")
                        
                        # Confusion Matrix
                        cm = np.array(metrics['confusion_matrix'])
                        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', 
                                      labels=dict(x="Predicted", y="Actual"), x=['Fail', 'Pass'], y=['Fail', 'Pass'])
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(str(e))

# --- SECTION 4: PREDICTION ---
elif options == "Prediction Engine":
    st.title("AI Prediction Engine")
    
    model = st.session_state.model
    if not model and os.path.exists("models/student_model.pkl"):
        model = load_trained_model("models/student_model.pkl")
        st.session_state.model = model
    
    if not model:
        st.warning("Please train a model first.")
    else:
        main_col, side_col = st.columns([1, 1], gap="large")
        
        with main_col:
            st.markdown("### 👤 Student Profile")
            with st.container():
                st.markdown("Adjust the parameters to simulate a student profile.")
                att = st.slider("Attendance", 0, 100, 75, format="%d%%")
                marks = st.slider("Internal Assessment", 0, 100, 60)
                act = st.radio("Extracurriculars", ["Yes", "No"], horizontal=True)
                
                predict = st.button("Generate Prediction", type="primary")

        with side_col:
            st.markdown("### 🔮 Forecast")
            if predict:
                act_enc = 1 if act == "Yes" else 0
                pred, prob = predict_single(model, [att, marks, act_enc])
                explain = explain_prediction(pred, att, marks, act_enc)
                
                # Save to History
                result_str = "PASS" if pred == 1 else "FAIL"
                st.session_state.history.append({
                    "result": result_str,
                    "conf": prob if pred == 1 else (1-prob),
                    "att": att,
                    "marks": marks
                })
                
                if pred == 1:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; border-radius: 12px; padding: 20px; text-align: center;">
                        <h1 style="color: #34d399; margin: 0;">PASS</h1>
                        <p style="color: #a7f3d0; margin: 0;">Confidence: {prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 12px; padding: 20px; text-align: center;">
                        <h1 style="color: #f87171; margin: 0;">FAIL</h1>
                        <p style="color: #fca5a5; margin: 0;">Confidence: {1-prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"**Analysis:** {explain}")
                
                # Radar
                fig = px.line_polar(r=[att, marks, act_enc*100], theta=['Attendance', 'Marks', 'Activities'], line_close=True)
                fig.update_traces(fill='toself', line_color='#3b82f6')
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
