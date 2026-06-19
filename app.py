import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_crop_yield_data
from pipeline import CropYieldPipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
        .main-header {
            font-size: 40px;
            font-weight: 700;
            color: #2E7D32;
            text-align: center;
        }
        .sub-header {
            font-size: 20px;
            color: #555;
            text-align: center;
        }
        .metric-container {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .stButton>button {
            background-color: #2E7D32;
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1B5E20;
        }
        .stProgress .st-bo {
            background-color: #2E7D32;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================
if "data" not in st.session_state:
    st.session_state.data = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "clusters" not in st.session_state:
    st.session_state.clusters = None
if "trained" not in st.session_state:
    st.session_state.trained = False


# =========================
# HELPER FUNCTIONS
# =========================
@st.cache_data
def load_default_data(n_samples=2000):
    return generate_crop_yield_data(n_samples=n_samples)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1598/1598196.png", width=120)
    st.markdown("## 🌾 Crop Yield ML")
    st.markdown("---")

    nav = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset", "🔍 EDA", "🤖 Train Models", "⚖️ Model Comparison", "🔮 Predict"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    sample_size = st.slider("Dataset Size", 500, 5000, 2000, 500)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Custom dataset uploaded!")
    else:
        st.session_state.data = load_default_data(sample_size)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")


# =========================
# HOME PAGE
# =========================
if nav == "🏠 Home":
    st.markdown('<p class="main-header">🌾 Crop Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Approaches for Agricultural Decision Support</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>🌱 Supervised</h4>
            <p>Random Forest & XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>🧠 Unsupervised</h4>
            <p>K-Means + PCA</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>🔄 Semi-Supervised</h4>
            <p>Self-Training Regression</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### 🎯 Project Objectives
    - Predict crop yield using environmental & agricultural features.
    - Compare supervised, unsupervised, semi-supervised, and neural network approaches.
    - Provide an interactive interface for farmers/agronomists.

    ### 📂 Dataset Features
    | Feature | Description |
    |---------|-------------|
    | State | Indian state region |
    | Crop | Crop type (Rice, Wheat, etc.) |
    | Season | Kharif / Rabi / Zaid / Whole Year |
    | Area | Cultivation area (ha) |
    | Annual Rainfall | Rainfall in mm |
    | Fertilizer | Fertilizer usage (kg/ha) |
    | Pesticide | Pesticide usage (kg/ha) |
    | Yield | Target: Crop yield (tons/ha) |
    """)

    if st.session_state.data is not None:
        st.success(f"Loaded dataset with {st.session_state.data.shape[0]} rows and {st.session_state.data.shape[1]} columns.")


# =========================
# DATASET PAGE
# =========================
elif nav == "📊 Dataset":
    st.markdown('<p class="main-header">📊 Dataset Overview</p>', unsafe_allow_html=True)

    df = st.session_state.data

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Crops", df["Crop"].nunique())
    c4.metric("Avg Yield", f"{df['Yield'].mean():.2f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Statistics", "Correlation"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True)

    with tab2:
        st.write(df.describe())

    with tab3:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Feature Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# EDA PAGE
# =========================
elif nav == "🔍 EDA":
    st.markdown('<p class="main-header">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    df = st.session_state.data

    tab1, tab2, tab3, tab4 = st.tabs(["Yield by Crop", "Yield vs Rainfall", "Distribution", "Feature Pairplot"])

    with tab1:
        crop_yield = df.groupby("Crop")["Yield"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(
            crop_yield,
            x="Yield",
            y="Crop",
            orientation="h",
            color="Yield",
            color_continuous_scale="Greens",
            title="Average Yield by Crop"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.scatter(
            df,
            x="Annual_Rainfall",
            y="Yield",
            color="Crop",
            size="Area",
            opacity=0.7,
            title="Yield vs Annual Rainfall"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.histogram(
            df,
            x="Yield",
            nbins=40,
            color="Season",
            title="Yield Distribution by Season",
            barmode="overlay"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        numeric_df = df.select_dtypes(include=[np.number])
        fig = px.scatter_matrix(
            numeric_df,
            dimensions=numeric_df.columns,
            color="Yield",
            color_continuous_scale="RdYlGn",
            title="Feature Pairplot"
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # Unsupervised cluster visualization
    if st.session_state.trained and st.session_state.clusters is not None:
        st.markdown("---")
        st.subheader("🧠 Unsupervised K-Means Clusters (PCA 2D Projection)")
        fig = px.scatter(
            st.session_state.clusters,
            x="PC1",
            y="PC2",
            color="Cluster",
            size="Yield",
            title="K-Means Clustering on Agricultural Features",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# TRAIN MODELS PAGE
# =========================
elif nav == "🤖 Train Models":
    st.markdown('<p class="main-header">🤖 Train ML Models</p>', unsafe_allow_html=True)

    df = st.session_state.data

    col1, col2 = st.columns(2)
    with col1:
        st.info("Models to train: Random Forest, XGBoost, Neural Network (MLP), Semi-Supervised Self-Training")
    with col2:
        labeled_ratio = st.slider("Semi-Supervised Labeled Ratio", 0.1, 0.5, 0.2, 0.05)

    if st.button("🚀 Start Training", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        # Initialize pipeline
        status.text("Preparing data...")
        pipe = CropYieldPipeline(df)
        pipe.split_data(test_size=test_size)
        progress.progress(10)

        # Supervised
        status.text("Training Random Forest & XGBoost...")
        sup_results = pipe.train_supervised()
        progress.progress(40)

        # Neural Network
        status.text("Training Neural Network (MLP)...")
        nn_results = pipe.train_neural_network()
        progress.progress(70)

        # Semi-Supervised
        status.text("Training Semi-Supervised model...")
        semi_results = pipe.train_semi_supervised(labeled_ratio=labeled_ratio)
        progress.progress(85)

        # Unsupervised
        status.text("Running K-Means clustering...")
        clusters = pipe.get_unsupervised_clusters()
        progress.progress(100)

        # Store results
        all_results = pd.concat([sup_results, nn_results, semi_results], ignore_index=True)
        st.session_state.results = all_results
        st.session_state.pipeline = pipe
        st.session_state.clusters = clusters
        st.session_state.trained = True

        status.success("Training complete! Go to 'Model Comparison' or 'Predict'.")

    if st.session_state.trained:
        st.markdown("---")
        st.subheader("📋 Evaluation Results")
        st.dataframe(st.session_state.results, use_container_width=True)

        # Cross-validation
        st.subheader("🔄 Cross-Validation R² Scores")
        cv_results = []
        for model_name in st.session_state.pipeline.models.keys():
            scores = st.session_state.pipeline.cross_validate(model_name, cv=5)
            cv_results.append({
                "Model": model_name,
                "Mean R²": f"{scores.mean():.3f}",
                "Std R²": f"{scores.std():.3f}"
            })
        st.dataframe(pd.DataFrame(cv_results), use_container_width=True)

        # Feature importance
        st.subheader("⭐ Feature Importance (Random Forest)")
        fi = st.session_state.pipeline.get_feature_importance("Random Forest")
        if fi is not None:
            fig = px.bar(
                fi.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================
# MODEL COMPARISON PAGE
# =========================
elif nav == "⚖️ Model Comparison":
    st.markdown('<p class="main-header">⚖️ Model Comparison</p>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("Please train models first from the '🤖 Train Models' page.")
    else:
        results = st.session_state.results

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model (R²)", results.loc[results["R²"].idxmax(), "Model"])
        col2.metric("Best R² Score", f"{results['R²'].max():.3f}")
        col3.metric("Lowest RMSE", f"{results['RMSE'].min():.3f}")

        st.markdown("---")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("R² Score", "RMSE", "MAE"),
            horizontal_spacing=0.1
        )

        fig.add_trace(go.Bar(x=results["Model"], y=results["R²"], marker_color="green"), row=1, col=1)
        fig.add_trace(go.Bar(x=results["Model"], y=results["RMSE"], marker_color="orange"), row=1, col=2)
        fig.add_trace(go.Bar(x=results["Model"], y=results["MAE"], marker_color="blue"), row=1, col=3)

        fig.update_layout(height=500, showlegend=False, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        fig2 = go.Figure()
        for _, row in results.iterrows():
            fig2.add_trace(go.Scatterpolar(
                r=[row["R²"], 1/(1+row["RMSE"]), 1/(1+row["MAE"]), row["R²"]],
                theta=["R²", "1/(1+RMSE)", "1/(1+MAE)", "R²"],
                fill="toself",
                name=row["Model"]
            ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Model Performance Radar"
        )
        st.plotly_chart(fig2, use_container_width=True)


# =========================
# PREDICT PAGE
# =========================
elif nav == "🔮 Predict":
    st.markdown('<p class="main-header">🔮 Predict Crop Yield</p>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("Please train models first from the '🤖 Train Models' page.")
    else:
        df = st.session_state.data

        model_choice = st.selectbox("Select Model", list(st.session_state.pipeline.models.keys()))

        st.markdown("### 📝 Input Agricultural Features")

        col1, col2, col3 = st.columns(3)
        with col1:
            state = st.selectbox("State", sorted(df["State"].unique()))
            crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
        with col2:
            season = st.selectbox("Season", sorted(df["Season"].unique()))
            area = st.slider("Area (ha)", 1.0, 100.0, 25.0)
        with col3:
            rainfall = st.slider("Annual Rainfall (mm)", 400.0, 2500.0, 1200.0)
            fertilizer = st.slider("Fertilizer (kg/ha)", 50.0, 500.0, 200.0)
            pesticide = st.slider("Pesticide (kg/ha)", 0.0, 50.0, 15.0)

        input_data = pd.DataFrame([{
            "State": state,
            "Crop": crop,
            "Season": season,
            "Area": area,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
        }])

        st.markdown("---")

        if st.button("🔮 Predict Yield", use_container_width=True):
            prediction = st.session_state.pipeline.predict(model_choice, input_data)
            pred_value = prediction[0]

            st.success(f"Predicted Yield: **{pred_value:.2f} tons/ha**")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_value,
                title={"text": "Predicted Yield (tons/ha)"},
                gauge={
                    "axis": {"range": [0, max(df['Yield'].max(), pred_value * 1.5)]},
                    "bar": {"color": "#2E7D32"},
                    "steps": [
                        {"range": [0, 2], "color": "lightgray"},
                        {"range": [2, 5], "color": "yellow"},
                        {"range": [5, 10], "color": "lightgreen"},
                        {"range": [10, max(df['Yield'].max(), pred_value * 1.5)], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": df['Yield'].mean()
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 Input Details"):
                st.dataframe(input_data, use_container_width=True)


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🌾 Crop Yield Prediction ML App | "
    "Built with Streamlit, Scikit-Learn, XGBoost & Plotly</p>",
    unsafe_allow_html=True
)