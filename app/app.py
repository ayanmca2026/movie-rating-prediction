import os
import pandas as pd
import streamlit as st
from PIL import Image

# Add root folder to sys.path so we can import src modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_rating
from src.utils import get_project_root

# Page configuration
st.set_page_config(
    page_title="Movie Rating Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    .prediction-header {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        color: white;
    }
    .badge-low { background-color: #dc3545; }
    .badge-avg { background-color: #ffc107; color: black; }
    .badge-good { background-color: #28a745; }
    .badge-excellent { background-color: #007bff; }
    </style>
""", unsafe_allow_html=True)

# Project paths
project_root = get_project_root()
model_path = project_root / "models" / "movie_rating_model.pkl"
results_path = project_root / "reports" / "model_results.csv"
figures_dir = project_root / "reports" / "figures"

# Title header
st.title("🎬 Movie Rating Prediction")
st.markdown("""
Predict the IMDb rating of Indian movies based on historical movie characteristics. 
This application utilizes a trained machine learning pipeline that preprocesses raw inputs and performs predictions.
""")

# Sidebar info
st.sidebar.header("About Project")
st.sidebar.info("""
**Dataset:** IMDb Movies India
**Target Variable:** Rating (0 to 10)
**Task:** Supervised Regression
**Model Type:** Hist Gradient Boosting Regressor (Tuned)
""")

st.sidebar.header("Features Used")
st.sidebar.markdown("""
- **Year:** Year of release (Numerical)
- **Duration:** Movie runtime in minutes (Numerical)
- **Votes:** Total votes cast on IMDb (Numerical)
- **Genre:** Primary and secondary genres (Categorical)
- **Director:** Movie Director (High-cardinality Categorical)
- **Actors (1, 2, 3):** Lead casting list (High-cardinality Categorical)
""")

st.sidebar.warning("⚠️ **Note:** Prediction results represent model association based on historical training data, not guaranteed outcomes.")

# Create tabs for application features
tab_predict, tab_analytics, tab_model_info = st.tabs([
    "🔮 Predict Rating", 
    "📊 Dataset Analytics", 
    "⚙️ Model & Training Performance"
])

with tab_predict:
    st.subheader("Enter Movie Details")
    
    # Form for input
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            movie_name = st.text_input("Movie Name", value="Dilwale Dulhania Le Jayenge", help="The movie title (displayed only, not processed directly).")
            year = st.text_input("Year of Release", value="1995", help="Format: YYYY (e.g. 1995)")
            duration = st.text_input("Duration (min)", value="189", help="Format: minutes (e.g. 189)")
            
        with col2:
            votes = st.text_input("Number of Votes", value="120000", help="Number of cast votes (e.g. 120000)")
            genre = st.text_input("Genre", value="Romance, Drama", help="Comma-separated genres (e.g. Romance, Drama)")
            director = st.text_input("Director", value="Aditya Chopra")
            
        with col3:
            actor1 = st.text_input("Actor 1 (Lead)", value="Shah Rukh Khan")
            actor2 = st.text_input("Actor 2", value="Kajol")
            actor3 = st.text_input("Actor 3", value="Amrish Puri")
            
        submit_button = st.form_submit_button(label="🔮 Predict Rating")

    if submit_button:
        # Input Validation
        errors = []
        
        # Validate Year
        try:
            val_year = int(year.strip())
            if val_year < 1900 or val_year > 2030:
                errors.append("Year must be between 1900 and 2030.")
        except ValueError:
            errors.append("Year must be a valid integer.")
            
        # Validate Duration
        try:
            val_duration = float(duration.strip())
            if val_duration <= 0:
                errors.append("Duration must be a positive number.")
        except ValueError:
            errors.append("Duration must be a valid numeric value.")
            
        # Validate Votes
        try:
            val_votes = float(votes.strip().replace(',', ''))
            if val_votes < 0:
                errors.append("Votes cannot be negative.")
        except ValueError:
            errors.append("Votes must be a valid numeric value.")
            
        # Check string fields
        if not genre.strip():
            errors.append("Genre cannot be empty.")
        if not director.strip():
            errors.append("Director cannot be empty.")
        if not actor1.strip():
            errors.append("Lead Actor 1 cannot be empty.")
            
        # Model existence check
        if not os.path.exists(model_path):
            errors.append("Trained model pipeline file not found! Please run the training script first.")
            
        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                # Perform prediction
                with st.spinner("Calculating rating..."):
                    pred_rating = predict_rating(
                        year=year,
                        duration=duration,
                        genre=genre,
                        votes=votes,
                        director=director,
                        actor1=actor1,
                        actor2=actor2,
                        actor3=actor3,
                        movie_name=movie_name
                    )
                
                # Show results inside a container card
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"### Predicted IMDb Rating for **{movie_name}**")
                st.markdown(f"<span class='prediction-header'>{pred_rating:.2f} / 10</span>", unsafe_allow_html=True)
                
                # Add category interpretation badge
                if pred_rating < 4.0:
                    badge_html = "<span class='badge badge-low'>Low Rating (0-4)</span>"
                elif pred_rating < 6.0:
                    badge_html = "<span class='badge badge-avg'>Average Rating (4-6)</span>"
                elif pred_rating < 8.0:
                    badge_html = "<span class='badge badge-good'>Good Rating (6-8)</span>"
                else:
                    badge_html = "<span class='badge badge-excellent'>Excellent Rating (8-10)</span>"
                    
                st.markdown(f"**Interpretation:** {badge_html}", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab_analytics:
    st.subheader("Exploratory Data Analysis Figures")
    st.markdown("These figures illustrate key features and correlation patterns analyzed in the dataset.")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        feat_imp_path = figures_dir / "feature_importance.png"
        if os.path.exists(feat_imp_path):
            st.image(Image.open(feat_imp_path), caption="Permutation Feature Importance on Test Set (Shows features contributing to model R²).")
        else:
            st.info("Feature importance plot not found. Run model training to generate it.")
            
    with col_img2:
        model_comp_path = figures_dir / "model_comparison.png"
        if os.path.exists(model_comp_path):
            st.image(Image.open(model_comp_path), caption="Test RMSE Comparison across Candidate Models.")
        else:
            st.info("Model comparison plot not found. Run model training to generate it.")

with tab_model_info:
    st.subheader("Model Configuration & Training Metrics")
    
    # Show comparison table if exists
    if os.path.exists(results_path):
        st.markdown("### Performance Comparison Table (Test Set)")
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df.style.highlight_min(subset=['RMSE', 'MAE'], color='#d4edda').highlight_max(subset=['R2'], color='#d4edda'), use_container_width=True)
    else:
        st.info("Comparison metrics table not found. Run model training to generate it.")
        
    st.markdown("### Evaluation Charts")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        act_vs_pred = figures_dir / "actual_vs_predicted.png"
        if os.path.exists(act_vs_pred):
            st.image(Image.open(act_vs_pred), caption="Actual vs Predicted Rating scatter plot.")
            
    with col_c2:
        residuals_p = figures_dir / "residuals_plot.png"
        if os.path.exists(residuals_p):
            st.image(Image.open(residuals_p), caption="Residual plot (errors vs predicted values).")

    # Explain metrics
    st.markdown("""
    #### Understanding the Metrics:
    - **MAE (Mean Absolute Error):** Average of the absolute differences between actual ratings and predictions. E.g. a MAE of 0.78 means the model is off by about 0.78 rating points on average.
    - **RMSE (Root Mean Squared Error):** Quantifies prediction errors by squaring them (penalizing larger errors more heavily).
    - **R² Score (Coefficient of Determination):** Explains the proportion of variance in the target variable explained by input features. An R² of 0.43 means features explain 43% of rating variance.
    """)
