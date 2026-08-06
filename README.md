# Movie Rating Prediction with Python and Machine Learning

This project implements an end-to-end Machine Learning regression pipeline that predicts the IMDb rating of an Indian movie based on its historical features, including Year, Duration, Genre, Votes, Director, and Lead Actor casting details.

## 1. Project Overview
IMDb ratings are a standard measure of movie quality and audience reception, heavily influencing viewership and distribution choices. This project builds a supervised regression system utilizing scikit-learn to analyze the relationship between movie metadata characteristics and audience ratings, saving the trained pipeline for serving through an interactive Streamlit UI.

## 2. Problem Statement
Audience reception is highly multi-dimensional. Predicting a continuous score like an IMDb rating requires modeling interactions between temporal changes (release year), runtime characteristics, audience interest (votes count), and creative inputs (directors, genre types, and lead cast members).

## 3. Objective
Build a robust, end-to-end machine learning system that:
- Cleans and prepares messy Indian IMDb historical data.
- Handles missing feature values securely using pipeline imputations to prevent leakage.
- Compares multiple regression algorithms using 5-fold cross-validation.
- Serializes the winning pipeline to serve predictions in a Streamlit web app.

## 4. Dataset
The project is built on the **IMDb Movies India** dataset (`IMDb Movies India.csv`).
- **Total Ingested Records:** 15,509
- **Usable Training Records:** 7,919 (after dropping records missing the target rating)

## 5. Dataset Features
- **Name:** Title of the movie (Metadata, dropped by model).
- **Year:** Release year of the movie (e.g. `(2019)` -> cleaned to numeric `2019`).
- **Duration:** Runtime length (e.g. `109 min` -> cleaned to numeric `109`).
- **Genre:** Comma-separated categories (e.g. `Drama, Romance`).
- **Votes:** Total votes cast on IMDb (e.g. `1,086` -> cleaned to numeric `1086`).
- **Director:** Creative director name.
- **Actor 1, 2, 3:** Lead actors in the casting hierarchy.
- **Rating (Target):** Continuous value from 1.0 to 10.0 representing the IMDb rating.

## 6. Technologies Used
- **Python 3.13**
- **Pandas** & **NumPy** for data structures and vector operations.
- **Scikit-Learn** for modeling, preprocessing pipeline, and cross-validation.
- **Matplotlib** & **Seaborn** for exploratory plots and model evaluation curves.
- **Joblib** for serialization of trained pipelines.
- **Streamlit** for the interactive user interface.
- **Pytest** for unit testing.

## 7. Project Architecture
```
movie-rating-prediction/
│
├── data/
│   └── IMDb Movies India.csv
│
├── notebooks/
│   └── movie_rating_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── movie_rating_model.pkl
│
├── reports/
│   ├── figures/
│   │   ├── actual_vs_predicted.png
│   │   ├── feature_importance.png
│   │   ├── model_comparison.png
│   │   ├── prediction_error_distribution.png
│   │   └── residuals_plot.png
│   └── model_results.csv
│
├── app/
│   └── app.py
│
├── tests/
│   └── test_prediction.py
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── PROJECT_REPORT.md
```

## 8. Data Preprocessing
- **Target `Rating` Check:** Dropped rows where `Rating` is missing because supervised learning requires target labels.
- **Numeric Parsing:**
  - `Year` extracted using regex `(\d{4})` to handle formatting like `(2019)`.
  - `Duration` extracted digits to clean formats like `"109 min"`.
  - `Votes` removed commas and parsed to float values.
- **Imputation Strategy:**
  - Numeric columns like `Duration` are imputed using the **median** value of the training split inside the pipeline.
  - Categorical columns (`Genre`, `Director`, and `Actors`) are imputed with a constant value `'Unknown'`.

## 9. Exploratory Data Analysis (EDA)
EDA graphs are saved to `reports/figures/` during training:
- **Distribution of Ratings:** Reveals a bell-shaped distribution centered around 5.8.
- **Rating vs Votes:** Shows higher density and slightly higher scores for movies with high vote counts.
- **Rating vs Duration:** Displays the spread of movie runtimes and ratings.

## 10. Feature Engineering
To prevent data leakage, all transformations are fitted solely on training splits:
- **Multi-Genre Binarization:** Custom `GenreBinarizer` wraps a `CountVectorizer` splitting comma-separated text strings (e.g. `"Drama, Musical"`) into binary flags.
- **High-Cardinality Categorical Columns:** `Director`, `Actor 1`, `Actor 2`, and `Actor 3` have thousands of categories. Naive one-hot encoding is avoided. Instead, `scikit-learn`'s `TargetEncoder` is used to map categories to continuous values representing target rating averages safely inside the cross-validation loops.
- **Numeric Scaling:** Numerical columns are scaled using `StandardScaler`.

## 11. Machine Learning Models
This is a regression task. We compared:
1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 Regularized)
3. **Random Forest Regressor** (Bagging)
4. **Gradient Boosting Regressor** (Boosting)
5. **HistGradientBoostingRegressor** (Histogram-based Boosting, optimized for large datasets and missing values)

## 12. Model Evaluation
Model comparison metrics on the Test set (20%):

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| Hist Gradient Boosting (Tuned) | 0.78 | 1.03 | 0.429 |
| Hist Gradient Boosting | 0.78 | 1.03 | 0.433 |
| Gradient Boosting | 0.79 | 1.04 | 0.419 |
| Random Forest | 0.80 | 1.05 | 0.408 |
| Ridge Regression | 0.88 | 1.13 | 0.309 |
| Linear Regression | 0.88 | 1.13 | 0.309 |

## 13. Best Model
The **Hist Gradient Boosting Regressor (Tuned)** was selected as the final model due to its high generalization score, robust handling of category encoding, and minimal test RMSE.
- **Final Test RMSE:** 1.03
- **Final Test MAE:** 0.78
- **Final Test R² Score:** 0.43

*Note: Permutation importance shows that the **Year of Release** and **Number of Votes** have the strongest relationships with predicted rating, followed by **Genre** and **Director**.*

## 14. How to Run

### Step 1: Create a Virtual Environment
```bash
python -m venv venv
```

### Step 2: Activate the Virtual Environment
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Preprocessing & Training
```bash
python -m src.train_model
```
This script runs cross-validation, hyperparameter grid search, plots figures in `reports/figures/`, and serializes the model to `models/movie_rating_model.pkl`.

### Step 5: Run tests
```bash
pytest
```
This runs unit tests in `tests/test_prediction.py` checking predictions bounds and missing input formatting.

## 15. Streamlit Application
Start the Streamlit user interface:
```bash
streamlit run app/app.py
```

## 16. Example Prediction
- **Year:** 2024
- **Duration:** 120
- **Genre:** Drama
- **Votes:** 10000
- **Director:** Aditya Chopra
- **Actor 1:** Shah Rukh Khan
- **Actor 2:** Kajol
- **Actor 3:** Amrish Puri
- **Output:** Predicted IMDb Rating: `7.10 / 10`

## 17. Project Screenshots
*(Screenshot files can be saved in reports/figures/ during execution)*

## 18. Results
The model is able to predict movie ratings with an average deviation of 0.78 points. High-vote movies release trends and genre indicators correlate strongly with predictions.

## 19. Limitations
- **Data Completeness:** The dataset relies on cast billing ordering (Actor 1, 2, 3), which might be incomplete for older movies.
- **Feature Sparsity:** New actors or directors are mapped to the overall mean because they don't have historical entries (handle_unknown).

## 20. Future Improvements
- Integrate budget and box office collection statistics to enrich the model features.
- Explore textual metadata such as movie summaries using embeddings.

## 21. Author
Senior Machine Learning Engineer & Data Scientist

## 22. License
Distributed under the MIT License. See `LICENSE` for more information.
