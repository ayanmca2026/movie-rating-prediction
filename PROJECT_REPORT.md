# Project Report: Movie Rating Prediction with Python and Machine Learning

**Course/Project:** Major Project / Data Science & Machine Learning Portfolio  
**Domain:** Predictive Analysis, Supervised Learning, Natural Language & Categorical Processing  

---

## 1. Introduction
In the contemporary entertainment industry, predicting a movie's critical reception and viewer popularity holds substantial commercial interest. IMDb (Internet Movie Database) ratings are a primary metric globally representing public and critical consensus. Predicting ratings using historical attributes allows producers, distributors, and streaming platforms to assess risk and plan marketing strategies. This project builds a supervised regression framework that learns relationships between features such as genre combinations, director profile, casting, release timeline, and duration to predict movie ratings.

## 2. Background
Historically, film performance was judged qualitatively. With the advent of web portals like IMDb, Rotten Tomatoes, and Letterboxd, quantitative audience feedback became accessible at scale. By compiling user reviews and votes, IMDb establishes a rating system. Machine Learning algorithms can uncover non-linear associations from this tabular metadata, mapping variables like director historical rating trends, year dynamics, and lead actors to the final rating.

## 3. Problem Statement
Predicting IMDb ratings is structured as a supervised regression task where the target variable is a continuous rating between 1.0 and 10.0. The challenges include:
- Processing raw text fields (e.g. Years written as parentheses, Votes containing commas).
- High cardinality in categorical variables (e.g., thousands of unique directors and actors).
- Safe handling of missing values without introducing data leakage.
- Modeling multi-genre tags (e.g., "Action, Comedy, Sci-Fi") which cannot be naively one-hot encoded without exploding dimensionality.

## 4. Objectives
The objectives of this project are:
1. Conduct programmatic data cleaning and preprocessing on Indian cinema data.
2. Develop a leak-free pipeline utilizing scikit-learn for numeric scaling, target encoding of directors/actors, and count binarization of multi-genres.
3. Compare five different regressor algorithms using 5-Fold Cross-Validation.
4. Perform grid search tuning on the best model to optimize prediction performance.
5. Create visual diagnostics (Residuals plots, Error distributions, Actual vs Predicted scatters).
6. Build a production-ready Streamlit application.

## 5. Dataset Description
The model is trained on the **IMDb Movies India** dataset (`IMDb Movies India.csv`). The schema consists of:
- **Name:** Movie title (text).
- **Year:** Year of release (text/parentheses).
- **Duration:** Runtime (text/minutes suffix).
- **Genre:** Comma-separated strings of genre categories.
- **Rating:** Continuous target variable (float).
- **Votes:** Total votes cast by users (text/commas).
- **Director:** Name of the movie director.
- **Actor 1, Actor 2, Actor 3:** Top three billed cast members.

## 6. Data Collection
The dataset represents historical Indian cinema metadata spanning over a century (1913 to 2022). It reflects structural variations in Indian movies (Bollywood, Tollywood, Kollywood, etc.) across genres and time periods.

## 7. Data Preprocessing
Supervised regression requires valid training targets. Out of 15,509 original rows:
1. **Target Filtering:** 7,590 rows missing `Rating` were dropped.
2. **De-duplication:** Duplicate entries were removed, resulting in a clean core set of **7,919 records**.
3. **Data Cleaning:**
   - **Year:** Stripped of parentheses and parsed to float.
   - **Duration:** Stripped of `"min"` suffix and parsed to float.
   - **Votes:** Stripped of commas and parsed to float.
4. **Pipeline Imputation:**
   - Missing runtimes (26.1% of `Duration`) were imputed inside the pipeline using the **median** value of training splits.
   - Missing values in `Genre`, `Director`, and `Actors` were imputed with a constant value `'Unknown'`.

## 8. Exploratory Data Analysis (EDA)
EDA was performed to evaluate target behavior and feature relationships:
- **Rating Distribution:** Follows a normal distribution centered at a mean of 5.84 with a standard deviation of 1.38.
- **Year Trends:** A massive rise in movie releases is observed starting from the 1990s.
- **Votes vs Rating:** A logarithmic relationship shows that movies with larger voting pools tend to cluster around slightly higher ratings (6.0 - 8.0).
- **Runtimes:** The majority of Indian movies span between 110 to 150 minutes, representing the signature theatrical run length.

## 9. Feature Engineering
A custom preprocessing pipeline was built using scikit-learn's `ColumnTransformer`:
- **Standard Scaling:** Scaled `Year`, `Duration`, and `Votes` using `StandardScaler` to bring numeric values to a mean of 0 and variance of 1.
- **Multi-Genre Binarization:** Custom `GenreBinarizer` splits genres by comma and fits a `CountVectorizer` to create binary columns for each genre.
- **Target Encoding:** `Director`, `Actor 1`, `Actor 2`, and `Actor 3` have thousands of categories. `TargetEncoder` maps each name to the mean target rating safely using internal cross-validation folds, preventing target leakage.

## 10. Machine Learning Methodology
To ensure model reproducibility and generalizability, the dataset was split into **80% training** and **20% testing** with `random_state=42`. All data-driven transformations (imputation, scaling, target encoding) were fitted strictly on the training folds inside 5-Fold Cross-Validation:
- **Number of Splits:** 5
- **Shuffle:** True
- **Random State:** 42

## 11. Algorithms Used
We implemented and compared five regression algorithms:
1. **Linear Regression:** Standard ordinary least squares regression.
2. **Ridge Regression:** L2 regularized linear model minimizing coefficients.
3. **Random Forest Regressor:** Bagging ensemble of decision trees.
4. **Gradient Boosting Regressor:** Boosting ensemble building sequential trees to minimize residuals.
5. **HistGradientBoostingRegressor:** Histogram-based gradient boosting optimized for large datasets and native missing value supports.

## 12. Training Process
The candidate models were evaluated on the training set using Cross-Validation:

- **Hist Gradient Boosting** showed the lowest cross-validation RMSE (1.026).
- **Hyperparameter Grid Search** was performed on `HistGradientBoostingRegressor` over the parameter space:
  - `learning_rate`: `[0.05, 0.1, 0.2]`
  - `max_iter`: `[100, 150]`
  - `max_depth`: `[5, 10, None]`
- **Best tuned configuration:** `learning_rate=0.1`, `max_depth=5`, `max_iter=100`.

## 13. Model Evaluation
The final evaluation metrics computed on the test set:

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| Hist Gradient Boosting (Tuned) | 0.78 | 1.03 | 0.429 |
| Hist Gradient Boosting | 0.78 | 1.03 | 0.433 |
| Gradient Boosting | 0.79 | 1.04 | 0.419 |
| Random Forest | 0.80 | 1.05 | 0.408 |
| Ridge Regression | 0.88 | 1.13 | 0.309 |
| Linear Regression | 0.88 | 1.13 | 0.309 |

- **MAE (Mean Absolute Error):** Measures average absolute deviation. The best model deviaties by 0.78 points from the actual rating.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors. The best model achieves a test RMSE of 1.03.
- **R² Score:** Explains 43% of the variance in movie ratings based on the metadata features.

## 14. Results & Interpretability
Permutation feature importance calculated on the test split shows:
- **Year** (Release Year) is the most critical feature, explaining the bulk of rating shifts.
- **Votes** is the second most important, confirming that public consensus size strongly correlates with ratings.
- **Genre** and **Director** contribute significantly to the model's predictive capability.
- **Duration** and actors play a smaller relative role in general predictions.

*Disclaimer: Feature importance shows model association, not causation.*

## 15. Best Model Summary
The **Hist Gradient Boosting Regressor** is the final model. The complete preprocessing and prediction logic were serialized together to `models/movie_rating_model.pkl`.

## 16. Application Development
A web application was built in Streamlit (`app/app.py`). It features:
- **Predict Tab:** Interactive form verifying parameters and rendering predicted ratings.
- **Interpretation Card:** Maps predicted ratings to qualitative buckets (Low: 0-4, Average: 4-6, Good: 6-8, Excellent: 8-10).
- **Diagnostics Tab:** Renders actual vs predicted and residual figures generated during model training.
- **Model Info Tab:** Renders the metrics comparison table.

## 17. Limitations
- **External Variables:** Features like marketing budgets, script quality, soundtrack success, and franchise value are not captured in the dataset but heavily influence audience rating.
- **Historical Bias:** Rating scales and voting behavior differ between historical eras (e.g. 1950s vs 2020s).

## 18. Future Scope
- Incorporate textual analysis of scripts or movie descriptions using Natural Language Processing (NLP) models.
- Collect financial figures (budgets, collections) to model the relationship between ratings and financial success.

## 19. Conclusion
This project successfully demonstrates the construction of a professional-grade regression system to predict IMDb ratings. The preprocessing pipeline effectively handles messy categorical and numeric structures without introducing target leakage. The Streamlit app provides an intuitive, robust, and responsive interface for serving predictions and examining model analytics.
