import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

def split_genres(x):
    """
    Module-level function to split and clean genres.
    Must be module-level to support model serialization (pickling).
    """
    if pd.isna(x) or not isinstance(x, str):
        return []
    return [g.strip() for g in x.split(',')]


class MovieDataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean Year, Duration, and Votes columns.
    Parses them from string representations to numeric values.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Set fitted flag for scikit-learn check_is_fitted compatibility
        self.is_fitted_ = True
        return self

    def transform(self, X):
        # Create a copy to avoid SettingWithCopyWarning
        X = pd.DataFrame(X).copy()

        # Clean Year: extracts 4 digits, e.g. "(2019)" -> 2019.0
        if 'Year' in X.columns:
            # Convert to string, strip whitespace, then extract digits
            year_str = X['Year'].astype(str).str.strip()
            X['Year'] = year_str.str.extract(r'(\d{4})').astype(float)

        # Clean Duration: extracts numeric part, e.g. "109 min" -> 109.0
        if 'Duration' in X.columns:
            duration_str = X['Duration'].astype(str).str.strip()
            X['Duration'] = duration_str.str.extract(r'(\d+)').astype(float)

        # Clean Votes: removes commas and extracts digits, e.g. "1,086" -> 1086.0
        if 'Votes' in X.columns:
            votes_str = X['Votes'].astype(str).str.strip().str.replace(',', '', regex=False)
            X['Votes'] = votes_str.str.extract(r'(\d+)').astype(float)

        # Clean whitespace for categorical columns
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].astype(str).str.strip()

        return X


class GenreBinarizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Genre multi-hot encoding.
    Splits comma-separated genres and creates binary flags for each genre.
    """
    def __init__(self):
        # Initialize the vectorizer with module-level tokenizer
        self.vectorizer = CountVectorizer(
            tokenizer=split_genres,
            token_pattern=None,
            binary=True
        )

    def fit(self, X, y=None):
        # Flatten X to a 1D pandas Series of strings
        x_flat = pd.Series(np.array(X).ravel()).astype(str).fillna('Unknown')
        self.vectorizer.fit(x_flat)
        # Set fitted flag for scikit-learn check_is_fitted compatibility
        self.is_fitted_ = True
        return self

    def transform(self, X):
        x_flat = pd.Series(np.array(X).ravel()).astype(str).fillna('Unknown')
        # Return dense matrix for consistency and easy modeling
        return self.vectorizer.transform(x_flat).toarray()

    def get_feature_names_out(self, input_features=None):
        # Needed for pipeline feature tracking
        return [f"Genre_{name}" for name in self.vectorizer.get_feature_names_out()]

def clean_target_and_duplicates(df):
    """
    Performs initial top-level dataset cleaning:
    1. Removes rows where target 'Rating' is missing.
    2. Removes duplicate rows.
    Returns cleaned DataFrame.
    """
    df_cleaned = df.copy()
    
    # 1. Drop rows missing the target 'Rating'
    df_cleaned = df_cleaned.dropna(subset=['Rating'])
    
    # 2. Drop duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned
