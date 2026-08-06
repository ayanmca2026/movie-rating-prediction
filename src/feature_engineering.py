from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from src.data_preprocessing import MovieDataCleaner, GenreBinarizer

def build_preprocessor():
    """
    Builds and returns the full preprocessing pipeline.
    Includes data cleaning, numerical scaling/imputation, genre binarization,
    and high-cardinality target encoding.
    """
    # 1. Numerical Columns
    numeric_features = ['Year', 'Duration', 'Votes']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Genre Column (comma separated)
    genre_feature = ['Genre']
    genre_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('binarizer', GenreBinarizer())
    ])
    
    # 3. High-cardinality Categorical Columns
    categorical_features = ['Director', 'Actor 1', 'Actor 2', 'Actor 3']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', TargetEncoder(target_type='continuous', random_state=42))
    ])
    
    # Combine features via ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('genre', genre_transformer, genre_feature),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Create the complete pipeline including the initial data cleaner
    full_pipeline = Pipeline(steps=[
        ('cleaner', MovieDataCleaner()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline
