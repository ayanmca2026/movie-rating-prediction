import os
import pandas as pd
import numpy as np
from src.utils import get_project_root, load_pipeline

# Path to the serialized model pipeline
project_root = get_project_root()
MODEL_PATH = project_root / "models" / "movie_rating_model.pkl"

# Global cache for the loaded pipeline model
_pipeline = None

def _get_model():
    """
    Loads and caches the model pipeline.
    """
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run model training first: python src/train_model.py"
            )
        _pipeline = load_pipeline(MODEL_PATH)
    return _pipeline

def predict_rating(
    year,
    duration,
    genre,
    votes,
    director,
    actor1,
    actor2,
    actor3,
    movie_name="Unknown"
) -> float:
    """
    Loads the trained pipeline and predicts the IMDb movie rating.
    
    Parameters:
    -----------
    year : int or str
        Release year of the movie (e.g. 2024 or "(2024)")
    duration : int or str
        Duration in minutes (e.g. 120 or "120 min")
    genre : str
        Genre(s) of the movie (e.g. "Drama, Action")
    votes : int or str
        Number of votes (e.g. 10000 or "10,000")
    director : str
        Director of the movie
    actor1 : str
        Primary lead actor
    actor2 : str
        Secondary lead actor
    actor3 : str
        Tertiary lead actor
    movie_name : str, optional
        Name of the movie (for documentation, not used by the model itself)
        
    Returns:
    --------
    float
        Predicted IMDb rating rounded to 2 decimal places.
    """
    # Create a single row DataFrame matching training features
    input_data = pd.DataFrame([{
        'Name': movie_name,
        'Year': year,
        'Duration': duration,
        'Genre': genre,
        'Votes': votes,
        'Director': director,
        'Actor 1': actor1,
        'Actor 2': actor2,
        'Actor 3': actor3
    }])
    
    # Load pipeline
    pipeline = _get_model()
    
    # Predict rating
    prediction = pipeline.predict(input_data)[0]
    
    # Clamp predicted rating between 1.0 and 10.0 (IMDb range)
    prediction = max(1.0, min(10.0, prediction))
    
    return float(round(prediction, 2))

if __name__ == '__main__':
    # Simple prediction test run
    try:
        pred = predict_rating(
            year=2024,
            duration=120,
            genre="Drama",
            votes=10000,
            director="Example Director",
            actor1="Actor A",
            actor2="Actor B",
            actor3="Actor C",
            movie_name="Example Movie"
        )
        print(f"Sample Prediction: {pred} / 10")
    except Exception as e:
        print(f"Prediction failed: {e}")
