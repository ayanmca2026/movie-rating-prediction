import os
import pytest
from src.predict import predict_rating, _get_model
from src.utils import get_project_root

def test_model_loading():
    """
    Test that the trained model pipeline exists and loads correctly.
    """
    model_path = get_project_root() / "models" / "movie_rating_model.pkl"
    assert model_path.exists(), "Trained model pkl file does not exist. Run training first!"
    
    pipeline = _get_model()
    assert pipeline is not None, "Pipeline failed to load."
    assert hasattr(pipeline, "predict"), "Loaded pipeline does not have predict method."

def test_prediction_returns_numeric():
    """
    Test that the predict_rating function returns a float value.
    """
    pred = predict_rating(
        year=2024,
        duration=120,
        genre="Drama",
        votes=10000,
        director="Aditya Chopra",
        actor1="Shah Rukh Khan",
        actor2="Kajol",
        actor3="Amrish Puri"
    )
    assert isinstance(pred, float), f"Expected float, got {type(pred)}"

def test_prediction_within_bounds():
    """
    Test that predictions are clamped within a valid IMDb rating range (1.0 to 10.0).
    """
    # Sample 1
    pred1 = predict_rating(
        year=2020,
        duration=150,
        genre="Action",
        votes=500,
        director="Unknown Director",
        actor1="Unknown Actor",
        actor2="Unknown Actor",
        actor3="Unknown Actor"
    )
    assert 1.0 <= pred1 <= 10.0, f"Rating prediction {pred1} out of bounds"

    # Extreme sample (checking clamping mechanism)
    pred_extreme = predict_rating(
        year=1900,
        duration=1000,
        genre="InvalidGenre",
        votes=-100,  # Negative votes, parsed by cleaner
        director="Bad Director",
        actor1="Bad Actor 1",
        actor2="Bad Actor 2",
        actor3="Bad Actor 3"
    )
    assert 1.0 <= pred_extreme <= 10.0, f"Clamped prediction {pred_extreme} out of bounds"

def test_missing_features_imputation():
    """
    Test that missing or string representation features are parsed/imputed correctly.
    For example: Year "(2015)", Duration "130 min", and Votes with commas "5,000".
    """
    pred = predict_rating(
        year="(2015)",
        duration="130 min",
        genre="Action, Thriller",
        votes="5,000",
        director="Rohit Shetty",
        actor1="Ajay Devgn",
        actor2="Kareena Kapoor",
        actor3="Anupam Kher"
    )
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 10.0
