import os
import logging
from pathlib import Path
import joblib

def get_project_root() -> Path:
    """
    Returns the absolute path to the root of the movie-rating-prediction directory.
    """
    # This file is in src/utils.py, so parent is src/, grandparent is project root
    return Path(__file__).resolve().parent.parent

def setup_logging(name="movie_rating"):
    """
    Sets up custom logging configuration.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def save_pipeline(model, file_path):
    """
    Saves the trained machine learning pipeline using joblib.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Pipeline successfully saved to {file_path}")

def load_pipeline(file_path):
    """
    Loads and returns the serialized machine learning pipeline.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")
    return joblib.load(file_path)
