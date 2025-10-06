import joblib
import os

def save_model(model, path):
    """
    Save trained model (pipeline) to disk.
    
    Parameters
    ----------
    model : sklearn model or pipeline
        Trained model to save.
    path : str
        File path to save model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """
    Load model from disk.
    
    Parameters
    ----------
    path : str
        Path to model file.
        
    Returns
    -------
    model : sklearn model or pipeline
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found at {path}")
    return joblib.load(path)


