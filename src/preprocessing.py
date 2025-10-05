import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------
# Function: Get Preprocessor (scaler + one-hot)
# ---------------------------------------------------
def get_preprocessor(num_features, cat_features, scaler_type="standard"):
    """
    Builds a preprocessing pipeline:
    - Scales numeric features
    - OneHotEncodes categorical features
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    encoder = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, num_features),
            ("cat", encoder, cat_features)
        ]
    )
    return preprocessor


# ---------------------------------------------------
# Function: Preprocess Data (X, y, pipeline)
# ---------------------------------------------------
def preprocess_data(df, target_col, scaler_type="standard"):
    """
    Splits dataframe into X, y and returns preprocessor pipeline.
    - df: input dataframe
    - target_col: target variable
    - scaler_type: 'standard' or 'minmax'
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Define categorical + numeric features
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]  # raw categorical
    num_features = [col for col in X.columns if col not in cat_features]

    preprocessor = get_preprocessor(num_features, cat_features, scaler_type)
    return X, y, preprocessor


# ---------------------------------------------------
# Function: Save Processed Data
# ---------------------------------------------------
def save_processed_data(df, filename):
    """
    Save processed dataframe to data/processed/
    """
    path = f"data/processed/{filename}"
    df.to_csv(path, index=False)
    print(f"✅ Saved processed data: {path}")
