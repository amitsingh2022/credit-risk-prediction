from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    """
    Preprocessor for loan default dataset.
    - Scales numeric features.
    - SEX, EDUCATION_*, MARRIAGE_* are already numeric (0/1).
    """
    numeric_features = [
        "AGE", "SEX", "LIMIT_BAL", "PAY_0",
        "avg_utilization", "avg_payment_ratio",
        "EDUCATION_Graduate", "EDUCATION_HighSchool",
        "EDUCATION_Others", "EDUCATION_University",
        "MARRIAGE_Married", "MARRIAGE_Others", "MARRIAGE_Single"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="drop"
    )
    return preprocessor
