import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
# ignoring the harmless warning given by the sklearn

class PowerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        print(f"DEBUG: PowerFeatures(degree={self.degree}).fit called")
        # Mark as fitted
        self.fitted_ = True
        self.n_features_in_ = np.array(X).shape[1]
        return self

    def transform(self, X):
        print(f"DEBUG: PowerFeatures(degree={self.degree}).transform called")
        X = np.asarray(X)
        feats = [X]
        for p in range(2, self.degree+1):
            feats.append(X**p)
        return np.concatenate(feats, axis=1)

def build_models(numeric_features, categorical_features):

    cat = OneHotEncoder(handle_unknown="ignore")
    num_linear = Pipeline([("scaler", StandardScaler())])

    preprocess_linear = ColumnTransformer([
        ("num", num_linear, numeric_features),
        ("cat", cat, categorical_features)
    ], n_jobs=1)

    models = {}
    # Linear Model
    models["linear"] = Pipeline([
        ("preprocess", preprocess_linear),
        ("regressor", LinearRegression())
    ])

    # Polynomial Models without interaction terms
    for d in [2, 3, 4]:
        num_poly = Pipeline([
            ("scaler", StandardScaler()),
            ("powers", PowerFeatures(degree=d))
        ])

        preprocess_poly = ColumnTransformer([
            ("num", num_poly, numeric_features),
            ("cat", cat, categorical_features)
        ], n_jobs=1)

        models[f"poly_deg_{d}_no_inter"] = Pipeline([
            ("preprocess", preprocess_poly),
            ("regressor", LinearRegression())
        ])

    # Quadratic Model with interaction terms
    num_quad = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])

    preprocess_quad = ColumnTransformer([
        ("num", num_quad, numeric_features),
        ("cat", cat, categorical_features)
    ], n_jobs=1)

    models["quadratic_with_interactions"] = Pipeline([
        ("preprocess", preprocess_quad),
        ("regressor", LinearRegression())
    ])

    return models
