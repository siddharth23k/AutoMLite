from sklearn.preprocessing import PolynomialFeatures
from feature_engine.selection import DropConstantFeatures

class FeatureEngineer:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        # Dropping constant features
        X = DropConstantFeatures().fit_transform(X)
        # Adding polynomial features
        poly = PolynomialFeatures(self.degree, include_bias=False)
        X_poly = poly.fit_transform(X.select_dtypes(include=['int64', 'float64']))
        return X_poly