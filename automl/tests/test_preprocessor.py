import pandas as pd
from src.preprocessing.preprocessor import DataPreprocessor

def test_preprocessor():
    df = pd.DataFrame({'A': [1, 2, None], 'B': ['x', 'y', 'z']})
    pre = DataPreprocessor()
    X = pre.fit_transform(df)
    assert X.shape[0] == 3