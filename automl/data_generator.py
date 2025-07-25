import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200,      # number of rows
    n_features=6,       # number of features
    n_informative=4,    # number of informative features
    n_redundant=0,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
df['target'] = y

df.to_csv('your_data.csv', index=False)
