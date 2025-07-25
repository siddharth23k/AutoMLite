import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=200,      # number of rows
    n_features=6,       # number of features
    n_informative=4,    # number of informative features
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Create a DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
df['target'] = y

# Save to CSV
df.to_csv('your_data.csv', index=False)
print("Sample dataset saved as your_data.csv")