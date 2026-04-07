# Example

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

X = pd.DataFrame([
    [1, 2, 3, 4],
])

y = pd.DataFrame([
    [5, 6, 7, 8]
])

model = RandomForestClassifier()

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Mean Score:", scores.mean())