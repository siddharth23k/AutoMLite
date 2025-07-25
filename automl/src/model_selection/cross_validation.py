from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y, scoring='accuracy', cv=3):
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    return scores.mean(), scores.std()