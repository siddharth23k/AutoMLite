import optuna

def optimize_model(model_class, param_space, X, y, scoring='accuracy', n_trials=10):
    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) if isinstance(v, list) else trial.suggest_float(k, *v)
                  for k, v in param_space.items()}
        model = model_class(**params)
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(model, X, y, scoring=scoring, cv=3).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params