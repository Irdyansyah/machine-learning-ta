import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier

from data.generator import STYLES, generate

ARTIFACTS_DIR = Path(__file__).parent.parent / 'artifacts'


def train_and_evaluate() -> dict:
    print('Generating synthetic data...')
    X, y = generate(n_per_class=800)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)} samples | Test: {len(X_test)} samples\n')

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8,
            random_state=42,
        ),
        'Naive Bayes': CategoricalNB(
            # min_categories ensures all categories seen even if missing in train split
            min_categories=4,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        results[name] = {
            'accuracy': round(acc, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4),
            'report': classification_report(y_test, y_pred, target_names=STYLES),
        }

        print(f'--- {name} ---')
        print(f'Test accuracy : {acc:.4f}')
        print(f'CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
        print(results[name]['report'])

    # Save Random Forest (primary model)
    rf_model = models['Random Forest']
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    joblib.dump(rf_model, ARTIFACTS_DIR / 'vark_rf.pkl')
    print(f'Model saved → {ARTIFACTS_DIR / "vark_rf.pkl"}')

    # Feature importance
    importances = rf_model.feature_importances_
    fi = {f'q{i+1}': round(float(v), 4) for i, v in enumerate(importances)}
    print('\nFeature importance (per question):')
    for q, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f'  {q}: {imp}')

    # Persist metrics
    metrics = {
        name: {k: v for k, v in r.items() if k != 'report'}
        for name, r in results.items()
    }
    metrics['feature_importance'] = fi
    with open(ARTIFACTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return results
