from pathlib import Path

import joblib
import numpy as np

from data.generator import STYLE_TO_INT, INT_TO_STYLE, STYLES

ARTIFACTS_DIR = Path(__file__).parent.parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'vark_rf.pkl'


class VARKPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f'Model not found at {MODEL_PATH}. Run train.py first.'
            )
        self._model = joblib.load(MODEL_PATH)

    def predict(self, answers: list[str]) -> dict:
        """
        answers: list of 8 style strings, e.g.
                 ['visual', 'auditory', 'visual', ...]
        Returns dominant_style, probabilities per class, and raw scores.
        """
        if len(answers) != 8:
            raise ValueError(f'Expected 8 answers, got {len(answers)}')

        encoded = np.array([[STYLE_TO_INT[a] for a in answers]])
        dominant_idx = int(self._model.predict(encoded)[0])
        proba = self._model.predict_proba(encoded)[0]

        # Map class indices to style names (RF stores classes in sorted order)
        class_indices = self._model.classes_
        probabilities = {
            INT_TO_STYLE[int(idx)]: round(float(p), 4)
            for idx, p in zip(class_indices, proba)
        }

        # Raw score counts (same as current rule-based, included for transparency)
        scores = {style: 0 for style in STYLES}
        for answer in answers:
            scores[answer] += 1

        return {
            'dominant_style': INT_TO_STYLE[dominant_idx],
            'probabilities': probabilities,
            'scores': scores,
        }
