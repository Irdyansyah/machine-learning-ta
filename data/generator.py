import numpy as np
import pandas as pd

STYLES = ['visual', 'auditory', 'read_write', 'kinesthetic']
STYLE_TO_INT = {s: i for i, s in enumerate(STYLES)}
INT_TO_STYLE = {i: s for i, s in enumerate(STYLES)}
N_QUESTIONS = 8


def generate(n_per_class: int = 800, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic VARK questionnaire data.

    Each sample = 8 answers, encoded as int (0=visual,1=auditory,2=read_write,3=kinesthetic).
    A learner with dominant style X answers mostly X, with realistic noise.
    """
    rng = np.random.default_rng(random_state)
    X, y = [], []

    for style_idx, style in enumerate(STYLES):
        for _ in range(n_per_class):
            # Dominant answers: 4-8 out of 8 questions match dominant style
            n_dominant = rng.integers(4, N_QUESTIONS + 1)
            dominant_positions = rng.choice(N_QUESTIONS, size=n_dominant, replace=False)

            answer = rng.integers(0, len(STYLES), size=N_QUESTIONS)
            answer[dominant_positions] = style_idx

            X.append(answer)
            y.append(style_idx)

    return np.array(X), np.array(y)


def to_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=[f'q{i+1}' for i in range(N_QUESTIONS)])
    df['label'] = [INT_TO_STYLE[i] for i in y]
    return df
