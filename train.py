"""Run this once to train the model before starting the API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.trainer import train_and_evaluate

if __name__ == '__main__':
    train_and_evaluate()
    print('\nTraining complete. You can now start the API with:')
    print('  uvicorn main:app --host 0.0.0.0 --port 8001 --reload')
