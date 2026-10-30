"""Make sure a trained model exists before the API module is imported."""
from src.train import train

if not train.MODEL_PATH.exists():
    train.main()
