from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent

# Paths to various resources
IMAGES_DIR = BASE_DIR / 'images'
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = DATA_DIR / 'results'
SAVED_MODELS_DIR = DATA_DIR / 'saved_models'

# from google.colab import userdata
# OPENAI_API_KEY = userdata.get('OpenAIKey')
