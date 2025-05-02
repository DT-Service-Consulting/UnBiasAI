from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent.parent

# Paths to various resources
DATA_DIR = BASE_DIR / 'data'

# from google.colab import userdata
# OPENAI_API_KEY = userdata.get('OpenAIKey')
