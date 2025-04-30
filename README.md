# UnBiasAI

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:DT-Service-Consulting/UnBiasAI.git
   cd UnBiasAI
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate  # For Unix-based systems
   .venv\Scripts\activate  # For Windows
   ```

4. pip install extra needs
```
pip install -r requirements.txt
```

5. Set your API key for OpenAI:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   
   #or 
   cat OPENAI_API_KEY="your_openai_api_key" > .env
   ```
   
## Usage

1. Example usage of the `get_embedding` function:

```python
from dotenv import load_dotenv
import os
from unbiasai.utils import get_embedding

# Load the variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

query = "What are the cafeteria plan benefits?"
query_embedding = get_embedding(query, OPENAI_API_KEY)

print(query_embedding)

from unbiasai.language import detect_language

print(detect_language)
from dtsc_queries.language import test_queries

print(test_queries)
```
