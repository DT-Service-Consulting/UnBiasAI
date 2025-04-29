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

## Usage

```
from unbiasai.utils import get_embedding
query = "What are the cafeteria plan benefits?",
query_embedding = get_embedding(query)
print(query_embedding)
```
