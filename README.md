# UnBiasAI

UnBiasAI is a Python package developed by BRAIN-by DTSC. 
It is designed to help users identify and mitigate bias in AI-generated text. 
It provides tools for analyzing text, detecting bias, and generating unbiased responses.

## Features
UnBiasAI analyzes five possible biases among five leading Large Language Models (LLMs)—GPT-4, Claude, Cohere, Mistral, and DeepSeek.
Leveraging a controlled Retrieval-Augmented Generation (RAG) pipeline and over 250 real-world queries grounded in corporate documentation, 
we examined four critical bias types: 

- **retrieval bias**,
- **reinforcement drift**, 
- **language bias**, 
- **hallucination**


## Pre-requisites

- Python 3.8 or higher
- Poetry for dependency management
- OpenAI API and other LLMs API access key 

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
   ```bash
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
# Load local variables from .env
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

2. Notebooks 
    - The `notebooks` directory contains Jupyter notebooks that demonstrate how to use the package and analyze bias in AI-generated text.
    - You can run these notebooks in a Jupyter environment or convert them to Python scripts.


## Authors & Credits

- [Brain by DTSC](https://www.brain.dtsc.be)
- [Shreya Bhattacharya](https://www.linkedin.com/in/dr-shreyab/)
- [Vincent Hagenow](https://www.linkedin.com/in/vincent-hagenow-6621082b7/)
- [Marco Di Gennaro](https://www.linkedin.com/in/marcodig/)

## License

This project is licensed under the Apache License Version 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DT Service Consulting](https://www.dtsc.be/)
