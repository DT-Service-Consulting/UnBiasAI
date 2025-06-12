# UnBiasAI

UnBiasAI is a Python package developed by [BRAIN](https://brain.dtsc.be/) (the [DTSC](https://www.dtsc.be) AI-research lab). 
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
- Supabase setup

## Setting up Supabase

 **Step 1: Create a Supabase Project**

1. Go to [https://app.supabase.com](https://app.supabase.com).
2. Click **New Project**.
3. Fill in:
   - **Project Name**
   - **Database Password**
   - **Region**
4. Click **Create Project**.
5. Once your project is ready, go to **Settings > API** and copy:
   - `Project URL`
   - `anon` or `service_role` API Key
  
** Step 2: Enable pgvector and Create Table**

1. In your Supabase project, go to **SQL Editor**.
2. Paste and run the following SQL:

```sql
create extension if not exists vector;

create table documents (
  id uuid default gen_random_uuid() primary key,
  content text,
  embedding vector(1536)
);
```
create index on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);


**Step 3: Add a Retrieval Function**
```
create or replace function match_documents(
  query_embedding vector(1536),
  match_count int
)
returns table (
  id uuid,
  content text,
  similarity float
)
language sql
as $$
  select
    id,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

**Step 4:  Set Up Google Colab**
```
!pip install supabase openai tiktoken numpy --quiet
```
**Step 5:  Connect to Supabase and OpenAI**
```
from google.colab import userdata
from supabase import create_client
import openai
import numpy as np

SUPABASE_URL = userdata.get("SUPABASE_URL")
SUPABASE_KEY = userdata.get("SUPABASE_KEY")
OPENAI_KEY = userdata.get("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_KEY
```
** Step 6: Define Functions to Embed and Insert Data**
```
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def insert_document(text):
    embedding = get_embedding(text)
    supabase.table("documents").insert({
        "content": text,
        "embedding": embedding
    }).execute()
```
** Step 7: Define Retrieval Function
```
def search_documents(query, top_k=5):
    query_embedding = get_embedding(query)
    response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()
    return response.data
```

** Step 8: Example Usage**
```
# Insert a document
insert_document("Supabase is an open-source Firebase alternative.")

# Search for similar documents
results = search_documents("What is Supabase?", top_k=3)

# Display results
for doc in results:
    print(f"Score: {doc['similarity']:.4f}")
    print(f"Content: {doc['content']}\n")
```




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

2. Connect to the supabase database

``` python
from dotenv import load_dotenv
from supabase import create_client, Client
from unbiasai.config import ENVFILE
load_dotenv(ENVFILE)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
```

3. Load and initialize all LLMs

```python
from unbiasai.utils import initialize_llm

initialized_models = {}

for model_name in ["gpt", "mistral", "cohere", "deepseek"]:
    initialized_models[model_name] = initialize_llm(model_name)
```

4. Notebooks 
    - The `notebooks` directory contains Jupyter notebooks that demonstrate how to use the package and analyze bias in AI-generated text.
    - You can run these notebooks in a Jupyter environment or convert them to Python scripts.


## Authors & Credits

- [Brain by DTSC](https://brain.dtsc.be/)
- [Shreya Bhattacharya](https://www.linkedin.com/in/dr-shreyab/)
- [Vincent Hagenow](https://www.linkedin.com/in/vincent-hagenow-6621082b7/)
- [Marco Di Gennaro](https://www.linkedin.com/in/marcodig/)

## License

This project is licensed under the Apache License Version 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DT Services and Consulting](https://www.dtsc.be/)
