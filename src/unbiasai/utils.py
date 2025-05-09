# New package as per deprecation warning
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import langchain_openai
from langchain_deepseek import ChatDeepSeek
from langchain_cohere import ChatCohere
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
import re
from datetime import datetime


def generate_embeddings(text):
    """
    Generate embeddings for a given text using OpenAI's embedding model.

    Parameters:
    text (str): The input text for which embeddings need to be generated.

    Environment Variables:
    OPENAI_API_KEY (str): The API key required to authenticate with OpenAI's services.
                          This must be set in the environment variables.

    Returns:
    list[float]: A list of floating-point numbers representing the embedding vector for the input text.

    Raises:
    Exception: If the `OPENAI_API_KEY` environment variable is not set.

    Example:
    >>> import os
    >>> os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    >>> embeddings = generate_embeddings("Artificial Intelligence is transforming the world.")
    >>> print(embeddings)
    [0.123, 0.456, 0.789, ...]  # Example embedding vector
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception("OPENAI_API_KEY environment variable not set")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,
                                   model="text-embedding-3-large")

    return embeddings.embed_query(text)


def insert_documents(df: pd.DataFrame, client, table_name: str = "unbiasai_test"):
    """
    Insert documents from a pandas DataFrame into a database table.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the documents to be inserted.
                       Each row should have the following columns:
                       - 'id': Unique identifier for the document (int).
                       - 'content': The content of the document (str).
                       - 'metadata' (optional): Additional metadata for the document.
                       - 'embedding': The embedding vector for the document.
    client: The database client used to interact with the database.
    table_name (str): The name of the table where the documents will be inserted.
                      Defaults to "retrieval_Recency".

    Returns:
    None

    Example:
    >>> import pandas as pd
    >>> from some_database_client import Client
    >>> data = {
    ...     "id": [1, 2],
    ...     "content": ["Document 1 content", "Document 2 content"],
    ...     "metadata": [{"author": "Alice"}, {"author": "Bob"}],
    ...     "embedding": [[0.1, 0.2], [0.3, 0.4]]
    ... }
    >>> df = pd.DataFrame(data)
    >>> client = Client()
    >>> insert_documents(df, client, table_name="my_table")
    """

    for index, row in df.iterrows():
        print(f"Inserting document with ID: {int(row['id'])}")
        data = {
            "id": int(row["id"]),
            "content": row["content"],
            "metadata": row.get("metadata", None),
            "embedding": row["embedding"]
        }
        response = client.table(table_name).upsert(data).execute()
    return



def initialize_llm(model_name, api_key):
    # Initialize LLM
    model_name = model_name.lower()
    if model_name == "gpt":
        llm = ChatOpenAI(model_name="gpt-4o-2024-11-20",
                         openai_api_key=api_key)
    elif model_name == "claude":
        llm = ChatAnthropic(model="claude-3-7-sonnet-latest",
                            anthropic_api_key=api_key)
    elif model_name == "mistral":
        llm = ChatMistralAI(model="mistral-small-latest",
                            mistral_api_key=api_key)
    elif model_name == "cohere":
        llm = ChatCohere(model="command-a-03-2025",
                         cohere_api_key=api_key)
    elif model_name == "deepseek":
        llm = ChatDeepSeek(model="deepseek-chat")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f'LLM initialized correctly: {model_name}, llm: {llm}')
    return llm

def get_documents_from_supabase(query, supabase_client, function_name='match_documents_recency_no_filter', k=10):
    """Get document embeddings from Supabase."""
    try:
        query_embedding = generate_embeddings(query)
        response = supabase_client.rpc(
            function_name,
            {
                'query_embedding': query_embedding,
                'match_count': k
            }
        ).execute()
        
        if not response.data or len(response.data) == 0:
            print("No relevant documents found.")
            return []
        return response.data
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


def convert_to_doc_objects(raw_docs):
    """Convert raw document data to Doc objects."""
    return [
        type("Doc", (object,), {
            "id": doc.get("id"),
            "page_content": doc.get("content", ""),
            "metadata": doc.get("metadata", {})
        })()
        for doc in raw_docs
    ]


def create_reranking_prompt(query, docs, k):
    """Create prompt for LLM re-ranking."""
    actual_k = min(k, len(docs))
    documents_text = "\n\n".join([
        f"Document {i+1} (ID: {docs[i].id}):\n{docs[i].page_content}"
        for i in range(actual_k)
    ])
    
    # Using the exact original prompt
    return f"""
 Query: {query}
 You will be given {actual_k} documents retrieved via semantic search.
 Your task is to re-rank these documents in order of their relevance to the query.
 Please return EXACTLY {actual_k} document numbers in order, from MOST to LEAST relevant,
 separated by commas (e.g., "3,1,2").
 Documents:
{documents_text}
 """


def perform_llm_reranking(llm, prompt, docs, k):
    """Re-rank documents using LLM."""
    actual_k = min(k, len(docs))
    try:
        # Using the exact original system message
        messages = [
            SystemMessage(content="You are a helpful assistant skilled at ranking document relevance."),
            HumanMessage(content=prompt)
        ]
        
        llm_response = llm.invoke(messages)
        ranking_text = llm_response.content.strip()
        ranking_order = [int(num.strip()) - 1 for num in re.findall(r'\d+', ranking_text)]
        
        if len(ranking_order) != actual_k or sorted(ranking_order) != list(range(actual_k)):
            print(f"Invalid ranking received: {ranking_text}. Using default order.")
            return docs
            
        return [docs[i] for i in ranking_order]
    except Exception as e:
        print(f"Re-ranking failed: {e}. Using initial ranking.")
        return docs


def format_results(docs):
    """Format documents into final result structure."""
    return [
        {
            "id": doc.id,
            "rank": idx + 1,
            "content": doc.page_content
        }
        for idx, doc in enumerate(docs)
    ]


def retrieve(query, llm, supabase_client, function_name, k=10, re_rank=False):
    """
    Retrieve top-k documents for a query using Supabase vector search with optional LLM re-ranking.
    Parameters:
    query (str): The search query.
    model_name (str): One of 'gpt-4o-mini', 'claude', 'mistral', etc.
    k (int): Number of top documents to retrieve.
    re_rank (bool): Whether to re-rank the documents using the LLM.
    Returns:
    List[dict]: A list of dictionaries with document 'id', 'rank', and 'content'.
    """
    # Step 1: Get raw documents from Supabase
    raw_docs = get_documents_from_supabase(query, supabase_client, function_name, k)
    if not raw_docs:
        return []

    # Step 2: Convert to Doc objects
    docs = convert_to_doc_objects(raw_docs)

    # Step 3: Re-rank if requested
    if re_rank and len(docs) > 1:
        reranking_prompt = create_reranking_prompt(query, docs, k)
        docs = perform_llm_reranking(llm, reranking_prompt, docs, k)

    # Step 4: Format and return results
    return format_results(docs)


def extract_created_datetime(content, pattern=r'createdDateTime[":]*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)'):
    # Try the pattern
    match = re.search(pattern, content)
    if match:
        # Handle both with and without milliseconds
        datetime_str = match.group(1)
        if '.' in datetime_str:
            return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")
    return None

# length
def analyze_content_word_count(df):
    """
    Analyzes content column and counts words after "output.content: "

    Parameters:
    df (pandas.DataFrame): DataFrame with a 'content' column

    Returns:
    pandas.DataFrame: DataFrame with 'word_count' column added
    """
    # Create a copy of the dataframe
    result_df = df.copy()

    # Count words after marker
    def count_words(text):
        if not isinstance(text, str):
            return 0

        marker = "output.content: "
        pos = text.find(marker)

        if pos == -1:
            return len(text.split())  # Count all words if marker not found

        # Extract content after marker and count words
        content_after_marker = text[pos + len(marker):]
        return len(content_after_marker.split())

    # Apply word counting
    result_df['word_count'] = result_df['content'].apply(count_words)

    return result_df


def categorize_word_count(df):
    """
    Categorizes an existing word_count column as 'short', 'medium', or 'long'
    within each Model and Query group based on relative length.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'word_count', 'Model', and 'Query' columns

    Returns:
    pandas.DataFrame: DataFrame with 'length_category' column added
    """
    result_df = df.copy()

    # Define the length categories
    length_categories = ['short', 'medium', 'long']

    # Sort and assign length categories within each group
    result_df['length_category'] = (
        result_df.sort_values(by='word_count', ascending=True)
        .groupby(['Model', 'Query'])
        .cumcount()
        .apply(lambda x: length_categories[min(x, len(length_categories)-1)])
    )

    return result_df

# language
def detect_language(content):
    """
    Detects the language of a given text content.

    Args:
        content (str): The text to analyze.

    Returns:
        str: Detected language code (e.g., 'en', 'fr', 'de', 'nl'), or None if detection fails.
    """
    try:
        return detect(content)
    except Exception:
        return None