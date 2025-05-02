# New package as per deprecation warning
from langchain_openai import OpenAIEmbeddings

def generate_embedding(text, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                  model="text-embedding-3-large")  # Use the appropriate model
    # Correctly call the method to generate embeddings
    response = embeddings.embed_query(text)
    embedding = response
    return embedding

def insert_documents(df: pd.DataFrame):
    for index, row in df.iterrows():
        print(f"Inserting document with ID: {int(row['id'])}")
        data = {
            "id": int(row["id"]),
            "content": row["content"],
            "metadata": row.get("metadata", None),
            "embedding": row["embedding"]
        }
        response = supabase_client.table("retrieval_Recency").insert(data).execute()

def get_embedding(text, api_key):
    """Get embeddings for a text using OpenAI's embedding model"""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                  model="text-embedding-3-small")
    response = embeddings.embed_query(text)
    embedding = response
    return embedding

# V: from Retrieval Bias-Recency

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
        llm = ChatMistralAI(model="mistral-large-latest",
                            mistral_api_key=api_key)
    elif model_name == "cohere":
        llm = ChatCohere(model="command-a-03-2025",
                         cohere_api_key=api_key)
    elif model_name == "deepseek":
        import os
        os.environ["DEEPSEEK_API_KEY"] = api_key
        llm = ChatDeepSeek(model="deepseek-v3-chat")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return llm

# VH: replaces retrieve function
def get_documents_from_supabase(query, k=10):
    """Get document embeddings from Supabase."""
    try:
        query_embedding = get_embedding(query)
        response = supabase.rpc(
            'match_documents_recency_no_filter',
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


def retrieve(query, llm, k=10, re_rank=False):
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
    raw_docs = get_documents_from_supabase(query, k)
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