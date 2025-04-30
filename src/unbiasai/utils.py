# New package as per deprecation warning
from langchain_openai import OpenAIEmbeddings

# V: from from Retrieval Bias-Recency
def generate_embedding(text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-large")  # Use the appropriate model
    response = embeddings.embed_query(text)  # Correctly call the method to generate embeddings
    embedding = response
    return embedding

def get_embedding(text, api_key):
    """Get embeddings for a text using OpenAI's embedding model"""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                 model="text-embedding-3-small")
    response = embeddings.embed_query(text)
    embedding = response
    return embedding

# V: from from Retrieval Bias-Recency
def retrieve(query, model_name, k=10, re_rank=False):
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

    # Initialize LLM
    model_name = model_name.lower()
    if model_name == "gpt":
        llm = ChatOpenAI(model_name="gpt-4o-2024-11-20", openai_api_key=OPENAI_API_KEY)
    elif model_name == "claude":
        llm = ChatAnthropic(model="claude-3-7-sonnet-latest", anthropic_api_key=CLAUDE_API_KEY)
    elif model_name == "mistral":
        llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY)
    elif model_name == "cohere":
        llm = ChatCohere(model="command-a-03-2025", cohere_api_key=COHERE_API_KEY)
    elif model_name == "deepseek":
        import os
        os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
        llm = ChatDeepSeek(model="deepseek-v3-chat")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        # Step 1: Get embedding
        query_embedding = get_embedding(query)

        # Step 2: Retrieve documents from Supabase via RPC
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

        # Step 3: Build document list
        docs = [
            type("Doc", (object,), {
                "id": doc.get("id"),
                "page_content": doc.get("content", ""),
                "metadata": doc.get("metadata", {})
            })()
            for doc in response.data
        ]

        actual_k = min(k, len(docs))

        # Step 4: Optional re-ranking
        if re_rank and actual_k > 1:
            try:
                documents_text = "\n\n".join([
                    f"Document {i+1} (ID: {docs[i].id}):\n{docs[i].page_content}"
                    for i in range(actual_k)
                ])

                prompt = f"""
                Query: {query}

                You will be given {actual_k} documents retrieved via semantic search.
                Your task is to re-rank these documents in order of their relevance to the query.
                Please return EXACTLY {actual_k} document numbers in order, from MOST to LEAST relevant,
                separated by commas (e.g., "3,1,2").

                Documents:
                {documents_text}
                """

                messages = [
                    SystemMessage(content="You are a helpful assistant skilled at ranking document relevance."),
                    HumanMessage(content=prompt)
                ]

                llm_response = llm.invoke(messages)
                ranking_text = llm_response.content.strip()
                ranking_order = [int(num.strip()) - 1 for num in re.findall(r'\d+', ranking_text)]

                if len(ranking_order) != actual_k or sorted(ranking_order) != list(range(actual_k)):
                    print(f"Invalid ranking received: {ranking_text}. Using default order.")
                    ranking_order = list(range(actual_k))

                docs = [docs[i] for i in ranking_order]

            except Exception as e:
                print(f"Re-ranking failed: {e}. Using initial ranking.")

        # Step 5: Return formatted result
        results = [
            {
                "id": doc.id,
                "rank": idx + 1,
                "content": doc.page_content
            }
            for idx, doc in enumerate(docs)
        ]
        return results

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
