from langchain_openai import OpenAIEmbeddings  # New package as per deprecation warning


def get_embedding(text):
    """Get embeddings for a text using OpenAI's embedding model"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                  model="text-embedding-3-small")  # Use the appropriate model
    response = embeddings.embed_query(text)  # Correctly call the method to generate embeddings
    embedding = response
    return embedding
