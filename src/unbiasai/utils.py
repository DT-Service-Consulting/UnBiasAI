# New package as per deprecation warning
from langchain_openai import OpenAIEmbeddings


def get_embedding(text):
    """Get embeddings for a text using OpenAI's embedding model"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                  model="text-embedding-3-small")  # Use the appropriate model
    # Correctly call the method to generate embeddings
    response = embeddings.embed_query(text)
    embedding = response
    return embedding
