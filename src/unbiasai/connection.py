import supabase
from dotenv import load_dotenv
import os
import langchain_openai

from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import SupabaseVectorStore
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Shreya's URL
# SUPABASE_URL = "https://rrjbrtbsvdoxndchvchq.supabase.co"
# Vincent's URL
SUPABASE_URL = "https://wuxtoyrimqwohizxcmzf.supabase.co"

def create_supabase_client():
    # Create a Supabase client using the URL and key
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    # Initialize your embeddings using the new import:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Define the table name where your documents will be stored.
    table_name = "unbiasai_test" # new table name
    # Create the Supabase vector store by providing the client, embeddings, and table_name.
    vector_store = SupabaseVectorStore(client=supabase_client, embedding=embeddings, table_name=table_name)

    return supabase_client, embeddings, vector_store
