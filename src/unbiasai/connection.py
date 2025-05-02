import supabase
from

SUPABASE_URL = "https://rrjbrtbsvdoxndchvchq.supabase.co"
SUPABASE_KEY = supabase_key
# OPENAI_API_KEY = openai_key # VH delete?

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
# Initialize your embeddings using the new import:
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# Define the table name where your documents will be stored.
table_name = "retrieval_recency"
# Create the Supabase vector store by providing the client, embeddings, and table_name.
vector_store = SupabaseVectorStore(client=supabase_client, embedding=embeddings, table_name=table_name)