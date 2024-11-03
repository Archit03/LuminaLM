import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables from api.env
load_dotenv("api.env")

# Initialize Pinecone client with API key
api_key = os.getenv("api_key")
pc = Pinecone(api_key=api_key, environment="us-east-1")

# Define index details
index_name = "luminalm-embeddings"

# Create the Pinecone index (only do this once)
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=512,  # Replace with your model's embedding dimension
        metric="cosine",  # Common metric for similarity search
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Define function to save embeddings to Pinecone
def save_embeddings_to_pinecone(embeddings, batch_ids, index_name=index_name):
    # Connect to the existing Pinecone index
    index = pc.Index(index_name)
    
    # Prepare data to upload in batches
    data_to_upsert = [(str(batch_id), embedding.tolist()) for batch_id, embedding in zip(batch_ids, embeddings)]
    
    # Upsert data to Pinecone
    index.upsert(vectors=data_to_upsert)
    print(f"Upserted {len(data_to_upsert)} embeddings to Pinecone.")

