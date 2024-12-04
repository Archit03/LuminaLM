import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv # type: ignore
import os
import torch

# Load environment variables from api.env
load_dotenv("api.env")
# Initialize Pinecone client with API key
api_key = os.getenv("api_key")
pc = Pinecone(api_key=api_key, environment="us-east-1")

# Define index details
index_name = "luminalm-embeddings"

# Define function to save embeddings to Pinecone
def save_embeddings_to_pinecone(embeddings, batch_ids, index_name=index_name):
    # Connect to the existing Pinecone index
    index = pc.Index(index_name)
    
    # Prepare data to upload in batches
    data_to_upsert = [(str(batch_id), embedding.tolist()) for batch_id, embedding in zip(batch_ids, embeddings)]
    
    # Upsert data to Pinecone
    index.upsert(vectors=data_to_upsert)
    print(f"Upserted {len(data_to_upsert)} embeddings to Pinecone.")

def fetch_embeddings(self, input_ids):
        # Convert token IDs to string keys for Pinecone
        keys = [f"token_{token_id.item()}" for token_id in input_ids.flatten()]
        response = self.pinecone_index.fetch(ids=keys)
        embeddings = torch.tensor([response['vectors'][key] for key in keys], dtype=torch.float32)
        return embeddings.view(input_ids.size(0), input_ids.size(1), -1)