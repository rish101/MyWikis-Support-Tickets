import pandas as pd
import time
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import AutoTokenizer, AutoModel

# Replace these with your actual Pinecone API key and host link
pinecone_api_key = 'pcsk_DbBMc_2YvzpLaQhMpkUd2nfu84LQK3FUfKMDbRFhDP7ZJKmxnQKh6Q15p1Dzzj4LgsmZP'
pinecone_host = 'https://support-tickets-dwsy0x6.svc.aped-4627-b74a.pinecone.io'
    
# Step 1: Initialize Pinecone client
print("Initializing Pinecone...")
pc = Pinecone(api_key=pinecone_api_key)
print("Initialized Pinecone successfully.")

# Step 2: Define your index name and dimension
index_name = 'support-tickets'
correct_dimension = 384  # This matches the model output dimension

# Step 3: Check if the index exists; if not, create it with the correct dimension
indexes = pc.list_indexes().names()
print("Existing indexes:", indexes)
if index_name not in indexes:
    print(f"Creating index '{index_name}' with dimension {correct_dimension}...")
    pc.create_index(
        name=index_name,
        dimension=correct_dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
else:
    print(f"Index '{index_name}' already exists.")

# Step 4: Connect to the existing index using the provided host link
try:
    index = pc.Index(index_name, host=pinecone_host)
    print(f"Successfully connected to the index '{index_name}'")
    print(f"Index object type: {type(index)}")  # Debug to ensure correct type
except Exception as e:
    print(f"Failed to connect to the index '{index_name}': {e}")

# Load transformer model and tokenizer for generating embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading transformer model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

# Function to generate embeddings
def embed_text(text):
    """Generates embeddings for a given text using a transformer model."""
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    with torch.no_grad():
        model_output = model(**tokens)
    embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze().tolist()
    return embedding

# Step 5: Load the ticket data from Excel file
input_file = 'support_tickets_full.xlsx'  # Make sure this file is in the same directory or provide the full path
df = pd.read_excel(input_file)
print(f"Loaded {len(df)} tickets from {input_file}.")

# Step 6: Generate embeddings and upsert to Pinecone in batches
batch_size = 50  # Define the batch size
vectors_to_upsert = []

for idx, row in df.iterrows():
    ticket_id = str(row['Ticket ID'])
    combined_text = f"Subject: {row['Subject']}. Messages: {row['Messages']}"
    embedding = embed_text(combined_text)
    
    # Prepare vector data for upsert, including the full message as metadata
    vectors_to_upsert.append({
        "id": ticket_id,
        "values": embedding,
        "metadata": {
            "subject": row['Subject'],
            "client_name": row['Client Name'],
            "priority": row['Priority'],
            "status": row['Status'],
            "message": row['Messages']  # Add the full message as part of the metadata
        }
    })
    
    # Upsert in batches
    if len(vectors_to_upsert) >= batch_size:
        try:
            print(f"Upserting a batch of {len(vectors_to_upsert)} vectors...")
            index.upsert(vectors=vectors_to_upsert)
            print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")
        except Exception as e:
            print(f"Failed to upsert batch of vectors: {e}")
        # Clear the batch after upserting
        vectors_to_upsert = []

# Upsert any remaining vectors
if vectors_to_upsert:
    try:
        print(f"Upserting the remaining {len(vectors_to_upsert)} vectors...")
        index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully upserted {len(vectors_to_upsert)} remaining vectors.")
    except Exception as e:
        print(f"Failed to upsert remaining batch of vectors: {e}")

print("All ticket data uploaded to Pinecone successfully.")

