from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load pre-trained text model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_text_embedding(text):
    """Extract text embeddings using a pre-trained model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Example text data
descriptions = [
    "A 3-bedroom house with a garden in Tokyo.",
    "A modern apartment in Osaka close to public transport.",
    "A traditional house in Kyoto with a beautiful backyard."
]

# Generate embeddings for the descriptions
embeddings = np.array([get_text_embedding(desc)[0] for desc in descriptions])

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)

# Query example
query = "Looking for a house in Tokyo with a garden."
query_embedding = get_text_embedding(query)

# Search the index
distances, indices = index.search(query_embedding, k=2)
print("Query:", query)
print("Top matches:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {descriptions[idx]} (distance: {distances[0][i]:.2f})")
