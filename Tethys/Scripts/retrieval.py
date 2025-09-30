# retrieval.py
# Searches the chroma dB and loads the embedding MODEL.
from sentence_transformers import SentenceTransformer
import chromadb

# --- SETUP (should match your ingestion script) ---
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "argo_profiles"

print("Loading sentence transformer model...")
model = SentenceTransformer(MODEL_NAME)

print(f"Connecting to persistent ChromaDB at: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)

print(f"Getting collection: {COLLECTION_NAME}")
collection = client.get_collection(COLLECTION_NAME)

# --- THE RETRIEVAL FUNCTION ---
def retrieve_docs(query, k=5):
    """
    Takes a text query, embeds it, and retrieves the top k most similar documents from ChromaDB.
    """
    print(f"\nSearching for '{query}'...")
    # Encode the query using the same model
    q_emb = model.encode([query]).tolist()
    
    # Query the collection
    res = collection.query(
        query_embeddings=q_emb, 
        n_results=k, 
        include=["documents", "distances"] # "metadatas", "ids" are also available
    )
    
    return res

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Test with a query
    user_query = "show me profiles with high pressure"
    results = retrieve_docs(user_query, k=3)
    
    print("\n--- Top 3 Results ---")
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        print(f"Result {i+1} (Distance: {distance:.4f}):")
        print(f"  {doc}\n")