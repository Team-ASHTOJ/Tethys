#To process source data, create embeddings, and store them in the vector DB.
# Pulls data from Postgre, processes and stores them into a vector dB
import chromadb 
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import pandas as pd


# --- Step 1: Connect to PostgreSQL and fetch data ---
print("Connecting to PostgreSQL and fetching data...")
engine = create_engine("postgresql://shaikmohammedomar@localhost:5432/argo") # Use your correct username
# Make sure the table name 'profiles' is correct
df = pd.read_sql("SELECT * FROM profiles LIMIT 100", engine)

# Convert the 'juld' column to a more readable date format
df['juld'] = pd.to_datetime(df['juld']).dt.strftime('%Y-%m-%d')


# --- Step 2: Generate summaries using CORRECT column names ---
print("Generating text summaries for each profile...")
summaries = [
    f"Argo float platform {row['platform_number']}, profile index {row['profile_idx']}, "
    f"located at latitude {row['latitude']:.3f}, longitude {row['longitude']:.3f} on {row['juld']}. "
    f"Measurement: Temperature {row['temperature']:.2f}°C, Salinity {row['salinity']:.2f} PSU, "
    f"Pressure {row['pressure']:.1f} dbar."
    for _, row in df.iterrows()
]
# Create unique IDs for ChromaDB
doc_ids = [f"{row['platform_number']}_{row['profile_idx']}_{row['pressure']}" for _, row in df.iterrows()]


# --- Step 3: Load embedding model and create embeddings ---
print("Loading sentence transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Creating embeddings for summaries...")
embeddings = model.encode(summaries, show_progress_bar=True)


# --- Step 4: Store in ChromaDB (efficiently) ---
print("Connecting to ChromaDB and storing data...")
# Use a persistent client to save data to disk
client = chromadb.PersistentClient(path="./chroma_db") 


# Get or create the collection
collection = client.get_or_create_collection("argo_profiles")

# Bulk add all data at once (much faster than a loop)
collection.add(
    ids=doc_ids,
    documents=summaries,
    embeddings=embeddings.tolist() # Convert numpy array to list
)

print(f"✅ Pushed {collection.count()} profiles into Chroma vector DB")

# --- Example Query to Test ---
# query_text = "ocean temperature near the equator"
# query_embedding = model.encode(query_text).tolist()

# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=3
# )
# print("\n🔍 Example Query Results for:", query_text)
# print(results['documents'])

# query = "show me profiles near 12N, 68E in September 2025"
# q_emb = model.encode([query])

# results = collection.query(query_embeddings=q_emb, n_results=3)
# print(results)

## Queries are just for TESTING here.