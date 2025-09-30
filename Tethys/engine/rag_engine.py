# rag_engine.py
# A self-contained module for the RAG (Retrieval-Augmented Generation) pipeline.

import os
import re
import sqlparse
import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import InferenceClient
from typing import Optional

# --- 1. CONFIGURATION ---
# Load secrets and configurations from environment variables for security
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://shaikmohammedomar@localhost:5432/argo")

# Model and DB path configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "argo_profiles"


# --- 2. PROMPT TEMPLATES ---
SQL_PROMPT_TEMPLATE = """
[INST] You are an expert assistant that translates natural language into a simple and efficient PostgreSQL query.

Follow these examples to construct your query.
---
Good Query Examples:
User Query: "Show me the temperature profile of the deepest measurements in the South Atlantic in 2004"
SQL: SELECT pressure, temperature FROM profiles WHERE latitude BETWEEN -40 AND -10 AND juld BETWEEN '2004-01-01' AND '2004-12-31' AND temp_qc = '1' AND psal_qc = '1' ORDER BY pressure DESC LIMIT 50;

User Query: "Find the location of float 1900121 in late 2002"
SQL: SELECT latitude, longitude, juld FROM profiles WHERE platform_number = 1900121 AND juld BETWEEN '2002-10-01' AND '2002-12-31' AND temp_qc = '1' AND psal_qc = '1' LIMIT 50;
---

Database schema:
Table: profiles (platform_number int, juld timestamptz, latitude double, longitude double, pressure double, temperature double, salinity double, temp_qc varchar, psal_qc varchar, pres_qc varchar)

User request: {user_query}
Relevant context documents (for guidance only): {context}

Instructions:
- CRITICAL: Generate ONLY the SQL query. No explanations, no markdown, no comments.
- Use simple `WHERE ... BETWEEN ...` conditions for ranges.
- Always filter for good QC: `temp_qc='1'` AND `psal_qc='1'`.
- Limit results to 50 rows for performance.
- For "deepest", use `ORDER BY pressure DESC`.
[/INST]
"""

SUMMARY_PROMPT_TEMPLATE = """
[INST] You are a data science assistant. Based on the user's original query and the data returned, provide a concise, one-sentence summary of the findings.

Original User Query: "{user_query}"

Data Returned (first 5 rows):
{data_head}

Instructions:
- Summarize the key information from the data in a single, easy-to-understand sentence.
- Do not describe the columns. Describe the data's meaning.
- Example: "The data shows 5 deep-sea temperature measurements in the South Atlantic, with temperatures around 2.5°C at pressures over 5000 dbar."
[/INST]
"""


# --- 3. THE RAG ENGINE CLASS ---
class RAG_Engine:
    """A class to encapsulate all components of the RAG pipeline."""

    def __init__(self):
        """Initializes all clients and models once."""
        print("Initializing RAG Engine...")
        # LLM Client
        self.llm_client = InferenceClient(model=LLM_MODEL_NAME, token=HF_TOKEN)
        # Database Engine
        self.db_engine = create_engine(DB_CONNECTION_STRING)
        # Embedding Model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # ChromaDB Client and Collection
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = chroma_client.get_collection(COLLECTION_NAME)
        print("✅ RAG Engine Initialized Successfully.")

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieves relevant document summaries from ChromaDB."""
        print(f"Retrieving context for query: '{query}'")
        q_emb = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents"]
        )
        context = "\n".join(results["documents"][0])
        return context

    def generate_sql(self, query: str, context: str) -> str:
        """Generates a SQL query using the LLM based on the user query and context."""
        print("Generating SQL query...")
        prompt = SQL_PROMPT_TEMPLATE.format(user_query=query, context=context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm_client.chat_completion(messages, max_tokens=500, temperature=0.0)
        sql_query = response.choices[0].message.content.strip()

        # Clean the response to ensure it's only a SQL query
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].strip()
        if "```" in sql_query:
            sql_query = sql_query.split("```")[0].strip()
        if ';' in sql_query:
            sql_query = sql_query.split(';')[0].strip() + ';'
            
        return sql_query

    def _sanitize_sql(self, sql: str) -> str:
        """Strips comments and checks for forbidden keywords."""
        parsed = sqlparse.format(sql, keyword_case='upper', strip_comments=True).strip()
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
        for f in forbidden:
            if re.search(r"\b" + f + r"\b", parsed, flags=re.IGNORECASE):
                raise ValueError(f"Forbidden SQL keyword found: {f}")
        if not re.search(r"^\s*SELECT\b", parsed, flags=re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        return parsed

    def execute_sql(self, sql: str) -> Optional[pd.DataFrame]:
        """Executes the sanitized SQL query against the database."""
        print(f"Executing SQL:\n{sql}")
        try:
            safe_sql = self._sanitize_sql(sql)
            with self.db_engine.connect() as conn:
                df = pd.read_sql(text(safe_sql), conn)
            return df
        except Exception as e:
            print(f"❌ SQL Execution Error: {e}")
            # In a real app, you might want to raise the exception or return a specific error message
            return None

    def summarize_results(self, query: str, df: pd.DataFrame) -> str:
        """Generates a natural language summary of the query results."""
        if df.empty:
            return "The query returned no results."
        
        print("Summarizing results...")
        prompt = SUMMARY_PROMPT_TEMPLATE.format(user_query=query, data_head=df.head().to_string())
        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm_client.chat_completion(messages, max_tokens=150, temperature=0.1)
        summary = response.choices[0].message.content.strip()
        return summary
        
# --- 4. EXAMPLE USAGE ---
if __name__ == "__main__":
    # This block allows you to test the engine directly
    engine = RAG_Engine()
    
    # test_query = "show me profiles with high pressure"
    test_query = "What were the deepest temperature measurements in the South Atlantic in 2004?"

    print("\n--- 1. Retrieving Context ---")
    retrieved_context = engine.retrieve_context(test_query)
    print(f"Context:\n{retrieved_context}")

    print("\n--- 2. Generating SQL ---")
    generated_sql = engine.generate_sql(test_query, retrieved_context)
    print(f"SQL:\n{generated_sql}")

    print("\n--- 3. Executing SQL ---")
    results_df = engine.execute_sql(generated_sql)
    if results_df is not None:
        print(f"Results DataFrame ({len(results_df)} rows):")
        print(results_df.head())

        print("\n--- 4. Summarizing Results ---")
        summary = engine.summarize_results(test_query, results_df)
        print(f"Summary:\n{summary}")
    else:
        print("Could not retrieve results to summarize.")