# main_app.py

# === IMPORTS from all components ===
import os
import re
import sqlparse
import pandas as pd
from sqlalchemy import create_engine, text
from huggingface_hub import InferenceClient
from retrieval import retrieve_docs # Your existing retrieval function

# === CONFIGURATION ===
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
DB_CONNECTION_STRING = "postgresql://shaikmohammedomar@localhost:5432/argo"

# --- COMPONENT 1: The Translator (LLM) ---
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

### CHANGED: The prompt is now smarter about how to use context ###
PROMPT_TEMPLATE = """
[INST] You are an expert assistant that translates natural language into SQL queries for an oceanographic Postgres database.

Database schema:
Table: profiles
Columns:
- platform_number (int)
- profile_idx (int)
- juld (timestamptz)
- latitude (double precision)
- longitude (double precision)
- pressure (double precision)
- temperature (double precision)
- salinity (double precision)
- temp_qc (varchar)
- psal_qc (varchar)
- pres_qc (varchar)

User request:
{user_query}

Relevant context documents to help you:
{retrieved_docs}

Instructions:
- Use the context documents for guidance on locations, dates, or typical values, but DO NOT filter by `platform_number` unless the user explicitly asks for a specific float.
- Select only the columns most relevant to the user's request, not always `SELECT *`.
- Always filter for good QC: `temp_qc='1'` AND `psal_qc='1'`.
- Limit results to 20 rows.
- Respond ONLY with a single, complete SQL query. Do not add explanations or surrounding text.
[/INST]
"""

def generate_sql(user_query):
    docs = retrieve_docs(user_query, k=3)
    retrieved_docs = "\n".join(docs["documents"][0])
    prompt = PROMPT_TEMPLATE.format(user_query=user_query, retrieved_docs=retrieved_docs)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(messages, max_tokens=500, temperature=0.1)
    sql_query = response.choices[0].message.content
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].strip()
    if "```" in sql_query:
        sql_query = sql_query.split("```")[0].strip()
    return sql_query

# --- COMPONENT 2: The Security Guard (SQL Sanitizer) ---
# This component is well-written and requires no changes.
ALLOWED_TABLES = {"profiles"}
def sanitize_sql(sql: str) -> str:
    parsed = sqlparse.format(sql, keyword_case='upper', strip_comments=True).strip()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE", ";"]
    for f in forbidden:
        if re.search(r"\b" + f + r"\b", parsed, flags=re.IGNORECASE):
            raise ValueError(f"Forbidden SQL keyword found: {f}")
    if not re.search(r"^\s*SELECT\b", parsed, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    from_clause = False
    for token in sqlparse.parse(parsed)[0].tokens:
        if token.is_keyword and token.value.upper() == 'FROM':
            from_clause = True; continue
        if from_clause and token.ttype is None:
             if token.value.lower() not in ALLOWED_TABLES:
                 raise ValueError(f"Table '{token.value}' is not allowed.")
             from_clause = False
    if "LIMIT" not in parsed.upper():
        parsed += " LIMIT 1000"
    return parsed

# --- COMPONENT 3: The Executor (DB Runner) ---
# This component is well-written and requires no changes.
engine = create_engine(DB_CONNECTION_STRING)
def run_sql_fetch(sql: str) -> pd.DataFrame:
    print("\n--- Sanitizing and Executing SQL ---")
    safe_sql = sanitize_sql(sql)
    print(f"Safe SQL to be executed:\n{safe_sql}")
    with engine.connect() as conn:
        df = pd.read_sql(text(safe_sql), conn)
    return df

# === THE MAIN WORKFLOW ===
if __name__ == "__main__":
    ### CHANGED: Made the workflow interactive ###
    print("Welcome to the Argo Natural Language Query Interface.")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        # 1. Get user input
        query = input("\nPlease enter your query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        try:
            # 2. The Translator
            print(f"\n--- Generating SQL for query: '{query}' ---")
            raw_sql = generate_sql(query)
            print(f"Raw SQL from LLM:\n{raw_sql}")
            
            # 3 & 4. The Security Guard and Executor
            results_df = run_sql_fetch(raw_sql)
            
            # 5. The Presenter
            print("\n--- Results from Database ---")
            if results_df.empty:
                print("The query returned no results.")
            else:
                print(results_df)

        except ValueError as ve:
            print(f"\n--- SECURITY ERROR ---: {ve}")
        except Exception as e:
            print(f"\n--- An unexpected error occurred ---: {e}")