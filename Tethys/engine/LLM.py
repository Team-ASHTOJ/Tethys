# LLM.py
from retrieval import retrieve_docs
from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

# BEST PRACTICE: Use the model's preferred instruction format ([INST])
PROMPT_TEMPLATE = """
[INST] You are an assistant that translates natural language into SQL queries for an oceanographic Postgres database.

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
- Only use the columns listed in the schema.
- Always filter for good QC: `temp_qc='1'` AND `psal_qc='1'`.
- Limit results to 20 rows.
- Respond ONLY with a single, complete SQL query. Do not add explanations or surrounding text.
[/INST]
"""

def generate_sql(user_query):
    # Step 1: retrieve top-k context
    docs = retrieve_docs(user_query, k=3)
    retrieved_docs = "\n".join(docs["documents"][0])
    
    # Step 2: build the prompt string
    prompt = PROMPT_TEMPLATE.format(user_query=user_query, retrieved_docs=retrieved_docs)
    
    # --- CHANGE 1: Format the prompt for the chat_completion method ---
    # It expects a list of messages, not a single string.
    messages = [{"role": "user", "content": prompt}]
    
    # --- CHANGE 2: Call the correct method: chat_completion ---
    response = client.chat_completion(
        messages,
        max_tokens=500,
        temperature=0.1,
    )
    
    # --- CHANGE 3: Extract the content from the response object ---
    sql_query = response.choices[0].message.content
    
    # Optional: Clean up potential markdown formatting from the model
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].strip()
    if "```" in sql_query:
        sql_query = sql_query.split("```")[0].strip()

    return sql_query

if __name__ == "__main__":
    query = "Show me salinity profiles near the equator in March 2023"
    sql_query = generate_sql(query)
    print("--- Generated SQL ---")
    print(sql_query)