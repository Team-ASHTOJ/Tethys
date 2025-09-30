# streamlit_app.py

# === IMPORTS (No changes) ===
import os, re, sqlparse, pandas as pd, streamlit as st, plotly.express as px
from sqlalchemy import create_engine, text
from huggingface_hub import InferenceClient
from retrieval import retrieve_docs

# === PAGE CONFIGURATION (No changes) ===
st.set_page_config(page_title="ARGO Ocean Data Assistant", page_icon="🌊", layout="wide")

# === CONFIGURATION & CONNECTIONS (No changes) ===
@st.cache_resource
def get_connections():
    HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
    DB_CONNECTION_STRING = "postgresql://shaikmohammedomar@localhost:5432/argo"
    llm_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)
    db_engine = create_engine(DB_CONNECTION_STRING)
    return llm_client, db_engine
llm_client, db_engine = get_connections()

# === BACKEND LOGIC ===

# Updated prompt to be more explicit about output format
PROMPT_TEMPLATE = """
[INST] You are an expert assistant that translates natural language into PostgreSQL queries.

Database schema:
Table: profiles
Columns:
- platform_number (int), juld (timestamptz), latitude (double precision), longitude (double precision),
  pressure (double precision), temperature (double precision), salinity (double precision),
  temp_qc (varchar), psal_qc (varchar), pres_qc (varchar)

Good Query Examples:
1. "Show me temperature profiles in South Atlantic"
   SQL: SELECT pressure, temperature FROM profiles WHERE latitude BETWEEN -40 AND -10 AND temp_qc = '1' ORDER BY pressure DESC LIMIT 50;

2. "Find float 1900121 locations in 2002"
   SQL: SELECT latitude, longitude, juld FROM profiles WHERE platform_number = 1900121 AND juld BETWEEN '2002-01-01' AND '2002-12-31' LIMIT 50;

Data Summary:
{data_summary}

User request: {user_query}

Relevant context: {retrieved_docs}

Instructions:
- Generate ONLY a SQL query, nothing else
- Use simple WHERE clauses with BETWEEN for ranges
- Always include temp_qc='1' AND psal_qc='1' for quality filtering
- Limit to 50 rows
- For deepest measurements, use ORDER BY pressure DESC
- NO explanations, NO comments, NO markdown, ONLY the SQL query
[/INST]
"""

@st.cache_data
def get_data_summary(_engine):
    query = "SELECT MIN(juld) as min_date, MAX(juld) as max_date, MIN(latitude) as min_lat, MAX(latitude) as max_lat FROM profiles;"
    with _engine.connect() as conn: return pd.read_sql(text(query), conn)

# FIXED: Enhanced SQL extraction to remove any non-SQL content
@st.cache_data
def generate_sql(_engine, user_query):
    summary_df = get_data_summary(_engine)
    summary_text = summary_df.to_string(index=False)
    docs = retrieve_docs(user_query, k=3)
    retrieved_docs = "\n".join(docs["documents"][0])
    prompt = PROMPT_TEMPLATE.format(data_summary=summary_text, user_query=user_query, retrieved_docs=retrieved_docs)
    messages = [{"role": "user", "content": prompt}]
    response = llm_client.chat_completion(messages, max_tokens=500, temperature=0.0)
    sql_query = response.choices[0].message.content.strip()
    
    # Clean the SQL query more aggressively
    # Remove code blocks
    if "```sql" in sql_query: 
        sql_query = sql_query.split("```sql")[1].strip()
    if "```" in sql_query: 
        sql_query = sql_query.split("```")[0].strip()
    
    # Remove any lines that start with [ (explanations)
    lines = sql_query.split('\n')
    clean_lines = []
    for line in lines:
        # Stop if we hit an explanation marker
        if line.strip().startswith('['):
            break
        clean_lines.append(line)
    sql_query = '\n'.join(clean_lines).strip()
    
    # Remove trailing semicolon and any text after it
    if ';' in sql_query:
        sql_query = sql_query.split(';')[0] + ';'
    
    return sql_query

def sanitize_sql(sql: str) -> str:
    parsed = sqlparse.format(sql, keyword_case='upper', strip_comments=True).strip()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    for f in forbidden:
        if re.search(r"\b" + f + r"\b", parsed, flags=re.IGNORECASE): 
            raise ValueError(f"Forbidden SQL keyword found: {f}")
    if not re.search(r"^\s*(SELECT|WITH)\b", parsed, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    if "LIMIT" not in parsed.upper(): 
        parsed += " LIMIT 1000"
    return parsed

@st.cache_data
def run_sql_fetch(_engine, sql: str) -> pd.DataFrame:
    safe_sql = sanitize_sql(sql)
    with _engine.connect() as conn:
        df = pd.read_sql(text(safe_sql), conn)
    return df

def visualize(df):
    st.write("### Visualization")
    if df.empty: return
    if "latitude" in df.columns and "longitude" in df.columns: 
        st.map(df, latitude='latitude', longitude='longitude')
    elif "pressure" in df.columns and "temperature" in df.columns:
        fig = px.line(df.sort_values('pressure'), x="temperature", y="pressure", title="Temperature Profile")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else: 
        st.dataframe(df)

# === STREAMLIT UI (No changes needed) ===
st.title("🌊 ARGO Ocean Data Assistant")
st.caption("A data-aware assistant for the global ARGO ocean float database.")
with st.expander("ℹ️ Click to see available data ranges"):
    summary = get_data_summary(db_engine)
    if not summary.empty:
        summary['min_date'] = pd.to_datetime(summary['min_date']).dt.strftime('%Y-%m-%d')
        summary['max_date'] = pd.to_datetime(summary['max_date']).dt.strftime('%Y-%m-%d')
        st.dataframe(summary)
if "messages" not in st.session_state: 
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])
if prompt := st.chat_input("Ask about ocean data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                raw_sql = generate_sql(db_engine, prompt)
                st.write("🔍 **Generated SQL Query:**")
                st.code(raw_sql, language="sql")
                results_df = run_sql_fetch(db_engine, raw_sql)
            row_count = len(results_df)
            st.metric(label="Rows Returned", value=row_count)
            if results_df.empty:
                st.warning("**The query returned no results.** Try asking a question that fits within the available data ranges shown above.")
            else:
                st.dataframe(results_df)
                visualize(results_df)
            st.session_state.messages.append({"role": "assistant", "content": f"Found {row_count} results for your query."})
        except ValueError as ve: 
            st.error(f"Security Error: {ve}")
        except Exception as e: 
            st.error(f"An error occurred: {e}")