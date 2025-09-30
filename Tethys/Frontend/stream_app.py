# streamlit_app.py

# === 1. IMPORTS ===
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from rag_engine import RAG_Engine # Our new, clean backend

# === 2. CONFIGURATION ===
# Directory where the initial Parquet file is stored for the overview map
PARQUET_DIR = './' # Assuming a file like 'argo_all_profiles_combined.parquet' is here

st.set_page_config(
    page_title="ARGO Conversational Dashboard",
    page_icon="🌊",
    layout="wide"
)
st.title("🌊 ARGO Conversational Dashboard")
st.caption("An AI-powered assistant for the global ARGO ocean float database.")


# === 3. INITIALIZE THE RAG ENGINE (CACHED) ===
@st.cache_resource
def get_engine():
    """Initializes the RAG_Engine once and caches it."""
    return RAG_Engine()

engine = get_engine()


# === 4. HELPER FUNCTIONS (from the dashboard app) ===
def safe_column(df: pd.DataFrame, candidates: list) -> str:
    """Finds the first valid column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_db_metadata(_engine: RAG_Engine):
    """Fetches metadata directly from the RAG engine's ChromaDB collection."""
    try:
        count = _engine.collection.count()
        return {"n_profiles": count}
    except Exception:
        return None

def visualize(df: pd.DataFrame):
    """Intelligently visualizes the dataframe based on its columns."""
    st.write("### 📊 Visualization")
    if df.empty:
        st.info("No data to visualize.")
        return

    lat_col = safe_column(df, ["latitude", "LATITUDE"])
    lon_col = safe_column(df, ["longitude", "LONGITUDE"])
    temp_col = safe_column(df, ["temperature", "TEMP"])
    pressure_col = safe_column(df, ["pressure", "PRES"])

    # Priority 1: If lat/lon are present, show a map
    if lat_col and lon_col:
        st.map(df, latitude=lat_col, longitude=lon_col)
    # Priority 2: If pressure/temp are present, show a profile plot
    elif pressure_col and temp_col:
        fig = px.line(df.sort_values(pressure_col), x=temp_col, y=pressure_col, title="Temperature vs. Pressure Profile")
        fig.update_yaxes(autorange="reversed", title_text="Pressure (dbar)")
        fig.update_xaxes(title_text="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Could not generate a specific plot. Displaying raw data.")


# === 5. SIDEBAR UI ===
with st.sidebar:
    st.header("Chat with ARGO Data")
    user_query = st.text_area("Ask a question about ocean data:", height=100)
    k_slider = st.slider("Number of context documents to retrieve:", 1, 10, 3)

    if st.button("Submit Query"):
        if user_query.strip():
            st.session_state.processing = True # Flag to trigger main page processing
            st.session_state.user_query = user_query
            st.session_state.k = k_slider
        else:
            st.warning("Please enter a question.")

    st.markdown("---")
    st.subheader("Vector DB Info")
    meta = get_db_metadata(engine)
    if meta:
        st.write(f"Profiles Indexed: **{meta.get('n_profiles', 'N/A')}**")
    else:
        st.error("Could not connect to the vector database.")


# === 6. MAIN PAGE LAYOUT ===

### --- MODIFIED SECTION: Interactive Overview Map --- ###
st.subheader("Global ARGO Profile Locations Overview")
try:
    parquet_files = [f for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet')]
    if parquet_files:
        overview_df = pd.read_parquet(os.path.join(PARQUET_DIR, parquet_files[0]))

        # Find all necessary columns for the interactive map
        lat_col = safe_column(overview_df, ["latitude", "LATITUDE"])
        lon_col = safe_column(overview_df, ["longitude", "LONGITUDE"])
        platform_col = safe_column(overview_df, ["platform_number", "PLATFORM_NUMBER"])
        date_col = safe_column(overview_df, ["juld", "JULD", "date"])
        temp_col = safe_column(overview_df, ["temperature", "TEMP"])

        if lat_col and lon_col and platform_col and date_col:
            # Use a smaller sample for performance if the file is large
            sample_df = overview_df.sample(n=10000) if len(overview_df) > 10000 else overview_df
            
            # Ensure date column is in a readable format
            sample_df[date_col] = pd.to_datetime(sample_df[date_col]).dt.strftime('%Y-%m-%d')
            
            # Create a custom hover text column
            sample_df['hover_text'] = (
                "<b>Float ID:</b> " + sample_df[platform_col].astype(str) + "<br>" +
                "<b>Date:</b> " + sample_df[date_col].astype(str)
            )

            # Create the interactive map with Plotly
            fig = px.scatter_mapbox(
                sample_df,
                lat=lat_col,
                lon=lon_col,
                color=temp_col,
                hover_name='hover_text',
                hover_data={lat_col: ':.3f', lon_col: ':.3f', temp_col: ':.2f'}, # Show extra data on hover
                color_continuous_scale=px.colors.sequential.Viridis,
                zoom=1,
                mapbox_style="carto-positron",
                title="Hover over points for details"
            )
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Could not find required columns (latitude, longitude, platform_number, juld) for an interactive map.")
            # Fallback to the simple map if essential columns are missing
            if lat_col and lon_col:
                 st.map(overview_df.sample(n=5000) if len(overview_df) > 5000 else overview_df, latitude=lat_col, longitude=lon_col)

    else:
        st.warning("No Parquet files found for overview map. Run the ingestion script.")
except FileNotFoundError:
    st.error(f"Parquet directory not found: {PARQUET_DIR}")
except Exception as e:
    st.error(f"Error loading overview data: {e}")

st.markdown("---")
### --- END OF MODIFIED SECTION --- ###


# --- Chat Response Area ---
st.subheader("Query Results")

# This block runs only when the "Submit" button is clicked in the sidebar
if st.session_state.get('processing', False):
    query = st.session_state.user_query
    k = st.session_state.k

    with st.spinner("Analyzing your query... This may take a moment."):
        try:
            # 1. Retrieve context
            context = engine.retrieve_context(query, k=k)
            
            # 2. Generate SQL
            sql_query = engine.generate_sql(query, context)
            
            # 3. Execute SQL
            results_df = engine.execute_sql(sql_query)
            
            # 4. Summarize results
            summary = "Query executed successfully." # Default summary
            if results_df is not None and not results_df.empty:
                summary = engine.summarize_results(query, results_df)

            # Store results to display
            st.session_state.last_result = {
                "summary": summary,
                "sql": sql_query,
                "df": results_df,
            }

        except Exception as e:
            st.error(f"An error occurred in the RAG pipeline: {e}")
            st.session_state.last_result = None

    # Reset the processing flag
    st.session_state.processing = False


# Display the last result stored in the session state
if 'last_result' in st.session_state and st.session_state.last_result:
    result = st.session_state.last_result
    
    st.markdown(f"##### 💬 Summary")
    st.info(result['summary'])
    
    with st.expander("🔍 View Generated SQL Query"):
        st.code(result['sql'], language="sql")
        
    df = result['df']
    if df is not None and not df.empty:
        st.markdown(f"##### 📈 Data ({len(df)} rows)")
        st.dataframe(df)
        visualize(df)
    else:
        st.warning("The query returned no results. Try rephrasing your question or adjusting the data ranges.")
else:
    st.info("Ask a question in the sidebar to get started!")