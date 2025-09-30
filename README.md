Tethys: Conversational ARGO Float Data Explorer 🌊💬

Tethys is a conversational AI system designed to make exploring ARGO float oceanographic data simple, intuitive, and interactive. Instead of wrestling with NetCDF files or SQL queries, users can ask questions in plain English and receive answers, summaries, and visualizations.

This project bridges the gap between vast ocean datasets and the researchers, students, and enthusiasts who want to understand them quickly and effectively.

⸻

🚀 Features
	•	Natural Language Queries
Ask: “Show me active floats in the Indian Ocean with high salinity readings last month.”
Get: A direct, readable answer.
	•	Interactive Map Visualizations
View float locations, trajectories, and parameters (temperature, salinity, pressure, etc.) on a world map.
	•	Data Summarization
Generate quick summaries of float data, missions, or regional trends.
	•	Semantic Search (Vector DB)
Retrieve relevant information using embeddings, even if your query doesn’t match exact keywords.

🏗️ System Architecture

Tethys is powered by a Retrieval-Augmented Generation (RAG) pipeline:
	1.	Data Ingestion
	•	Download and preprocess ARGO NetCDF files.
	•	Convert them into Parquet format and store them in PostgreSQL.
	•	Create embeddings and push them into a Chroma/FAISS vector database.
	2.	User Query
	•	User interacts through the frontend dashboard (chat + map).
	3.	Backend Engine
	•	Query is embedded → relevant chunks retrieved from vector DB + PostgreSQL.
	4.	LLM Augmentation
	•	Retrieved data + query are sent to a language model.
	•	Generates a coherent, human-readable response.
	5.	Response & Visualization
	•	Final answer and visual data (maps/graphs) are sent back to the frontend.

⸻

📊 Example Queries
	•	“Show me floats near the Bay of Bengal with high surface temperature.”
	•	“Summarize salinity trends in the Arabian Sea in 2023.”
	•	“List active floats with depth > 1000m in the Indian Ocean.”

  ⚙️ Tech Stack
	•	Data Handling: Python, xarray, pandas, parquet
	•	Databases: PostgreSQL (relational), Chroma/FAISS (vector)
	•	Embeddings/LLM: Sentence Transformers / HuggingFace models
	•	Frontend: Streamlit, Plotly, Leaflet
	•	Backend: Retrieval-Augmented Generation (RAG)
