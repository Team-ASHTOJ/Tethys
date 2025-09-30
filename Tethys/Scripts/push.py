from sqlalchemy import create_engine, text
import pandas as pd

# Load parquet
df = pd.read_parquet("argo_profile.parquet")

# --- FIX IS HERE ---
# Ensure the QC flag columns are treated as strings
# This tells pandas/SQLAlchemy to create a text-based column in PostgreSQL
for col in ['temp_qc', 'psal_qc', 'pres_qc']:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Connect to PostgreSQL
engine = create_engine("postgresql://shaikmohammedomar@localhost:5432/argo")

# Use "replace" to drop the old table and create a new one with the correct types
df.to_sql("profiles", engine, if_exists="replace", index=False)

print("✅ Data pushed into PostgreSQL with corrected data types!")

# Verify the result from Python
with engine.connect() as connection:
    result = connection.execute(text("SELECT temp_qc, psal_qc, pres_qc FROM profiles LIMIT 5;"))
    print("\nVerification from DB:")
    for row in result:
        print(row)