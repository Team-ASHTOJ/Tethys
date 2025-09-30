import xarray as xr
import pandas as pd

# Load the profile file
ds = xr.open_dataset("7902251_prof.nc")

# Extract the main parameters (check if they exist first)
pres = ds["PRES"].values.flatten() if "PRES" in ds else None
temp = ds["TEMP"].values.flatten() if "TEMP" in ds else None
psal = ds["PSAL"].values.flatten() if "PSAL" in ds else None

# Create a DataFrame for readability
data = {
    "Pressure(dbar)": pres,
    "Temperature(°C)": temp,
    "Salinity(PSU)": psal
}

df = pd.DataFrame(data)

# Save to a TXT file in table format
with open("table1.txt", "w") as f:
    f.write(df.to_string(index=False))

print("✅ Saved profile data to table1.txt")