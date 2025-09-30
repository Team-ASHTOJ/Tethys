import xarray as xr

ds = xr.open_dataset("D2900228_470.nc")

print(ds)

# print("\nParameters recorded:\n", ds["DATA_TYPE"].values)